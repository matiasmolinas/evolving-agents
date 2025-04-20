# evolving_agents/workflow/generate_workflow_tool.py

import json
import logging
from typing import Dict, Any, List, Optional
import re # Added for cleaning output
import yaml # Import yaml for basic validation during generation (optional)

# Pydantic for input validation
from pydantic import BaseModel, Field

# BeeAI Framework imports for Tool structure
from beeai_framework.tools.tool import Tool, StringToolOutput
from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter

# Project-specific imports
from evolving_agents.core.llm_service import LLMService
from evolving_agents.smart_library.smart_library import SmartLibrary
# Import the specific Input model and the config module
from evolving_agents.tools.intent_review.workflow_design_review_tool import WorkflowDesignReviewTool, WorkflowDesignInput
from evolving_agents import config # Import the config module

logger = logging.getLogger(__name__)

# --- Helper Function for Safe JSON Dumping ---
def safe_json_dumps(data: Any, indent: int = 2) -> str:
    """Safely dump data to JSON, handling common non-serializable types."""
    def default_serializer(obj):
        if isinstance(obj, set): return list(obj)
        if obj is Ellipsis: return None
        # Let the default raise the error for other unhandled types
        try:
            # Use ensure_ascii=False for broader character support if needed, otherwise keep default
            return json.JSONEncoder(ensure_ascii=True).encode(obj)
        except TypeError:
             raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    try:
        return json.dumps(data, indent=indent, default=default_serializer)
    except TypeError as e:
         logger.error(f"JSON serialization failed even with custom handler: {e}", exc_info=True)
         return f'{{"error": "Data not fully serializable: {e}"}}'
# ---------------------------------------------


# --- Input Schema ---
class GenerateWorkflowInput(BaseModel):
    """Input schema for the GenerateWorkflowTool."""
    task_objective: str = Field(description="The main goal of the workflow")
    domain: str = Field(description="The application domain (e.g., finance, healthcare)")
    workflow_design: Optional[Dict[str, Any]] = Field(None, description="Detailed workflow design including sequence and data flow (use this OR requirements)")
    library_entries: Optional[Dict[str, Any]] = Field(None, description="Information about components to reuse, evolve, or create (used with workflow_design)")
    requirements: Optional[str] = Field(None, description="Natural language requirements if design/entries are not provided (use this OR workflow_design)")
    workflow_name: Optional[str] = Field("generated_workflow", description="A name for the generated workflow scenario")

# --- Tool Implementation ---
class GenerateWorkflowTool(Tool[GenerateWorkflowInput, None, StringToolOutput]):
    """
    Tool for generating executable YAML workflows based on requirements or designs.
    It uses LLM capabilities to translate high-level descriptions into structured workflow steps.
    """
    name = "GenerateWorkflowTool"
    description = "Generate YAML workflows from requirements or a structured design, using library information"
    input_schema = GenerateWorkflowInput

    def __init__(
        self,
        llm_service: LLMService,
        smart_library: SmartLibrary,
        options: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes the GenerateWorkflowTool.
        """
        super().__init__(options=options or {})
        self.llm = llm_service
        self.library = smart_library
        logger.info("GenerateWorkflowTool initialized")

    def _create_emitter(self) -> Emitter:
        """Creates the event emitter for this tool."""
        return Emitter.root().child(
            namespace=["tool", "workflow", "generate"],
            creator=self,
        )

    async def _run(self, input: GenerateWorkflowInput, options: Optional[Dict[str, Any]] = None, context: Optional[RunContext] = None) -> StringToolOutput:
        """
        Generate a complete YAML workflow based on the provided input.
        """
        logger.info(f"Received request to generate workflow '{input.workflow_name}' in domain '{input.domain}'")
        try:
            # If workflow_design is provided, review it before generating YAML
            if input.workflow_design:
                # First check if review is enabled globally
                intent_review_enabled = getattr(config, "INTENT_REVIEW_ENABLED", False)
                review_levels = getattr(config, "INTENT_REVIEW_LEVELS", [])

                # Check if 'design' review level is enabled
                design_review_active = intent_review_enabled and ("design" in review_levels)

                if design_review_active and context and hasattr(context, "get_value"):
                    # Allow override from context, defaulting to True if design review is active
                    human_review_workflow = context.get_value("human_review_workflow", True)

                    if human_review_workflow:
                        # Create a temporary WorkflowDesignReviewTool
                        design_review_tool = WorkflowDesignReviewTool()

                        logger.info("Workflow design intercepted for human review")

                        # Submit the design for human review using the correct input model
                        review_input = WorkflowDesignInput(
                            design=input.workflow_design,
                            interactive=True # Or determine from context/config
                        )

                        try:
                            review_result = await design_review_tool._run(review_input, options, context)
                            review_data = json.loads(review_result.get_text_content())

                            # Check if the design was approved
                            if review_data.get("status") != "approved":
                                # Design was rejected
                                return StringToolOutput(json.dumps({
                                    "status": "design_rejected",
                                    "message": "The workflow design was rejected and YAML generation was aborted.",
                                    "reason": review_data.get("reason", "No reason provided"),
                                    "original_design": input.workflow_design
                                }, indent=2)) # Added indent for readability

                            # If we get here, the design was approved
                            logger.info("Workflow design approved by human reviewer, proceeding to YAML generation")

                            # Continue with any design modifications from review
                            if "modified_design" in review_data:
                                input.workflow_design = review_data["modified_design"]

                        except Exception as e:
                            logger.error(f"Error during workflow design review: {str(e)}")
                            # Continue with generation despite review error? Decide policy here.
                            # For now, let's continue.
                            logger.warning("Continuing YAML generation despite review error.")

            # Proceed with the existing workflow generation logic

            # Prioritize structured input if available
            if input.workflow_design and input.library_entries:
                logger.info("Generating workflow from structured design and library entries.")
                yaml_workflow_str = await self._generate_from_design(
                    input.workflow_name,
                    input.task_objective,
                    input.domain,
                    input.workflow_design,
                    input.library_entries
                )
            elif input.requirements:
                 logger.info("Generating workflow from natural language requirements.")
                 yaml_workflow_str = await self._generate_from_requirements(
                     input.workflow_name,
                     input.task_objective,
                     input.domain,
                     input.requirements
                 )
            else:
                # If using GenerateWorkflowTool directly, one of the inputs is required
                # If called by SystemAgent internally based on a design, workflow_design should always be present
                # Add a check or rely on SystemAgent to provide correct input
                logger.warning("GenerateWorkflowTool called without design or requirements. This might happen if SystemAgent failed to pass the design.")
                # Attempt generation based on objective/domain as a fallback
                if input.task_objective and input.domain:
                     yaml_workflow_str = await self._generate_from_requirements(
                         input.workflow_name,
                         input.task_objective,
                         input.domain,
                         f"Generate a workflow to achieve the objective: {input.task_objective}" # Fallback requirements
                     )
                else:
                    raise ValueError("Insufficient input: Need workflow_design/library_entries OR requirements OR task_objective/domain.")


            # Clean YAML response from LLM
            cleaned_yaml = self._clean_yaml_output(yaml_workflow_str)
            if not cleaned_yaml:
                 raise ValueError("LLM response did not contain extractable YAML content.")

            # **Optional Basic Validation:** Try parsing the generated YAML here to catch errors early
            try:
                yaml.safe_load(cleaned_yaml)
                logger.info("Generated YAML parsed successfully (basic validation).")
            except yaml.YAMLError as e:
                 logger.error(f"Generated YAML is invalid: {e}")
                 logger.debug(f"Invalid YAML content:\n---\n{cleaned_yaml}\n---")
                 # Return error or attempt correction? For now, return error.
                 raise ValueError(f"Generated YAML failed basic validation: {e}") from e

            logger.info(f"Successfully generated YAML for workflow '{input.workflow_name}'.")
            # Return the YAML string directly
            return StringToolOutput(cleaned_yaml) # Return raw YAML string

        except Exception as e:
            import traceback
            logger.error(f"Error generating workflow: {str(e)}")
            # Return error message as JSON string
            error_output = safe_json_dumps({
                "status": "error",
                "message": f"Error generating workflow: {str(e)}",
                "details": traceback.format_exc()
            })
            return StringToolOutput(error_output)


    async def _generate_from_design(
        self,
        workflow_name: str,
        task_objective: str,
        domain: str,
        workflow_design: Dict[str, Any],
        library_entries: Dict[str, Any]
    ) -> str:
        """Generate YAML from structured design and library info using LLM."""
        reuse_components = library_entries.get("reuse", [])
        evolve_components = library_entries.get("evolve", [])
        create_components = library_entries.get("create", [])

        # Construct the prompt with EVEN MORE detailed instructions, especially for RETURN
        workflow_prompt = f"""
        Generate a complete YAML workflow for the task: '{task_objective}' in domain '{domain}'.

        Workflow Name: {workflow_name}

        Components to Reuse: {safe_json_dumps(reuse_components)}
        Components to Evolve: {safe_json_dumps(evolve_components)}
        Components to Create: {safe_json_dumps(create_components)}

        Workflow Logic Sequence: {safe_json_dumps(workflow_design.get("workflow", {}).get("sequence", []))} # Adjusted key based on observed JSON
        Data Flow (for context): {safe_json_dumps(workflow_design.get("workflow", {}).get("data_flow", []))} # Adjusted key

        **YAML STRUCTURE RULES (VERY STRICT):**
        1.  The entire output MUST be valid YAML.
        2.  The top level MUST be a mapping (dictionary) with keys: `scenario_name`, `domain`, `description`, and `steps`.
        3.  `steps:` MUST contain a list (`-` syntax in YAML).
        4.  **EACH item in the `steps` list MUST be a mapping (dictionary).** Start each step with `- ` followed by keys on new lines with indentation.
        5.  **EACH step mapping MUST have a `type` key.** Valid types are "DEFINE", "CREATE", "EXECUTE", "RETURN".
        6.  **DEFINE steps:** Mapping MUST include keys: `item_type` (string: "AGENT" or "TOOL"), `name` (string), `description` (string), `code_snippet` (string, use `|` for multiline). Optional: `from_existing_snippet` (string).
        7.  **CREATE steps:** Mapping MUST include keys: `item_type` (string: "AGENT" or "TOOL"), `name` (string). Optional: `config` (mapping).
        8.  **EXECUTE steps:** Mapping MUST include keys: `item_type` (string: "AGENT" or "TOOL"), `name` (string).
            *   Optional: `input` (mapping). **IF `input` is present, its value MUST be a mapping (dictionary).** Keys are parameter names, values are parameter values (strings, numbers, bools, or `{{{{...}}}}` placeholders).
            *   Optional: `output_var` (string).
            *   Optional: `condition` (string).
        9.  **RETURN steps:** Mapping MUST include key: `value`.
            *   **CRITICAL: The `value` key MUST be followed by a SINGLE string**, typically a placeholder like `{{{{variable_name}}}}` referencing the output variable of a previous step.
            *   **DO NOT** put a nested mapping (dictionary) directly under the `value:` key. If you need to return multiple pieces of data, create them as a dictionary in a previous `EXECUTE` step and return the variable holding that dictionary.

        **CRITICAL YAML FORMATTING:**
        *   Use 2 spaces for indentation.
        *   Ensure correct list (`- `) and mapping (`key: value`) syntax.
        *   Placeholders MUST be enclosed in double curly braces: `{{{{params.var}}}}` or `{{{{step_output_var}}}}`.

        **EXAMPLE OF CORRECT RETURN STEP:**
        ```yaml
        steps:
          # ... other steps ...
          - type: EXECUTE
            item_type: AGENT
            name: FinalProcessor
            input:
              data1: "{{{{step1_output}}}}"
              data2: "{{{{step2_output}}}}"
            output_var: final_structured_result # This variable holds the dictionary
          - type: RETURN # Correct: value is a single placeholder string
            value: {{{{final_structured_result}}}}
        ```

        **EXAMPLE OF INCORRECT RETURN STEP (DO NOT DO THIS):**
        ```yaml
        steps:
          # ... other steps ...
          - type: RETURN # Incorrect: value has a nested mapping
            value:
              field1: "{{{{step1_output}}}}"
              field2: "{{{{step2_output}}}}"
        ```

        Generate the complete YAML based on the design and logic. Ensure every step mapping and the overall structure is valid YAML. Pay meticulous attention to indentation, the structure of the `input` field in EXECUTE steps, and the SINGLE string value required for the `RETURN` step's `value` key.
        Return *only* the raw YAML content, starting with `scenario_name:` and ending after the last step. Do NOT include ```yaml ``` markers.
        """
        logger.debug("Generating workflow YAML from design prompt (strict formatting, RETURN emphasis)...")
        return await self.llm.generate(workflow_prompt)

    async def _generate_from_requirements(
        self,
        workflow_name: str,
        task_objective: str,
        domain: str,
        requirements: str
    ) -> str:
        """Generate YAML directly from natural language requirements using LLM."""
        # Search logic
        logger.debug(f"Searching library for components related to requirements: {requirements[:100]}...")
        try:
            search_results = await self.library.semantic_search(
                query=requirements,
                domain=domain,
                limit=10
            )
            # --- FIX IS HERE ---
            # Correctly unpack the 4-element tuple from semantic_search
            potential_components = [{
                "name": r["name"], "type": r["record_type"],
                "description": r["description"], "similarity": round(final_score, 3)
            } for r, final_score, content_score, task_score in search_results] # Unpack all 4 values
            # --------------------
            logger.debug(f"Found {len(potential_components)} potential components for prompt context.")
        except Exception as search_error:
            logger.error(f"Error during semantic search within GenerateWorkflowTool: {search_error}", exc_info=True)
            potential_components = [] # Proceed without potential components if search fails

        # Construct the prompt with EVEN MORE detailed instructions, especially for RETURN
        workflow_prompt = f"""
        Generate a complete YAML workflow based on these requirements:

        REQUIREMENTS: "{requirements}"
        TASK OBJECTIVE: {task_objective}
        DOMAIN: {domain}
        WORKFLOW NAME: {workflow_name}

        POTENTIAL EXISTING COMPONENTS (for context, you decide if they fit):
        {safe_json_dumps(potential_components)}

        INSTRUCTIONS:
        1. Analyze the requirements to determine the necessary steps.
        2. Identify the components (agents/tools) needed for each step.
        3. Decide whether to reuse existing components, evolve them, or define new ones.
        4. Structure the workflow in YAML format adhering STRICTLY to the rules below.
        5. Ensure the workflow logically fulfills the requirements.

        **YAML STRUCTURE RULES (VERY STRICT):**
        1.  The entire output MUST be valid YAML.
        2.  The top level MUST be a mapping (dictionary) with keys: `scenario_name`, `domain`, `description`, and `steps`.
        3.  `steps:` MUST contain a list (`-` syntax in YAML).
        4.  **EACH item in the `steps` list MUST be a mapping (dictionary).** Start each step with `- ` followed by keys on new lines with indentation.
        5.  **EACH step mapping MUST have a `type` key.** Valid types are "DEFINE", "CREATE", "EXECUTE", "RETURN".
        6.  **DEFINE steps:** Mapping MUST include keys: `item_type` (string: "AGENT" or "TOOL"), `name` (string), `description` (string), `code_snippet` (string, use `|` for multiline). Optional: `from_existing_snippet` (string). Generate plausible code snippets if defining new components.
        7.  **CREATE steps:** Mapping MUST include keys: `item_type` (string: "AGENT" or "TOOL"), `name` (string). Optional: `config` (mapping).
        8.  **EXECUTE steps:** Mapping MUST include keys: `item_type` (string: "AGENT" or "TOOL"), `name` (string).
            *   Optional: `input` (mapping). **IF `input` is present, its value MUST be a mapping (dictionary).** Keys are parameter names, values are parameter values (strings, numbers, bools, or `{{{{...}}}}` placeholders).
            *   Optional: `output_var` (string).
            *   Optional: `condition` (string).
        9.  **RETURN steps:** Mapping MUST include key: `value`.
            *   **CRITICAL: The `value` key MUST be followed by a SINGLE string**, typically a placeholder like `{{{{variable_name}}}}` referencing the output variable of a previous step.
            *   **DO NOT** put a nested mapping (dictionary) directly under the `value:` key. If you need to return multiple pieces of data, create them as a dictionary in a previous `EXECUTE` step and return the variable holding that dictionary.

        **CRITICAL YAML FORMATTING:**
        *   Use 2 spaces for indentation.
        *   Ensure correct list (`- `) and mapping (`key: value`) syntax.
        *   Placeholders MUST be enclosed in double curly braces: `{{{{params.var}}}}` or `{{{{step_output_var}}}}`.

        **EXAMPLE OF CORRECT RETURN STEP:**
        ```yaml
        steps:
          # ... other steps ...
          - type: EXECUTE
            item_type: AGENT
            name: FinalProcessor
            input:
              data1: "{{{{step1_output}}}}"
              data2: "{{{{step2_output}}}}"
            output_var: final_structured_result # This variable holds the dictionary
          - type: RETURN # Correct: value is a single placeholder string
            value: {{{{final_structured_result}}}}
        ```

        **EXAMPLE OF INCORRECT RETURN STEP (DO NOT DO THIS):**
        ```yaml
        steps:
          # ... other steps ...
          - type: RETURN # Incorrect: value has a nested mapping
            value:
              field1: "{{{{step1_output}}}}"
              field2: "{{{{step2_output}}}}"
        ```

        Generate the complete YAML. Ensure every step mapping and the overall structure is valid YAML. Pay meticulous attention to indentation, the structure of the `input` field in EXECUTE steps, and the SINGLE string value required for the `RETURN` step's `value` key.
        Return *only* the raw YAML content, starting with `scenario_name:` and ending after the last step. Do NOT include ```yaml ``` markers.
        """
        logger.debug("Generating workflow YAML from requirements prompt (strict formatting, RETURN emphasis)...")
        return await self.llm.generate(workflow_prompt)

    def _clean_yaml_output(self, yaml_output: str) -> Optional[str]:
        """Extract YAML content from LLM response, removing markdown blocks and preamble."""
        if not yaml_output:
             return None

        # Remove markdown blocks first
        yaml_output = re.sub(r"```yaml\s*([\s\S]*?)\s*```", r"\1", yaml_output, flags=re.MULTILINE)
        yaml_output = re.sub(r"```\s*([\s\S]*?)\s*```", r"\1", yaml_output, flags=re.MULTILINE)

        # Find the start of the actual YAML content
        lines = yaml_output.strip().split('\n')
        start_index = -1
        for i, line in enumerate(lines):
            # Look for common top-level keys
            if line.strip().startswith("scenario_name:") or \
               line.strip().startswith("domain:") or \
               line.strip().startswith("description:") or \
               line.strip().startswith("steps:"):
                start_index = i
                break
            # Also handle if it starts directly with a step list item
            elif line.strip().startswith("- type:"):
                logger.warning("YAML seems to start directly with steps list. Wrapping with default top-level keys.")
                reconstructed_yaml = f"scenario_name: Recovered Workflow\ndomain: general\ndescription: Recovered from steps list\nsteps:\n{yaml_output.strip()}"
                return reconstructed_yaml.strip()


        if start_index != -1:
            # Return from the identified start line onwards
            return "\n".join(lines[start_index:]).strip()
        else:
             logger.warning("Could not find standard YAML start keys. Returning cleaned response.")
             return yaml_output.strip()