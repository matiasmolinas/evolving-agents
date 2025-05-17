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
        if obj is Ellipsis: return None # Handle Ellipsis if it appears
        # Let the default raise the error for other unhandled types
        try:
            # Use ensure_ascii=False for broader character support if needed, otherwise keep default
            return json.JSONEncoder(ensure_ascii=True).encode(obj)
        except TypeError:
             # Attempt to convert common non-serializable types to string representation
             if hasattr(obj, '__str__'):
                 logger.warning(f"Object of type {obj.__class__.__name__} is not JSON serializable, using str(). Value: {str(obj)[:100]}")
                 return str(obj)
             raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable and has no __str__ method.")

    try:
        return json.dumps(data, indent=indent, default=default_serializer)
    except TypeError as e:
         logger.error(f"JSON serialization failed even with custom handler: {e}", exc_info=True)
         # Return a string indicating the error, making it valid JSON
         return json.dumps({"error": f"Data not fully serializable: {str(e)}"})


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

    async def _run(self, tool_input: GenerateWorkflowInput, options: Optional[Dict[str, Any]] = None, context: Optional[RunContext] = None) -> StringToolOutput:
        """
        Generate a complete YAML workflow based on the provided input.
        """
        logger.info(f"Received request to generate workflow '{tool_input.workflow_name}' in domain '{tool_input.domain}'")
        try:
            # If workflow_design is provided, review it before generating YAML
            if tool_input.workflow_design:
                intent_review_enabled = getattr(config, "INTENT_REVIEW_ENABLED", False)
                review_levels = getattr(config, "INTENT_REVIEW_LEVELS", [])
                design_review_active = intent_review_enabled and ("design" in review_levels)

                if design_review_active and context and hasattr(context, "get_value"):
                    human_review_workflow = context.get_value("human_review_workflow", True)
                    if human_review_workflow:
                        design_review_tool = WorkflowDesignReviewTool()
                        logger.info("Workflow design intercepted for human review")
                        review_input_model = WorkflowDesignInput(
                            design=tool_input.workflow_design,
                            interactive=True 
                        )
                        try:
                            review_result = await design_review_tool._run(tool_input=review_input_model, options=options, context=context)
                            review_data = json.loads(review_result.get_text_content())
                            if review_data.get("status") != "approved":
                                return StringToolOutput(json.dumps({
                                    "status": "design_rejected",
                                    "message": "The workflow design was rejected and YAML generation was aborted.",
                                    "reason": review_data.get("reason", "No reason provided"),
                                    "original_design": tool_input.workflow_design
                                }, indent=2))
                            logger.info("Workflow design approved by human reviewer, proceeding to YAML generation")
                            if "modified_design" in review_data: # If review suggested modifications
                                tool_input.workflow_design = review_data["modified_design"]
                        except Exception as e:
                            logger.error(f"Error during workflow design review: {str(e)}")
                            logger.warning("Continuing YAML generation despite review error.")

            # Proceed with workflow generation
            yaml_workflow_str = ""
            if tool_input.workflow_design and tool_input.library_entries:
                logger.info("Generating workflow from structured design and library entries.")
                yaml_workflow_str = await self._generate_from_design(
                    tool_input.workflow_name,
                    tool_input.task_objective,
                    tool_input.domain,
                    tool_input.workflow_design,
                    tool_input.library_entries
                )
            elif tool_input.requirements:
                 logger.info("Generating workflow from natural language requirements.")
                 yaml_workflow_str = await self._generate_from_requirements(
                     tool_input.workflow_name,
                     tool_input.task_objective,
                     tool_input.domain,
                     tool_input.requirements
                 )
            else:
                logger.warning("GenerateWorkflowTool called without design/library_entries or requirements.")
                if tool_input.task_objective and tool_input.domain:
                     yaml_workflow_str = await self._generate_from_requirements(
                         tool_input.workflow_name,
                         tool_input.task_objective,
                         tool_input.domain,
                         f"Generate a workflow to achieve the objective: {tool_input.task_objective}"
                     )
                else:
                    raise ValueError("Insufficient input: Need workflow_design/library_entries OR requirements OR task_objective/domain.")

            cleaned_yaml = self._clean_yaml_output(yaml_workflow_str)
            if not cleaned_yaml:
                 raise ValueError("LLM response did not contain extractable YAML content.")

            try:
                yaml.safe_load(cleaned_yaml) # Basic validation
                logger.info("Generated YAML parsed successfully (basic validation).")
            except yaml.YAMLError as e:
                 logger.error(f"Generated YAML is invalid: {e}")
                 logger.debug(f"Invalid YAML content:\n---\n{cleaned_yaml}\n---")
                 raise ValueError(f"Generated YAML failed basic validation: {e}") from e

            logger.info(f"Successfully generated YAML for workflow '{tool_input.workflow_name}'.")
            return StringToolOutput(cleaned_yaml)

        except Exception as e:
            import traceback
            logger.error(f"Error generating workflow: {str(e)}")
            error_output = safe_json_dumps({ # Use safe_json_dumps here
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
        reuse_components = library_entries.get("reuse", [])
        evolve_components = library_entries.get("evolve", [])
        create_components = library_entries.get("create", [])

        workflow_prompt = f"""
        Generate a complete YAML workflow for the task: '{task_objective}' in domain '{domain}'.

        Workflow Name: {workflow_name}

        Components to Reuse: {safe_json_dumps(reuse_components)}
        Components to Evolve: {safe_json_dumps(evolve_components)}
        Components to Create: {safe_json_dumps(create_components)}

        Workflow Logic Sequence: {safe_json_dumps(workflow_design.get("workflow", {}).get("sequence", []))}
        Data Flow (for context): {safe_json_dumps(workflow_design.get("workflow", {}).get("data_flow", []))}

        **YAML STRUCTURE RULES (VERY STRICT):**
        1.  The entire output MUST be valid YAML.
        2.  The top level MUST be a mapping (dictionary) with keys: `scenario_name`, `domain`, `description`, and `steps`.
        3.  `steps:` MUST contain a list (`-` syntax in YAML).
        4.  **EACH item in the `steps` list MUST be a mapping (dictionary).** Start each step with `- ` followed by keys on new lines with indentation.
        5.  **EACH step mapping MUST have a `type` key.** Valid types are "DEFINE", "CREATE", "EXECUTE", "RETURN".
        6.  **DEFINE steps:** Mapping MUST include keys: `item_type` (string: "AGENT" or "TOOL"), `name` (string), `description` (string), `code_snippet` (string, use `|` for literal block scalar style for multiline Python code. Ensure correct indentation for the Python code *within* the YAML block scalar). Optional: `from_existing_snippet` (string).
        7.  **CREATE steps:** Mapping MUST include keys: `item_type` (string: "AGENT" or "TOOL"), `name` (string). Optional: `config` (mapping).
        8.  **EXECUTE steps:** Mapping MUST include keys: `item_type` (string: "AGENT" or "TOOL"), `name` (string).
            *   Optional: `input` (mapping). **IF `input` is present, its value MUST be a mapping (dictionary).** Keys are parameter names, values are parameter values (strings, numbers, bools, or `{{{{...}}}}` placeholders). For multi-line string inputs (like an invoice document), use the YAML literal block scalar style (`|`) and ensure the string content is correctly indented relative to the `|`. Example:
                ```yaml
                input:
                  invoice_document: |
                    INVOICE #123
                    Date: ...
                    Total: ...
                  another_param: value
                ```
            *   Optional: `output_var` (string).
            *   Optional: `condition` (string).
        9.  **RETURN steps:** Mapping MUST include key: `value`.
            *   **CRITICAL: The `value` key MUST be followed by a SINGLE string**, typically a placeholder like `{{{{variable_name}}}}` referencing the output variable of a previous `EXECUTE` step.
            *   **DO NOT** put a nested mapping (dictionary) directly under the `value:` key. Example of correct RETURN: `value: {{{{final_result}}}}`. Example of INCORRECT RETURN: `value: {{ field: {{{{var}}}} }}`. If you need to return a dictionary, create it in a previous `EXECUTE` step and return the variable holding that dictionary.

        **CRITICAL YAML FORMATTING:**
        *   Use 2 spaces for indentation.
        *   Ensure correct list (`- `) and mapping (`key: value`) syntax.
        *   Placeholders MUST be enclosed in double curly braces: `{{{{params.var}}}}` or `{{{{step_output_var}}}}`.

        Generate the complete YAML based on the design and logic. Ensure every step mapping and the overall structure is valid YAML. Pay meticulous attention to indentation, the structure of the `input` field in EXECUTE steps, and the SINGLE string value required for the `RETURN` step's `value` key.
        Return *only* the raw YAML content, starting with `scenario_name:` and ending after the last step. Do NOT include ```yaml ``` markers or any other explanatory text.
        """
        logger.debug("Generating workflow YAML from design prompt...")
        return await self.llm.generate(workflow_prompt)

    async def _generate_from_requirements(
        self,
        workflow_name: str,
        task_objective: str,
        domain: str,
        requirements: str
    ) -> str:
        logger.debug(f"Searching library for components related to requirements: {requirements[:100]}...")
        try:
            search_results = await self.library.semantic_search(
                query=requirements,
                domain=domain,
                limit=10 # Fetch a few relevant components to provide context to the LLM
            )
            potential_components = [{
                "name": r_dict["name"], "type": r_dict["record_type"],
                "description": r_dict["description"], "similarity": round(final_score, 3)
            } for r_dict, final_score, content_score, task_score in search_results]
            logger.debug(f"Found {len(potential_components)} potential components for prompt context.")
        except Exception as search_error:
            logger.error(f"Error during semantic search within GenerateWorkflowTool: {search_error}", exc_info=True)
            potential_components = []

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
        6.  **DEFINE steps:** Mapping MUST include keys: `item_type` (string: "AGENT" or "TOOL"), `name` (string), `description` (string), `code_snippet` (string, use `|` for literal block scalar style for multiline Python code. Ensure correct indentation for the Python code *within* the YAML block scalar). Optional: `from_existing_snippet` (string). Generate plausible code snippets if defining new components.
        7.  **CREATE steps:** Mapping MUST include keys: `item_type` (string: "AGENT" or "TOOL"), `name` (string). Optional: `config` (mapping).
        8.  **EXECUTE steps:** Mapping MUST include keys: `item_type` (string: "AGENT" or "TOOL"), `name` (string).
            *   Optional: `input` (mapping). **IF `input` is present, its value MUST be a mapping (dictionary).** Keys are parameter names, values are parameter values (strings, numbers, bools, or `{{{{...}}}}` placeholders). For multi-line string inputs (like an invoice document), use the YAML literal block scalar style (`|`) and ensure the string content is correctly indented relative to the `|`. Example:
                ```yaml
                input:
                  invoice_document: |
                    INVOICE #123
                    Date: ...
                    Total: ...
                  another_param: value
                ```
            *   Optional: `output_var` (string).
            *   Optional: `condition` (string).
        9.  **RETURN steps:** Mapping MUST include key: `value`.
            *   **CRITICAL: The `value` key MUST be followed by a SINGLE string**, typically a placeholder like `{{{{variable_name}}}}` referencing the output variable of a previous `EXECUTE` step.
            *   **DO NOT** put a nested mapping (dictionary) directly under the `value:` key. Example of correct RETURN: `value: {{{{final_result}}}}`. Example of INCORRECT RETURN: `value: {{ field: {{{{var}}}} }}`. If you need to return a dictionary, create it in a previous `EXECUTE` step and return the variable holding that dictionary.

        **CRITICAL YAML FORMATTING:**
        *   Use 2 spaces for indentation.
        *   Ensure correct list (`- `) and mapping (`key: value`) syntax.
        *   Placeholders MUST be enclosed in double curly braces: `{{{{params.var}}}}` or `{{{{step_output_var}}}}`.

        Generate the complete YAML. Ensure every step mapping and the overall structure is valid YAML. Pay meticulous attention to indentation, the structure of the `input` field in EXECUTE steps, and the SINGLE string value required for the `RETURN` step's `value` key.
        Return *only* the raw YAML content, starting with `scenario_name:` (or the first top-level key) and ending after the last step. Do NOT include ```yaml ``` markers or any other explanatory text.
        """
        logger.debug("Generating workflow YAML from requirements prompt...")
        return await self.llm.generate(workflow_prompt)

    def _clean_yaml_output(self, yaml_output: str) -> Optional[str]:
        if not yaml_output:
             return None

        # Remove markdown code blocks (```yaml ... ``` or ``` ... ```)
        yaml_output = re.sub(r"```yaml\s*([\s\S]*?)\s*```", r"\1", yaml_output, flags=re.MULTILINE | re.DOTALL)
        yaml_output = re.sub(r"```\s*([\s\S]*?)\s*```", r"\1", yaml_output, flags=re.MULTILINE | re.DOTALL)

        lines = yaml_output.strip().split('\n')
        
        # Remove potential leading/trailing explanatory text more carefully
        # Find the first line that looks like a YAML top-level key or a list item
        start_index = 0
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if stripped_line and (stripped_line.split(':')[0] in ["scenario_name", "domain", "description", "steps"] or stripped_line.startswith("- type:")):
                start_index = i
                break
        
        # Find the last line that looks like YAML content
        # This is harder, but we can try to find a reasonable end point.
        # For now, we'll just take from the start_index if found.
        # If YAML starts with a list, wrap it.
        if start_index < len(lines) and lines[start_index].strip().startswith("- type:"):
            logger.warning("YAML seems to start directly with steps list. Wrapping with default top-level keys.")
            yaml_content_from_steps = "\n".join(lines[start_index:])
            reconstructed_yaml = f"scenario_name: Recovered Workflow\ndomain: general\ndescription: Recovered from steps list\nsteps:\n{yaml_content_from_steps.strip()}"
            return reconstructed_yaml.strip()
        elif start_index < len(lines):
            return "\n".join(lines[start_index:]).strip()
        else:
             logger.warning("Could not find standard YAML start keys. Returning entire cleaned response as is.")
             return yaml_output.strip()