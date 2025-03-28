# evolving_agents/workflow/generate_workflow_tool.py

import json
import logging
from typing import Dict, Any, List, Optional
import re # Added for cleaning output

# Pydantic for input validation
from pydantic import BaseModel, Field

# BeeAI Framework imports for Tool structure
from beeai_framework.tools.tool import Tool, StringToolOutput
from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter

# Project-specific imports
from evolving_agents.core.llm_service import LLMService
from evolving_agents.smart_library.smart_library import SmartLibrary

logger = logging.getLogger(__name__)

# --- Helper Function for Safe JSON Dumping ---
def safe_json_dumps(data: Any, indent: int = 2) -> str:
    """Safely dump data to JSON, handling common non-serializable types like ellipsis and sets."""
    def default_serializer(obj):
        if isinstance(obj, set): # Handle sets
            return list(obj)
        if obj is Ellipsis: # Handle ellipsis
            return None # Replace ellipsis with None (or choose another representation like '...')
        # Let the default raise the error for other unhandled types
        try:
            # Try standard JSONEncoder first
            return json.JSONEncoder().encode(obj)
        except TypeError:
             # If standard encoder fails, raise the error clearly
             raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    try:
        # Use the custom handler
        return json.dumps(data, indent=indent, default=default_serializer)
    except TypeError as e:
         logger.error(f"JSON serialization failed even with custom handler: {e}", exc_info=True)
         # Fallback: return a string representation indicating the error
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

        Args:
            llm_service: The LLM service instance for generation.
            smart_library: The SmartLibrary instance for component context.
            options: Optional configuration for the tool.
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

        Args:
            input: The validated input parameters.
            options: Optional runtime options.
            context: The run context.

        Returns:
            A StringToolOutput containing the status and the generated YAML workflow (or error message).
        """
        logger.info(f"Received request to generate workflow '{input.workflow_name}' in domain '{input.domain}'")
        try:
            # Prioritize structured input if available
            if input.workflow_design and input.library_entries:
                logger.info("Generating workflow from structured design and library entries.")
                yaml_workflow = await self._generate_from_design(
                    input.workflow_name,
                    input.task_objective,
                    input.domain,
                    input.workflow_design,
                    input.library_entries
                )
            elif input.requirements:
                 logger.info("Generating workflow from natural language requirements.")
                 yaml_workflow = await self._generate_from_requirements(
                     input.workflow_name,
                     input.task_objective,
                     input.domain,
                     input.requirements
                 )
            else:
                raise ValueError("Either workflow_design/library_entries OR requirements must be provided.")

            # Clean YAML response from LLM
            cleaned_yaml = self._clean_yaml_output(yaml_workflow)
            if not cleaned_yaml:
                 raise ValueError("LLM response did not contain extractable YAML content.")

            logger.info(f"Successfully generated YAML for workflow '{input.workflow_name}'.")
            return StringToolOutput(safe_json_dumps({ # Use safe dump for final output too
                "status": "success",
                "workflow_name": input.workflow_name,
                "yaml_workflow": cleaned_yaml
            }))

        except Exception as e:
            import traceback
            logger.error(f"Error generating workflow: {str(e)}")
            return StringToolOutput(safe_json_dumps({ # Use safe dump for error output
                "status": "error",
                "message": f"Error generating workflow: {str(e)}",
                "details": traceback.format_exc()
            }))

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

        # Construct the prompt with detailed instructions and safe JSON dumping
        workflow_prompt = f"""
        Generate a complete YAML workflow for the task: '{task_objective}' in domain '{domain}'.

        Workflow Name: {workflow_name}

        Components to Reuse: {safe_json_dumps(reuse_components)}
        Components to Evolve: {safe_json_dumps(evolve_components)}
        Components to Create: {safe_json_dumps(create_components)}

        Workflow Logic Sequence: {safe_json_dumps(workflow_design.get("sequence", []))}
        Data Flow (for context): {safe_json_dumps(workflow_design.get("data_flow", []))}

        **YAML STRUCTURE RULES:**
        1. The top level must have `scenario_name`, `domain`, and `description`.
        2. Under the `steps:` key, provide a list of dictionaries.
        3. **EACH step dictionary MUST have a `type` key.** Valid types are "DEFINE", "CREATE", "EXECUTE", "RETURN".
        4. **DEFINE steps:** MUST include `item_type` ("AGENT" or "TOOL"), `name`, `description`, and `code_snippet`. Can optionally include `from_existing_snippet` for evolution.
        5. **CREATE steps:** MUST include `item_type` ("AGENT" or "TOOL") and `name`. Can include `config`.
        6. **EXECUTE steps:** MUST include `item_type` ("AGENT" or "TOOL"), `name`.
           - **Input Specification:** The `input` field for EXECUTE steps MUST be a dictionary mapping input parameter names (like 'text' or specific schema fields) to their values. Use `{{{{params.param_name}}}}` for workflow parameters or `{{{{variable_name}}}}` for outputs from previous steps (`output_var`).
           - Optional keys: `input`, `method`, `output_var`, `condition`.
        7. **RETURN steps:** MUST include `value` (often referencing an `output_var` like `{{{{variable_name}}}}`).

        **EXAMPLE STEP FORMATS:**
        ```yaml
          - type: DEFINE
            item_type: TOOL
            name: MyNewTool
            description: Does something specific.
            code_snippet: |
              # Python code here...
          - type: CREATE
            item_type: AGENT
            name: MyAgent
            config:
              memory_type: token
          - type: EXECUTE # Example showing correct input structure
            item_type: TOOL
            name: MyNewTool
            input: # Input MUST be a dictionary
              text: "{{{{params.some_input}}}}" # Parameter substitution
              mode: "detailed" # Static value
            output_var: tool_result
          - type: EXECUTE # Example using output from previous step
            item_type: AGENT
            name: ProcessingAgent
            input:
              data_to_process: "{{{{tool_result}}}}" # Using output variable
            output_var: final_data
          - type: RETURN
            value: {{{{final_data}}}}
        ```

        Generate the complete YAML based on the design and logic. Ensure every step strictly follows the required format, especially the `type` key and the `input` structure for EXECUTE steps.
        Return *only* the YAML content, enclosed in ```yaml ... ``` blocks.
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
        """Generate YAML directly from natural language requirements using LLM."""
        # Search for potentially relevant components in the library
        logger.debug(f"Searching library for components related to requirements: {requirements[:100]}...")
        search_results = await self.library.semantic_search(
            query=requirements,
            domain=domain,
            limit=10 # Get more results for the LLM to consider
        )
        potential_components = [{
            "name": r["name"],
            "type": r["record_type"],
            "description": r["description"],
            "similarity": round(sim, 3) # Round similarity for prompt
        } for r, sim in search_results]

        # Construct the prompt with detailed instructions and safe JSON dumping
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
        3. Decide whether to reuse existing components (from POTENTIAL list), evolve them, or define new ones.
        4. Structure the workflow in YAML format adhering STRICTLY to the rules below.
        5. Ensure the workflow logically fulfills the requirements.

        **YAML STRUCTURE RULES:**
        1. Top level: `scenario_name`, `domain`, `description`.
        2. `steps:` key holds a list of dictionaries.
        3. **EACH step dictionary MUST have a `type` key ("DEFINE", "CREATE", "EXECUTE", "RETURN").**
        4. **DEFINE steps:** MUST include `item_type`, `name`, `description`, `code_snippet`. Optional `from_existing_snippet`. Generate plausible code snippets if defining new components.
        5. **CREATE steps:** MUST include `item_type`, `name`. Optional `config`.
        6. **EXECUTE steps:** MUST include `item_type`, `name`.
           - **Input Specification:** The `input` field MUST be a dictionary mapping input parameter names (like 'text') to values (using `{{{{params.param_name}}}}` or `{{{{variable_name}}}}`).
           - Optional keys: `input`, `method`, `output_var`, `condition`.
        7. **RETURN steps:** MUST include `value` (e.g., `{{{{variable_name}}}}`).

        **EXAMPLE STEP FORMATS:**
        ```yaml
          - type: DEFINE
            item_type: TOOL
            name: MyNewTool
            description: Does something specific.
            code_snippet: |
              # Python code here...
          - type: CREATE
            item_type: AGENT
            name: MyAgent
            config:
              memory_type: token
          - type: EXECUTE # Example showing correct input structure
            item_type: TOOL
            name: MyNewTool
            input: # Input MUST be a dictionary
              text: "{{{{params.some_input}}}}" # Parameter substitution
              mode: "detailed" # Static value
            output_var: tool_result
          - type: RETURN
            value: {{{{tool_result}}}}
        ```

        Generate the complete YAML. Ensure every step strictly follows the required format, especially the `type` key and the `input` structure for EXECUTE steps.
        Return *only* the YAML content, enclosed in ```yaml ... ``` blocks.
        """
        logger.debug("Generating workflow YAML from requirements prompt...")
        return await self.llm.generate(workflow_prompt)

    def _clean_yaml_output(self, yaml_output: str) -> Optional[str]:
        """Extract YAML content from LLM response, handling markdown blocks."""
        if not yaml_output:
             return None

        # 1. Look for ```yaml ... ``` markdown blocks explicitly
        match = re.search(r"```yaml\s*([\s\S]*?)\s*```", yaml_output, re.MULTILINE)
        if match:
            logger.debug("Extracted YAML using ```yaml block.")
            return match.group(1).strip()

        # 2. Look for generic ``` ... ``` blocks if no explicit yaml block
        match = re.search(r"```\s*([\s\S]*?)\s*```", yaml_output, re.MULTILINE)
        if match:
             potential_yaml = match.group(1).strip()
             # Basic check if it looks like YAML
             if potential_yaml.startswith("scenario_name:") or potential_yaml.startswith("- type:"):
                  logger.debug("Extracted YAML using generic ``` block.")
                  return potential_yaml

        # 3. Fallback: Assume the whole string might be YAML if it starts like it
        cleaned_output = yaml_output.strip()
        lines = cleaned_output.split('\n')
        if lines and (lines[0].strip().startswith("scenario_name:") or lines[0].strip().startswith("- type:")):
            # Try to remove potential preamble/postamble text from LLM
            yaml_lines = []
            in_yaml = False
            for line in lines:
                 if line.strip().startswith("scenario_name:") or line.strip().startswith("- type:"):
                      in_yaml = True
                 if in_yaml:
                      # Basic heuristic: stop if a line looks like explanatory text again
                      if not line.startswith(' ') and ':' not in line and not line.strip().startswith('-') and len(yaml_lines) > 1:
                           if not line.strip().startswith('#'): # Allow comments
                                break
                      yaml_lines.append(line)
            if yaml_lines:
                 logger.debug("Extracted YAML using start pattern heuristic.")
                 return "\n".join(yaml_lines).strip()

        logger.warning("Could not reliably extract YAML from LLM response. Returning raw response.")
        # Return the original output if no reliable extraction worked, maybe with a comment
        return f"# WARNING: Could not reliably extract YAML.\n{yaml_output.strip()}"