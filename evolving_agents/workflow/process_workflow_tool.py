# evolving_agents/workflow/process_workflow_tool.py

import logging
import yaml
from typing import Dict, Any, Optional, List
import json
import html # For unescaping HTML entities
import re # For parameter substitution regex

# Pydantic for input validation
from pydantic import BaseModel, Field

# BeeAI Framework imports for Tool structure
from beeai_framework.tools.tool import Tool, StringToolOutput
from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter

logger = logging.getLogger(__name__)

# --- Input Schema ---
class ProcessWorkflowInput(BaseModel):
    """Input schema for the ProcessWorkflowTool."""
    workflow_yaml: str = Field(description="YAML string defining the workflow")
    params: Optional[Dict[str, Any]] = Field(None, description="Parameters to substitute into the workflow (key-value pairs)")

# --- Tool Implementation ---
class ProcessWorkflowTool(Tool[ProcessWorkflowInput, None, StringToolOutput]):
    """
    Tool for parsing and preparing YAML-based workflows for execution.
    It validates the workflow structure, substitutes parameters, and returns
    a structured plan (list of steps) for the SystemAgent to execute.
    """
    name = "ProcessWorkflowTool"
    description = "Parse, validate, and prepare YAML workflows for execution by returning a structured plan"
    input_schema = ProcessWorkflowInput

    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """
        Initialize the workflow processing tool.
        No direct dependencies needed as it primarily parses YAML.

        Args:
            options: Optional configuration for the tool.
        """
        super().__init__(options=options or {})
        logger.info("ProcessWorkflowTool initialized")

    def _create_emitter(self) -> Emitter:
        """Creates the event emitter for this tool."""
        return Emitter.root().child(
            namespace=["tool", "workflow", "process"],
            creator=self,
        )

    async def _run(self, input: ProcessWorkflowInput, options: Optional[Dict[str, Any]] = None, context: Optional[RunContext] = None) -> StringToolOutput:
        """
        Parse the workflow YAML and return a structured execution plan.

        Args:
            input: The validated input parameters (YAML string and params dict).
            options: Optional runtime options.
            context: The run context.

        Returns:
            A StringToolOutput containing the status and the structured plan (or error message).
        """
        logger.info("Processing workflow YAML...")
        try:
            # 1. Clean HTML entities from YAML before parsing
            # This handles cases where LLM might output " instead of "
            cleaned_yaml_string = html.unescape(input.workflow_yaml)
            logger.debug("Cleaned YAML string (first 500 chars):\n%s", cleaned_yaml_string[:500])

            # 2. Load YAML using safe_load
            try:
                workflow = yaml.safe_load(cleaned_yaml_string)
                if not isinstance(workflow, dict):
                    # Try recovery if it loaded as just the steps list
                    if isinstance(workflow, list):
                        logger.warning("YAML loaded as a list, assuming it's the 'steps'. Wrapping.")
                        workflow = {"scenario_name": "Recovered Workflow", "steps": workflow}
                    else:
                         raise ValueError("Workflow YAML must load as a dictionary (object).")
                if "steps" not in workflow or not isinstance(workflow["steps"], list):
                    raise ValueError("Invalid workflow structure: Missing 'steps' list key.")
            except yaml.YAMLError as e:
                error_line = getattr(e, 'problem_mark', None)
                line_num = error_line.line + 1 if error_line else 'unknown'
                problem = getattr(e, 'problem', str(e))
                context_msg = getattr(e, 'context', '')
                snippet_start = max(0, error_line.index - 40) if error_line else 0
                snippet_end = error_line.index + 40 if error_line else 80
                snippet = cleaned_yaml_string[snippet_start:snippet_end]
                raise ValueError(f"Error parsing workflow YAML near line {line_num}: {problem}. {context_msg}. Snippet: ...{snippet}...") from e
            except Exception as e:
                 raise ValueError(f"Unexpected error loading workflow YAML: {str(e)}") from e

            logger.info(f"Successfully parsed YAML for '{workflow.get('scenario_name', 'N/A')}'")

            # 3. Substitute parameters
            try:
                 # Ensure params is a dict
                 params_dict = input.params if isinstance(input.params, dict) else {}
                 processed_steps = self._substitute_params(workflow["steps"], params_dict)
                 logger.info("Parameter substitution complete.")
            except Exception as e:
                 logger.error("Error during parameter substitution.", exc_info=True)
                 raise ValueError(f"Error during parameter substitution: {e}") from e


            # 4. Validate steps structure
            try:
                 validated_steps = self._validate_steps(processed_steps)
                 logger.info("Workflow steps validated.")
            except Exception as e:
                 logger.error("Error during step validation.", exc_info=True)
                 raise ValueError(f"Error during step validation: {e}") from e


            # 5. Prepare the structured plan for the SystemAgent
            plan = {
                "status": "success",
                "scenario_name": workflow.get("scenario_name", "Unnamed Workflow"),
                "domain": workflow.get("domain", "general"),
                "description": workflow.get("description", ""),
                "steps": validated_steps,
                "execution_guidance": "The SystemAgent should now execute these steps sequentially using its available tools."
            }

            # Use safe_json_dumps for the final output
            return StringToolOutput(safe_json_dumps(plan))

        except Exception as e: # Catch errors from this _run method itself
            import traceback
            logger.error(f"Error processing workflow in _run: {str(e)}")
            # Use safe_json_dumps for error output
            return StringToolOutput(safe_json_dumps({
                "status": "error",
                "message": f"Error processing workflow: {str(e)}",
                "details": traceback.format_exc() # Include traceback for debugging
            }))

    def _substitute_params(self, obj: Any, params: Dict[str, Any]) -> Any:
        """
        Recursively substitute {{params.key}} placeholders in strings within a nested structure.
        Uses JSON dumping/loading internally to handle complex values correctly.
        """
        # Convert the entire structure (list or dict) to a JSON string first
        try:
             structure_str = json.dumps(obj)
        except TypeError as e:
             raise ValueError(f"Workflow structure contains non-JSON serializable data before substitution: {e}") from e

        # Define the regex pattern to find {{params.key}}
        # It captures the 'key' part.
        pattern = r'{{\s*params\.(\w+)\s*}}'

        # Use re.sub with a function to handle replacements
        def replace_match(match):
            key = match.group(1)
            if key in params:
                value = params[key]
                # IMPORTANT: Dump the value to a JSON string representation.
                # This ensures that strings get quotes, numbers don't, booleans become true/false,
                # and dicts/lists become valid JSON object/array strings.
                try:
                    return json.dumps(value)
                except TypeError as e:
                     logger.warning(f"Parameter '{key}' value is not JSON serializable ({type(value)}): {e}. Substituting as null.")
                     return 'null' # Substitute non-serializable params as null
            else:
                logger.warning(f"Parameter '{key}' not found in provided params. Leaving placeholder.")
                return match.group(0) # Return the original placeholder if key not found

        substituted_str = re.sub(pattern, replace_match, structure_str)

        # Now, parse the resulting string (which should be valid JSON) back into a Python object
        try:
            processed_obj = json.loads(substituted_str)
        except json.JSONDecodeError as e:
            # This might happen if replacements resulted in invalid JSON
            logger.error(f"JSON decode error after parameter substitution. Substituted string snippet: {substituted_str[:500]}...")
            raise ValueError(f"Error parsing structure after parameter substitution: {e}") from e

        return processed_obj

    def _validate_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Perform basic validation on workflow steps.
        Ensures each step is a dict with a valid 'type' and required keys for that type.
        """
        if not isinstance(steps, list):
             raise ValueError("Workflow 'steps' must be a list.")

        validated_steps_list = []
        for i, step in enumerate(steps):
            step_num = i + 1
            if not isinstance(step, dict):
                 # Attempt simple recovery for common LLM errors like forgetting the dict structure
                 if isinstance(step, str) and ':' in step:
                      try:
                           # Example: "- type: CREATE" -> {"- type": "CREATE"} - needs more robust parsing maybe
                           logger.warning(f"Step {step_num} is a string, attempting simple parse: '{step}'")
                           # This simple split is likely insufficient for complex steps
                           parts = step.split(':', 1)
                           if len(parts) == 2:
                                step = {parts[0].strip().lstrip('-').strip(): parts[1].strip()}
                           else:
                                raise ValueError("String step could not be parsed into key-value.")
                      except Exception as parse_err:
                           raise ValueError(f"Step {step_num} is not a dictionary and could not be recovered: {step}. Parse error: {parse_err}")
                 else:
                      raise ValueError(f"Step {step_num} is not a dictionary: {step}")

            # Check for mandatory 'type' key
            if "type" not in step:
                raise ValueError(f"Step {step_num} is missing the required 'type' field. Step content: {step}")

            step_type = step.get("type") # Use .get() for safety

            # Validate step type
            valid_types = ["DEFINE", "CREATE", "EXECUTE", "RETURN"]
            if step_type not in valid_types:
                raise ValueError(f"Step {step_num} has an invalid type: '{step_type}'. Must be one of {valid_types}.")

            # Validate required keys based on type
            required_keys = set()
            if step_type == "DEFINE":
                required_keys = {"item_type", "name", "description", "code_snippet"}
            elif step_type == "CREATE":
                required_keys = {"item_type", "name"}
            elif step_type == "EXECUTE":
                required_keys = {"item_type", "name"} # Input is optional but recommended
            elif step_type == "RETURN":
                required_keys = {"value"}

            missing_keys = required_keys - set(step.keys())
            if missing_keys:
                 raise ValueError(f"Step {step_num} (type: {step_type}) is missing required keys: {', '.join(missing_keys)}. Step content: {step}")

            # Basic type checking for common fields (optional but good practice)
            if "item_type" in step and step.get("item_type") not in ["AGENT", "TOOL"]:
                 logger.warning(f"Step {step_num}: 'item_type' should typically be 'AGENT' or 'TOOL', found '{step.get('item_type')}'.")
            if "name" in step and not isinstance(step.get("name"), str):
                 logger.warning(f"Step {step_num}: 'name' should be a string, found {type(step.get('name'))}.")

            validated_steps_list.append(step)

        return validated_steps_list

# Helper function (if needed outside the class, otherwise keep inside)
def safe_json_dumps(data: Any, indent: int = 2) -> str:
    """Safely dump data to JSON, handling common non-serializable types like ellipsis and sets."""
    def default_serializer(obj):
        if isinstance(obj, set): return list(obj)
        if obj is Ellipsis: return None
        try: return json.JSONEncoder().encode(obj) # Try standard first
        except TypeError: raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
    try: return json.dumps(data, indent=indent, default=default_serializer)
    except TypeError as e: logger.error(f"JSON serialization error: {e}"); return f'{{"error": "Data not fully serializable: {e}"}}'