# evolving_agents/workflow/process_workflow_tool.py

import logging
import yaml
from typing import Dict, Any, Optional, List, Union
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

# --- Helper Function for Safe JSON Dumping ---
# (Keep the existing safe_json_dumps function)
def safe_json_dumps(data: Any, indent: int = 2) -> str:
    """Safely dump data to JSON, handling common non-serializable types."""
    def default_serializer(obj):
        if isinstance(obj, set): return list(obj)
        if obj is Ellipsis: return None
        try: return json.JSONEncoder(ensure_ascii=True).encode(obj)
        except TypeError: raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
    try: return json.dumps(data, indent=indent, default=default_serializer)
    except TypeError as e: logger.error(f"JSON serialization error: {e}"); return f'{{"error": "Data not fully serializable: {e}"}}'


# --- Tool Implementation ---
class ProcessWorkflowTool(Tool[ProcessWorkflowInput, None, StringToolOutput]):
    """
    Tool for parsing and preparing YAML-based workflows for execution.
    It validates the workflow structure, substitutes parameters recursively,
    and returns a structured plan (list of steps) for the SystemAgent to execute.
    """
    name = "ProcessWorkflowTool"
    description = "Parse, validate, and prepare YAML workflows for execution by returning a structured plan"
    input_schema = ProcessWorkflowInput

    def __init__(self, options: Optional[Dict[str, Any]] = None):
        super().__init__(options=options or {})
        logger.info("ProcessWorkflowTool initialized")

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "workflow", "process"],
            creator=self,
        )

    async def _run(self, input: ProcessWorkflowInput, options: Optional[Dict[str, Any]] = None, context: Optional[RunContext] = None) -> StringToolOutput:
        logger.info("Processing workflow YAML...")
        try:
            # 1. Clean and Load YAML
            cleaned_yaml_string = html.unescape(input.workflow_yaml)
            logger.debug("Cleaned YAML string (first 500 chars):\n%s", cleaned_yaml_string[:500])
            try:
                workflow = yaml.safe_load(cleaned_yaml_string)
                if not isinstance(workflow, dict):
                    if isinstance(workflow, list):
                        logger.warning("YAML loaded as a list, assuming 'steps'. Wrapping.")
                        workflow = {"scenario_name": "Recovered Workflow", "steps": workflow}
                    else: raise ValueError("Workflow YAML must load as a dictionary.")
                if "steps" not in workflow or not isinstance(workflow["steps"], list):
                    raise ValueError("Invalid workflow: Missing 'steps' list.")
            except yaml.YAMLError as e:
                # (Keep detailed YAML error reporting)
                error_line = getattr(e, 'problem_mark', None); line_num = error_line.line + 1 if error_line else 'unknown'
                problem = getattr(e, 'problem', str(e)); context_msg = getattr(e, 'context', '')
                snippet_start = max(0, error_line.index - 40) if error_line else 0; snippet_end = error_line.index + 40 if error_line else 80
                snippet = cleaned_yaml_string[snippet_start:snippet_end]
                raise ValueError(f"Error parsing YAML near line {line_num}: {problem}. {context_msg}. Snippet: ...{snippet}...") from e
            logger.info(f"Successfully parsed YAML for '{workflow.get('scenario_name', 'N/A')}'")

            # 2. Substitute Parameters Recursively **ON THE PYTHON OBJECT**
            try:
                 params_dict = input.params if isinstance(input.params, dict) else {}
                 processed_workflow_steps = self._substitute_params_recursive(workflow["steps"], params_dict)
                 logger.info("Parameter substitution complete.")
            except Exception as e:
                 logger.error("Error during recursive parameter substitution.", exc_info=True)
                 raise ValueError(f"Error during parameter substitution: {e}") from e

            # 3. Validate Steps Structure
            try:
                 validated_steps = self._validate_steps(processed_workflow_steps)
                 logger.info("Workflow steps validated.")
            except Exception as e:
                 logger.error("Error during step validation.", exc_info=True)
                 raise ValueError(f"Error during step validation: {e}") from e

            # 4. Prepare the Structured Plan
            plan = {
                "status": "success",
                "scenario_name": workflow.get("scenario_name", "Unnamed Workflow"),
                "domain": workflow.get("domain", "general"),
                "description": workflow.get("description", ""),
                "steps": validated_steps, # Use the validated & substituted steps
                "execution_guidance": "The SystemAgent should now execute these steps sequentially."
            }
            return StringToolOutput(safe_json_dumps(plan))

        except Exception as e:
            import traceback
            logger.error(f"Error processing workflow in _run: {str(e)}")
            return StringToolOutput(safe_json_dumps({
                "status": "error",
                "message": f"Error processing workflow: {str(e)}",
                "details": traceback.format_exc()
            }))

    def _substitute_params_recursive(self, obj: Any, params: Dict[str, Any]) -> Any:
        """
        Recursively walks through a nested Python object (dict, list, str)
        and substitutes '{{params.key}}' placeholders found in strings.
        """
        pattern = r'{{\s*params\.(\w+)\s*}}' # Regex to find {{params.key}}

        if isinstance(obj, dict):
            # Create a new dict to avoid modifying the original during iteration
            new_dict = {}
            for key, value in obj.items():
                # Recursively substitute in both key (if string) and value
                new_key = self._substitute_params_recursive(key, params) if isinstance(key, str) else key
                new_dict[new_key] = self._substitute_params_recursive(value, params)
            return new_dict
        elif isinstance(obj, list):
            # Recursively substitute in list items
            return [self._substitute_params_recursive(item, params) for item in obj]
        elif isinstance(obj, str):
            # --- Perform substitution on the string value ---
            matches = list(re.finditer(pattern, obj))
            if not matches:
                return obj # No placeholders found

            # Case 1: String consists ONLY of a single placeholder
            if len(matches) == 1 and matches[0].group(0) == obj.strip():
                 key = matches[0].group(1)
                 if key in params:
                      logger.debug(f"Substituting exact placeholder '{obj}' with value for '{key}' (type: {type(params[key])})")
                      # Return the actual value, preserving its type (int, bool, list, dict, etc.)
                      return params[key]
                 else:
                      logger.warning(f"Parameter '{key}' for exact match placeholder '{obj}' not found. Returning placeholder string.")
                      return obj # Return original placeholder string if key not found

            # Case 2: String contains placeholders mixed with other text
            else:
                def replace_match_in_string(match):
                    key = match.group(1)
                    if key in params:
                        # Convert param value to string for embedding within the larger string
                        return str(params[key])
                    else:
                        logger.warning(f"Parameter '{key}' for embedded placeholder '{match.group(0)}' not found. Leaving placeholder.")
                        return match.group(0) # Keep the placeholder string

                return re.sub(pattern, replace_match_in_string, obj)
        else:
            # Return non-dict/list/str types as is (numbers, booleans, None)
            return obj

    def _validate_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Perform basic validation on workflow steps.
        """
        if not isinstance(steps, list):
             raise ValueError("Workflow 'steps' must be a list.")

        validated_steps_list = []
        for i, step in enumerate(steps):
            step_num = i + 1
            if not isinstance(step, dict):
                 raise ValueError(f"Step {step_num} is not a dictionary: {step}")

            if "type" not in step:
                raise ValueError(f"Step {step_num} missing required 'type' field: {step}")
            step_type = step.get("type")
            valid_types = ["DEFINE", "CREATE", "EXECUTE", "RETURN"]
            if step_type not in valid_types:
                raise ValueError(f"Step {step_num} invalid type '{step_type}'. Must be one of {valid_types}.")

            required_keys = set()
            if step_type == "DEFINE": required_keys = {"item_type", "name", "description", "code_snippet"}
            elif step_type == "CREATE": required_keys = {"item_type", "name"}
            elif step_type == "EXECUTE": required_keys = {"item_type", "name"}
            elif step_type == "RETURN": required_keys = {"value"}

            missing_keys = required_keys - set(step.keys())
            if missing_keys:
                 raise ValueError(f"Step {step_num} (type: {step_type}) missing required keys: {missing_keys}. Step: {step}")

            # *** Crucial: Validate EXECUTE input is a dict ***
            if step_type == "EXECUTE" and "input" in step:
                 if not isinstance(step["input"], dict):
                      # Log the problematic step for debugging
                      logger.error(f"Validation Error in Step {step_num}: 'input' field is not a dictionary. Found type: {type(step['input'])}. Step content: {step}")
                      raise ValueError(f"Step {step_num} (EXECUTE): 'input' field MUST be a dictionary (mapping), but found type {type(step['input'])}.")

            validated_steps_list.append(step)
        return validated_steps_list