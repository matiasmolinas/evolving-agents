# evolving_agents/workflow/process_workflow_tool.py

import logging
import yaml
from typing import Dict, Any, Optional, List, Union
import json
import html # For unescaping HTML entities
import re # For parameter substitution regex
import uuid # For intent IDs
from datetime import datetime # For intent timestamps

# Pydantic for input validation
from pydantic import BaseModel, Field

# BeeAI Framework imports for Tool structure
from beeai_framework.tools.tool import Tool, StringToolOutput
from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter

# Import config for intent mode check (adjust path if necessary)
from evolving_agents import config

logger = logging.getLogger(__name__)

# --- Input Schema ---
class ProcessWorkflowInput(BaseModel):
    """Input schema for the ProcessWorkflowTool."""
    workflow_yaml: str = Field(description="YAML string defining the workflow")
    params: Optional[Dict[str, Any]] = Field(None, description="Parameters to substitute into the workflow (key-value pairs)")

# --- Helper Function for Safe JSON Dumping ---
def safe_json_dumps(data: Any, indent: int = 2) -> str:
    """Safely dump data to JSON, handling common non-serializable types."""
    def default_serializer(obj):
        if isinstance(obj, set): return list(obj)
        if obj is Ellipsis: return None
        # Add datetime serialization
        if isinstance(obj, datetime): return obj.isoformat()
        try: return json.JSONEncoder(ensure_ascii=True).encode(obj)
        except TypeError: raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
    try: return json.dumps(data, indent=indent, default=default_serializer)
    except TypeError as e: logger.error(f"JSON serialization error: {e}"); return f'{{"error": "Data not fully serializable: {e}"}}'


# --- Tool Implementation ---
class ProcessWorkflowTool(Tool[ProcessWorkflowInput, None, StringToolOutput]):
    """
    Tool for parsing and preparing YAML-based workflows for execution.
    It validates the workflow structure, substitutes parameters recursively.
    If Intent Mode is enabled, it converts the steps into an intent plan for review.
    Otherwise, it returns a structured plan (list of steps) for direct execution.
    """
    name = "ProcessWorkflowTool"
    description = "Parse, validate, prepare YAML workflows, optionally generating an intent plan for review, or returning steps for execution."
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

            # 3. Determine if Intent Mode is Active
            intent_mode = False
            if context and hasattr(context, "get_value"):
                # Check context first for override
                intent_mode = context.get_value("intent_review_mode", False)
                logger.debug(f"Intent mode from context: {intent_mode}")

            # Check global setting if context doesn't specify true, or if context doesn't exist
            if not intent_mode and config.INTENT_REVIEW_ENABLED:
                 # Check if the specific level "intents" is enabled in the config
                 intent_mode = "intents" in getattr(config, "INTENT_REVIEW_LEVELS", [])
                 logger.debug(f"Intent mode from global config: {intent_mode} (INTENT_REVIEW_ENABLED={config.INTENT_REVIEW_ENABLED}, INTENT_REVIEW_LEVELS={getattr(config, 'INTENT_REVIEW_LEVELS', [])})")

            # 4. Validate Steps / Convert to Intent Plan
            try:
                 # Pass the processed steps and intent_mode flag
                 validated_output = self._validate_steps(processed_workflow_steps, intent_mode=intent_mode)
                 logger.info("Workflow steps validated." + (" (Intent plan generated)" if intent_mode and isinstance(validated_output, dict) else ""))
            except Exception as e:
                 logger.error("Error during step validation/conversion.", exc_info=True)
                 raise ValueError(f"Error during step validation/conversion: {e}") from e

            # 5. Handle Output Based on Mode
            # --- INTENT MODE OUTPUT ---
            if intent_mode and isinstance(validated_output, dict) and "intents" in validated_output:
                intent_plan = validated_output

                # Save the intent plan to the context for potential review/use by other components
                if context and hasattr(context, "set_value"):
                    try:
                        context.set_value("intent_plan", safe_json_dumps(intent_plan)) # Store as JSON string
                        logger.info(f"Intent plan '{intent_plan.get('plan_id')}' saved to context.")
                    except Exception as ctx_err:
                        logger.warning(f"Failed to save intent plan to context: {ctx_err}", exc_info=True)

                # Return the intent plan itself, indicating review is needed
                return StringToolOutput(safe_json_dumps({
                    "status": "intent_plan_created",
                    "message": "Intent plan created successfully. Review is required before execution.",
                    "intent_plan": intent_plan
                }))

            # --- DIRECT EXECUTION MODE OUTPUT ---
            elif isinstance(validated_output, list): # Should be a list if not intent mode or conversion failed gracefully
                plan = {
                    "status": "success",
                    "scenario_name": workflow.get("scenario_name", "Unnamed Workflow"),
                    "domain": workflow.get("domain", "general"),
                    "description": workflow.get("description", ""),
                    "steps": validated_output, # Use the validated (and potentially substituted) steps
                    "execution_guidance": "The SystemAgent should now execute these steps sequentially."
                }
                return StringToolOutput(safe_json_dumps(plan))
            
            # --- UNEXPECTED VALIDATION OUTPUT ---
            else:
                raise TypeError(f"Unexpected output type from _validate_steps: {type(validated_output)}. Expected list or dict (for intent plan).")


        except Exception as e:
            import traceback
            logger.error(f"Error processing workflow in _run: {str(e)}", exc_info=True)
            # Use safe_json_dumps for error output as well
            return StringToolOutput(safe_json_dumps({
                "status": "error",
                "message": f"Error processing workflow: {str(e)}",
                "details": traceback.format_exc() # Keep traceback for debugging
            }))

    def _substitute_params_recursive(self, obj: Any, params: Dict[str, Any]) -> Any:
        """
        Recursively walks through a nested Python object (dict, list, str)
        and substitutes '{{params.key}}' placeholders found in strings.
        Handles exact matches by returning the original parameter type.
        """
        pattern = r'{{\s*params\.(\w+)\s*}}' # Regex to find {{params.key}}

        if isinstance(obj, dict):
            new_dict = {}
            for key, value in obj.items():
                new_key = self._substitute_params_recursive(key, params) if isinstance(key, str) else key
                new_dict[new_key] = self._substitute_params_recursive(value, params)
            return new_dict
        elif isinstance(obj, list):
            return [self._substitute_params_recursive(item, params) for item in obj]
        elif isinstance(obj, str):
            matches = list(re.finditer(pattern, obj))
            if not matches:
                return obj

            # Case 1: String consists ONLY of a single placeholder '{{params.key}}'
            if len(matches) == 1 and matches[0].group(0) == obj.strip():
                 key = matches[0].group(1)
                 if key in params:
                      logger.debug(f"Substituting exact placeholder '{obj}' with value for '{key}' (type: {type(params[key])})")
                      return params[key] # Return the actual value, preserving type
                 else:
                      logger.warning(f"Parameter '{key}' for exact match placeholder '{obj}' not found. Returning placeholder string.")
                      return obj # Return original placeholder if key not found

            # Case 2: String contains placeholders mixed with other text
            else:
                def replace_match_in_string(match):
                    key = match.group(1)
                    if key in params:
                        # Convert param value to string for embedding
                        return str(params[key])
                    else:
                        logger.warning(f"Parameter '{key}' for embedded placeholder '{match.group(0)}' not found. Leaving placeholder.")
                        return match.group(0) # Keep the placeholder

                return re.sub(pattern, replace_match_in_string, obj)
        else:
            # Return non-dict/list/str types as is
            return obj

    # --- NEW METHOD: Convert Steps to Intent Plan ---
    def _convert_steps_to_intents(self, workflow_steps: List[Dict[str, Any]], objective: str) -> Dict[str, Any]:
        """
        Convert validated workflow steps to a structured intent plan.
        """
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        logger.info(f"Converting {len(workflow_steps)} steps to intent plan '{plan_id}' with objective: '{objective}'")

        intents = []
        # Keep track of which intent corresponds to which original step index
        step_index_to_intent_id = {}

        for i, step in enumerate(workflow_steps):
            step_type = step.get("type", "UNKNOWN")
            intent_id = f"intent_{uuid.uuid4().hex[:8]}"
            step_index_to_intent_id[i] = intent_id # Map original index to new ID

            # Determine parameters/input field based on step type
            params_or_input = {}
            if step_type == "EXECUTE":
                params_or_input = step.get("input", {})
            elif step_type in ["DEFINE", "CREATE"]: # Add other types that might have params if needed
                 # Define/Create might have parameters separate from core definition
                 # This assumes params are stored directly, adjust if needed
                 params_or_input = {k: v for k, v in step.items() if k not in ["type", "item_type", "name", "description", "code_snippet", "output_var"]} # Crude way to get extra params
            elif step_type == "RETURN":
                 # Return usually has 'value', treat it like a param for consistency?
                 params_or_input = {"value": step.get("value")}


            # Map dependencies based on output_var references in params/input
            depends_on = []
            # Convert params dict to string to search for placeholders - might be fragile
            # A better approach might involve recursively searching string values within the params dict
            params_str = safe_json_dumps(params_or_input) # Use safe dump to handle complex types
            # Simpler pattern to find variable names like {{var_name}} - assumes no 'params.' prefix needed here
            # as substitution should have happened before this step if using {{params.xyz}}
            # If substitution uses different syntax or output_vars are used directly, adjust pattern
            var_pattern = r'{{\s*([\w_]+)\s*}}'
            found_vars = set(re.findall(var_pattern, params_str))

            if found_vars:
                 logger.debug(f"Found potential variable references in step {i+1} params: {found_vars}")
                 for var_name in found_vars:
                     # Find the intent (from previous steps) that sets this variable via output_var
                     found_dependency = False
                     for j, prev_step in enumerate(workflow_steps[:i]): # Look only at previous steps
                         if prev_step.get("output_var") == var_name:
                             if j in step_index_to_intent_id:
                                 depends_on.append(step_index_to_intent_id[j])
                                 logger.debug(f"Mapped dependency: Step {i+1} ({intent_id}) depends on Step {j+1} ({step_index_to_intent_id[j]}) via var '{var_name}'")
                                 found_dependency = True
                                 break # Assume first match is the correct one
                             else:
                                 logger.warning(f"Could not find intent ID for dependency step index {j} providing var '{var_name}'")
                     if not found_dependency:
                         logger.warning(f"Could not find previous step defining output_var '{var_name}' needed by step {i+1}")


            # Create the intent dictionary
            intent = {
                "intent_id": intent_id,
                "step_type": step_type,
                "component_type": step.get("item_type", "UNKNOWN"),
                "component_name": step.get("name", "N/A"), # Provide default
                "action": step_type.lower(), # Use step type as action
                "params": params_or_input, # Use the extracted params/input
                "justification": step.get("description", f"Step {i+1} in workflow: {step_type} {step.get('item_type', '')} '{step.get('name', '')}'"),
                "output_var": step.get("output_var"), # Include output_var if present
                "depends_on": sorted(list(set(depends_on))), # Ensure unique, sorted dependencies
                "status": "PENDING" # Initial status for all intents
            }
            # Remove None value for output_var if not present
            if intent["output_var"] is None:
                del intent["output_var"]

            intents.append(intent)

        # Create the full intent plan structure
        intent_plan = {
            "plan_id": plan_id,
            "title": f"Workflow Intent Plan ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
            "description": f"Automatically generated from workflow steps for scenario: {workflow_steps[0].get('scenario_name', 'N/A') if workflow_steps else 'N/A'}", # Extract scenario name if possible
            "objective": objective if objective else "Objective not explicitly found in workflow parameters.",
            "intents": intents,
            "status": "PENDING_REVIEW", # Overall plan status
            "created_at": datetime.now().isoformat()
        }

        return intent_plan

    # --- MODIFIED: Validate Steps and Optionally Convert to Intents ---
    def _validate_steps(self, steps: List[Dict[str, Any]], intent_mode: bool = False) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Perform basic validation on workflow steps.
        If intent_mode is True and intent review is globally enabled,
        convert the steps into an intent plan dictionary.
        Otherwise, return the validated list of steps.
        """
        if not isinstance(steps, list):
             raise ValueError("Workflow 'steps' must be a list.")

        validated_steps_list = []
        objective = "" # Initialize objective extraction

        for i, step in enumerate(steps):
            step_num = i + 1
            if not isinstance(step, dict):
                 raise ValueError(f"Step {step_num} is not a dictionary: {step}")

            # --- Basic Step Structure Validation ---
            if "type" not in step:
                raise ValueError(f"Step {step_num} missing required 'type' field: {step}")
            step_type = step.get("type")
            valid_types = ["DEFINE", "CREATE", "EXECUTE", "RETURN"] # Add any other valid types
            if step_type not in valid_types:
                raise ValueError(f"Step {step_num} invalid type '{step_type}'. Must be one of {valid_types}.")

            required_keys = set()
            # Define required keys based on type (adjust as needed)
            if step_type == "DEFINE": required_keys = {"item_type", "name", "description"} # Code snippet might be optional or part of description/params
            elif step_type == "CREATE": required_keys = {"item_type", "name"}
            elif step_type == "EXECUTE": required_keys = {"item_type", "name"} # Input might be optional
            elif step_type == "RETURN": required_keys = {"value"}

            present_keys = set(step.keys())
            missing_keys = required_keys - present_keys
            if missing_keys:
                 raise ValueError(f"Step {step_num} (type: {step_type}) missing required keys: {missing_keys}. Step: {step}")

            # Validate EXECUTE input is a dict if present
            if step_type == "EXECUTE" and "input" in step:
                 if not isinstance(step["input"], dict):
                      logger.error(f"Validation Error in Step {step_num}: 'input' field is not a dictionary. Found type: {type(step['input'])}. Step content: {step}")
                      raise ValueError(f"Step {step_num} (EXECUTE): 'input' field MUST be a dictionary (mapping), but found type {type(step['input'])}.")

                 # --- Try to extract objective if in intent mode (only need to find it once) ---
                 if intent_mode and not objective:
                     input_params = step["input"]
                     for key, value in input_params.items():
                         # Check common keys for objective/goal, case-insensitive
                         if isinstance(key, str) and key.lower() in ["objective", "goal", "user_request", "task_description"]:
                             if isinstance(value, str) and value: # Ensure it's a non-empty string
                                objective = value
                                logger.debug(f"Extracted objective from step {step_num}, key '{key}': '{objective}'")
                                break # Found it, stop searching in this step's input
                     if objective:
                         pass # Continue validation, but don't search for objective anymore


            validated_steps_list.append(step) # Add validated step to the list

        # --- Conditional Conversion to Intent Plan ---
        # Check both the flag passed to the function AND the global config again
        # This ensures conversion only happens if explicitly requested *and* globally enabled
        if intent_mode and config.INTENT_REVIEW_ENABLED:
            logger.info("Intent mode active and enabled, converting validated steps to intent plan.")
            return self._convert_steps_to_intents(validated_steps_list, objective)
        else:
            # If not in intent mode, or intent review is disabled globally, return the validated steps list
            if intent_mode and not config.INTENT_REVIEW_ENABLED:
                logger.warning("Intent mode requested but INTENT_REVIEW_ENABLED is False in config. Returning direct execution steps.")
            logger.info("Returning validated steps list for direct execution.")
            return validated_steps_list