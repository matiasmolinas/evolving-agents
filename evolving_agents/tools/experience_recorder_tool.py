import json
import logging
from typing import List, Optional, Dict, Any

from evolving_agents.agent_bus.smart_agent_bus import SmartAgentBus
from evolving_agents.tools.agent_bus.request_agent_tool import RequestAgentTool


class ExperienceRecorderTool:
    """
    Records the details and outcome of a completed task or workflow
    into the persistent agent memory via the MemoryManagerAgent.
    """
    name: str = "experience_recorder_tool"
    description: str = (
        "Records the details and outcome of a completed task or workflow "
        "into the persistent agent memory."
    )

    def __init__(self, smart_agent_bus: SmartAgentBus):
        """
        Initializes the ExperienceRecorderTool.

        Args:
            smart_agent_bus: An instance of SmartAgentBus.
        """
        self.smart_agent_bus = smart_agent_bus
        self.request_agent_tool = RequestAgentTool(self.smart_agent_bus) # Corrected instantiation
        self.logger = logging.getLogger(__name__)
        # Ensure logger is configured (moved to main script or higher level for actual usage)
        # logging.basicConfig(level=logging.INFO) 

    async def record_experience(
        self,
        primary_goal_description: str,
        sub_task_description: Optional[str],
        involved_components: List[Dict[str, str]],
        input_context_summary: str,
        key_decisions_made: List[Dict[str, str]],
        final_outcome: str, 
        final_outcome_reason: Optional[str],
        output_summary: str,
        feedback_signals: Optional[List[Dict[str, Any]]] = None, 
        tags: Optional[List[str]] = None,
        agent_version: Optional[str] = None,
        tool_versions: Optional[List[Dict[str, str]]] = None, 
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        initiating_agent_id: Optional[str] = None 
    ) -> Dict[str, Any]:
        """
        Structures and sends task/workflow completion data to MemoryManagerAgent.
        
        Args:
            primary_goal_description: The main goal.
            sub_task_description: The specific sub-task this experience relates to.
            involved_components: List of dicts, e.g.,
                                 `{"component_id": "id", "component_name": "name",
                                   "component_type": "AGENT/TOOL", "usage_description": "how it was used"}`.
            input_context_summary: Text summary of the initial context.
            key_decisions_made: List of dicts, e.g.,
                                `{"decision_summary": "text", "decision_reasoning": "text",
                                  "timestamp": "ISO_datetime_str_optional"}`.
            final_outcome: Text (e.g., "success", "failure").
            final_outcome_reason: Optional text explaining the outcome.
            output_summary: Text summary of the final result/output.
            feedback_signals: Optional list of dicts, e.g.,
                              `{"feedback_source": "user", "feedback_content": "text",
                                "feedback_rating": 4.5, "timestamp": "ISO_datetime_str_optional"}`.
            tags: Optional list of strings for categorization.
            agent_version: Optional version of the agent recording/performing the task.
            tool_versions: Optional list of tools and their versions used.
            session_id: Optional ID for the session this experience belongs to.
            run_id: Optional ID for the specific run/execution.
            initiating_agent_id: Optional ID of the agent that initiated this task/workflow.
        """
        self.logger.info(f"Recording experience for goal: {primary_goal_description}")

        experience_data = {
            "primary_goal_description": primary_goal_description,
            "sub_task_description": sub_task_description,
            "involved_components": involved_components or [],
            "input_context_summary": input_context_summary,
            "key_decisions_made": key_decisions_made or [],
            "status": final_outcome, 
            "status_reason": final_outcome_reason,
            "output_summary": output_summary,
            "feedback": feedback_signals or [], 
            "tags": tags or [],
            "agent_version": agent_version,
            "tool_versions": tool_versions or [],
            "session_id": session_id,
            "run_id": run_id,
            "initiating_agent_id": initiating_agent_id
        }
        
        experience_data_cleaned = {k: v for k, v in experience_data.items() if v is not None}

        # Similar to ContextBuilderTool, assuming RequestAgentTool.run expects a dictionary
        # that matches its input schema (RequestInput).
        request_input = {
            "capability": "store_agent_experience",
            "content": { # Content for 'store_agent_experience' capability
                "experience_data": experience_data_cleaned
            },
            "specific_agent": "MemoryManagerAgent"
        }

        try:
            self.logger.debug(f"Sending experience data to MemoryManagerAgent: {json.dumps(request_input, indent=2)}")
            response_output = await self.request_agent_tool.run(request_input)
            response_str = response_output.result if hasattr(response_output, 'result') else str(response_output)
            response_json = json.loads(response_str) # RequestAgentTool returns JSON string


            if response_json and response_json.get("status") == "success":
                self.logger.info(
                    f"Successfully recorded experience. Response from MMA: {response_json}"
                )
            elif response_json:
                self.logger.warning(
                    f"MemoryManagerAgent (via RequestAgentTool) returned an issue for recording experience: {response_json.get('message', str(response_json))}"
                )
            else:
                 self.logger.error("Received no valid JSON response from MemoryManagerAgent via RequestAgentTool.")
                 return {"status": "error", "message": "No valid response from MemoryManagerAgent (via RequestAgentTool)"}
            
            return response_json

        except Exception as e:
            self.logger.error(f"Error calling MemoryManagerAgent to record experience: {e}", exc_info=True)
            return {"status": "error", "message": f"Failed to record experience due to an internal error: {str(e)}"}

# Example (Conceptual)
# (example_usage removed for brevity in subtask, it's for local testing)
