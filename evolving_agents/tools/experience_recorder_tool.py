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
        self.request_agent_tool = RequestAgentTool(smart_agent_bus=self.smart_agent_bus)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO) # Ensure logger is configured

    async def record_experience(
        self,
        primary_goal_description: str,
        sub_task_description: Optional[str],
        involved_components: List[Dict[str, str]],
        input_context_summary: str,
        key_decisions_made: List[Dict[str, str]],
        final_outcome: str, # e.g., "success", "failure", "partial_success"
        final_outcome_reason: Optional[str],
        output_summary: str,
        # Defaulting feedback_signals and tags to None if not provided
        feedback_signals: Optional[List[Dict[str, Any]]] = None, # Note: Value can be Any for rating
        tags: Optional[List[str]] = None,
        # Optional fields that might be part of a comprehensive experience schema
        agent_version: Optional[str] = None,
        tool_versions: Optional[List[Dict[str, str]]] = None, # e.g., [{"tool_name": "x", "version": "1.0"}]
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        initiating_agent_id: Optional[str] = None # ID of the agent that initiated the task
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


        Returns:
            A dictionary containing the response from MemoryManagerAgent
            (e.g., {"status": "success", "experience_id": "xyz..."} or
             {"status": "error", "message": "..."}).
        """
        self.logger.info(f"Recording experience for goal: {primary_goal_description}")

        experience_data = {
            "primary_goal_description": primary_goal_description,
            "sub_task_description": sub_task_description,
            "involved_components": involved_components or [],
            "input_context_summary": input_context_summary,
            "key_decisions_made": key_decisions_made or [],
            "status": final_outcome,  # Directly using final_outcome as status
            "status_reason": final_outcome_reason,
            "output_summary": output_summary,
            "feedback": feedback_signals or [], # Renamed from feedback_signals to feedback for schema
            "tags": tags or [],
            # Optional fields from parameters
            "agent_version": agent_version,
            "tool_versions": tool_versions or [],
            "session_id": session_id,
            "run_id": run_id,
            "initiating_agent_id": initiating_agent_id
            # Fields like 'timestamp' and 'experience_id' are typically generated by the
            # MongoExperienceStoreTool/MemoryManagerAgent upon storage.
            # Embedding fields are also generated there.
        }
        
        # Remove None values for optional fields not provided, to keep payload clean
        experience_data_cleaned = {k: v for k, v in experience_data.items() if v is not None}


        prompt_payload = {
            "capability": "store_agent_experience",
            "args": {
                "experience_data": experience_data_cleaned
            }
        }

        try:
            self.logger.debug(f"Sending experience data to MemoryManagerAgent: {json.dumps(prompt_payload, indent=2)}")
            response_json = await self.request_agent_tool.run(
                agent_id="MemoryManagerAgent",
                prompt=json.dumps(prompt_payload)
            )

            if response_json and response_json.get("status") == "success":
                self.logger.info(
                    f"Successfully recorded experience with ID: {response_json.get('experience_id')}"
                )
            elif response_json:
                self.logger.warning(
                    f"MemoryManagerAgent returned an issue for recording experience: {response_json.get('message', str(response_json))}"
                )
            else:
                 self.logger.error("Received no valid JSON response from MemoryManagerAgent.")
                 return {"status": "error", "message": "No valid response from MemoryManagerAgent"}
            
            return response_json # Return the full response from MemoryManagerAgent

        except Exception as e:
            self.logger.error(f"Error calling MemoryManagerAgent to record experience: {e}", exc_info=True)
            return {"status": "error", "message": f"Failed to record experience due to an internal error: {str(e)}"}

# Example (Conceptual - requires running async environment and mocked dependencies)
async def example_usage():
    # Mock dependencies
    class MockSmartAgentBus:
        async def publish_message(self, topic, message): pass
        async def subscribe(self, topic, callback): pass
        async def request_agent_sync(self, agent_id: str, prompt: str, timeout: Optional[float] = 10.0) -> Any: pass


    logging.basicConfig(level=logging.DEBUG)
    mock_bus = MockSmartAgentBus()

    # Patch RequestAgentTool.run for this example
    original_request_agent_run = RequestAgentTool.run
    async def mock_request_agent_run(self, agent_id: str, prompt: str):
        prompt_data = json.loads(prompt)
        if agent_id == "MemoryManagerAgent" and prompt_data["capability"] == "store_agent_experience":
            exp_data = prompt_data["args"]["experience_data"]
            logging.info(f"Mock MemoryManagerAgent received for storage: {exp_data.get('primary_goal_description')}")
            return {"status": "success", "experience_id": f"mock_exp_{hash(exp_data.get('primary_goal_description'))}"}
        return {"status": "error", "message": "Unknown capability or agent in mock"}
    RequestAgentTool.run = mock_request_agent_run


    recorder = ExperienceRecorderTool(smart_agent_bus=mock_bus)

    # Record an experience
    result = await recorder.record_experience(
        primary_goal_description="Develop a customer onboarding workflow.",
        sub_task_description="Implement email verification step.",
        involved_components=[
            {"component_id": "email_service_v1", "component_name": "EmailSender", "component_type": "TOOL", "usage_description": "Sent verification email."},
            {"component_id": "UserDB_v3", "component_name": "UserDatabase", "component_type": "SERVICE", "usage_description": "Updated user status."}
        ],
        input_context_summary="New user signed up; requires email verification before full access.",
        key_decisions_made=[
            {"decision_summary": "Used SendGrid for email delivery.", "decision_reasoning": "High deliverability rates and existing integration.", "timestamp": "2023-10-26T10:00:00Z"},
            {"decision_summary": "Verification link valid for 24 hours.", "decision_reasoning": "Security best practice."}
        ],
        final_outcome="success",
        final_outcome_reason="Email verification step completed and user status updated.",
        output_summary="User 'john.doe@example.com' successfully verified their email.",
        feedback_signals=[
            {"feedback_source": "system_metric", "feedback_content": "Email delivery success rate: 99.8%", "feedback_rating": 5.0}
        ],
        tags=["onboarding", "email_verification", "customer_workflow"],
        agent_version="SystemAgent_v1.2",
        session_id="session_abc123",
        run_id = "run_xyz789",
        initiating_agent_id = "UserRequestRouterAgent_v0.5"
    )

    print("\n--- Record Experience Result ---")
    print(json.dumps(result, indent=2))
    
    # Example for a failed task
    result_failure = await recorder.record_experience(
        primary_goal_description="Process batch payment file.",
        sub_task_description="Validate payment data.",
        involved_components=[{"component_id": "validator_v1", "component_name": "PaymentValidator", "component_type": "TOOL", "usage_description": "Used for data validation."}],
        input_context_summary="Received batch payment file 'payments_20231026.csv'.",
        key_decisions_made=[],
        final_outcome="failure",
        final_outcome_reason="Validation failed: File contained records with missing account numbers.",
        output_summary="Batch processing aborted due to validation errors.",
        tags=["payment_processing", "validation", "error"],
        agent_version="PaymentProcessorAgent_v2.0"
    )
    print("\n--- Record Experience Result (Failure) ---")
    print(json.dumps(result_failure, indent=2))

    # Restore original methods if other tests follow in same session
    RequestAgentTool.run = original_request_agent_run

if __name__ == "__main__":
    import asyncio
    # asyncio.run(example_usage())
    pass
