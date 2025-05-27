import json
import logging
from typing import List, Optional, Dict, Any

from evolving_agents.agent_bus.smart_agent_bus import SmartAgentBus
from evolving_agents.core.llm_service import LLMService
from evolving_agents.core.smart_context import SmartContext, Message, ContextEntry
from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.tools.agent_bus.request_agent_tool import RequestAgentTool
from evolving_agents.tools.smart_library.search_component_tool import SearchComponentTool


class ContextBuilderTool:
    """
    Dynamically constructs a task-specific SmartContext for an agent,
    incorporating relevant past experiences, summarized message history,
    and related library components.
    """
    name: str = "context_builder_tool"
    description: str = (
        "Dynamically constructs a task-specific SmartContext for an agent, "
        "incorporating relevant past experiences, summarized message history, "
        "and related library components."
    )

    def __init__(
        self,
        smart_agent_bus: SmartAgentBus,
        smart_library: SmartLibrary,
        llm_service: LLMService  # llm_service might be used for future enhancements
    ):
        """
        Initializes the ContextBuilderTool.

        Args:
            smart_agent_bus: An instance of SmartAgentBus.
            smart_library: An instance of SmartLibrary.
            llm_service: An instance of LLMService (currently unused but available for future).
        """
        self.smart_agent_bus = smart_agent_bus
        self.smart_library = smart_library
        self.llm_service = llm_service  # Retain for potential future use

        # Instantiate dependent tools
        # Assuming RequestAgentTool and SearchComponentTool do not require async init
        self.request_agent_tool = RequestAgentTool(smart_agent_bus=self.smart_agent_bus)
        self.search_component_tool = SearchComponentTool(smart_library=self.smart_library)

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO) # Ensure logger is configured

    async def build_context(
        self,
        target_agent_id: str,
        assigned_sub_task_goal_description: str,
        workflow_context: SmartContext,
        historical_message_limit: int = 20,
        relevant_experience_limit: int = 3,
        relevant_component_limit: int = 5
    ) -> SmartContext:
        """
        Dynamically constructs an optimized SmartContext object for a target agent.

        Args:
            target_agent_id: ID of the agent for whom this context is being built.
            assigned_sub_task_goal_description: The specific goal/task description.
            workflow_context: The current broader SmartContext of the calling agent.
            historical_message_limit: Number of recent messages for summarization.
            relevant_experience_limit: Number of relevant past experiences to fetch.
            relevant_component_limit: Number of relevant library components to fetch.

        Returns:
            A new SmartContext object tailored for the sub-task.
        """
        self.logger.info(f"Building context for agent '{target_agent_id}' for task: '{assigned_sub_task_goal_description}'")

        new_smart_context = SmartContext(current_task=assigned_sub_task_goal_description)
        new_smart_context.metadata["target_agent_id"] = target_agent_id
        # Assuming workflow_context.metadata contains the ID of the agent owning that context
        new_smart_context.metadata["caller_agent_id"] = workflow_context.metadata.get("agent_id", "unknown_caller")
        new_smart_context.metadata["parent_workflow_id"] = workflow_context.metadata.get("workflow_id", "unknown_workflow")


        # 1. Retrieve Relevant Experiences
        try:
            experience_prompt_payload = {
                "capability": "retrieve_relevant_experiences",
                "args": {
                    "goal_description": assigned_sub_task_goal_description,
                    "limit": relevant_experience_limit
                }
            }
            self.logger.debug(f"Requesting experiences from MemoryManagerAgent with payload: {experience_prompt_payload}")
            # Assuming RequestAgentTool.run executes the request and returns the parsed JSON response
            # from the target agent's capability method.
            response_json = await self.request_agent_tool.run(
                agent_id="MemoryManagerAgent",  # Target agent ID
                prompt=json.dumps(experience_prompt_payload)
            )
            
            if response_json and response_json.get("status") == "success" and response_json.get("experiences"):
                experiences = response_json["experiences"]
                # Format experiences (e.g., list of concise summaries or key fields)
                formatted_experiences = [
                    {
                        "experience_id": exp.get("experience_id"),
                        "primary_goal": exp.get("primary_goal_description"),
                        "sub_task": exp.get("sub_task_description"),
                        "output_summary": exp.get("output_summary"),
                        "status": exp.get("status")
                    } for exp in experiences
                ]
                new_smart_context.add_data_entry(ContextEntry(
                    value=formatted_experiences,
                    semantic_key="relevant_past_experiences",
                    description="Past experiences relevant to the current sub-task."
                ))
                self.logger.info(f"Added {len(experiences)} relevant experiences to context.")
            elif response_json and response_json.get("status") == "error":
                 self.logger.warning(f"MemoryManagerAgent returned error for experiences: {response_json.get('message')}")
            else:
                self.logger.warning(f"No relevant experiences found or unexpected response: {response_json}")

        except Exception as e:
            self.logger.error(f"Error retrieving relevant experiences: {e}", exc_info=True)

        # 2. Summarize Message History
        try:
            if workflow_context.messages and historical_message_limit > 0:
                recent_messages = workflow_context.messages[-historical_message_limit:]
                
                # Serialize messages (assuming Message objects have a to_dict method or are dicts)
                serializable_messages = []
                for msg in recent_messages:
                    if hasattr(msg, 'to_dict'):
                        serializable_messages.append(msg.to_dict())
                    elif isinstance(msg, dict):
                         serializable_messages.append(msg)
                    else:
                        self.logger.warning(f"Message of type {type(msg)} could not be serialized, skipping.")


                if serializable_messages:
                    summary_prompt_payload = {
                        "capability": "summarize_message_history",
                        "args": {
                            "messages": serializable_messages,
                            "target_goal": assigned_sub_task_goal_description,
                            "max_summary_tokens": 300 # Example token limit
                        }
                    }
                    self.logger.debug(f"Requesting message summary from MemoryManagerAgent with payload: {summary_prompt_payload}")
                    response_json = await self.request_agent_tool.run(
                        agent_id="MemoryManagerAgent",
                        prompt=json.dumps(summary_prompt_payload)
                    )

                    if response_json and response_json.get("status") == "success" and response_json.get("summary"):
                        summary_text = response_json["summary"]
                        new_smart_context.add_data_entry(ContextEntry(
                            value=summary_text,
                            semantic_key="summarized_message_history",
                            description="Summary of recent messages relevant to the current sub-task."
                        ))
                        self.logger.info("Added summarized message history to context.")
                    elif response_json and response_json.get("status") == "error":
                        self.logger.warning(f"MemoryManagerAgent returned error for summary: {response_json.get('message')}")
                    else:
                        self.logger.warning(f"Failed to get message summary or unexpected response: {response_json}")
                else:
                    self.logger.info("No serializable messages found to summarize.")
            else:
                self.logger.info("No messages in workflow context or historical_message_limit is 0, skipping summarization.")
        except Exception as e:
            self.logger.error(f"Error summarizing message history: {e}", exc_info=True)

        # 3. Query SmartLibrary for Relevant Components
        try:
            self.logger.debug(f"Querying SmartLibrary for components related to: '{assigned_sub_task_goal_description}'")
            # Assuming SearchComponentTool.run returns a list of component dicts
            # and handles async execution if necessary.
            # Also assuming it takes 'query', 'limit', and 'search_type'
            components = await self.search_component_tool.run(
                query=assigned_sub_task_goal_description,
                limit=relevant_component_limit,
                search_type="semantic" # Or another appropriate search type
            )

            if components:
                # Format components (e.g., list of dicts with name, id, description)
                formatted_components = [
                    {
                        "id": comp.get("id") or comp.get("component_id"), # Handle potential variations in ID field name
                        "name": comp.get("name"),
                        "description": comp.get("description"),
                        "version": comp.get("version"),
                        "type": comp.get("type")
                    } for comp in components # Assuming components is a list of dicts
                ]
                new_smart_context.add_data_entry(ContextEntry(
                    value=formatted_components,
                    semantic_key="relevant_library_components",
                    description="Library components relevant to the current sub-task."
                ))
                self.logger.info(f"Added {len(components)} relevant library components to context.")
            else:
                self.logger.info("No relevant library components found.")
        except Exception as e:
            self.logger.error(f"Error querying SmartLibrary: {e}", exc_info=True)
        
        # Add the original assigned sub-task goal to the context data as well for clarity
        new_smart_context.add_data_entry(ContextEntry(
            value=assigned_sub_task_goal_description,
            semantic_key="assigned_sub_task_goal",
            description="The specific goal description for the target agent."
        ))

        self.logger.info(f"Context building complete for agent '{target_agent_id}'.")
        return new_smart_context

# Example (Conceptual - requires running async environment and mocked dependencies)
async def example_usage():
    # Mock dependencies
    class MockSmartAgentBus:
        async def publish_message(self, topic, message): # Added to satisfy SmartAgentBus ABC if needed
            pass
        async def subscribe(self, topic, callback): # Added to satisfy SmartAgentBus ABC if needed
            pass
        async def request_agent_sync(self, agent_id: str, prompt: str, timeout: Optional[float] = 10.0) -> Any: # Added
            pass

    class MockSmartLibrary:
        def __init__(self):
            self.logger = logging.getLogger(__name__)
        async def add_component(self, component_data: Dict[str, Any]): self.logger.info("Mock Add Comp")
        async def get_component(self, component_id: str, version: Optional[str] = None): return None
        async def search_components(self, query: str, limit: int = 10, search_type: str = "semantic", filters: Optional[Dict] = None):
            self.logger.info(f"Mock search_components: {query}")
            if "login" in query:
                return [{"id": "comp1", "name": "OAuth2Client", "description": "Handles OAuth2 login."}]
            return []
        async def update_component(self, component_id: str, update_data: Dict[str, Any]): self.logger.info("Mock Update")
        async def delete_component(self, component_id: str): self.logger.info("Mock Delete")


    class MockLLMService(LLMService):
        async def generate(self, prompt: str, **kwargs) -> str: return "Mocked LLM generation."
        async def embed(self, text: str) -> List[float]: return [0.1] * 10


    # Setup
    logging.basicConfig(level=logging.DEBUG)
    mock_bus = MockSmartAgentBus()
    mock_library = MockSmartLibrary()
    mock_llm = MockLLMService(api_key="dummy", default_model="dummy")

    # Patch RequestAgentTool.run and SearchComponentTool.run for this example
    original_request_agent_run = RequestAgentTool.run
    original_search_tool_run = SearchComponentTool.run

    async def mock_request_agent_run(self, agent_id: str, prompt: str):
        prompt_data = json.loads(prompt)
        if agent_id == "MemoryManagerAgent":
            if prompt_data["capability"] == "retrieve_relevant_experiences":
                return {"status": "success", "experiences": [{"experience_id": "exp1", "primary_goal_description": "Old login task"}]}
            if prompt_data["capability"] == "summarize_message_history":
                return {"status": "success", "summary": "Messages were about login requirements."}
        return {"status": "error", "message": "Unknown capability or agent"}

    async def mock_search_component_run(self, query: str, limit: int, search_type: str):
        if "login" in query:
            return [{"id": "comp1", "name": "OAuth2Client", "description": "Handles OAuth2 login."}]
        return []

    RequestAgentTool.run = mock_request_agent_run
    SearchComponentTool.run = mock_search_component_run


    context_builder = ContextBuilderTool(smart_agent_bus=mock_bus, smart_library=mock_library, llm_service=mock_llm)

    # Create a dummy workflow_context
    wf_context = SmartContext(current_task="Main workflow task")
    wf_context.metadata["agent_id"] = "SystemAgent"
    wf_context.metadata["workflow_id"] = "wf_main_123"
    wf_context.add_message(Message(sender_id="user", content="We need a login page."))
    wf_context.add_message(Message(sender_id="dev", content="Okay, any specific requirements?"))

    # Build context
    new_ctx = await context_builder.build_context(
        target_agent_id="AuthAgent",
        assigned_sub_task_goal_description="Implement the new login page feature.",
        workflow_context=wf_context
    )

    print("\n--- Built Context ---")
    print(f"Current Task: {new_ctx.current_task}")
    print(f"Target Agent ID: {new_ctx.metadata.get('target_agent_id')}")
    print(f"Caller Agent ID: {new_ctx.metadata.get('caller_agent_id')}")
    print(f"Parent Workflow ID: {new_ctx.metadata.get('parent_workflow_id')}")

    if "relevant_past_experiences" in new_ctx.data:
        print(f"Relevant Experiences: {new_ctx.data['relevant_past_experiences'].value}")
    if "summarized_message_history" in new_ctx.data:
        print(f"Summarized History: {new_ctx.data['summarized_message_history'].value}")
    if "relevant_library_components" in new_ctx.data:
        print(f"Relevant Components: {new_ctx.data['relevant_library_components'].value}")
    if "assigned_sub_task_goal" in new_ctx.data:
        print(f"Assigned Goal in Data: {new_ctx.data['assigned_sub_task_goal'].value}")
        
    # Restore original methods if other tests follow in same session
    RequestAgentTool.run = original_request_agent_run
    SearchComponentTool.run = original_search_tool_run


if __name__ == "__main__":
    import asyncio
    # To run this example, you'd need to ensure the SmartContext, Message, ContextEntry
    # classes are defined and that the mocked dependencies are sufficient.
    # asyncio.run(example_usage())
    pass
