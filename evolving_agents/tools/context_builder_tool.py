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
        self.request_agent_tool = RequestAgentTool(self.smart_agent_bus) # Corrected instantiation
        self.search_component_tool = SearchComponentTool(smart_library=self.smart_library) # Assuming this is correct

        self.logger = logging.getLogger(__name__)
        # Ensure logger is configured (moved to main script or higher level for actual usage)
        # logging.basicConfig(level=logging.INFO) 

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
        new_smart_context.metadata["caller_agent_id"] = workflow_context.metadata.get("agent_id", "unknown_caller")
        new_smart_context.metadata["parent_workflow_id"] = workflow_context.metadata.get("workflow_id", "unknown_workflow")


        # 1. Retrieve Relevant Experiences
        try:
            experience_request_input = {
                "capability": "retrieve_relevant_experiences",
                "content": { 
                    "goal_description": assigned_sub_task_goal_description,
                    "limit": relevant_experience_limit
                },
                "specific_agent": "MemoryManagerAgent" 
            }
            
            self.logger.debug(f"Requesting experiences from MemoryManagerAgent with input: {experience_request_input}")
            
            response_output = await self.request_agent_tool.run(experience_request_input)
            response_str = response_output.result if hasattr(response_output, 'result') else str(response_output)
            response_json = json.loads(response_str) 
            
            if response_json and response_json.get("status") == "success":
                retrieved_content = response_json.get("content", {})
                experiences = retrieved_content.get("experiences", []) 
                                                                      
                if experiences:
                    formatted_experiences = [
                        {
                            "experience_id": exp.get("experience_id"),
                            "primary_goal": exp.get("primary_goal_description"), 
                            "sub_task": exp.get("sub_task_description"),
                            "output_summary": exp.get("output_summary"),
                            "status": exp.get("status") 
                        } for exp in experiences
                    ]
                    new_smart_context.data["relevant_past_experiences"] = ContextEntry(
                        value=formatted_experiences,
                        semantic_key="relevant_past_experiences"
                    )
                    self.logger.info(f"Added {len(experiences)} relevant experiences to context.")
                else:
                    self.logger.info("No experiences found in successful response from MemoryManagerAgent.")
            elif response_json and response_json.get("status") == "error":
                 self.logger.warning(f"MemoryManagerAgent (via RequestAgentTool) returned error for experiences: {response_json.get('message')}")
            else:
                self.logger.warning(f"No relevant experiences found or unexpected response: {response_json}")

        except Exception as e:
            self.logger.error(f"Error retrieving relevant experiences: {e}", exc_info=True)

        # 2. Summarize Message History
        try:
            if workflow_context.messages and historical_message_limit > 0:
                recent_messages = workflow_context.messages[-historical_message_limit:]
                
                serializable_messages = []
                for msg in recent_messages:
                    if isinstance(msg, Message): 
                         serializable_messages.append({"sender_id": msg.sender_id, "receiver_id": msg.receiver_id, "content": msg.content, "timestamp": msg.timestamp, "version": msg.version})
                    elif isinstance(msg, dict):
                         serializable_messages.append(msg)
                    else:
                        self.logger.warning(f"Message of type {type(msg)} could not be serialized, skipping.")

                if serializable_messages:
                    summary_request_input = {
                        "capability": "summarize_message_history",
                        "content": { 
                            "messages": serializable_messages,
                            "target_goal": assigned_sub_task_goal_description
                        },
                        "specific_agent": "MemoryManagerAgent"
                    }
                    self.logger.debug(f"Requesting message summary from MemoryManagerAgent with input: {summary_request_input}")
                    response_output = await self.request_agent_tool.run(summary_request_input)
                    response_str = response_output.result if hasattr(response_output, 'result') else str(response_output)
                    response_json = json.loads(response_str)

                    if response_json and response_json.get("status") == "success":
                        retrieved_content = response_json.get("content", {})
                        summary_text = retrieved_content.get("summary") 
                        if summary_text:
                            new_smart_context.data["summarized_message_history"] = ContextEntry(
                                value=summary_text,
                                semantic_key="summarized_message_history"
                            )
                            self.logger.info("Added summarized message history to context.")
                        else:
                            self.logger.info("No summary found in successful response from MemoryManagerAgent.")
                    elif response_json and response_json.get("status") == "error":
                        self.logger.warning(f"MemoryManagerAgent (via RequestAgentTool) returned error for summary: {response_json.get('message')}")
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
            
            search_tool_input = {"query": assigned_sub_task_goal_description, "limit": relevant_component_limit}
            search_output = await self.search_component_tool.run(search_tool_input) 
            
            components_data_str = search_output.result if hasattr(search_output, 'result') else str(search_output)
            components_data = json.loads(components_data_str)

            if components_data and components_data.get("status") == "success":
                components = components_data.get("results", [])
                if components:
                    formatted_components = [
                        {
                            "id": comp.get("id"), 
                            "name": comp.get("name"),
                            "description": comp.get("description"),
                            "version": comp.get("version"),
                            "record_type": comp.get("record_type") 
                        } for comp in components 
                    ]
                    new_smart_context.data["relevant_library_components"] = ContextEntry(
                        value=formatted_components,
                        semantic_key="relevant_library_components"
                    )
                    self.logger.info(f"Added {len(components)} relevant library components to context.")
                else:
                    self.logger.info("No relevant library components found by SearchComponentTool.")
            else:
                self.logger.warning(f"SearchComponentTool did not return success or results: {components_data}")
        except Exception as e:
            self.logger.error(f"Error querying SmartLibrary: {e}", exc_info=True)
        
        new_smart_context.data["assigned_sub_task_goal"] = ContextEntry(
            value=assigned_sub_task_goal_description,
            semantic_key="assigned_sub_task_goal"
        )

        self.logger.info(f"Context building complete for agent '{target_agent_id}'.")
        return new_smart_context

# Example (Conceptual)
# (example_usage removed for brevity in subtask, it's for local testing)
