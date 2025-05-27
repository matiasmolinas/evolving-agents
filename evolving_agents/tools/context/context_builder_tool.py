import logging
import json
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

from pydantic import BaseModel, Field, field_validator

from evolving_agents.core.base import BaseTool
from evolving_agents.agent_bus.smart_agent_bus import SmartAgentBus
from evolving_agents.smart_library.smart_library import SmartLibrary # Assuming this path
from evolving_agents.core.llm_service import LLMService
# from evolving_agents.core.smart_context import SmartContext, Message # If we were to build the object

# Configure logging
logger = logging.getLogger(__name__)

class ContextBuilderInput(BaseModel):
    """
    Input schema for the ContextBuilderTool.
    """
    target_agent_id: str = Field(..., description="ID or description of the agent for whom the context is being built.")
    sub_task_description: str = Field(..., description="The specific sub-task description requiring context.")
    current_messages: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="List of message dicts from current context.")
    
    max_relevant_experiences: int = Field(default=3, gt=0, le=10, description="Max number of relevant experiences to retrieve.")
    max_messages_to_summarize: int = Field(default=20, gt=0, le=100, description="Max number of recent messages to consider for summarization.")
    
    include_message_summary: bool = Field(default=True, description="Whether to include a summary of current messages.")
    include_relevant_experiences: bool = Field(default=True, description="Whether to include relevant past experiences.")
    include_library_components: bool = Field(default=True, description="Whether to include relevant library components.")

class ContextBuilderTool(BaseTool):
    """
    Dynamically constructs an optimized SmartContext for a sub-task by
    retrieving relevant past experiences, summarizing message history,
    and finding relevant library components.
    """
    name: str = "ContextBuilderTool"
    description: str = (
        "Dynamically constructs an optimized SmartContext for a sub-task by retrieving relevant "
        "past experiences, summarizing message history, and finding relevant library components."
    )
    input_schema: Dict[str, Any] = ContextBuilderInput.model_json_schema()
    # Output is a dictionary representing SmartContext data
    output_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "current_task": {"type": "string"},
            "data": {"type": "object"},
            "messages": {"type": "array", "items": {"type": "object"}},
            "metadata": {"type": "object"}
        },
        "description": "A dictionary representing the structured data for a SmartContext."
    }

    def __init__(
        self,
        agent_bus: SmartAgentBus,
        smart_library: SmartLibrary,
        memory_manager_agent_id: str,
        llm_service: Optional[LLMService] = None, # For any local LLM tasks, if needed in future
    ):
        super().__init__()
        if agent_bus is None:
            raise ValueError("agent_bus cannot be None for ContextBuilderTool")
        if smart_library is None:
            raise ValueError("smart_library cannot be None for ContextBuilderTool")
        if not memory_manager_agent_id:
            raise ValueError("memory_manager_agent_id must be provided.")

        self.agent_bus = agent_bus
        self.smart_library = smart_library
        self.memory_manager_agent_id = memory_manager_agent_id
        self.llm_service = llm_service # Currently unused, but available for future enhancements

    async def _request_mma_capability(
        self,
        task_description: str,
        timeout: Optional[int] = 60 # TODO: Implement timeout in SmartAgentBus if needed
    ) -> Any:
        """
        Helper method to send a task to the MemoryManagerAgent via the SmartAgentBus.
        """
        bus_payload = {"task_description": task_description}
        bus_capability_name = "process_task" # Generic capability for ReAct agent

        logger.debug(f"Requesting MMA ({self.memory_manager_agent_id}) for task: {task_description[:200]}...")
        try:
            response = await self.agent_bus.request_capability(
                target_agent_id=self.memory_manager_agent_id,
                capability=bus_capability_name,
                content=bus_payload,
                # timeout=timeout # Pass timeout if bus supports it
            )
            logger.debug(f"MMA response received: {type(response)}")
            return response
        except Exception as e:
            logger.error(f"Error requesting MMA capability for task '{task_description[:100]}...': {e}", exc_info=True)
            return None # Or a more specific error structure

    async def run(self, **kwargs: Any) -> Dict[str, Any]:
        try:
            config = ContextBuilderInput(**kwargs)
        except Exception as e: # Pydantic's ValidationError
            logger.error(f"Input validation error for ContextBuilderTool: {e}", exc_info=True)
            # Return a minimal context indicating the error
            return {
                "current_task": kwargs.get("sub_task_description", "Unknown due to validation error"),
                "data": {"error": f"Input validation error: {e}"},
                "messages": [],
                "metadata": {"source_tool": self.name, "timestamp": datetime.now(timezone.utc).isoformat()}
            }

        new_smart_context_data: Dict[str, Any] = {
            "current_task": config.sub_task_description,
            "data": {}, # For structured data like experiences, library components
            "messages": [], # For summaries or important original messages
            "metadata": {
                "source_tool": self.name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "target_agent_id": config.target_agent_id,
                "builder_config": config.model_dump(exclude_none=True)
            }
        }

        # 1. Retrieve Relevant Experiences
        if config.include_relevant_experiences:
            experience_task = (
                f"Find experiences relevant to the task: '{config.sub_task_description}'. "
                f"Return top {config.max_relevant_experiences} results. "
                f"Include experience_id, sub_task_description, and output_summary for each."
            )
            mma_response = await self._request_mma_capability(experience_task)
            
            if mma_response:
                # MemoryManagerAgent (ReAct) might return a list of dicts directly if its final answer is structured,
                # or a string that needs parsing, or a dict containing the results.
                # Assuming SemanticExperienceSearchTool (used by MMA) returns a list of dicts
                # and MMA's ReAct prompt is set up to return this list as the final answer.
                if isinstance(mma_response, list) and all(isinstance(item, dict) for item in mma_response):
                    new_smart_context_data['data']['relevant_experiences_summary'] = mma_response
                    logger.info(f"Added {len(mma_response)} relevant experiences to context.")
                elif isinstance(mma_response, dict) and "final_answer" in mma_response and isinstance(mma_response["final_answer"], list): # Common ReAct pattern
                    new_smart_context_data['data']['relevant_experiences_summary'] = mma_response["final_answer"]
                    logger.info(f"Added {len(mma_response['final_answer'])} relevant experiences to context from MMA's final_answer.")
                else:
                    logger.warning(f"Unexpected format for relevant experiences from MMA: {type(mma_response)}. Response: {str(mma_response)[:500]}")
                    new_smart_context_data['data']['relevant_experiences_error'] = f"Unexpected response format from MMA for experiences: {str(mma_response)[:200]}"
            else:
                logger.warning("No response or error while retrieving relevant experiences from MMA.")
                new_smart_context_data['data']['relevant_experiences_error'] = "Failed to retrieve experiences from MemoryManagerAgent."

        # 2. Summarize Message History
        if config.include_message_summary and config.current_messages:
            messages_to_summarize = config.current_messages[-config.max_messages_to_summarize:]
            if messages_to_summarize:
                # Format messages for the prompt to MMA. JSON is less ambiguous than raw list of dicts in a string.
                messages_json_string = json.dumps(messages_to_summarize)
                
                summary_task = (
                    f"Summarize the following message history (JSON format): {messages_json_string}. "
                    f"The summary should focus on information relevant to the goal: '{config.sub_task_description}'. "
                    f"Provide a concise summary text."
                )
                mma_response = await self._request_mma_capability(summary_task)

                if mma_response:
                    # MemoryManagerAgent (ReAct) using MessageSummarizationTool should return a string summary.
                    summary_text = None
                    if isinstance(mma_response, str):
                        summary_text = mma_response
                    elif isinstance(mma_response, dict) and "final_answer" in mma_response and isinstance(mma_response["final_answer"], str):
                        summary_text = mma_response["final_answer"]
                    
                    if summary_text and not summary_text.startswith("Error summarizing messages:") and not summary_text.startswith("Summary generation resulted in an empty response."):
                        summary_message = {
                            "sender_id": self.name, # Or "ContextBuilderTool"
                            "content": {"summary": summary_text, "original_task_for_summary": config.sub_task_description},
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "type": "context_summary"
                        }
                        new_smart_context_data['messages'].append(summary_message)
                        logger.info(f"Added message summary to context: {summary_text[:100]}...")
                    else:
                        logger.warning(f"Failed to get a valid summary from MMA. Response: {summary_text or mma_response}")
                        new_smart_context_data['data']['message_summary_error'] = summary_text or f"Unexpected response from MMA for summary: {str(mma_response)[:200]}"
                else:
                    logger.warning("No response or error while requesting message summary from MMA.")
                    new_smart_context_data['data']['message_summary_error'] = "Failed to get message summary from MemoryManagerAgent."
        
        # 3. Query SmartLibrary
        if config.include_library_components:
            try:
                logger.debug(f"Querying SmartLibrary with: '{config.sub_task_description}'")
                # Assuming limit is part of semantic_search args or a default. Let's use 5 as per prompt.
                library_components = await self.smart_library.semantic_search(
                    text_query=config.sub_task_description,
                    limit=5
                )
                if library_components:
                    # Store metadata: name, description, id (adapt to actual SmartLibraryItem structure)
                    new_smart_context_data['data']['relevant_library_components'] = [
                        {"id": comp.id, "name": comp.name, "description": comp.description, "score": comp.score} 
                        for comp in library_components # Assuming SmartLibraryItem has these attrs
                    ]
                    logger.info(f"Added {len(library_components)} library components to context.")
                else:
                    logger.info("No relevant library components found by SmartLibrary.")
                    new_smart_context_data['data']['relevant_library_components'] = []
            except Exception as e:
                logger.error(f"Error querying SmartLibrary: {e}", exc_info=True)
                new_smart_context_data['data']['library_query_error'] = f"SmartLibrary query failed: {e}"
        
        return new_smart_context_data

# Example (Conceptual)
# async def main_example():
#     # ... (Requires extensive mocking of SmartAgentBus, MemoryManagerAgent, SmartLibrary) ...
#     # class MockSmartAgentBus(SmartAgentBus): ...
#     # class MockSmartLibrary(SmartLibrary): ...
#     # class MockLLMService(LLMService): ...

#     # mock_bus = MockSmartAgentBus(None)
#     # mock_library = MockSmartLibrary(None, None, None) # type: ignore
#     # mock_llm = MockLLMService()

#     # context_tool = ContextBuilderTool(
#     #     agent_bus=mock_bus,
#     #     smart_library=mock_library,
#     #     memory_manager_agent_id="mma_001",
#     #     llm_service=mock_llm
#     # )

#     # task_details = {
#     #     "target_agent_id": "test_agent_007",
#     #     "sub_task_description": "Develop a new feature for user authentication using OAuth2.",
#     #     "current_messages": [
#     #         {"sender_id": "manager", "content": "We need to prioritize OAuth2 for the login system."},
#     #         {"sender_id": "developer", "content": "Okay, I'll start looking into suitable Python libraries."}
#     #     ],
#     #     "max_relevant_experiences": 2,
#     #     "include_message_summary": True,
#     #     "include_relevant_experiences": True,
#     #     "include_library_components": True
#     # }
#     # built_context = await context_tool.run(**task_details)
#     # print(json.dumps(built_context, indent=2))
#     pass

# if __name__ == "__main__":
#     import asyncio
#     # asyncio.run(main_example())
#     pass
