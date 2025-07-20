# evolving_agents/agents/memory_manager_agent.py
import logging
from typing import List, Dict, Optional, Any, Type

from beeai_framework.agents.tool_calling import ToolCallingAgent # Changed from ReActAgent
from beeai_framework.agents.types import AgentMeta
from beeai_framework.memory import UnconstrainedMemory, TokenMemory
from beeai_framework.backend.chat import ChatModel
from pydantic import BaseModel, Field # Added for Output schema

from evolving_agents.core.llm_service import LLMService
from evolving_agents.tools.internal.mongo_experience_store_tool import MongoExperienceStoreTool
from evolving_agents.tools.internal.semantic_experience_search_tool import SemanticExperienceSearchTool
from evolving_agents.tools.internal.message_summarization_tool import MessageSummarizationTool

logger = logging.getLogger(__name__)

class MemoryOperationOutput(BaseModel):
    """Output schema for MemoryManagerAgent's operations."""
    status: str = Field(description="Status of the memory operation, e.g., 'success', 'error', 'not_found'.")
    message: Optional[str] = Field(None, description="A message detailing the outcome or error.")
    data: Optional[Any] = Field(None, description="Data retrieved or context relevant to the operation, e.g., experience_id, search results, summary.")

class MemoryManagerAgent(ToolCallingAgent):
    def __init__(
        self,
        llm_service: LLMService,
        mongo_experience_store_tool: MongoExperienceStoreTool,
        semantic_search_tool: SemanticExperienceSearchTool,
        message_summarization_tool: MessageSummarizationTool,
        agent_meta_override: Optional[AgentMeta] = None,
        memory_override: Optional[Any] = None,
    ):
        if llm_service is None:
            raise ValueError("LLMService instance is required for MemoryManagerAgent.")
        if llm_service.chat_model is None:
            raise ValueError("LLMService's chat_model is not initialized.")

        internal_tools = [
            mongo_experience_store_tool,
            semantic_search_tool,
            message_summarization_tool,
        ]

        if agent_meta_override:
            final_agent_meta = agent_meta_override
            # Ensure tools are correctly set in the meta object
            # This logic ensures our internal tools are present if an override is given
            # but might be simplified if AgentMeta's tools attribute is a mutable list.
            current_meta_tools = list(getattr(final_agent_meta, 'tools', []) or [])
            tools_to_add = [tool for tool in internal_tools if tool not in current_meta_tools]
            if tools_to_add:
                if hasattr(final_agent_meta, 'tools') and isinstance(final_agent_meta.tools, list):
                    final_agent_meta.tools.extend(tools_to_add)
                else: # If tools attribute doesn't exist or isn't a list, overwrite/create it.
                    final_agent_meta.tools = current_meta_tools + tools_to_add
                logger.debug(f"Extended/Set tools in agent_meta_override for MemoryManagerAgent.")
        else:
            final_agent_meta = AgentMeta(
                name="MemoryManagerAgent", # Explicitly set name here
                description=(
                    "Manages persistent storage and retrieval of agent experiences and "
                    "contextual facts. It uses internal tools for storing, searching, and summarizing."
                ),
                extra_description=(
                    "This agent is called by other agents (like SystemAgent) "
                    "via the SmartAgentBus. Tasks should be natural language requests "
                    "specifying the memory operation."
                ),
                tools=internal_tools,
            )

        final_memory = memory_override
        if final_memory is None:
            final_memory = TokenMemory(llm_service.chat_model)

        # Call ToolCallingAgent's __init__
        super().__init__(
            llm=llm_service.chat_model,
            tools=final_agent_meta.tools,
            memory=final_memory,
            meta=final_agent_meta
        )

        # After super().__init__(), ToolCallingAgent should have populated self.name, self.description, etc.
        # from the 'meta' object. If self.name is still not available, it indicates a deeper issue
        # with ReActAgent's initialization or an incorrect assumption about its behavior.

        # The logger line that caused the error:
        # logger.info(f"MemoryManagerAgent '{self.name}' initialized with LLM: {type(self.llm).__name__}, Memory: {type(self.memory).__name__}, Tools: {[t.name for t in self.tools_map.values()]}")
        # Let's make the logging more robust to potential missing attributes for now
        
        agent_name_for_log = getattr(self, 'name', 'UnknownName(MMA)')
        llm_type_for_log = type(getattr(self, 'llm', None)).__name__
        memory_type_for_log = type(getattr(self, 'memory', None)).__name__
        tools_map_for_log = getattr(self, 'tools_map', {})
        tool_names_for_log = [t.name for t in tools_map_for_log.values() if hasattr(t, 'name')]

        logger.info(f"MemoryManagerAgent '{agent_name_for_log}' initialized with LLM: {llm_type_for_log}, Memory: {memory_type_for_log}, Tools: {tool_names_for_log}")


    async def run(self, task_description: str) -> MemoryOperationOutput: # Updated return type
        logger.info(f"MemoryManagerAgent '{getattr(self, 'name', 'UnnamedMMA')}' received task: {task_description[:200]}...")
        try:
            # super().run for ToolCallingAgent will return the Pydantic model instance (MemoryOperationOutput)
            # if the LLM response correctly maps to the schema.
            # If it can't parse, it might raise an error or return a default/error object.
            raw_output = await super().run(prompt=task_description)

            if isinstance(raw_output, MemoryOperationOutput):
                logger.info(f"MemoryManagerAgent '{getattr(self, 'name', 'UnnamedMMA')}' completed task. Status: {raw_output.status}")
                return raw_output
            else:
                # This case implies the LLM response didn't match the MemoryOperationOutput schema
                # or ToolCallingAgent's run method returned something unexpected.
                logger.warning(f"MemoryManagerAgent '{getattr(self, 'name', 'UnnamedMMA')}' received unexpected output type: {type(raw_output)}. Content: {str(raw_output)[:200]}")
                return MemoryOperationOutput(
                    status="error",
                    message=f"Unexpected output format from agent: {str(raw_output)[:200]}",
                    data=str(raw_output)
                )
        except Exception as e:
            logger.error(f"Error during MemoryManagerAgent '{getattr(self, 'name', 'UnnamedMMA')}' execution for task '{task_description[:100]}...': {e}", exc_info=True)
            return MemoryOperationOutput(
                status="error",
                message=f"Execution error: {str(e)}",
                data={"task_description": task_description}
            )