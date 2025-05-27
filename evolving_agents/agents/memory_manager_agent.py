import logging
from typing import List, Dict, Optional, Any

# Assuming these paths are correct based on the project structure
from beeai_framework.agents.react import ReActAgent
from beeai_framework.agents.types import AgentMeta
from beeai_framework.memory import UnconstrainedMemory # Using UnconstrainedMemory as a default if TokenMemory is not specifically needed

from evolving_agents.core.llm_service import LLMService
from evolving_agents.tools.internal.mongo_experience_store_tool import MongoExperienceStoreTool
from evolving_agents.tools.internal.semantic_experience_search_tool import SemanticExperienceSearchTool
from evolving_agents.tools.internal.message_summarization_tool import MessageSummarizationTool

# Configure logging
logger = logging.getLogger(__name__)

class MemoryManagerAgent(ReActAgent):
    """
    Manages persistent storage and retrieval of agent experiences and contextual facts.
    It provides capabilities to store new experiences, search for relevant past
    experiences, and summarize message histories using its internal tools.
    This agent is designed to be invoked via the SmartAgentBus.
    """

    def __init__(
        self,
        llm_service: LLMService,
        mongo_experience_store_tool: MongoExperienceStoreTool,
        semantic_search_tool: SemanticExperienceSearchTool,
        message_summarization_tool: MessageSummarizationTool,
        memory: Optional[Any] = None, # e.g., TokenMemory or UnconstrainedMemory
        agent_meta: Optional[AgentMeta] = None,
        # debug_enabled: bool = False, # ReActAgent might have this
    ):
        """
        Initializes the MemoryManagerAgent.

        Args:
            llm_service: The language model service for ReAct decision making.
            mongo_experience_store_tool: Tool for storing and managing experiences in MongoDB.
            semantic_search_tool: Tool for semantic search of experiences.
            message_summarization_tool: Tool for summarizing message histories.
            memory: Optional memory component for the ReActAgent. Defaults to UnconstrainedMemory.
            agent_meta: Optional agent metadata. If None, it's created internally.
            # debug_enabled: Whether to enable debug logging for the ReAct process.
        """
        self.mongo_experience_store_tool = mongo_experience_store_tool
        self.semantic_search_tool = semantic_search_tool
        self.message_summarization_tool = message_summarization_tool

        internal_tools = [
            self.mongo_experience_store_tool,
            self.semantic_search_tool,
            self.message_summarization_tool,
        ]

        if agent_meta is None:
            agent_meta = AgentMeta(
                name="MemoryManagerAgent",
                description=(
                    "Manages persistent storage and retrieval of agent experiences and "
                    "contextual facts. It provides capabilities to store new experiences, "
                    "search for relevant past experiences, and summarize message histories."
                ),
                extra_description=(
                    "This agent uses internal tools to interact with a MongoDB database "
                    "for experience storage and a language model for summarization tasks. "
                    "It is designed to be called by other agents (like SystemAgent) "
                    "via the SmartAgentBus. The tasks provided to this agent should be "
                    "phrased as clear natural language requests that specify the desired "
                    "action (store, search, summarize) and any necessary parameters."
                ),
                tools=internal_tools,
                # system_prompt_template can be customized if needed for ReAct
            )
        else:
            # Ensure the provided agent_meta includes the internal tools
            # This might involve creating a new AgentMeta if the tools list is immutable
            # or if we want to preserve the original agent_meta for other purposes.
            # For simplicity, we'll assume agent_meta.tools can be extended or is already set up.
            # A more robust approach might be:
            # current_tools = list(agent_meta.tools or [])
            # current_tools.extend(internal_tools)
            # agent_meta = agent_meta.copy(update={"tools": current_tools}) # If AgentMeta is a Pydantic model or similar
            # For now, let's assume if agent_meta is provided, its tools list is what we want,
            # but the requirement was to use these three tools.
            agent_meta.tools = internal_tools


        if memory is None:
            memory = UnconstrainedMemory()

        super().__init__(
            llm_service=llm_service,
            agent_meta=agent_meta,
            memory=memory,
            # debug_enabled=debug_enabled,
        )
        # The ReActAgent's _handle_tool_call method will use the tools defined in agent_meta.
        # The prompts for this agent should guide its LLM to select one of these tools.

    async def run(self, task_description: str) -> Any:
        """
        Main entry point for the MemoryManagerAgent.
        The ReActAgent's run method will process the task_description using its LLM
        to decide which internal tool to use and what parameters to pass.

        Args:
            task_description: A natural language description of the task to perform.
                              Examples:
                              - "Store the following agent experience: {experience_data_json_string}"
                              - "Find relevant experiences for the query: 'how to optimize database queries' and return top 3 results."
                              - "Summarize this conversation: [{...}, {...}] with the goal of 'identifying action items'."

        Returns:
            The result of the ReAct execution, which typically is the output of
            the last executed internal tool or a final thought from the agent.
        """
        logger.info(f"MemoryManagerAgent received task: {task_description}")
        try:
            # The ReActAgent's run method handles the thought-action-observation loop.
            # It will use the LLM to parse task_description, select one of its
            # configured tools (mongo_experience_store_tool, semantic_search_tool,
            # or message_summarization_tool), and execute it.
            result = await super().run(task_description=task_description)
            logger.info(f"MemoryManagerAgent completed task. Result: {type(result)}")
            return result
        except Exception as e:
            logger.error(f"Error during MemoryManagerAgent execution for task '{task_description}': {e}", exc_info=True)
            # Depending on how SmartAgentBus expects errors, either re-raise or return an error structure.
            # For now, re-raising to make it visible.
            raise

# Example usage (conceptual, would require setting up dependencies)
# async def main_example():
#     # --- Mock/Real Dependencies ---
#     class MockLLMService(LLMService):
#         async def generate(self, prompt: str, **kwargs) -> str:
#             # Simplified ReAct LLM simulation
#             print(f"\n--- MemoryManagerAgent LLM (ReAct) Prompt ---\n{prompt}\n--- End Prompt ---")
#             if "Store the following agent experience" in prompt and "store_experience" in prompt:
#                 # Simulate LLM deciding to use mongo_experience_store_tool
#                 # Extracting params is complex; ReAct handles this.
#                 # For this mock, assume it correctly identifies the tool and params.
#                 # This is a very simplified mock of the ReAct LLM's output.
#                 # In reality, ReAct would make multiple calls for thought, action, observation.
#                 if "'primary_goal': 'test_goal'" in prompt:
#                     return "Action: MongoExperienceStoreTool.store_experience\nAction Input: {'experience_data': {'primary_goal': 'test_goal', 'sub_task_description': 'testing storage'}}"
#             elif "Find relevant experiences for the query" in prompt and "search_relevant_experiences" in prompt:
#                 if "'how to optimize database queries'" in prompt:
#                     return "Action: SemanticExperienceSearchTool.search_relevant_experiences\nAction Input: {'query_string': 'how to optimize database queries', 'top_k': 3}"
#             elif "Summarize this conversation" in prompt and "summarize_messages" in prompt:
#                  if "'identifying action items'" in prompt:
#                     return "Action: MessageSummarizationTool.summarize_messages\nAction Input: {'messages': [{'sender': 'user', 'content': 'hello'}], 'target_goal': 'identifying action items'}"
#             return "Final Answer: I was unable to determine the correct action for the request."

#         async def embed(self, text: str) -> List[float]:
#             return [0.1] * 768 # Dummy embedding

#     class MockMongoExperienceStoreTool(MongoExperienceStoreTool):
#         def __init__(self): self.name = "MongoExperienceStoreTool"; self.description="Stores experiences"; super().__init__(None, None) # type: ignore
#         async def store_experience(self, experience_data: Dict[str, Any]) -> str:
#             print(f"MockMongoExperienceStoreTool: Storing {experience_data}")
#             return f"exp_{uuid.uuid4().hex}"
#         async def _execute(self, experience_data: Dict[str, Any]) -> str: # ReActAgent might call _execute
#             return await self.store_experience(experience_data)


#     class MockSemanticSearchTool(SemanticExperienceSearchTool):
#         def __init__(self): self.name = "SemanticExperienceSearchTool"; self.description="Searches experiences"; super().__init__(None, None) # type: ignore
#         async def search_relevant_experiences(self, query_string: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
#             print(f"MockSemanticSearchTool: Searching for '{query_string}', top_k={top_k}")
#             return [{"experience_id": "exp_123", "primary_goal": query_string, "similarity_score": 0.9}]
#         async def _execute(self, query_string: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
#             return await self.search_relevant_experiences(query_string, top_k, **kwargs)

#     class MockMessageSummarizationTool(MessageSummarizationTool):
#         def __init__(self): self.name = "MessageSummarizationTool"; self.description="Summarizes messages"; super().__init__(None) # type: ignore
#         async def summarize_messages(self, messages: List[Dict[str, Any]], target_goal: str, **kwargs) -> str:
#             print(f"MockMessageSummarizationTool: Summarizing messages for goal '{target_goal}'")
#             return f"Summary of {len(messages)} messages for {target_goal}."
#         async def _execute(self, messages: List[Dict[str, Any]], target_goal: str, **kwargs) -> str:
#             return await self.summarize_messages(messages, target_goal, **kwargs)
    
#     import uuid # for mock tool

#     # --- Instantiate ---
#     mock_llm_service = MockLLMService()
#     mock_store_tool = MockMongoExperienceStoreTool()
#     mock_search_tool = MockSemanticSearchTool()
#     mock_summary_tool = MockMessageSummarizationTool()

#     memory_manager = MemoryManagerAgent(
#         llm_service=mock_llm_service,
#         mongo_experience_store_tool=mock_store_tool,
#         semantic_search_tool=mock_search_tool,
#         message_summarization_tool=mock_summary_tool,
#         # debug_enabled=True # Enable ReAct debug logs
#     )

#     # --- Test Cases ---
#     print("\n--- Test Case 1: Store Experience ---")
#     task1_desc = "Store the following agent experience: {'primary_goal': 'test_goal', 'sub_task_description': 'testing storage'}"
#     # In a real ReAct agent, the LLM would output "Action: MongoExperienceStoreTool.store_experience \n Action Input: {...}"
#     # The ReActAgent base class would parse this, call the tool, get observation, and feed back to LLM.
#     # For this example, we are simplifying as the internal ReAct loop is complex.
#     # The `run` method of ReActAgent triggers this loop.
#     # The mocked LLM above tries to simulate the *final* action choice part.
#     result1 = await memory_manager.run(task_description=task1_desc)
#     print(f"Result for Task 1: {result1}") # ReAct usually returns the final answer or observation

#     print("\n--- Test Case 2: Search Experiences ---")
#     task2_desc = "Find relevant experiences for the query: 'how to optimize database queries' and return top 3 results."
#     result2 = await memory_manager.run(task_description=task2_desc)
#     print(f"Result for Task 2: {result2}")

#     print("\n--- Test Case 3: Summarize Messages ---")
#     task3_desc = "Summarize this conversation: [{'sender': 'user', 'content': 'hello'}] with the goal of 'identifying action items'."
#     result3 = await memory_manager.run(task_description=task3_desc)
#     print(f"Result for Task 3: {result3}")

# if __name__ == "__main__":
#     import asyncio
#     # This example requires a running event loop and proper async setup
#     # For actual execution, ensure beeai_framework and other dependencies are installed and configured.
#     # asyncio.run(main_example())
#     pass
