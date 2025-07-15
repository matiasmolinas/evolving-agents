import unittest
from unittest.mock import MagicMock, AsyncMock, patch

# EAT Components
from evolving_agents.agents.memory_manager_agent import MemoryManagerAgent # Assumed to exist
from evolving_agents.tools.context.context_builder_tool import ContextBuilderTool
from evolving_agents.tools.memory.experience_recorder_tool import ExperienceRecorderTool
from evolving_agents.core.llm_service import LLMService
from evolving_agents.core.mongodb_client import MongoDBClient # SmartMemory uses this
from evolving_agents.config import EvolvingAgentsConfig

# BeeAI Components (MemoryManagerAgent might be based on ReAct/ToolCalling)
# from beeai_framework.agents.react import ReActAgent # Or ToolCallingAgent
# from beeai_framework.core.run_context import RunContext
# from beeai_framework.core.tool import ToolOutput

# Placeholders
class PlaceholderReActAgent(MagicMock): # Or ToolCallingAgent
    def __init__(self, llm_service, tools, meta, **kwargs):
        super().__init__(**kwargs)
        self.name = meta.get("name", "PlaceholderReActAgent")
        self.meta = meta
        self.llm_service = llm_service
        self.tools = tools
        self.run = AsyncMock()
        # self.run.return_value.get_output.return_value = "MemoryManagerAgent task processed"
        mock_run_instance = AsyncMock() # spec=Run
        mock_run_instance.get_output = AsyncMock(return_value="MemoryManagerAgent task processed")
        self.run.return_value = mock_run_instance


class PlaceholderRunContext(MagicMock):
    pass

class PlaceholderToolOutput(MagicMock):
    def __init__(self, content, **kwargs):
        super().__init__(**kwargs)
        self.content = content


try:
    from beeai_framework.agents.react import ReActAgent # Or ToolCallingAgent
    from beeai_framework.core.run_context import RunContext
    from beeai_framework.core.tool import ToolOutput

    # If MemoryManagerAgent is a concrete class we can import
    # from evolving_agents.agents.memory_manager_agent import MemoryManagerAgent
    # For this test, we'll assume MemoryManagerAgent is instantiated and its tools are called.

except ImportError:
    ReActAgent = PlaceholderReActAgent # Fallback
    RunContext = PlaceholderRunContext
    ToolOutput = PlaceholderToolOutput
    # MemoryManagerAgent = PlaceholderReActAgent # Fallback if class not found


# Mocking the actual SmartMemory interactions (which happen inside tools)
# ContextBuilderTool and ExperienceRecorderTool interact with MongoDB via db_client.
# We will mock the db_client calls made by these tools.

class TestSmartMemoryIntegration(unittest.IsolatedAsyncioTestCase):

    @patch('evolving_agents.core.mongodb_client.MongoDBClient')
    async def asyncSetUp(self, MockMongoDBClient):
        self.mock_mongo_client = MockMongoDBClient.return_value
        # Mock collections that SmartMemory tools would use
        self.mock_experiences_collection = AsyncMock()
        self.mock_context_collection = AsyncMock() # If ContextBuilderTool saves context state

        def get_collection_side_effect(collection_name):
            if collection_name == self.config.SMART_MEMORY_EXPERIENCES_COLLECTION_NAME:
                return self.mock_experiences_collection
            elif collection_name == "some_context_collection_name": # Fictitious
                return self.mock_context_collection
            return AsyncMock() # Default mock collection

        self.mock_mongo_client.get_collection = MagicMock(side_effect=get_collection_side_effect)

        self.config = EvolvingAgentsConfig()
        # Define collection names used by SmartMemory components
        self.config.SMART_MEMORY_DB_NAME = "eat_memory_test_db"
        self.config.SMART_MEMORY_EXPERIENCES_COLLECTION_NAME = "experiences_test"
        self.config.SMART_MEMORY_CONTEXT_WINDOW_SIZE = 10 # Example value

        self.mock_llm_service = MagicMock(spec=LLMService)
        self.mock_llm_service.embed = AsyncMock(return_value=[0.1, 0.2, 0.3]) # For semantic search/storage

        # Instantiate the tools
        self.experience_recorder_tool = ExperienceRecorderTool(
            db_client=self.mock_mongo_client,
            config=self.config,
            llm_service=self.mock_llm_service
        )
        self.context_builder_tool = ContextBuilderTool(
            db_client=self.mock_mongo_client,
            config=self.config,
            llm_service=self.mock_llm_service
        )

        # Mock MemoryManagerAgent
        # MemoryManagerAgent uses these tools.
        # We can test by directly invoking the tools' _run methods,
        # or by mocking the agent and asserting it calls the tools.
        # For "integration", let's assume MemoryManagerAgent calls these tools.

        memory_manager_meta = {"name": "TestMemoryManagerAgent"}
        # Use the actual MemoryManagerAgent if available and it's based on a BeeAI agent
        if 'MemoryManagerAgent' in globals() and MemoryManagerAgent != PlaceholderReActAgent:
            self.memory_manager_agent = MemoryManagerAgent(
                llm_service=self.mock_llm_service,
                tools=[self.experience_recorder_tool, self.context_builder_tool],
                meta=memory_manager_meta,
                config=self.config # Assuming it takes config
            )
        else: # Fallback
             self.memory_manager_agent = PlaceholderReActAgent(
                llm_service=self.mock_llm_service,
                tools=[self.experience_recorder_tool, self.context_builder_tool],
                meta=memory_manager_meta
            )
        # Ensure the agent's run method is an AsyncMock for testing its calls
        self.memory_manager_agent.run = AsyncMock()
        mock_mma_run_output = AsyncMock() # spec=Run
        mock_mma_run_output.get_output = AsyncMock(return_value="MMA processed with tool")
        self.memory_manager_agent.run.return_value = mock_mma_run_output


    async def test_experience_recorder_tool_via_memory_manager_agent(self):
        """
        Test that ExperienceRecorderTool correctly stores an experience.
        This is tested by directly calling the tool's _run, simulating agent usage.
        """
        await self.asyncSetUp()

        experience_data = {
            "event_type": "agent_interaction",
            "agent_name": "TestAgent",
            "prompt": "What is the weather?",
            "response": "It is sunny.",
            "tags": ["weather", "test"],
            "session_id": "session_123"
        }
        # Input for ExperienceRecorderTool is ExperienceRecorderToolInput model
        # For testing, we'll pass the dict and assume tool parses it.

        mock_run_context = MagicMock(spec=RunContext)
        mock_run_context.session_id = "session_123"

        # Mock DB calls for ExperienceRecorderTool
        self.mock_experiences_collection.insert_one = AsyncMock(return_value=MagicMock(inserted_id="exp_id_001"))

        # Simulate MemoryManagerAgent deciding to use ExperienceRecorderTool
        # Actual call would be: await self.memory_manager_agent.run(prompt="Record this experience: ...", context=...)
        # which would internally select and run ExperienceRecorderTool.
        # For this test, we directly invoke the tool's _run method.

        # Assuming ExperienceRecorderToolInput is a Pydantic model or similar
        # For direct _run call, we'd need to construct the input model instance.
        # Let's use a simplified approach if pydantic models are not easily mockable here.
        tool_input_data_obj = MagicMock() # Mocking the Pydantic input model instance
        tool_input_data_obj.event_type = experience_data["event_type"]
        tool_input_data_obj.agent_name = experience_data["agent_name"]
        tool_input_data_obj.data = {"prompt": experience_data["prompt"], "response": experience_data["response"]}
        tool_input_data_obj.tags = experience_data["tags"]
        tool_input_data_obj.session_id = experience_data["session_id"]
        tool_input_data_obj.timestamp = unittest.mock.ANY # Will be set by tool
        tool_input_data_obj.embedding = unittest.mock.ANY # Will be set by tool


        tool_output = await self.experience_recorder_tool._run(
            input_data=tool_input_data_obj,
            context=mock_run_context
        )

        self.mock_llm_service.embed.assert_called_once() # For embedding the experience text
        self.mock_experiences_collection.insert_one.assert_called_once()

        # Check the document that was inserted
        inserted_doc_call = self.mock_experiences_collection.insert_one.call_args[0][0]
        self.assertEqual(inserted_doc_call['event_type'], experience_data['event_type'])
        self.assertEqual(inserted_doc_call['session_id'], experience_data['session_id'])
        self.assertIn('embedding', inserted_doc_call)

        self.assertIsNotNone(tool_output)
        self.assertTrue(tool_output.content['success'])
        self.assertEqual(tool_output.content['experience_id'], "exp_id_001")


    async def test_context_builder_tool_via_memory_manager_agent(self):
        """
        Test that ContextBuilderTool retrieves relevant experiences for context.
        This is tested by directly calling the tool's _run, simulating agent usage.
        """
        await self.asyncSetUp()

        query = "relevant past interactions about weather"
        mock_run_context = MagicMock(spec=RunContext)
        mock_run_context.session_id = "session_456"

        # Mock DB calls for ContextBuilderTool (semantic search on experiences collection)
        query_embedding = [0.7, 0.8, 0.9]
        self.mock_llm_service.embed.return_value = query_embedding

        # Simulate MongoDB $vectorSearch result from experiences collection
        retrieved_experience_doc = {
            "event_type": "agent_interaction", "data": {"prompt": "Is it sunny?", "response": "Yes, very sunny."},
            "timestamp": "2023-01-01T10:00:00Z", "score": 0.95 # Vector search score
        }
        mock_aggregate_cursor = MagicMock()
        mock_aggregate_cursor.to_list = AsyncMock(return_value=[retrieved_experience_doc])
        self.mock_experiences_collection.aggregate = MagicMock(return_value=mock_aggregate_cursor)

        # Tool input object
        tool_input_data_obj = MagicMock()
        tool_input_data_obj.query = query
        tool_input_data_obj.session_id = mock_run_context.session_id
        tool_input_data_obj.limit = 3

        tool_output = await self.context_builder_tool._run(
            input_data=tool_input_data_obj,
            context=mock_run_context
        )

        self.mock_llm_service.embed.assert_called_once_with(query)
        self.mock_experiences_collection.aggregate.assert_called_once() # For vector search

        self.assertIsNotNone(tool_output)
        self.assertIn('context_summary', tool_output.content) # Or 'formatted_context'
        self.assertIn('retrieved_experiences', tool_output.content)
        self.assertEqual(len(tool_output.content['retrieved_experiences']), 1)
        self.assertEqual(tool_output.content['retrieved_experiences'][0]['data']['prompt'], "Is it sunny?")

    # TODO:
    # - Test MemoryManagerAgent actually choosing and using these tools based on a prompt.
    # - Test integration with ComponentExperienceTracker if these tools/agent emit events.
    # - Test context window management by ContextBuilderTool.

if __name__ == '__main__':
    unittest.main()
