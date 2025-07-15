import unittest
from unittest.mock import MagicMock, AsyncMock, patch

# Assume these are the refactored/new components
# from beeai_framework.agents.tool_calling import ToolCallingAgent
# from beeai_framework.core.tool import Tool

from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.core.llm_service import LLMService
from evolving_agents.core.mongodb_client import MongoDBClient
from evolving_agents.config import EvolvingAgentsConfig

# Placeholders if actual imports fail
class PlaceholderToolCallingAgent(MagicMock):
    def __init__(self, name="PlaceholderAgent", **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.meta = {"name": name, "type": "ToolCallingAgent", "version": "0.1"}
        # Add other attributes that SmartLibrary might access, e.g., description
        self.description = "A placeholder agent."


class PlaceholderTool(MagicMock):
    def __init__(self, name="PlaceholderTool", **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.description = "A placeholder tool."
        # SmartLibrary might store tool's schema or other metadata
        self.meta = {"name": name, "type": "Tool", "version": "0.1"}


try:
    from beeai_framework.agents.tool_calling import ToolCallingAgent
    from beeai_framework.core.tool import Tool
    # If EAT agents like SystemAgent are directly stored and retrieved
    # from evolving_agents.agents.system_agent import SystemAgent
except ImportError:
    ToolCallingAgent = PlaceholderToolCallingAgent
    Tool = PlaceholderTool
    # SystemAgent = PlaceholderToolCallingAgent # Assuming SystemAgent becomes a ToolCallingAgent


class TestSmartLibraryIntegration(unittest.IsolatedAsyncioTestCase):

    @patch('evolving_agents.core.mongodb_client.MongoDBClient')
    async def asyncSetUp(self, MockMongoDBClient):
        self.mock_mongo_client = MockMongoDBClient.return_value
        self.mock_mongo_client.get_collection = MagicMock() # For SmartLibrary init

        self.mock_llm_service = MagicMock(spec=LLMService)
        # Mock embedding functionality needed by SmartLibrary for semantic search
        self.mock_llm_service.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])

        self.config = EvolvingAgentsConfig()
        # SmartLibrary might require a specific collection name setup in config
        self.config.SMART_LIBRARY_DB_NAME = "eat_test_db"
        self.config.SMART_LIBRARY_COLLECTION_NAME = "smart_library_test_collection"

        self.smart_library = SmartLibrary(
            llm_service=self.mock_llm_service,
            config=self.config,
            db_client=self.mock_mongo_client
        )
        # Mock DB methods used by SmartLibrary
        self.mock_collection = self.mock_mongo_client.get_collection.return_value
        self.mock_collection.insert_one = AsyncMock()
        self.mock_collection.find_one = AsyncMock()
        self.mock_collection.find = MagicMock() # find returns a cursor
        self.mock_cursor = MagicMock()
        self.mock_cursor.to_list = AsyncMock(return_value=[]) # Default empty find result
        self.mock_collection.find.return_value = self.mock_cursor
        self.mock_collection.update_one = AsyncMock()
        self.mock_collection.delete_one = AsyncMock()
        self.mock_collection.create_index = AsyncMock() # For vector index creation


    async def test_add_and_retrieve_beeai_agent_component(self):
        """
        Test adding a BeeAI-based agent (e.g., ToolCallingAgent) to SmartLibrary
        and then retrieving it.
        """
        await self.asyncSetUp() # Ensure setup is called for each test

        # Agent to be added - this would be an instance of a BeeAI agent
        # or an EAT agent based on BeeAI's ToolCallingAgent
        agent_component_meta = { # This is what SmartLibrary's add_component expects for 'meta'
            "name": "MyBeeAIAgent",
            "version": "0.1.0",
            "description": "A test agent based on BeeAI's ToolCallingAgent.",
            "component_type": "agent", # EAT's classification
            "provider": "BeeAIProvider", # Indicates it's a BeeAI framework component
            "config": { # Agent-specific configuration
                "type": "ToolCallingAgent", # Actual BeeAI agent type
                "tools": ["Tool1", "Tool2"],
                "meta": {"internal_beeai_meta": "value"}
            },
            # Other fields like 'tags', 'access_level' etc.
        }
        # The 'component' itself for BeeAI might be just its config or a serialized form
        # SmartLibrary's add_component takes 'component' and 'meta'
        # Let's assume for BeeAI components, 'component' is the same as 'meta.config' or similar

        self.mock_collection.find_one.return_value = None # Simulate not existing initially

        component_id = await self.smart_library.add_component(
            component=agent_component_meta['config'], # Or a serialized representation
            meta=agent_component_meta
        )
        self.assertIsNotNone(component_id)
        self.mock_llm_service.embed.assert_called_once() # For semantic indexing
        self.mock_collection.insert_one.assert_called_once()

        # Now retrieve it
        # SmartLibrary's get_component_by_id usually reconstructs/returns the meta or full component doc
        db_entry_for_agent = {
            "_id": component_id,
            "embedding": [0.1,0.2,0.3],
            **agent_component_meta # meta is stored at top level
        }
        db_entry_for_agent["component_config"] = agent_component_meta['config'] # if stored separately

        self.mock_collection.find_one.return_value = db_entry_for_agent

        retrieved_agent_doc = await self.smart_library.get_component_by_id(component_id)

        self.assertIsNotNone(retrieved_agent_doc)
        self.assertEqual(retrieved_agent_doc["name"], "MyBeeAIAgent")
        self.assertEqual(retrieved_agent_doc["provider"], "BeeAIProvider")
        self.assertEqual(retrieved_agent_doc["config"]["type"], "ToolCallingAgent")

    async def test_search_beeai_components_semantically(self):
        """
        Test semantic search for BeeAI components in SmartLibrary.
        """
        await self.asyncSetUp()

        search_query = "Find an agent that processes invoices"
        self.mock_llm_service.embed.reset_mock() # Reset from setup
        query_embedding = [0.4, 0.5, 0.6]
        self.mock_llm_service.embed.return_value = query_embedding

        # Simulate MongoDB $vectorSearch result
        # SmartLibrary's _semantic_search_components constructs the pipeline
        # We mock the aggregate call which would use $vectorSearch
        mock_aggregate_cursor = MagicMock()

        # Example component that would be returned by vector search
        agent_doc = {
            "name": "InvoiceProcessorAgent", "description": "Agent for processing invoices.",
            "component_type": "agent", "provider": "BeeAIProvider",
            "config": {"type": "ToolCallingAgent"}, "score": 0.9 # Vector search score
        }
        mock_aggregate_cursor.to_list = AsyncMock(return_value=[agent_doc])
        self.mock_collection.aggregate = MagicMock(return_value=mock_aggregate_cursor)

        results = await self.smart_library.search_components(
            query=search_query,
            component_type="agent",
            limit=1
        )

        self.mock_llm_service.embed.assert_called_once_with(search_query)
        self.mock_collection.aggregate.assert_called_once() # Check that aggregation (for vector search) was called

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "InvoiceProcessorAgent")
        self.assertIn("score", results[0]) # Semantic search should add a score

    # TODO: Add tests for:
    # - Adding and retrieving BeeAI Tools from SmartLibrary
    # - Evolving a BeeAI component (if SmartLibrary handles this directly or via other services)


if __name__ == '__main__':
    unittest.main()
