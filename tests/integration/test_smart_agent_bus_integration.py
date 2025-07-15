import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

# from beeai_framework.agents.tool_calling import ToolCallingAgent
# from beeai_framework.core.run_context import RunContext
# from beeai_framework.core.run import Run

from evolving_agents.agent_bus.smart_agent_bus import SmartAgentBus
from evolving_agents.core.llm_service import LLMService # For agent instantiation
from evolving_agents.providers.beeai_provider import BeeAIProvider # To provide agents
from evolving_agents.config import EvolvingAgentsConfig
from evolving_agents.core.dependency_container import DependencyContainer

# Placeholders
class PlaceholderToolCallingAgent(MagicMock):
    def __init__(self, llm_service, tools, meta, **kwargs):
        super().__init__(**kwargs)
        self.name = meta.get("name", "PlaceholderAgent")
        self.meta = meta # SmartAgentBus uses this
        self.llm_service = llm_service
        self.tools = tools
        # Mock the run method for capability requests
        self.run = AsyncMock()
        mock_run_output = AsyncMock() # This is the Run object
        mock_run_output.get_output = AsyncMock(return_value=f"Response from {self.name}")
        self.run.return_value = mock_run_output


class PlaceholderRunContext(MagicMock):
    pass

class PlaceholderRun(MagicMock):
    async def get_output(self):
        return "Mocked Run output"

try:
    from beeai_framework.agents.tool_calling import ToolCallingAgent
    from beeai_framework.core.run_context import RunContext
    from beeai_framework.core.run import Run
except ImportError:
    ToolCallingAgent = PlaceholderToolCallingAgent
    RunContext = PlaceholderRunContext
    Run = PlaceholderRun


class TestSmartAgentBusIntegration(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.config = EvolvingAgentsConfig()
        self.mock_llm_service = MagicMock(spec=LLMService)

        # Mock BeeAIProvider
        self.mock_beeai_provider = MagicMock(spec=BeeAIProvider)
        self.mock_beeai_provider.get_name.return_value = "BeeAIProvider"

        # Setup DependencyContainer with mocked services
        # SmartAgentBus uses DependencyContainer to get providers
        self.container = DependencyContainer()
        self.container.register_provider("llm_service", self.mock_llm_service)
        self.container.register_provider("BeeAIProvider", self.mock_beeai_provider) # Register by name
        # SmartAgentBus might also try to get 'AgentFactory' or 'ToolFactory'
        self.container.register_provider("AgentFactory", MagicMock())
        self.container.register_provider("ToolFactory", MagicMock())


        self.agent_bus = SmartAgentBus(config=self.config, dependency_container=self.container)

        # Clean up bus before each test if needed (though instance is new)
        self.agent_bus.agents_registry = {}
        self.agent_bus.capabilities_registry = {}


    async def test_register_and_discover_beeai_agent(self):
        """
        Test registering a BeeAI-based agent with SmartAgentBus and then discovering it.
        """
        await self.asyncSetUp()

        agent_id = "beeai_agent_001"
        agent_meta = {
            "name": "MyBeeAIAgentForBus",
            "version": "0.1",
            "description": "A BeeAI agent registered on the bus.",
            "provider": "BeeAIProvider", # Critical for bus to use correct provider
            "capabilities": ["process_data", "generate_report"],
            "config": {"type": "ToolCallingAgent", "tools": []} # Config for BeeAIProvider
        }

        # Mock the agent instance that BeeAIProvider would create
        mock_agent_instance = PlaceholderToolCallingAgent(
            llm_service=self.mock_llm_service,
            tools=[],
            meta=agent_meta # Pass meta so instance has it
        )
        self.mock_beeai_provider.create_agent_instance.return_value = mock_agent_instance

        await self.agent_bus.register_agent(agent_id, agent_meta)

        # Verify BeeAIProvider was used (indirectly, by checking if agent is in registry)
        self.assertIn(agent_id, self.agent_bus.agents_registry)
        self.assertEqual(self.agent_bus.agents_registry[agent_id]['meta'], agent_meta)
        # Check that the actual agent instance is stored (or provider knows how to make it)
        self.assertIsNotNone(self.agent_bus.agents_registry[agent_id]['agent'])


        # Discover by capability
        discovered_agents = await self.agent_bus.discover_agents_by_capability("process_data")
        self.assertEqual(len(discovered_agents), 1)
        self.assertEqual(discovered_agents[0]['id'], agent_id)
        self.assertEqual(discovered_agents[0]['meta']['name'], "MyBeeAIAgentForBus")

    async def test_request_capability_from_beeai_agent(self):
        """
        Test requesting a capability from a registered BeeAI agent.
        This involves the bus finding the agent and its provider executing it.
        """
        await self.asyncSetUp()

        agent_id = "beeai_agent_002"
        agent_meta = {
            "name": "TaskExecutorBeeAIAgent",
            "version": "0.1",
            "description": "Executes tasks via BeeAI framework.",
            "provider": "BeeAIProvider",
            "capabilities": ["execute_task"],
            "config": {"type": "ToolCallingAgent", "tools": []}
        }

        mock_agent_instance = PlaceholderToolCallingAgent(
            llm_service=self.mock_llm_service,
            tools=[],
            meta=agent_meta
        )
        # mock_agent_instance.run = AsyncMock() # Already done in Placeholder
        # mock_run_obj = AsyncMock(spec=Run)
        # mock_run_obj.get_output.return_value = "Task executed successfully by BeeAI agent"
        # mock_agent_instance.run.return_value = mock_run_obj

        self.mock_beeai_provider.create_agent_instance.return_value = mock_agent_instance
        # Mock the provider's execute method which SmartAgentBus will call
        self.mock_beeai_provider.execute_agent_instance = AsyncMock(
            return_value="Task executed successfully by BeeAI agent via provider"
        )


        await self.agent_bus.register_agent(agent_id, agent_meta)

        prompt = "Please execute this important task."
        mock_run_context = MagicMock(spec=RunContext)

        # This call will go through:
        # SmartAgentBus.request_capability -> _execute_capability -> Provider.execute_agent_instance
        response = await self.agent_bus.request_capability(
            capability_name="execute_task",
            prompt=prompt,
            context=mock_run_context
        )

        self.assertIsNotNone(response)
        self.assertEqual(len(response), 1) # Assuming one agent provides the capability
        agent_response = response[0]

        self.assertEqual(agent_response['agent_id'], agent_id)
        self.assertEqual(agent_response['response'], "Task executed successfully by BeeAI agent via provider")

        # Verify that the BeeAIProvider's execute_agent_instance was called correctly
        self.mock_beeai_provider.execute_agent_instance.assert_called_once_with(
            agent=mock_agent_instance, # The instance created by the provider
            prompt=prompt,
            context=mock_run_context
        )

    # TODO:
    # - Test discovery with multiple agents, some BeeAI, some not.
    # - Test scenario where no agent provides the capability.
    # - Test error handling if agent execution fails.

if __name__ == '__main__':
    unittest.main()
