import unittest
from unittest.mock import MagicMock, AsyncMock, patch

# from beeai_framework.agents.react import ReActAgent
# from beeai_framework.agents.tool_calling import ToolCallingAgent
# from beeai_framework.core.run_context import RunContext
# from beeai_framework.core.run import Run
# from beeai_framework.core.tool import ToolOutput

from evolving_agents.providers.beeai_provider import BeeAIProvider
from evolving_agents.core.llm_service import LLMService
from evolving_agents.config import EvolvingAgentsConfig

# Placeholder for actual BeeAI components
class PlaceholderReActAgent:
    def __init__(self, llm_service, tools, meta, **kwargs):
        self.llm_service = llm_service
        self.tools = tools
        self.meta = meta
    async def run(self, prompt, context=None):
        run_mock = AsyncMock() # Simulate the Run object
        run_mock.get_output.return_value = "Mocked ReActAgent response"
        return run_mock

class PlaceholderToolCallingAgent:
    def __init__(self, llm_service, tools, meta, **kwargs):
        self.llm_service = llm_service
        self.tools = tools
        self.meta = meta
    async def run(self, prompt, context=None):
        run_mock = AsyncMock() # Simulate the Run object
        run_mock.get_output.return_value = "Mocked ToolCallingAgent response"
        return run_mock

class PlaceholderRunContext:
    pass

class PlaceholderRun:
    async def get_output(self):
        return "Mocked Run output"

class PlaceholderToolOutput:
    def __init__(self, content):
        self.content = content

try:
    from beeai_framework.agents.react import ReActAgent
    from beeai_framework.agents.tool_calling import ToolCallingAgent
    from beeai_framework.core.run_context import RunContext
    from beeai_framework.core.run import Run
    from beeai_framework.core.tool import ToolOutput
except ImportError:
    ReActAgent = PlaceholderReActAgent
    ToolCallingAgent = PlaceholderToolCallingAgent
    RunContext = PlaceholderRunContext
    Run = PlaceholderRun
    ToolOutput = PlaceholderToolOutput


class TestBeeAIProvider(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.mock_llm_service = MagicMock(spec=LLMService)
        self.mock_config = MagicMock(spec=EvolvingAgentsConfig)
        self.provider = BeeAIProvider(llm_service=self.mock_llm_service, config=self.mock_config)

    @patch('evolving_agents.providers.beeai_provider.ReActAgent', spec=ReActAgent)
    async def test_create_agent_instance_react(self, MockReActAgent):
        """
        Test creating a ReActAgent instance.
        Assumes AgentFactory/initializers are updated as per Phase 2.
        """
        mock_agent_instance = AsyncMock(spec=ReActAgent)
        MockReActAgent.return_value = mock_agent_instance

        agent_config = {
            "name": "TestReActAgent",
            "version": "0.1",
            "description": "A test ReAct agent.",
            "type": "ReActAgent", # This type would be resolved by AgentFactory
            "config": {
                "llm_service_id": "default_llm", # Handled by DI or factory
                "tools": ["Tool1", "Tool2"], # Tool names or instances
                "meta": {"some_meta_key": "some_value"}
            }
        }
        mock_tools = [MagicMock(), MagicMock()]

        # Simulate tool resolution if tools are passed as names
        with patch.object(self.provider, '_resolve_tools', return_value=mock_tools):
            agent = self.provider.create_agent_instance(agent_config)

        MockReActAgent.assert_called_once_with(
            llm_service=self.mock_llm_service,
            tools=mock_tools,
            meta=agent_config["config"]["meta"]
        )
        self.assertEqual(agent, mock_agent_instance)

    @patch('evolving_agents.providers.beeai_provider.ToolCallingAgent', spec=ToolCallingAgent)
    async def test_create_agent_instance_tool_calling(self, MockToolCallingAgent):
        """
        Test creating a ToolCallingAgent instance.
        Assumes SystemAgent is upgraded and AgentFactory handles this type.
        """
        mock_agent_instance = AsyncMock(spec=ToolCallingAgent)
        MockToolCallingAgent.return_value = mock_agent_instance

        agent_config = {
            "name": "TestToolAgent",
            "version": "0.1",
            "description": "A test ToolCalling agent.",
            "type": "ToolCallingAgent", # This type would be resolved by AgentFactory
            "config": {
                "llm_service_id": "default_llm",
                "tools": ["ToolA"],
                "meta": {"another_key": "another_value"}
            }
        }
        mock_tools = [MagicMock()]
        with patch.object(self.provider, '_resolve_tools', return_value=mock_tools):
            agent = self.provider.create_agent_instance(agent_config)

        MockToolCallingAgent.assert_called_once_with(
            llm_service=self.mock_llm_service,
            tools=mock_tools,
            meta=agent_config["config"]["meta"]
        )
        self.assertEqual(agent, mock_agent_instance)

    async def test_execute_agent_instance(self):
        """
        Test executing an agent instance.
        Assumes agent.run() returns a Run object as per Phase 2.
        """
        mock_agent = AsyncMock() # Can be ReAct or ToolCalling
        mock_run_obj = AsyncMock(spec=Run)
        mock_run_obj.get_output.return_value = "Agent execution result"
        mock_agent.run = AsyncMock(return_value=mock_run_obj)

        prompt = "Execute this task."
        mock_context = MagicMock(spec=RunContext)

        result = await self.provider.execute_agent_instance(mock_agent, prompt, context=mock_context)

        mock_agent.run.assert_called_once_with(prompt=prompt, context=mock_context)
        mock_run_obj.get_output.assert_called_once() # verify we awaited the output
        self.assertEqual(result, "Agent execution result")

    def test_resolve_tools_with_tool_instances(self):
        """Test _resolve_tools when actual tool instances are provided."""
        tool1 = MagicMock(name="Tool1Instance")
        tool2 = MagicMock(name="Tool2Instance")
        resolved_tools = self.provider._resolve_tools([tool1, tool2])
        self.assertEqual(resolved_tools, [tool1, tool2])

    @patch('evolving_agents.tools.tool_factory.ToolFactory.create_tool')
    def test_resolve_tools_with_tool_names(self, mock_create_tool):
        """Test _resolve_tools when tool names are provided and need creation."""
        mock_tool_instance = MagicMock(name="CreatedToolInstance")
        mock_create_tool.return_value = mock_tool_instance

        # Mock the tool_factory if it's directly accessed by the provider
        # If it's accessed via DependencyContainer, that would need more elaborate mocking
        self.provider.tool_factory = MagicMock()
        self.provider.tool_factory.create_tool = mock_create_tool

        tool_names = ["RegisteredToolName"]
        resolved_tools = self.provider._resolve_tools(tool_names)

        mock_create_tool.assert_called_once_with("RegisteredToolName")
        self.assertEqual(resolved_tools, [mock_tool_instance])

    def test_get_name(self):
        self.assertEqual(self.provider.get_name(), "BeeAIProvider")


if __name__ == '__main__':
    unittest.main()
