# evolving_agents/core/system_agent.py

import logging
from typing import Dict, Any, List, Optional

# BeeAI Framework imports
from beeai_framework.agents.react import ReActAgent
from beeai_framework.agents.types import AgentMeta
from beeai_framework.memory import TokenMemory, UnconstrainedMemory

# Import our specialized tools
from evolving_agents.tools.smart_library.search_component_tool import SearchComponentTool
from evolving_agents.tools.smart_library.create_component_tool import CreateComponentTool
from evolving_agents.tools.smart_library.evolve_component_tool import EvolveComponentTool
from evolving_agents.tools.agent_bus.register_provider_tool import RegisterProviderTool
from evolving_agents.tools.agent_bus.request_service_tool import RequestServiceTool
from evolving_agents.tools.agent_bus.discover_capability_tool import DiscoverCapabilityTool

from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.firmware.firmware import Firmware
from evolving_agents.core.llm_service import LLMService
from evolving_agents.agent_bus.smart_agent_bus import SmartAgentBus

# Import workflow components with fixed imports
from evolving_agents.workflow.workflow_processor import WorkflowProcessor
from evolving_agents.workflow.workflow_generator import WorkflowGenerator

logger = logging.getLogger(__name__)

class SystemAgentFactory:
    """
    Factory for creating a SystemAgent as a pure BeeAI ReActAgent.
    
    The SystemAgent is a specialized ReActAgent that manages the agent ecosystem
    using a set of specialized tools for component management and agent communication.
    """
    
    @staticmethod
    async def create_agent(
        llm_service: LLMService,
        smart_library: Optional[SmartLibrary] = None,
        agent_bus = None,
        memory_type: str = "token"
    ) -> ReActAgent:
        """
        Create and configure a SystemAgent as a pure BeeAI ReActAgent.
        
        Args:
            llm_service: LLM service for text generation
            smart_library: Optional SmartLibrary instance (will create if not provided)
            agent_bus: Optional AgentBus instance (will create if not provided)
            memory_type: Memory type for the agent ("token" or "unconstrained")
            
        Returns:
            Fully configured SystemAgent as a BeeAI ReActAgent
        """
        # Get the chat model from LLM service
        chat_model = llm_service.chat_model
        
        # Initialize SmartLibrary if not provided
        if not smart_library:
            smart_library = SmartLibrary("system_library.json")
        
        # Create firmware instance
        firmware = Firmware()
        
        # Initialize or use provided agent_bus
        if not agent_bus:
            # Create a simple agent bus
            agent_bus = SmartAgentBus(
                smart_library=smart_library,
                system_agent=None,  # We'll set this after system_agent is created
                storage_path="smart_agent_bus.json", 
                log_path="agent_bus_logs.json"
            )
        
        # Create Smart Library tools
        search_tool = SearchComponentTool(smart_library)
        create_tool = CreateComponentTool(smart_library, llm_service, firmware)
        evolve_tool = EvolveComponentTool(smart_library, llm_service, firmware)
        
        # Create Agent Bus tools
        register_tool = RegisterProviderTool(agent_bus)
        request_tool = RequestServiceTool(agent_bus)
        discover_tool = DiscoverCapabilityTool(agent_bus)
        
        # Initialize workflow components (without circular dependency)
        workflow_processor = WorkflowProcessor()
        workflow_generator = WorkflowGenerator(llm_service, smart_library)
        
        # Create the tools list for the agent
        tools = [
            search_tool,
            create_tool,
            evolve_tool,
            register_tool,
            request_tool,
            discover_tool
        ]
        
        # Create SystemAgent metadata with a clean description that doesn't hardcode strategies
        meta = AgentMeta(
            name="SystemAgent",
            description=(
                "I am the System Agent, responsible for orchestrating the agent ecosystem. "
                "I manage component discovery, creation, evolution, and communication using "
                "specialized tools that each have their own expertise and strategies."
            ),
            extra_description=(
                "I follow the agent-centric architecture principles where everything is an agent "
                "with capabilities. I coordinate between specialized tools, each of which handles "
                "its own domain of expertise with its own embedded strategies."
            ),
            tools=tools
        )
        
        # Create memory based on specified type
        if memory_type == "unconstrained":
            memory = UnconstrainedMemory()
        else:
            memory = TokenMemory(chat_model)
        
        # Create the SystemAgent as a pure BeeAI ReActAgent
        system_agent = ReActAgent(
            llm=chat_model,
            tools=tools,
            memory=memory,
            meta=meta
        )
        
        # Set the agent in the workflow components
        workflow_processor.set_agent(system_agent)
        workflow_generator.set_agent(system_agent)
        
        # Create a tools dictionary for convenience (not part of the ReActAgent interface)
        tools_dict = {
            "search_component": search_tool,
            "create_component": create_tool,
            "evolve_component": evolve_tool,
            "register_provider": register_tool,
            "request_service": request_tool,
            "discover_capability": discover_tool
        }
        
        # Add property for components that might be needed elsewhere
        system_agent.tools = tools_dict
        system_agent.workflow_processor = workflow_processor
        system_agent.workflow_generator = workflow_generator
        
        return system_agent