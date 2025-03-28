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
from evolving_agents.tools.agent_bus.register_agent_tool import RegisterAgentTool
from evolving_agents.tools.agent_bus.request_agent_tool import RequestAgentTool
from evolving_agents.tools.agent_bus.discover_agent_tool import DiscoverAgentTool

from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.firmware.firmware import Firmware
from evolving_agents.core.llm_service import LLMService
from evolving_agents.agent_bus.smart_agent_bus import SmartAgentBus
from evolving_agents.core.dependency_container import DependencyContainer

# Import workflow components
from evolving_agents.workflow.workflow_processor import WorkflowProcessor
from evolving_agents.workflow.workflow_generator import WorkflowGenerator

logger = logging.getLogger(__name__)

class SystemAgentFactory:
    """
    Factory for creating a SystemAgent as a pure BeeAI ReActAgent.
    """
    
    @staticmethod
    async def create_agent(
        llm_service: Optional[LLMService] = None,
        smart_library: Optional[SmartLibrary] = None,
        agent_bus = None,
        memory_type: str = "token",
        container: Optional[DependencyContainer] = None
    ) -> ReActAgent:
        # Resolve dependencies from container if provided
        if container:
            llm_service = llm_service or container.get('llm_service')
            smart_library = smart_library or container.get('smart_library')
            agent_bus = agent_bus or container.get('agent_bus')
            firmware = container.get('firmware') if container.has('firmware') else Firmware()
        else:
            if not llm_service:
                raise ValueError("LLM service must be provided")
            
            if not smart_library:
                smart_library = SmartLibrary("system_library.json")
            
            firmware = Firmware()
        
        chat_model = llm_service.chat_model
        
        if not agent_bus:
            agent_bus = SmartAgentBus(
                smart_library=smart_library,
                system_agent=None,  # Will be set later
                llm_service=llm_service,
                storage_path="smart_agent_bus.json", 
                log_path="agent_bus_logs.json"
            )
        
        # Create tools
        search_tool = SearchComponentTool(smart_library)
        create_tool = CreateComponentTool(smart_library, llm_service, firmware)
        evolve_tool = EvolveComponentTool(smart_library, llm_service, firmware)
        register_tool = RegisterAgentTool(agent_bus)
        request_tool = RequestAgentTool(agent_bus)
        discover_tool = DiscoverAgentTool(agent_bus)
        
        workflow_processor = WorkflowProcessor()
        workflow_generator = WorkflowGenerator(llm_service, smart_library)
        
        tools = [
            search_tool,
            create_tool,
            evolve_tool,
            register_tool,
            request_tool,
            discover_tool
        ]
        
        meta = AgentMeta(
            name="SystemAgent",
            description=(
                "I am the System Agent, responsible for orchestrating the agent ecosystem. "
                "I manage agent discovery, registration, and capability requests."
            ),
            extra_description=(
                "I follow the agent-centric architecture principles where everything is an agent "
                "with capabilities. I coordinate between specialized tools."
            ),
            tools=tools
        )
        
        memory = UnconstrainedMemory() if memory_type == "unconstrained" else TokenMemory(chat_model)
        
        system_agent = ReActAgent(
            llm=chat_model,
            tools=tools,
            memory=memory,
            meta=meta
        )
        
        workflow_processor.set_agent(system_agent)
        workflow_generator.set_agent(system_agent)
        
        tools_dict = {
            "search_component": search_tool,
            "create_component": create_tool,
            "evolve_component": evolve_tool,
            "register_agent": register_tool,
            "request_agent": request_tool,
            "discover_agent": discover_tool
        }
        
        system_agent.tools = tools_dict
        system_agent.workflow_processor = workflow_processor
        system_agent.workflow_generator = workflow_generator
        
        if hasattr(agent_bus, 'system_agent') and agent_bus.system_agent is None:
            agent_bus.system_agent = system_agent
        
        if container and not container.has('system_agent'):
            container.register('system_agent', system_agent)
            
        if hasattr(agent_bus, 'initialize_from_library'):
            await agent_bus.initialize_from_library()
        
        return system_agent