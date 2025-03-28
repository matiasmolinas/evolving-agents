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

from evolving_agents.workflow.generate_workflow_tool import GenerateWorkflowTool
from evolving_agents.workflow.process_workflow_tool import ProcessWorkflowTool

from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.firmware.firmware import Firmware
from evolving_agents.core.llm_service import LLMService
from evolving_agents.agent_bus.smart_agent_bus import SmartAgentBus
from evolving_agents.core.dependency_container import DependencyContainer

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
             # Ensure agent_bus uses the container if provided
            agent_bus = SmartAgentBus(
                smart_library=smart_library,
                system_agent=None, # Will be set later
                llm_service=llm_service,
                storage_path="smart_agent_bus.json",
                log_path="agent_bus_logs.json",
                container=container # Pass container here
            )

        # --- Create Tools ---
        # Library Tools
        search_tool = SearchComponentTool(smart_library)
        create_tool = CreateComponentTool(smart_library, llm_service, firmware)
        evolve_tool = EvolveComponentTool(smart_library, llm_service, firmware)
        # Agent Bus Tools
        register_tool = RegisterAgentTool(agent_bus)
        request_tool = RequestAgentTool(agent_bus)
        discover_tool = DiscoverAgentTool(agent_bus)
        # Workflow Tools (NEW)
        generate_workflow_tool = GenerateWorkflowTool(llm_service, smart_library)
        process_workflow_tool = ProcessWorkflowTool() # Doesn't need direct dependencies anymore

        # --- Assemble Tools for Agent ---
        tools = [
            search_tool,
            create_tool,
            evolve_tool,
            register_tool,
            request_tool,
            discover_tool,
            generate_workflow_tool, # Add new tool
            process_workflow_tool   # Add new tool
        ]

        # Define Agent Metadata
        meta = AgentMeta(
            name="SystemAgent",
            description=(
                "I am the System Agent, responsible for orchestrating the agent ecosystem. "
                "I manage component discovery, creation, evolution, registration, capability requests, "
                "and workflow generation/processing." # Updated description
            ),
            extra_description=(
                "I follow the agent-centric architecture principles where everything is an agent "
                "with capabilities. I coordinate between specialized tools, including workflow tools." # Updated
            ),
            tools=tools
        )

        # Define Memory
        memory = UnconstrainedMemory() if memory_type == "unconstrained" else TokenMemory(chat_model)

        # --- Create the ReActAgent ---
        system_agent = ReActAgent(
            llm=chat_model,
            tools=tools, # Pass the full list of tools
            memory=memory,
            meta=meta
        )

        # --- Store Tools for Easy Access (Optional but convenient) ---
        # Update the tools_dict mapping
        tools_dict = {
            "search_component": search_tool,
            "create_component": create_tool,
            "evolve_component": evolve_tool,
            "register_agent": register_tool,
            "request_agent": request_tool,
            "discover_agent": discover_tool,
            "generate_workflow": generate_workflow_tool, # Add new tool mapping
            "process_workflow": process_workflow_tool   # Add new tool mapping
        }
        # You can still attach this dict if needed for external access, but the agent uses the list passed in constructor
        system_agent.tools_map = tools_dict # Renamed to avoid conflict with internal 'tools'

        # --- Connect Agent Bus and Container ---
        # Set the created system_agent instance on the agent_bus if it wasn't already set
        if hasattr(agent_bus, 'system_agent') and agent_bus.system_agent is None:
            agent_bus.system_agent = system_agent

        # Register the final system_agent in the container if not already present
        if container and not container.has('system_agent'):
            container.register('system_agent', system_agent)

        # --- Final Initialization ---
        # Initialize agent bus from library *after* everything is set up
        if hasattr(agent_bus, 'initialize_from_library'):
            await agent_bus.initialize_from_library()

        logger.info("SystemAgent created successfully with workflow tools integrated.")
        return system_agent