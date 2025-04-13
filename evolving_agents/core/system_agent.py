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
from evolving_agents.tools.smart_library.task_context_tool import TaskContextTool
from evolving_agents.tools.smart_library.task_context_tool import ContextualSearchTool
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

from evolving_agents.core.base import IAgent

logger = logging.getLogger(__name__)

class SystemAgentFactory:
    @staticmethod
    async def create_agent(
        llm_service: Optional[LLMService] = None, # Keep allowing direct pass for flexibility
        smart_library: Optional[SmartLibrary] = None,
        agent_bus = None,
        memory_type: str = "token",
        container: Optional[DependencyContainer] = None
    ) -> ReActAgent:

        # --- Debugging Dependency Resolution ---
        logger.debug(f"SystemAgentFactory: Received container: {container is not None}")
        resolved_llm_service = llm_service # Start with potentially passed service
        if not resolved_llm_service and container:
            logger.debug("SystemAgentFactory: Attempting to get 'llm_service' from container.")
            if container.has('llm_service'):
                resolved_llm_service = container.get('llm_service')
                logger.debug(f"SystemAgentFactory: Retrieved 'llm_service' from container: {resolved_llm_service is not None}")
            else:
                logger.warning("SystemAgentFactory: Container does not have 'llm_service'.")
        elif resolved_llm_service:
             logger.debug("SystemAgentFactory: Using directly passed 'llm_service'.")
        else:
             logger.error("SystemAgentFactory: No LLM service provided directly or via container.")
             # Raise error early if no LLM service found
             raise ValueError("LLM Service is required to create SystemAgent, but none was found.")

        # Now use the resolved service, check again just in case
        if resolved_llm_service is None:
             # This should theoretically be caught by the ValueError above, but safety check.
             logger.critical("SystemAgentFactory: resolved_llm_service is None even after checks!")
             raise ValueError("Critical Error: LLM Service resolved to None unexpectedly.")

        # ---> This is the line that caused the error <---
        chat_model = resolved_llm_service.chat_model
        logger.debug(f"SystemAgentFactory: Successfully accessed chat_model: {chat_model is not None}")
        # ------------------------------------------


        # Resolve other dependencies similarly with logging
        resolved_smart_library = smart_library
        if not resolved_smart_library and container:
            logger.debug("SystemAgentFactory: Attempting to get 'smart_library' from container.")
            if container.has('smart_library'):
                resolved_smart_library = container.get('smart_library')
                logger.debug(f"SystemAgentFactory: Retrieved 'smart_library': {resolved_smart_library is not None}")
            else:
                 logger.warning("SystemAgentFactory: Container does not have 'smart_library', creating default.")
                 # Fallback or raise error? Let's fallback for now if needed elsewhere
                 resolved_smart_library = SmartLibrary(f"sys_agent_fallback_library_{uuid.uuid4().hex[:4]}.json")
        elif not resolved_smart_library:
             raise ValueError("Smart Library is required for SystemAgent.")

        resolved_agent_bus = agent_bus
        if not resolved_agent_bus and container:
            logger.debug("SystemAgentFactory: Attempting to get 'agent_bus' from container.")
            if container.has('agent_bus'):
                 resolved_agent_bus = container.get('agent_bus')
                 logger.debug(f"SystemAgentFactory: Retrieved 'agent_bus': {resolved_agent_bus is not None}")
            else:
                 logger.warning("SystemAgentFactory: Container does not have 'agent_bus', creating default.")
                 # Need to ensure this default bus gets proper deps later if created here
                 resolved_agent_bus = SmartAgentBus(container=container) # Use container for its deps

        resolved_firmware = None
        if container and container.has('firmware'):
             resolved_firmware = container.get('firmware')
        else:
             logger.warning("SystemAgentFactory: Firmware not in container, creating default.")
             resolved_firmware = Firmware()

        # --- Create Tools using RESOLVED dependencies ---
        search_tool = SearchComponentTool(resolved_smart_library)
        create_tool = CreateComponentTool(resolved_smart_library, resolved_llm_service, resolved_firmware)
        evolve_tool = EvolveComponentTool(resolved_smart_library, resolved_llm_service, resolved_firmware)
        register_tool = RegisterAgentTool(resolved_agent_bus)
        request_tool = RequestAgentTool(resolved_agent_bus)
        discover_tool = DiscoverAgentTool(resolved_agent_bus)
        generate_workflow_tool = GenerateWorkflowTool(resolved_llm_service, resolved_smart_library)
        process_workflow_tool = ProcessWorkflowTool()
        
        # New task context tools
        task_context_tool = TaskContextTool(resolved_llm_service)
        contextual_search_tool = ContextualSearchTool(task_context_tool, search_tool)


        # Prioritize the contextual search tool
        tools = [
            contextual_search_tool,  # This should come first for visibility
            task_context_tool,
            search_tool, 
            create_tool, 
            evolve_tool,
            register_tool, 
            request_tool, 
            discover_tool,
            generate_workflow_tool, 
            process_workflow_tool
        ]

        # --- Agent Meta (using updated description) ---
        meta = AgentMeta(
            name="SystemAgent",
            description=(
                "I am the System Agent, the central orchestrator for the agent ecosystem. "
                "My primary purpose is to help you reuse, evolve, and create agents and tools "
                "to solve your problems efficiently. I find the most relevant components by "
                "deeply understanding the specific task context you're working in. "
                "Whether implementing, testing, or documenting, I'll recommend the most appropriate "
                "existing components to reuse or evolve, or help create new ones when needed. "
                "Give me a goal, and I will design and run multi-step workflows using my available tools "
                "to achieve it, always prioritizing effective component reuse and evolution."
            ),
            extra_description=(
                "When faced with a complex task, I might need to break it down. This could involve analyzing requirements, "
                "identifying or creating necessary agents/tools via the SmartLibrary, coordinating their execution, "
                "and potentially generating an internal plan if multiple steps are needed. "
                "I can also use task context to find components specifically relevant to the current operation. "
                "My goal is to deliver the final result for your request."
            ),
            tools=tools
        )

        # --- Memory and Agent Creation ---
        memory = UnconstrainedMemory() if memory_type == "unconstrained" else TokenMemory(chat_model)
        system_agent = ReActAgent(
            llm=chat_model,
            tools=tools,
            memory=memory,
            meta=meta
        )

        # --- Tool Mapping (Optional) ---
        tools_dict = {
            "search_component": search_tool, 
            "create_component": create_tool, 
            "evolve_component": evolve_tool,
            "register_agent": register_tool, 
            "request_agent": request_tool, 
            "discover_agent": discover_tool,
            "generate_workflow": generate_workflow_tool, 
            "process_workflow": process_workflow_tool,
            "task_context": task_context_tool,  # Map the new tools
            "contextual_search": contextual_search_tool
        }
        system_agent.tools_map = tools_dict

        # --- Container Registration ---
        # Register the created agent instance
        if container and not container.has('system_agent'):
            container.register('system_agent', system_agent)
            logger.debug("SystemAgentFactory: Registered self in container.")

        # Ensure bus knows about system agent (important if bus was created before agent)
        # Accessing _system_agent_instance directly is okay here within the factory context
        if resolved_agent_bus and resolved_agent_bus._system_agent_instance is None:
             resolved_agent_bus._system_agent_instance = system_agent
             logger.debug("SystemAgentFactory: Set self as system_agent on the resolved AgentBus.")

        logger.info("SystemAgent created successfully.")
        return system_agent