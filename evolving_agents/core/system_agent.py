# evolving_agents/core/system_agent.py

import logging
import uuid # Added missing import potentially needed for fallback library name
from typing import Dict, Any, List, Optional

# BeeAI Framework imports
from beeai_framework.agents.react import ReActAgent
from beeai_framework.agents.types import AgentMeta
from beeai_framework.memory import TokenMemory, UnconstrainedMemory

# Import our specialized tools
# Standard Tools
from evolving_agents.tools.smart_library.search_component_tool import SearchComponentTool
from evolving_agents.tools.smart_library.create_component_tool import CreateComponentTool
from evolving_agents.tools.smart_library.evolve_component_tool import EvolveComponentTool
from evolving_agents.tools.smart_library.task_context_tool import TaskContextTool
from evolving_agents.tools.smart_library.task_context_tool import ContextualSearchTool
from evolving_agents.tools.agent_bus.register_agent_tool import RegisterAgentTool
from evolving_agents.tools.agent_bus.request_agent_tool import RequestAgentTool
from evolving_agents.tools.agent_bus.discover_agent_tool import DiscoverAgentTool

# Workflow Tools
from evolving_agents.workflow.generate_workflow_tool import GenerateWorkflowTool
from evolving_agents.workflow.process_workflow_tool import ProcessWorkflowTool

# Intent Review Tools (Added in the update)
from evolving_agents.tools.intent_review.workflow_design_review_tool import WorkflowDesignReviewTool
from evolving_agents.tools.intent_review.component_selection_review_tool import ComponentSelectionReviewTool
from evolving_agents.tools.intent_review.approve_plan_tool import ApprovePlanTool

# Core Components
from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.firmware.firmware import Firmware
from evolving_agents.core.llm_service import LLMService
from evolving_agents.agent_bus.smart_agent_bus import SmartAgentBus
from evolving_agents.core.dependency_container import DependencyContainer
from evolving_agents.core.base import IAgent # Ensure IAgent is imported if used

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

        # --- Resolve Dependencies ---
        logger.debug(f"SystemAgentFactory: Received container: {container is not None}")
        resolved_llm_service = llm_service
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
             raise ValueError("LLM Service is required to create SystemAgent, but none was found.")

        if resolved_llm_service is None:
             logger.critical("SystemAgentFactory: resolved_llm_service is None even after checks!")
             raise ValueError("Critical Error: LLM Service resolved to None unexpectedly.")

        # Access chat_model after ensuring resolved_llm_service is valid
        chat_model = resolved_llm_service.chat_model
        logger.debug(f"SystemAgentFactory: Successfully accessed chat_model: {chat_model is not None}")

        # Resolve other dependencies
        resolved_smart_library = smart_library
        if not resolved_smart_library and container:
            logger.debug("SystemAgentFactory: Attempting to get 'smart_library' from container.")
            if container.has('smart_library'):
                resolved_smart_library = container.get('smart_library')
                logger.debug(f"SystemAgentFactory: Retrieved 'smart_library': {resolved_smart_library is not None}")
            else:
                 logger.warning("SystemAgentFactory: Container does not have 'smart_library', creating default.")
                 resolved_smart_library = SmartLibrary(f"sys_agent_fallback_library_{uuid.uuid4().hex[:4]}.json")
        elif not resolved_smart_library:
             logger.error("SystemAgentFactory: Smart Library is required but none was provided or found.")
             raise ValueError("Smart Library is required for SystemAgent.")

        resolved_agent_bus = agent_bus
        if not resolved_agent_bus and container:
            logger.debug("SystemAgentFactory: Attempting to get 'agent_bus' from container.")
            if container.has('agent_bus'):
                 resolved_agent_bus = container.get('agent_bus')
                 logger.debug(f"SystemAgentFactory: Retrieved 'agent_bus': {resolved_agent_bus is not None}")
            else:
                 logger.warning("SystemAgentFactory: Container does not have 'agent_bus', creating default.")
                 resolved_agent_bus = SmartAgentBus(container=container) # Use container for its deps
        elif not resolved_agent_bus:
            logger.error("SystemAgentFactory: Agent Bus is required but none was provided or found.")
            raise ValueError("Agent Bus is required for SystemAgent.")

        resolved_firmware = None
        if container and container.has('firmware'):
             resolved_firmware = container.get('firmware')
        else:
             logger.warning("SystemAgentFactory: Firmware not in container, creating default.")
             resolved_firmware = Firmware()

        # --- Create Tools using RESOLVED dependencies ---
        # Standard Tools
        search_tool = SearchComponentTool(resolved_smart_library)
        create_tool = CreateComponentTool(resolved_smart_library, resolved_llm_service, resolved_firmware)
        evolve_tool = EvolveComponentTool(resolved_smart_library, resolved_llm_service, resolved_firmware)
        register_tool = RegisterAgentTool(resolved_agent_bus)
        request_tool = RequestAgentTool(resolved_agent_bus)
        discover_tool = DiscoverAgentTool(resolved_agent_bus)
        generate_workflow_tool = GenerateWorkflowTool(resolved_llm_service, resolved_smart_library)
        process_workflow_tool = ProcessWorkflowTool()

        # Task context tools
        task_context_tool = TaskContextTool(resolved_llm_service)
        contextual_search_tool = ContextualSearchTool(task_context_tool, search_tool)

        # Intent review tools (Added in the update)
        workflow_design_review_tool = WorkflowDesignReviewTool()
        component_selection_review_tool = ComponentSelectionReviewTool()
        approve_plan_tool = ApprovePlanTool(llm_service=resolved_llm_service) # Needs LLM

        # Define the list of tools for the agent (Include ALL necessary tools)
        tools = [
            contextual_search_tool,  # Prioritized for context-aware searching
            task_context_tool,
            search_tool,
            create_tool,
            evolve_tool,
            register_tool,
            request_tool,
            discover_tool,
            generate_workflow_tool,
            process_workflow_tool,
            # Intent Review Tools (Include these)
            workflow_design_review_tool,
            component_selection_review_tool,
            approve_plan_tool,
        ]

        # --- Agent Meta (Using the updated description) ---
        meta = AgentMeta(
            name="SystemAgent",
            description=(
                "I am the System Agent, the central orchestrator for the agent ecosystem. "
                "My primary purpose is to help you reuse, evolve, and create agents and tools "
                "to solve your problems efficiently. I find the most relevant components by "
                "deeply understanding the specific task context you're working in. "
                # Added line for human-in-the-loop
                "I can also operate in a human-in-the-loop workflow where my plans are reviewed "
                "before execution to ensure safety and appropriateness."
            ),
            extra_description=(
                "When faced with a complex task, I might need to break it down. This could involve analyzing requirements, "
                "identifying or creating necessary agents/tools via the SmartLibrary, coordinating their execution, "
                "and potentially generating an internal plan if multiple steps are needed. "
                "I can also use task context to find components specifically relevant to the current operation. "
                "My goal is to deliver the final result for your request."
            ),
            tools=tools # Pass the complete list of tools
        )

        # --- Memory and Agent Creation ---
        memory = UnconstrainedMemory() if memory_type == "unconstrained" else TokenMemory(chat_model)
        system_agent = ReActAgent(
            llm=chat_model,
            tools=tools, # Use the complete list
            memory=memory,
            meta=meta
        )

        # --- Tool Mapping (Include ALL tools) ---
        tools_dict = {
            # Standard Tools
            "search_component": search_tool,
            "create_component": create_tool,
            "evolve_component": evolve_tool,
            "register_agent": register_tool,
            "request_agent": request_tool,
            "discover_agent": discover_tool,
            "generate_workflow": generate_workflow_tool,
            "process_workflow": process_workflow_tool,
            "task_context": task_context_tool,
            "contextual_search": contextual_search_tool,
            # Intent Review Tools Map (Include these)
            "workflow_design_review": workflow_design_review_tool,
            "component_selection_review": component_selection_review_tool,
            "approve_plan": approve_plan_tool,
        }
        system_agent.tools_map = tools_dict # Assign the complete map

        # --- Container Registration & Bus Update (FIXED Section) ---
        if container and not container.has('system_agent'):
            container.register('system_agent', system_agent)
            logger.debug("SystemAgentFactory: Registered self (SystemAgent instance) in container.")

        # Ensure bus knows about system agent - USE DIRECT ASSIGNMENT
        if resolved_agent_bus and resolved_agent_bus._system_agent_instance is None:
            # resolved_agent_bus.set_system_agent(system_agent) # <-- Removed this line
            resolved_agent_bus._system_agent_instance = system_agent # <-- Use this direct assignment
            logger.debug("SystemAgentFactory: Set self as system_agent on the resolved AgentBus.")
        elif resolved_agent_bus and resolved_agent_bus._system_agent_instance is not system_agent:
            logger.warning("SystemAgentFactory: Agent Bus already has a system agent assigned, replacing.")
            # resolved_agent_bus.set_system_agent(system_agent) # <-- Removed this line
            resolved_agent_bus._system_agent_instance = system_agent # <-- Use this direct assignment
        # ---------------------------------------------------------

        logger.info("SystemAgent created successfully with intent review capabilities.")
        return system_agent