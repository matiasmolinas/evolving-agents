# evolving_agents/core/system_agent.py

import logging
import uuid # For fallback library name
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
from evolving_agents.tools.smart_library.task_context_tool import TaskContextTool, ContextualSearchTool # Combined import
from evolving_agents.tools.agent_bus.register_agent_tool import RegisterAgentTool
from evolving_agents.tools.agent_bus.request_agent_tool import RequestAgentTool
from evolving_agents.tools.agent_bus.discover_agent_tool import DiscoverAgentTool

# Workflow Tools
from evolving_agents.workflow.generate_workflow_tool import GenerateWorkflowTool
from evolving_agents.workflow.process_workflow_tool import ProcessWorkflowTool

# Intent Review Tools
from evolving_agents.tools.intent_review.workflow_design_review_tool import WorkflowDesignReviewTool
from evolving_agents.tools.intent_review.component_selection_review_tool import ComponentSelectionReviewTool
from evolving_agents.tools.intent_review.approve_plan_tool import ApprovePlanTool

# Newly Integrated Tools for Context and Experience Management
from evolving_agents.tools.context_builder_tool import ContextBuilderTool
from evolving_agents.tools.experience_recorder_tool import ExperienceRecorderTool

# Core Components
from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.firmware.firmware import Firmware
from evolving_agents.core.llm_service import LLMService
from evolving_agents.agent_bus.smart_agent_bus import SmartAgentBus
from evolving_agents.core.dependency_container import DependencyContainer
from evolving_agents.core.base import IAgent # Ensure IAgent is imported
from evolving_agents.core.mongodb_client import MongoDBClient # For passing to tools

logger = logging.getLogger(__name__)

class SystemAgentFactory:
    @staticmethod
    async def create_agent(
        # Direct passing still allowed for flexibility/testing, but container is preferred
        llm_service: Optional[LLMService] = None,
        smart_library: Optional[SmartLibrary] = None,
        agent_bus: Optional[SmartAgentBus] = None,
        mongodb_client: Optional[MongoDBClient] = None, # Added for explicit passing
        memory_type: str = "token",
        container: Optional[DependencyContainer] = None
    ) -> ReActAgent:

        # --- Resolve Dependencies ---
        logger.debug(f"SystemAgentFactory: Received container: {container is not None}")

        # Helper to resolve from container or use provided, with logging
        def _resolve_dependency(name: str, provided_instance: Optional[Any], default_factory: Optional[callable] = None):
            instance = provided_instance
            if not instance and container and container.has(name):
                instance = container.get(name)
                logger.debug(f"SystemAgentFactory: Retrieved '{name}' from container.")
            elif instance:
                logger.debug(f"SystemAgentFactory: Using directly passed '{name}'.")
            elif default_factory:
                logger.warning(f"SystemAgentFactory: '{name}' not in container or provided, creating default.")
                instance = default_factory()
                if container and instance: container.register(name, instance) # Register if created
            else:
                logger.error(f"SystemAgentFactory: Critical dependency '{name}' not found or provided, and no default factory.")
                raise ValueError(f"{name} is required but was not found or provided.")
            if instance is None: # Should be caught by the else above, but as a safeguard
                raise ValueError(f"Critical Error: {name} resolved to None unexpectedly.")
            return instance

        resolved_llm_service = _resolve_dependency("llm_service", llm_service, lambda: LLMService())
        chat_model = resolved_llm_service.chat_model
        logger.debug(f"SystemAgentFactory: Using LLMService with chat_model: {chat_model is not None}")

        # For SmartLibrary, the default factory now needs the container or llm_service
        # If container is present, SmartLibrary's __init__ can pull llm_service from it.
        # If only llm_service is passed, it uses that.
        def smart_lib_factory():
            # SmartLibrary expects llm_service if no container, or will try to get from container
            if container:
                return SmartLibrary(container=container) # This will get llm_service and mongodb_client from container
            else: # If no container, llm_service and mongodb_client must be explicitly passed
                # This branch is less ideal as it might create a new MongoDBClient if not passed explicitly
                resolved_mongo_for_lib = mongodb_client or (container.get('mongodb_client') if container and container.has('mongodb_client') else MongoDBClient())
                return SmartLibrary(llm_service=resolved_llm_service, mongodb_client=resolved_mongo_for_lib)


        resolved_smart_library = _resolve_dependency("smart_library", smart_library, smart_lib_factory)

        resolved_mongodb_client = _resolve_dependency("mongodb_client", mongodb_client, lambda: MongoDBClient())


        # For AgentBus, it also needs container or resolved dependencies
        def agent_bus_factory():
            if container:
                return SmartAgentBus(container=container) # Will get smart_library, llm_service, mongodb_client from container
            else: # Less ideal, direct dependency passing
                return SmartAgentBus(smart_library=resolved_smart_library,
                                     llm_service=resolved_llm_service,
                                     mongodb_client=resolved_mongodb_client)

        resolved_agent_bus = _resolve_dependency("agent_bus", agent_bus, agent_bus_factory)

        resolved_firmware = _resolve_dependency("firmware", None, lambda: Firmware()) # Assuming firmware doesn't have complex deps

        # --- Create Tools using RESOLVED dependencies ---
        logger.debug("SystemAgentFactory: Instantiating tools...")
        # Standard Tools
        search_tool = SearchComponentTool(resolved_smart_library)
        create_tool = CreateComponentTool(resolved_smart_library, resolved_llm_service, resolved_firmware)
        evolve_tool = EvolveComponentTool(resolved_smart_library, resolved_llm_service, resolved_firmware)

        # Agent Bus Tools
        register_tool = RegisterAgentTool(resolved_agent_bus)
        request_tool = RequestAgentTool(resolved_agent_bus)
        discover_tool = DiscoverAgentTool(resolved_agent_bus) # This tool now uses MongoDB via AgentBus

        # Workflow Tools
        generate_workflow_tool = GenerateWorkflowTool(resolved_llm_service, resolved_smart_library)
        # **MODIFIED: Pass mongodb_client or container to ProcessWorkflowTool**
        process_workflow_tool = ProcessWorkflowTool(mongodb_client=resolved_mongodb_client, container=container)

        # Task context tools
        task_context_tool = TaskContextTool(resolved_llm_service)
        contextual_search_tool = ContextualSearchTool(task_context_tool, search_tool)

        # Intent review tools
        workflow_design_review_tool = WorkflowDesignReviewTool() # Assumes no complex deps for now
        component_selection_review_tool = ComponentSelectionReviewTool() # Assumes no complex deps
        # **MODIFIED: Pass mongodb_client or container to ApprovePlanTool**
        approve_plan_tool = ApprovePlanTool(llm_service=resolved_llm_service, mongodb_client=resolved_mongodb_client, container=container)

        # Instantiate ContextBuilderTool and ExperienceRecorderTool
        context_builder_tool = ContextBuilderTool(
            smart_agent_bus=resolved_agent_bus,
            smart_library=resolved_smart_library,
            llm_service=resolved_llm_service  # Pass LLM service, even if for future use by the tool
        )
        experience_recorder_tool = ExperienceRecorderTool(
            smart_agent_bus=resolved_agent_bus
        )
        logger.debug("SystemAgentFactory: ContextBuilderTool and ExperienceRecorderTool instantiated.")

        tools = [
            contextual_search_tool, task_context_tool, search_tool, create_tool, evolve_tool,
            register_tool, request_tool, discover_tool,
            generate_workflow_tool, process_workflow_tool,
            workflow_design_review_tool, component_selection_review_tool, approve_plan_tool,
            # Add new tools
            context_builder_tool,
            experience_recorder_tool,
        ]
        logger.debug(f"SystemAgentFactory: {len(tools)} tools instantiated.")

        # --- Agent Meta ---
        meta = AgentMeta(
            name="SystemAgent",
            description=(
                "I am the System Agent, the central orchestrator for the agent ecosystem. "
                "My primary purpose is to help you reuse, evolve, and create agents and tools "
                "to solve your problems efficiently. I find the most relevant components by "
                "deeply understanding the specific task context you're working in. "
                "I can also operate in a human-in-the-loop workflow where my plans are reviewed "
                "before execution to ensure safety and appropriateness."
            ),
            extra_description=(
                "When faced with a complex task, I might need to break it down. This could involve analyzing requirements, "
                "identifying or creating necessary agents/tools via the SmartLibrary, coordinating their execution, "
                "and potentially generating an internal plan if multiple steps are needed. "
                "I use task context for relevant component discovery. My goal is to deliver the final result."
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

        # --- Tool Mapping ---
        system_agent.tools_map = {tool_instance.name: tool_instance for tool_instance in tools}
        # Log the actual mapped tools for verification
        logger.debug(f"SystemAgent tools_map contains: {list(system_agent.tools_map.keys())}")

        # --- Conceptual Usage Comments for SystemAgent (ReActAgent instance) ---
        # The SystemAgent, being a ReActAgent, will use its tools based on its reasoning loop.
        # The following comments outline where the new tools would conceptually fit into its operations.
        # Actual implementation depends on the ReAct prompt engineering and internal logic.

        # TODO: Integrate ContextBuilderTool before detailed planning by SystemAgent
        # Location: Within SystemAgent's ReAct logic when it receives a new complex goal and is about
        #           to generate a detailed plan (e.g., before calling GenerateWorkflowTool).
        # Action:
        #   current_smart_context = self.memory.get_current_context() # Or however SystemAgent accesses its context
        #   planning_context = await self.tools_map['context_builder_tool'].build_context(
        #       target_agent_id=self.meta.id, # SystemAgent's own ID
        #       assigned_sub_task_goal_description=complex_goal_description,
        #       workflow_context=current_smart_context
        #   )
        #   # Use planning_context.data (e.g., relevant_past_experiences, summarized_message_history)
        #   # to inform GenerateWorkflowTool or other planning steps, potentially by adding
        #   # this context to the prompt for the planning tool.
        # Benefit: The insights from past experiences and message history in the returned SmartContext
        #          should inform a more robust and contextually aware planning process.

        # TODO: Use ContextBuilderTool before SystemAgent delegates a sub-task
        # Location: When SystemAgent is about to delegate a sub-task to another agent using
        #           RequestAgentTool or a similar mechanism.
        # Action:
        #   current_smart_context = self.memory.get_current_context()
        #   sub_task_context = await self.tools_map['context_builder_tool'].build_context(
        #       target_agent_id=worker_agent_id, # ID of the agent receiving the sub-task
        #       assigned_sub_task_goal_description=sub_task_description,
        #       workflow_context=current_smart_context
        #   )
        #   # Pass sub_task_context (e.g., sub_task_context.to_json_string()) to the worker agent
        #   # as part of the 'prompt' or 'content' for the RequestAgentTool call.
        #   # Example:
        #   # request_prompt = json.dumps({
        #   #     "capability": "target_capability_of_worker_agent",
        #   #     "args": {
        #   #         "task_description": sub_task_description,
        #   #         "task_specific_context": sub_task_context.to_dict() # Serialize SmartContext
        #   #     }
        #   # })
        #   # result = await self.tools_map['request_agent_tool'].run(
        #   #     agent_id=worker_agent_id,
        #   #     prompt=request_prompt
        #   # )
        # Benefit: The worker agent receives a richer, pre-processed context, enabling it to
        #          perform its sub-task more effectively and with less redundant information gathering.

        # TODO: Use ExperienceRecorderTool after SystemAgent completes a significant task/workflow
        # Location: After a significant workflow or task orchestrated by SystemAgent has concluded.
        #           SystemAgent needs criteria to determine "significance".
        # Action:
        #   # if task_is_significant and (workflow_succeeded or workflow_failed):
        #   #     # SystemAgent gathers information from its memory, workflow logs, tool outputs.
        #   #     experience_details = {
        #   #         "primary_goal_description": "The overall goal SystemAgent was trying to achieve.",
        #   #         "sub_task_description": "Specific sub-task if applicable, or main task again.",
        #   #         "involved_components": [ # List of dicts: {"component_id": ..., "component_name": ..., "component_type": ..., "usage_description": ...}
        #   #             {"component_id": "GenerateWorkflowTool_v1", "component_name": "GenerateWorkflowTool", "component_type": "TOOL", "usage_description": "Used for initial plan creation."},
        #   #             {"component_id": "worker_agent_alpha", "component_name": "AlphaWorker", "component_type": "AGENT", "usage_description": "Executed data processing sub-task."}
        #   #         ],
        #   #         "input_context_summary": "Summary of the initial context/prompt SystemAgent received.",
        #   #         "key_decisions_made": [ # List of dicts: {"decision_summary": ..., "decision_reasoning": ..., "timestamp": ...}
        #   #             {"decision_summary": "Chose AlphaWorker over BetaWorker for data processing.", "decision_reasoning": "AlphaWorker had higher success rate on similar tasks from experience context."}
        #   #         ],
        #   #         "final_outcome": "success" if workflow_succeeded else "failure",
        #   #         "final_outcome_reason": "Detailed reason for success or failure.",
        #   #         "output_summary": "Summary of the final output or result of the workflow.",
        #   #         "feedback_signals": [ # Optional: from user feedback or system metrics
        #   #             {"feedback_source": "user_rating", "feedback_content": "Excellent work!", "feedback_rating": 5.0}
        #   #         ],
        #   #         "tags": ["complex_workflow", "data_processing", "user_initiated"],
        #   #         "agent_version": self.meta.version if hasattr(self.meta, 'version') else "N/A", # Assuming AgentMeta might have version
        #   #         "initiating_agent_id": self.meta.id # Or the agent that called SystemAgent
        #   #     }
        #   #     record_result = await self.tools_map['experience_recorder_tool'].record_experience(**experience_details)
        #   #     if record_result.get("status") == "success":
        #   #         logger.info(f"SystemAgent successfully recorded experience: {record_result.get('experience_id')}")
        #   #     else:
        #   #         logger.error(f"SystemAgent failed to record experience: {record_result.get('message')}")
        # Benefit: Captures valuable knowledge about workflow execution, component performance, and decision effectiveness,
        #          making this information available for future planning and task execution via ContextBuilderTool.

        # --- Container Registration & AgentBus Update ---
        if container and not container.has('system_agent'):
            container.register('system_agent', system_agent)
            logger.debug("SystemAgentFactory: Registered SystemAgent instance in container.")

        # Ensure AgentBus knows about the SystemAgent instance for potential internal calls or context
        # This uses a direct assignment to a property as discussed.
        if resolved_agent_bus: # Check if agent_bus was successfully resolved
            if resolved_agent_bus._system_agent_instance is None:
                resolved_agent_bus._system_agent_instance = system_agent
                logger.debug("SystemAgentFactory: Set SystemAgent instance on the resolved AgentBus.")
            elif resolved_agent_bus._system_agent_instance is not system_agent:
                logger.warning("SystemAgentFactory: AgentBus already had a different SystemAgent instance. Overwriting.")
                resolved_agent_bus._system_agent_instance = system_agent
        else:
            logger.error("SystemAgentFactory: resolved_agent_bus is None, cannot set system_agent property on it.")


        logger.info("SystemAgent created successfully with updated tool initializations.")
        return system_agent