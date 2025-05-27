import logging
from typing import Optional

from evolving_agents.core.dependency_container import DependencyContainer
from evolving_agents.agents.memory_manager_agent import MemoryManagerAgent
from evolving_agents.agent_bus.smart_agent_bus import SmartAgentBus
from evolving_agents.core.llm_service import LLMService
from evolving_agents.core.mongodb_client import MongoDBClient

logger = logging.getLogger(__name__)

async def initialize_and_register_memory_manager(container: DependencyContainer) -> Optional[MemoryManagerAgent]:
    """
    Instantiates the MemoryManagerAgent and registers it with the SmartAgentBus.

    This function should be called during the application's startup sequence
    after the DependencyContainer has been populated with essential services
    like LLMService, MongoDBClient, and SmartAgentBus.

    Args:
        container: The application's DependencyContainer instance.

    Returns:
        The instantiated MemoryManagerAgent if successful, None otherwise.
    """
    logger.info("Attempting to initialize and register MemoryManagerAgent...")

    try:
        # 1. Get Dependencies from Container
        llm_service = container.get("llm_service")
        if not llm_service:
            logger.error("LLMService not found in DependencyContainer. Cannot initialize MemoryManagerAgent.")
            return None

        mongodb_client = container.get("mongodb_client")
        if not mongodb_client:
            logger.error("MongoDBClient not found in DependencyContainer. Cannot initialize MemoryManagerAgent.")
            return None

        smart_agent_bus = container.get("smart_agent_bus")
        if not smart_agent_bus:
            logger.error("SmartAgentBus not found in DependencyContainer. Cannot register MemoryManagerAgent.")
            return None
            
        logger.debug("Successfully retrieved LLMService, MongoDBClient, and SmartAgentBus from container.")

        # 2. Instantiate MemoryManagerAgent
        logger.debug("Instantiating MemoryManagerAgent...")
        memory_manager_agent = MemoryManagerAgent(
            llm_service=llm_service,
            mongodb_client=mongodb_client
            # MemoryManagerAgent does not take the container directly in its constructor
        )
        logger.info(f"MemoryManagerAgent instantiated: {memory_manager_agent.agent_meta.name}")


        # 3. Register with SmartAgentBus
        # The AgentMeta is an attribute of the agent instance (self.agent_meta from ReActAgent base)
        # The register_agent method in SmartAgentBus expects:
        # agent_id (optional), name, description, capabilities (list of dicts),
        # agent_type, metadata, agent_instance, embed_capabilities.
        
        # Determine agent_id: use agent_meta.id if set, otherwise agent_meta.name
        # (Assuming AgentMeta from beeai_framework might have an 'id' field, or we default to name)
        agent_id_to_register = getattr(memory_manager_agent.agent_meta, 'id', memory_manager_agent.agent_meta.name)
        if not agent_id_to_register: # Fallback if name is also empty for some reason
            agent_id_to_register = "MemoryManagerAgent_DefaultID"
            logger.warning(f"AgentMeta.id and AgentMeta.name are empty. Using default ID: {agent_id_to_register}")


        logger.debug(f"Registering {memory_manager_agent.agent_meta.name} with SmartAgentBus...")
        await smart_agent_bus.register_agent(
            agent_id=agent_id_to_register,
            name=memory_manager_agent.agent_meta.name,
            description=memory_manager_agent.agent_meta.description,
            capabilities=memory_manager_agent.agent_meta.capabilities,
            agent_type="MemoryManagementServices", # A descriptive type for the agent's role
            metadata={"source": "EATCoreInitialization", "version": "1.0"}, # Example metadata
            agent_instance=memory_manager_agent,
            embed_capabilities=True # Recommended for discovery
        )
        logger.info(f"Successfully registered '{memory_manager_agent.agent_meta.name}' with SmartAgentBus (ID: {agent_id_to_register}).")
        
        # Optionally, add the initialized agent to the container if other components need direct access
        if not container.has("memory_manager_agent"):
            container.register("memory_manager_agent", memory_manager_agent)
            logger.info("Registered MemoryManagerAgent instance in DependencyContainer.")
            
        return memory_manager_agent

    except Exception as e:
        logger.error(f"Failed to initialize and register MemoryManagerAgent: {e}", exc_info=True)
        return None

# Conceptual example of how this might be called during application startup
async def example_main_startup():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 1. Initialize DependencyContainer and core services
    container = DependencyContainer()
    
    # Mock or initialize actual services
    # These would typically be more complex initializations
    mock_llm_service = LLMService(api_key="dummy_key", default_model="gpt-3.5-turbo") # Assuming LLMService structure
    mock_mongodb_client = MongoDBClient(connection_string="mongodb://localhost:27017/", database_name="eat_db") # Assuming MongoDBClient structure
    
    container.register("llm_service", mock_llm_service)
    container.register("mongodb_client", mock_mongodb_client)

    # SmartAgentBus itself might need other dependencies like SmartLibrary, LLMService from container
    # For this example, assume SmartAgentBus can be instantiated or retrieved
    # If SmartAgentBus uses the container for its own dependencies:
    try:
        # smart_library_for_bus = SmartLibrary(llm_service=mock_llm_service, mongodb_client=mock_mongodb_client)
        # container.register("smart_library", smart_library_for_bus) # If bus needs it
        mock_smart_agent_bus = SmartAgentBus(container=container) # Assuming bus can take container
    except Exception as e: # Catch if SmartLibrary or other deps are missing for bus
        logger.error(f"Failed to init SmartAgentBus for example: {e}")
        # Fallback if SmartAgentBus cannot be initialized with container
        # This is a simplified mock for the bus registration part to work
        class MockSmartAgentBus:
            async def register_agent(self, **kwargs): logger.info(f"MockBus: Agent {kwargs.get('name')} registered.")
        mock_smart_agent_bus = MockSmartAgentBus()

    container.register("smart_agent_bus", mock_smart_agent_bus)

    logger.info("Core services and DependencyContainer initialized for example startup.")

    # 2. Initialize and register MemoryManagerAgent
    mma = await initialize_and_register_memory_manager(container)
    if mma:
        logger.info("MemoryManagerAgent initialization and registration successful in example.")
    else:
        logger.error("MemoryManagerAgent initialization and registration failed in example.")

    # 3. Continue with other application setup, like initializing SystemAgent, etc.
    # from evolving_agents.core.system_agent import SystemAgentFactory
    # system_agent = await SystemAgentFactory.create_agent(container=container)
    # logger.info(f"SystemAgent created: {system_agent.meta.name if system_agent else 'Failed'}")


if __name__ == "__main__":
    import asyncio
    # This example_main_startup is conceptual and may require actual implementations
    # of LLMService, MongoDBClient, SmartAgentBus, SmartLibrary, etc.,
    # or more robust mocks to run without error.
    # asyncio.run(example_main_startup())
    pass
