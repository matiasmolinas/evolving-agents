import asyncio
import logging
import uuid
import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# EAT imports
from evolving_agents.core.mongodb_client import MongoDBClient
from evolving_agents.core.llm_service import LLMService
from evolving_agents.agent_bus.smart_agent_bus import SmartAgentBus # Assuming direct import
from evolving_agents.core.smart_context import SmartContext, Message, ContextEntry
from evolving_agents.agents.memory_manager_agent import MemoryManagerAgent
from evolving_agents.tools.context_builder_tool import ContextBuilderTool
from evolving_agents.tools.experience_recorder_tool import ExperienceRecorderTool
from evolving_agents.tools.internal.mongo_experience_store_tool import MongoExperienceStoreTool
from evolving_agents.core.dependency_container import DependencyContainer # For SmartAgentBus if it uses one
from evolving_agents.smart_library.smart_library import SmartLibrary # Import real SmartLibrary


async def preload_sample_component(smart_lib_instance: SmartLibrary):
    """
    Pre-loads a sample tool component into the SmartLibrary if it doesn't already exist.
    """
    logger = logging.getLogger(__name__)
    component_name = "GenericResearchTool"
    component_type = "TOOL"
    try:
        # SmartLibrary's find_record_by_name might need more specific parameters or might not exist.
        # Let's assume find_record_by_id is more common, or we try to create and handle potential errors.
        # For this demo, we'll attempt to find it by name and type, assuming such a method or similar logic exists.
        # A more robust way would be to assign a fixed UUID and check by ID.
        # For simplicity, we'll rely on name and type for this check.
        # SmartLibrary.find_record_by_name may not exist; using a more generic check or try-catch on create.
        # A simple approach: try to get it, if not found, create it.
        # The `SmartLibrary.create_record` should internally handle embedding generation.

        # Check if component exists (simplified check, real check might need version too)
        # As `find_record_by_name` is not a standard method in the provided SmartLibrary snippets,
        # we'll attempt creation and rely on potential unique constraints or just overwrite for demo simplicity if needed.
        # A better check for "already exists" would use a specific ID.
        # For now, we'll just log an attempt. A real app might query first.
        logger.info(f"Attempting to preload {component_name} into SmartLibrary...")
        
        # To truly check existence, one might need a method like:
        # existing_component = await smart_lib_instance.get_component_by_name_and_type(name=component_name, record_type=component_type)
        # This demo will proceed to create, assuming `create_record` is idempotent or it's okay for a demo.
        # For a more robust check, a dedicated `find_record_by_name_and_type` would be ideal in SmartLibrary.
        # Let's assume we try to create it and if it fails due to a unique constraint (e.g. name+type+version), it's fine.
        # Or, for the demo, we just create it. If it's run multiple times, it might create duplicates without a proper check.
        # The prompt example uses `find_record_by_name` - let's assume it exists for the purpose of the demo structure.

        existing_components = await smart_lib_instance.search_components(query=component_name, limit=5, filters={"name": component_name, "record_type": component_type})
        
        found_existing = False
        if existing_components:
            for comp in existing_components:
                if comp.get("name") == component_name and comp.get("record_type") == component_type:
                    logger.info(f"{component_name} already exists in SmartLibrary with ID: {comp.get('id')}.")
                    found_existing = True
                    break
        
        if not found_existing:
            logger.info(f"Preloading {component_name} into SmartLibrary...")
            tool_record_data = {
                "name": component_name,
                "record_type": component_type,
                "domain": "Research",
                "description": "A generic tool for performing research tasks, capable of searching and summarizing information.",
                # Functional text that SmartLibrary will use for E_orig embedding
                "code_snippet": "class GenericResearchTool:\n    def __init__(self, api_key: str):\n        self.api_key = api_key\n\n    def run(self, query: str, sources: list = ['web', 'arxiv']):\n        # Simulates searching various sources\n        print(f'GenericResearchTool: Researching query \"{query}\" across sources: {sources}')\n        results = f'Comprehensive research results for \"{query}\" from {sources}. Key findings include A, B, C.'\n        # In a real tool, this would involve API calls, data processing, etc.\n        return results\n\n    def summarize(self, text: str):\n        print(f'GenericResearchTool: Summarizing text of length {len(text)}')\n        return f'Summary of the provided text: First sentence. Last sentence.'",
                "version": "1.0.0",
                "status": "active", # Assuming 'status' is a relevant field for SmartLibrary records
                "tags": ["research", "utility", "generic_tool"],
                # Applicability text (T_raz) will be generated by SmartLibrary.create_record
                # if its internal logic includes call to _generate_applicability_text
                "metadata": { 
                    "dependencies": ["requests", "beautifulsoup4"],
                    "example_usage": "tool.run(query='latest advancements in AI')"
                }
            }
            # `create_record` in SmartLibrary is responsible for generating embeddings.
            created_component_doc = await smart_lib_instance.create_record(tool_record_data)
            logger.info(f"{component_name} preloaded with ID: {created_component_doc['id']}")
        
    except Exception as e:
        logger.error(f"Error preloading {component_name}: {e}", exc_info=True)


async def preload_dummy_experiences(
    mongo_tool: MongoExperienceStoreTool,
    num_experiences: int = 2
):
    """
    Pre-loads dummy experiences relevant to AI research into MongoDB.
    """
    logger = logging.getLogger(__name__)
    logger.info("Preloading dummy experiences...")

    common_fields = {
        "involved_components": [{"component_id": "WebSearchTool_v1", "component_name": "WebSearchTool", "component_type": "TOOL", "usage_description": "Used for initial info gathering."}],
        "input_context_summary": "Tasked with finding information on a new AI topic.",
        "key_decisions_made": [{"decision_summary": "Focused on papers from top conferences.", "decision_reasoning": "Higher quality signal."}],
        "agent_version": "DemoAgent_v0.1",
        "tags": ["ai_research", "literature_review", "dummy_data"]
    }

    experiences = [
        {
            **common_fields,
            "primary_goal_description": "Understand and summarize 'Transformer Networks'.",
            "sub_task_description": "Initial literature review of Transformer papers.",
            "final_outcome": "success",
            "status_reason": "Successfully gathered and summarized key papers.", # if MongoExperienceStoreTool uses this
            "output_summary": "Transformer networks rely on self-attention mechanisms and are foundational to many modern NLP models. Key papers include 'Attention Is All You Need'.",
            "feedback": [{"feedback_source": "self", "feedback_content": "Good overview achieved.", "feedback_rating": 4.0}]
        },
        {
            **common_fields,
            "primary_goal_description": "Investigate 'Federated Learning' privacy implications.",
            "sub_task_description": "Research privacy-preserving techniques in Federated Learning.",
            "final_outcome": "partial_success",
            "status_reason": "Found several techniques but detailed comparison is pending.",
            "output_summary": "Federated Learning inherently offers some privacy by not moving raw data. Techniques like secure aggregation and differential privacy can further enhance it. More research needed for specific trade-offs.",
            "feedback": [{"feedback_source": "self", "feedback_content": "Found relevant techniques but need deeper analysis.", "feedback_rating": 3.5}]
        }
    ]

    for i, exp_data in enumerate(experiences[:num_experiences]):
        try:
            # MongoExperienceStoreTool expects 'status' not 'final_outcome' for the schema
            exp_data_for_store = exp_data.copy()
            exp_data_for_store["status"] = exp_data_for_store.pop("final_outcome")
            
            # Ensure all embeddable fields are present, even if empty
            for field in mongo_tool.embed_fields:
                if field not in exp_data_for_store:
                    exp_data_for_store[field] = "" # Add empty string if missing
                elif exp_data_for_store[field] is None: # Ensure None is also empty string
                    exp_data_for_store[field] = ""


            exp_id = await mongo_tool.store_experience(exp_data_for_store)
            logger.info(f"Preloaded dummy experience {i+1} with ID: {exp_id}")
        except Exception as e:
            logger.error(f"Error preloading dummy experience {i+1}: {e}", exc_info=True)
    logger.info("Dummy experience preloading complete.")


async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Starting Smart Memory System Demo: Researching and Summarizing a New AI Technique")
    logger.info("---")
    logger.info("IMPORTANT: Ensure MONGODB_URI, MONGODB_DATABASE_NAME, and LLM API keys (e.g., OPENAI_API_KEY) are set in your environment.")
    logger.info("If MongoDB is not running or accessible, parts of this demo will fail.")
    logger.info("---")

    # --- 1. Setup Environment ---
    logger.info("STEP 1: Setting up environment (MongoDB, LLM, AgentBus, Tools)...")
    try:
        # These will typically be loaded from environment variables by the classes themselves
        mongodb_client = MongoDBClient() # Assumes MONGODB_URI and MONGODB_DATABASE_NAME are set
        llm_service = LLMService() # Assumes OPENAI_API_KEY (or other) is set

        # SmartAgentBus can use a container.
        container = DependencyContainer()
        container.register("llm_service", llm_service)
        container.register("mongodb_client", mongodb_client)
        
        # Instantiate real SmartLibrary
        logger.info("Initializing SmartLibrary...")
        smart_library = SmartLibrary(
            llm_service=llm_service,
            mongodb_client=mongodb_client
            # SmartLibrary's constructor might also accept/prefer `container=container`
            # For this demo, direct llm_service and mongodb_client passing is clear.
            # await smart_library.initialize() # If SmartLibrary has an async init method
        )
        container.register("smart_library", smart_library) # Register real SmartLibrary for AgentBus
        logger.info("SmartLibrary initialized.")

        # SmartAgentBus needs SmartLibrary for some operations (e.g. agent component discovery)
        smart_agent_bus = SmartAgentBus(container=container)
        container.register("smart_agent_bus", smart_agent_bus)

        # Instantiate MemoryManagerAgent
        memory_manager_agent = MemoryManagerAgent(
            llm_service=llm_service,
            mongodb_client=mongodb_client
        )
        # Register MemoryManagerAgent with SmartAgentBus
        # The agent_meta is stored as self.agent_meta by ReActAgent's __init__
        agent_id_to_register = getattr(memory_manager_agent.agent_meta, 'id', memory_manager_agent.agent_meta.name)
        await smart_agent_bus.register_agent(
            agent_id=agent_id_to_register,
            name=memory_manager_agent.agent_meta.name,
            description=memory_manager_agent.agent_meta.description,
            capabilities=memory_manager_agent.agent_meta.capabilities,
            agent_type="MemoryManagementServices",
            metadata={"source": "SmartMemoryDemoScript", "version": "1.0"},
            agent_instance=memory_manager_agent,
            embed_capabilities=True
        )
        logger.info(f"MemoryManagerAgent '{memory_manager_agent.agent_meta.name}' instantiated and registered with SmartAgentBus.")

        # Instantiate tools with real SmartLibrary
        context_builder_tool = ContextBuilderTool(
            smart_agent_bus=smart_agent_bus,
            smart_library=smart_library,    # Pass the real smart_library instance
            llm_service=llm_service
        )
        experience_recorder_tool = ExperienceRecorderTool(smart_agent_bus=smart_agent_bus)
        logger.info("Core services, MemoryManagerAgent, and tools initialized.")

    except Exception as e:
        logger.error(f"Error during environment setup: {e}", exc_info=True)
        logger.error("This demo may not function correctly. Please check your environment variables and service availability (MongoDB, LLM).")
        return

    # --- 2. Pre-load Data (Experiences & Components) ---
    logger.info("\nSTEP 2: Pre-loading data (dummy experiences and sample component)...")
    
    # Pre-load dummy experiences
    direct_mongo_tool = MongoExperienceStoreTool(mongodb_client, llm_service) # For direct preloading
    await preload_dummy_experiences(direct_mongo_tool)

    # Pre-load sample component into SmartLibrary
    await preload_sample_component(smart_library)
    
    logger.info("Pre-loading complete.")

    # --- 3. Simulate SystemAgent Starting a New Task ---
    logger.info("\nSTEP 3: Simulating SystemAgent starting a new research task...")
    main_goal = "Research and summarize a new AI technique: Hyperdimensional Computing."
    current_sub_task_goal = "Gather initial information and identify key concepts of Hyperdimensional Computing."
    
    simulated_workflow_context = SmartContext(current_task=main_goal)
    simulated_workflow_context.metadata["agent_id"] = "SimulatedSystemAgent"
    simulated_workflow_context.metadata["workflow_id"] = f"wf_{uuid.uuid4().hex[:8]}"
    simulated_workflow_context.add_message(Message(sender_id="UserRequest", content="I need a report on Hyperdimensional Computing."))
    simulated_workflow_context.add_message(Message(sender_id="SimulatedSystemAgent", content=f"Understood. Starting initial research on: {current_sub_task_goal}"))
    logger.info(f"Main Goal: {main_goal}")
    logger.info(f"Current Sub-task: {current_sub_task_goal}")
    logger.info(f"Simulated Workflow Context created with {len(simulated_workflow_context.messages)} messages.")

    # --- 4. Demonstrate ContextBuilderTool Usage ---
    logger.info("\nSTEP 4: Demonstrating ContextBuilderTool...")
    built_context: Optional[SmartContext] = None
    try:
        built_context = await context_builder_tool.build_context(
            target_agent_id="ResearchAgent_ChildInstance", # For whom this context is built
            assigned_sub_task_goal_description=current_sub_task_goal,
            workflow_context=simulated_workflow_context,
            historical_message_limit=5,
            relevant_experience_limit=2,
            relevant_component_limit=3
        )
        logger.info("ContextBuilderTool.build_context() executed.")

        if built_context:
            logger.info("--- Built Context Details ---")
            logger.info(f"  Target Agent ID: {built_context.metadata.get('target_agent_id')}")
            logger.info(f"  Assigned Task: {built_context.current_task}")
            
            summary_entry = built_context.get_data_entry("summarized_message_history")
            if summary_entry: logger.info(f"  Summarized Message History:\n    {summary_entry.value}")
            else: logger.info("  Summarized Message History: Not found or empty.")

            experiences_entry = built_context.get_data_entry("relevant_past_experiences")
            if experiences_entry and experiences_entry.value:
                logger.info(f"  Relevant Past Experiences ({len(experiences_entry.value)}):")
                for i, exp in enumerate(experiences_entry.value):
                    logger.info(f"    Exp {i+1}: Goal='{exp.get('primary_goal')}', SubTask='{exp.get('sub_task')}', Output='{exp.get('output_summary', '')[:50]}...'")
            else: logger.info("  Relevant Past Experiences: Not found or empty.")

            components_entry = built_context.get_data_entry("relevant_library_components")
            if components_entry and components_entry.value:
                logger.info(f"  Relevant Library Components (from SmartLibrary) ({len(components_entry.value)}):")
                for i, comp in enumerate(components_entry.value):
                    logger.info(f"    Comp {i+1}: Name='{comp.get('name')}', ID='{comp.get('id')}', Desc='{comp.get('description')[:60]}...'")
            else: logger.info("  Relevant Library Components: Not found or empty.")
            logger.info("--- End of Built Context ---")

    except Exception as e:
        logger.error(f"Error using ContextBuilderTool: {e}", exc_info=True)

    # --- 5. Simulate Task Execution Results ---
    logger.info("\nSTEP 5: Simulating task execution results for Hyperdimensional Computing research...")
    key_decisions_made_during_task = [
        {"decision_summary": "Focused on papers by Kanerva and Plate.", "decision_reasoning": "Identified as key figures from initial context.", "timestamp": datetime.now(timezone.utc).isoformat()},
        {"decision_summary": "Focused on papers by Kanerva and Plate.", "decision_reasoning": "Identified as key figures from initial context.", "timestamp": datetime.now(timezone.utc).isoformat()},
        {"decision_summary": "Selected GenericResearchTool for initial information gathering.", "decision_reasoning": "Tool identified as relevant by ContextBuilderTool via SmartLibrary."}
    ]
    output_of_research = (
        "Hyperdimensional Computing (HDC) represents data as high-dimensional vectors (hypervectors) "
        "and uses simple operations (binding, bundling, permutation) for computation. "
        "It's inspired by neural models and offers robustness to noise and efficient learning. "
        "Key applications include NLP, cognitive architectures, and biosignal processing."
    )
    final_task_outcome = "success" # Could be "failure", "partial_success"
    final_task_outcome_reason = "Successfully gathered and summarized core concepts of HDC."
    involved_components_in_task = [
        {"component_id": "GenericResearchTool_1.0.0", "component_name": "GenericResearchTool", "component_type": "TOOL", "usage_description": "Used for initial research query on HDC."}, # Assuming ID might include version or is auto-generated
        {"component_id": "SimulatedSystemAgent", "component_name": "SimulatedSystemAgent", "component_type": "AGENT", "usage_description": "Orchestrated the research task."}
    ]
    logger.info("Simulated task execution results defined.")

    # --- 6. Demonstrate ExperienceRecorderTool Usage ---
    logger.info("\nSTEP 6: Demonstrating ExperienceRecorderTool...")
    recorded_experience_id: Optional[str] = None
    try:
        record_response = await experience_recorder_tool.record_experience(
            primary_goal_description=main_goal,
            sub_task_description=current_sub_task_goal,
            involved_components=involved_components_in_task,
            input_context_summary=built_context.get_data_entry("summarized_message_history").value if built_context and built_context.get_data_entry("summarized_message_history") else "Initial prompt about HDC.",
            key_decisions_made=key_decisions_made_during_task,
            final_outcome=final_task_outcome,
            final_outcome_reason=final_task_outcome_reason,
            output_summary=output_of_research,
            feedback_signals=[{"feedback_source": "system_simulation", "feedback_content": "Task completed as expected.", "feedback_rating": 4.8}],
            tags=["ai_research", "hyperdimensional_computing", "hdc", "new_technique_summary", "demo_scenario"],
            agent_version="SimulatedSystemAgent_v1.0",
            initiating_agent_id=simulated_workflow_context.metadata.get("agent_id")
        )
        logger.info(f"ExperienceRecorderTool.record_experience() response: {record_response}")
        if record_response and record_response.get("status") == "success":
            recorded_experience_id = record_response.get("experience_id")
            logger.info(f"Successfully recorded experience with ID: {recorded_experience_id}")
        else:
            logger.warning(f"Failed to record experience or unexpected response: {record_response}")

    except Exception as e:
        logger.error(f"Error using ExperienceRecorderTool: {e}", exc_info=True)

    # --- 7. Verification Info ---
    logger.info("\nSTEP 7: Verification Info")
    if recorded_experience_id:
        logger.info(f"To verify the recorded experience, please check your MongoDB instance.")
        logger.info(f"  Database: '{os.getenv('MONGODB_DATABASE_NAME', 'your_db_name')}' (ensure this matches your config)")
        logger.info(f"  Collection: 'eat_agent_experiences'")
        logger.info(f"  Look for a document with 'experience_id': '{recorded_experience_id}'")
        logger.info(f"  You can use a MongoDB client (like MongoDB Compass or mongosh) with a query like:")
        logger.info(f"  `db.eat_agent_experiences.findOne({{ \"experience_id\": \"{recorded_experience_id}\" }})`")
    else:
        logger.info("Experience was not successfully recorded, so no specific ID to verify.")
    logger.info("--- Demo Complete ---")

if __name__ == "__main__":
    # Ensure that the asyncio event loop is managed correctly, especially if running in
    # environments like Jupyter notebooks where a loop might already be running.
    # For standard Python scripts, asyncio.run() is fine.
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "cannot be called when another loop is running" in str(e):
            logger_main = logging.getLogger(__name__)
            logger_main.warning("Asyncio loop already running. This might be an issue in some environments like Spyder/Jupyter if not handled.")
            # If you must run it, you might need nest_asyncio or careful loop management.
            # For this demo, we'll just log the warning.
        else:
            raise
