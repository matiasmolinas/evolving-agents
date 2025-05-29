# scripts/test_smart_memory.py
import asyncio
import json
import uuid
import logging
from datetime import datetime, timezone
import os

from evolving_agents.core.dependency_container import DependencyContainer
from evolving_agents.core.llm_service import LLMService
from evolving_agents.core.mongodb_client import MongoDBClient
from evolving_agents.agent_bus.smart_agent_bus import SmartAgentBus
from evolving_agents.tools.internal.mongo_experience_store_tool import MongoExperienceStoreTool
from evolving_agents.tools.internal.semantic_experience_search_tool import SemanticExperienceSearchTool
from evolving_agents.tools.internal.message_summarization_tool import MessageSummarizationTool
from evolving_agents.agents.memory_manager_agent import MemoryManagerAgent
from beeai_framework.memory import UnconstrainedMemory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://mongo:27017/evolving_agents_db")
MONGODB_DATABASE_NAME = os.getenv("MONGODB_DATABASE_NAME", "evolving_agents_db")
MEMORY_MANAGER_AGENT_ID = "memory_manager_agent_default_id"

async def setup_dependencies():
    container = DependencyContainer()
    logger.info(f"Initializing MongoDBClient (URI: {MONGODB_URI}, DB: {MONGODB_DATABASE_NAME})...")
    mongodb_client = MongoDBClient(uri=MONGODB_URI, db_name=MONGODB_DATABASE_NAME)
    if not await mongodb_client.ping_server(): raise ConnectionError("MongoDB ping failed.")
    container.register("mongodb_client", mongodb_client)

    logger.info("Initializing LLMService...")
    if not os.getenv("OPENAI_API_KEY"): logger.warning("OPENAI_API_KEY not found.")
    llm_service = LLMService(container=container)
    container.register("llm_service", llm_service)

    experience_store_tool = MongoExperienceStoreTool(mongodb_client=mongodb_client, llm_service=llm_service)
    semantic_search_tool = SemanticExperienceSearchTool(mongodb_client=mongodb_client, llm_service=llm_service)
    message_summarization_tool = MessageSummarizationTool(llm_service=llm_service)
    memory_manager_agent = MemoryManagerAgent(
        llm_service=llm_service, mongo_experience_store_tool=experience_store_tool,
        semantic_search_tool=semantic_search_tool, message_summarization_tool=message_summarization_tool,
        memory=UnconstrainedMemory()
    )
    agent_bus = SmartAgentBus(container=container)
    container.register("agent_bus", agent_bus)
    await agent_bus.register_agent(
        agent_id=MEMORY_MANAGER_AGENT_ID, name="MemoryManagerAgentForTest", description="Test MMA.",
        agent_type="MemoryManagement", capabilities=[{"id": "process_task", "name": "Process Task", "description": "Processes memory tasks.", "confidence": 1.0}],
        agent_instance=memory_manager_agent
    )
    return agent_bus, mongodb_client

async def main():
    logger.info("Starting Simplified Smart Memory Test Script...")
    agent_bus, mongodb_client = await setup_dependencies()
    experience_id_to_test = f"exp_simple_test_{uuid.uuid4().hex}"
    experience_data = {"experience_id": experience_id_to_test, "primary_goal_description": "Simple Docker Test Goal"}
    # Forcing json.dumps for the data part to ensure it's a well-formed string for the ReAct agent
    store_task_desc = f"Store experience: {json.dumps(experience_data)}"

    store_success = False
    logger.info(f"--- Test 1: Storing Experience {experience_id_to_test} ---")
    try:
        # The content for MemoryManagerAgent's 'process_task' should be a dictionary
        response = await agent_bus.request_capability(
            capability="process_task",
            content={"task_description": store_task_desc}, # task_description is the string MMA's run() expects
            specific_agent_id=MEMORY_MANAGER_AGENT_ID,
            timeout=180
        )
        logger.info(f"Store Response: {str(response)[:200]}") # Log snippet of response
        coll = mongodb_client.get_collection("eat_agent_experiences")
        if await coll.find_one({"experience_id": experience_id_to_test}):
            logger.info(f"DB check: Experience {experience_id_to_test} found. SUCCESS.")
            store_success = True
        else:
            # Log more info if MMA response was not confirming
            logger.error(f"DB check: Experience {experience_id_to_test} NOT found. MMA Response was: {response}. FAILED.")
    except Exception as e:
        logger.error(f"Store Test FAILED (API call): {e}", exc_info=True)

    if store_success:
        logger.info(f"--- Test 2: Retrieving Experience {experience_id_to_test} by direct tool query ---")
        # Using MongoExperienceStoreTool directly to verify data integrity simply.
        # Testing MMA's ReAct retrieval would be a separate, more complex assertion based on its LLM response.
        retrieved_exp = await MongoExperienceStoreTool(mongodb_client, llm_service).get_experience(experience_id_to_test)
        if retrieved_exp and retrieved_exp.get("primary_goal_description") == experience_data["primary_goal_description"]:
            logger.info(f"Direct Get Experience Test: SUCCESS. Found: {retrieved_exp.get('primary_goal_description')}")
        else:
            logger.error(f"Direct Get Experience Test: FAILED. Exp ID: {experience_id_to_test}. Retrieved: {retrieved_exp}")

    logger.info(f"--- Test Cleanup: Removing {experience_id_to_test} ---")
    try:
        coll = mongodb_client.get_collection("eat_agent_experiences")
        await coll.delete_one({"experience_id": experience_id_to_test})
        logger.info(f"Cleanup: Deleted {experience_id_to_test}.")
    except Exception as e:
        logger.error(f"Cleanup FAILED: {e}", exc_info=True)
    logger.info("Simplified Smart Memory Test Script Finished.")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"): print("ERROR: OPENAI_API_KEY missing.")
    else: asyncio.run(main())
