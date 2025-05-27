import logging
from typing import List, Optional, Dict, Any, Union

# Assuming beeai_framework is in the PYTHONPATH
# If not, this import will fail and needs to be adjusted based on actual project structure.
# For now, we proceed with the assumption it's available.
try:
    from beeai_framework.agents.react.react_agent import ReActAgent # Adjusted path
    from beeai_framework.agents.agent_meta import AgentMeta # Adjusted path
except ImportError:
    logging.warning(
        "beeai_framework not found. Using placeholder for ReActAgent and AgentMeta. "
        "Functionality will be limited. Ensure beeai_framework is installed and in PYTHONPATH."
    )
    # Define placeholder classes if beeai_framework is not available
    # This allows the rest of the file to be syntactically correct for development purposes.
    class AgentMeta:
        def __init__(self, name: str, description: str, capabilities: List[Dict[str, Any]], tools: Optional[List[Any]] = None):
            self.name = name
            self.description = description
            self.capabilities = capabilities
            self.tools = tools or []

    class ReActAgent:
        def __init__(self, llm_service: Any, agent_meta: AgentMeta, tools: List[Any]):
            self.llm_service = llm_service
            self.agent_meta = agent_meta
            self.tools = tools
            self.logger = logging.getLogger(agent_meta.name if agent_meta else __name__)
            self.logger.info(f"{self.agent_meta.name if self.agent_meta else 'Agent'} initialized.")

        async def run(self, prompt: str, **kwargs) -> Any:
            # This is a placeholder run method.
            # The actual ReActAgent would have a complex implementation.
            self.logger.info(f"Placeholder run method called with prompt: {prompt}")
            # In a real scenario, this would parse the prompt, identify a capability,
            # and call the corresponding method.
            # For now, it doesn't do anything functional.
            if "store_agent_experience" in prompt:
                # This is a simplistic check, real parsing would be needed
                # Also, arguments would need to be extracted
                return {"status": "error", "message": "Placeholder run: Cannot execute store_agent_experience without proper request parsing."}
            elif "retrieve_relevant_experiences" in prompt:
                return {"status": "error", "message": "Placeholder run: Cannot execute retrieve_relevant_experiences without proper request parsing."}
            elif "summarize_message_history" in prompt:
                return {"status": "error", "message": "Placeholder run: Cannot execute summarize_message_history without proper request parsing."}
            return {"status": "error", "message": "Placeholder run: Unknown capability or prompt not parsable."}


from evolving_agents.core.llm_service import LLMService
from evolving_agents.core.mongodb_client import MongoDBClient
from evolving_agents.tools.internal.mongo_experience_store_tool import MongoExperienceStoreTool
from evolving_agents.tools.internal.message_summarization_tool import MessageSummarizationTool
# from evolving_agents.core.smart_context import Message # Using Dict as per latest instruction


class MemoryManagerAgent(ReActAgent):
    """
    Manages persistent storage and retrieval of agent experiences and contextual facts,
    and provides message summarization capabilities.
    It uses internal tools to interact with MongoDB for experience storage/retrieval
    and an LLM for message summarization.
    """

    def __init__(self, llm_service: LLMService, mongodb_client: MongoDBClient):
        """
        Initializes the MemoryManagerAgent.

        Args:
            llm_service: LLMService for the agent's own ReAct thinking and summarization.
            mongodb_client: MongoDBClient for database interactions via tools.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing MemoryManagerAgent...")

        # Initialize internal tools
        # Assuming the same llm_service can be used for embeddings and summarization.
        # If specific models are needed, multiple LLMService instances might be required.
        try:
            self.mongo_experience_store_tool = MongoExperienceStoreTool(mongodb_client, llm_service)
            self.logger.info("MongoExperienceStoreTool initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize MongoExperienceStoreTool: {e}", exc_info=True)
            raise

        try:
            self.message_summarization_tool = MessageSummarizationTool(llm_service)
            self.logger.info("MessageSummarizationTool initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize MessageSummarizationTool: {e}", exc_info=True)
            raise

        capabilities = [
            {
                "id": "store_agent_experience",
                "name": "Store Agent Experience",
                "description": "Stores a structured record of a completed agent task or workflow.",
                "parameters": {
                    "type": "object",
                    "properties": {"experience_data": {"type": "object", "description": "The structured experience data to store."}},
                    "required": ["experience_data"]
                }
            },
            {
                "id": "retrieve_relevant_experiences",
                "name": "Retrieve Relevant Experiences",
                "description": "Retrieves past agent experiences relevant to a given goal or sub-task description using semantic search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "goal_description": {"type": "string", "description": "The primary goal to find relevant experiences for."},
                        "sub_task_description": {"type": ["string", "null"], "description": "Optional: A more specific sub-task description."},
                        "limit": {"type": "integer", "default": 5, "description": "Max number of experiences to return."}
                    },
                    "required": ["goal_description"]
                }
            },
            {
                "id": "summarize_message_history",
                "name": "Summarize Message History",
                "description": "Summarizes a list of messages, focusing on relevance to a target goal.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "messages": {"type": "array", "items": {"type": "object"}, "description": "List of message objects/dictionaries."},
                        "target_goal": {"type": "string", "description": "The goal to focus the summary on."}
                    },
                    "required": ["messages", "target_goal"]
                }
            }
        ]

        agent_meta = AgentMeta(
            name="MemoryManagerAgent",
            description="Manages persistent storage and retrieval of agent experiences and contextual facts, and provides message summarization.",
            capabilities=capabilities,
            tools=[] # Internal tools are not exposed directly to ReAct prompt builder
        )

        # Initialize ReActAgent base class
        # The llm_service passed here is for the ReAct thinking process itself.
        super().__init__(llm_service=llm_service, agent_meta=agent_meta, tools=[])
        self.logger.info("MemoryManagerAgent initialized successfully with ReActAgent as base.")

    async def store_agent_experience(self, experience_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Capability: Stores a new agent experience.
        Internally calls MongoExperienceStoreTool.store_experience.
        """
        self.logger.info(f"store_agent_experience called with data: {experience_data.get('experience_id', 'N/A')}")
        if not isinstance(experience_data, dict) or not experience_data:
            self.logger.warning("Invalid experience_data received: must be a non-empty dictionary.")
            return {"status": "error", "message": "Invalid input: experience_data must be a non-empty dictionary."}

        try:
            experience_id = await self.mongo_experience_store_tool.store_experience(experience_data)
            self.logger.info(f"Successfully stored experience: {experience_id}")
            return {"status": "success", "experience_id": experience_id}
        except Exception as e:
            self.logger.error(f"Error storing agent experience: {e}", exc_info=True)
            return {"status": "error", "message": f"Failed to store experience: {str(e)}"}

    async def retrieve_relevant_experiences(
        self,
        goal_description: str,
        sub_task_description: Optional[str] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Capability: Retrieves past experiences relevant to a current goal or task.
        Internally calls MongoExperienceStoreTool.find_similar_experiences.
        """
        self.logger.info(
            f"retrieve_relevant_experiences called with goal: '{goal_description}', "
            f"sub_task: '{sub_task_description}', limit: {limit}"
        )

        if not goal_description or not isinstance(goal_description, str):
            self.logger.warning("Invalid goal_description: must be a non-empty string.")
            return {"status": "error", "message": "Invalid input: goal_description must be a non-empty string."}
        if sub_task_description and not isinstance(sub_task_description, str):
            self.logger.warning("Invalid sub_task_description: must be a string if provided.")
            return {"status": "error", "message": "Invalid input: sub_task_description must be a string if provided."}
        if not isinstance(limit, int) or limit <= 0:
            self.logger.warning(f"Invalid limit value: {limit}. Using default of 5.")
            limit = 5


        text_to_match = goal_description
        text_field_to_search = "primary_goal_description"

        if sub_task_description and sub_task_description.strip():
            text_to_match = sub_task_description
            text_field_to_search = "sub_task_description"
        
        self.logger.info(f"Searching on field '{text_field_to_search}' with text: '{text_to_match}'")

        try:
            experiences = await self.mongo_experience_store_tool.find_similar_experiences(
                text_to_match=text_to_match,
                text_field_to_search=text_field_to_search,
                limit=limit
            )
            self.logger.info(f"Retrieved {len(experiences)} relevant experiences.")
            return {"status": "success", "experiences": experiences}
        except Exception as e:
            self.logger.error(f"Error retrieving relevant experiences: {e}", exc_info=True)
            return {"status": "error", "message": f"Failed to retrieve experiences: {str(e)}"}

    async def summarize_message_history(
        self,
        messages: List[Dict[str, Any]], # Conforms to List[Message] if Message is a Dict structure
        target_goal: str,
        max_summary_tokens: int = 250 # Added as per tool's signature
    ) -> Dict[str, Any]:
        """
        Capability: Summarizes a list of messages.
        Internally calls MessageSummarizationTool.summarize_messages.
        """
        self.logger.info(f"summarize_message_history called for goal: '{target_goal}' with {len(messages)} messages.")
        if not messages or not isinstance(messages, list):
            self.logger.warning("Invalid messages: must be a non-empty list of dictionaries.")
            return {"status": "error", "message": "Invalid input: messages must be a non-empty list."}
        if not target_goal or not isinstance(target_goal, str):
            self.logger.warning("Invalid target_goal: must be a non-empty string.")
            return {"status": "error", "message": "Invalid input: target_goal must be a non-empty string."}

        try:
            summary = await self.message_summarization_tool.summarize_messages(
                messages=messages,
                target_goal=target_goal,
                max_summary_tokens=max_summary_tokens
            )
            self.logger.info("Successfully generated message summary.")
            return {"status": "success", "summary": summary}
        except Exception as e:
            self.logger.error(f"Error summarizing message history: {e}", exc_info=True)
            return {"status": "error", "message": f"Failed to summarize messages: {str(e)}"}

    # The ReActAgent.run(self, prompt: str) method is inherited.
    # For this MemoryManagerAgent, the `prompt` is expected to be a structured request,
    # potentially a JSON string, from the SmartAgentBus specifying which capability
    # to invoke and its arguments.

    # The ReActAgent's `run` method (or a custom one overriding it) would need to:
    # 1. Parse the `prompt` (e.g., JSON string) to determine the intended capability
    #    (e.g., "store_agent_experience") and extract its arguments.
    # 2. Validate these arguments.
    # 3. Dispatch the call to the appropriate internal capability method:
    #    - self.store_agent_experience(**parsed_args)
    #    - self.retrieve_relevant_experiences(**parsed_args)
    #    - self.summarize_message_history(**parsed_args)
    # 4. The ReAct "thought" process here is less about complex tool chaining and more
    #    about request parsing and dispatching to the correct internal function.
    #    The "action" is calling one of its own methods.
    # 5. The "observation" is the result from that method.
    # 6. The final response is then formulated based on this observation.

    # If the ReActAgent base class's `run` method is generic enough to allow
    # defining tools that are actually agent's own methods, that would be one way.
    # Alternatively, `run` might be overridden here to implement this specific
    # capability dispatch logic if the base `run` is too restrictive or expects
    # external-like tools.

    # For now, we assume the base ReActAgent's `run` can be configured or is
    # flexible enough, or that the SmartAgentBus directly calls these capability
    # methods after an initial routing phase. If using SmartAgentBus, the bus
    # might directly call `agent.store_agent_experience(data)`, bypassing `run`
    # for such direct capability calls. The ReAct part of the agent would be for
    # more complex internal reasoning if ever needed, or if it's the standard
    # interface for all agent interactions.

# Example of how this agent might be used (conceptual):
# async def main():
#     # Setup (dummy MongoDBClient and LLMService for example)
#     class DummyLLMService(LLMService):
#         async def generate(self, prompt: str, **kwargs) -> str: return f"Summary for: {prompt[:50]}"
#         async def embed(self, text: str) -> List[float]: return [0.1] * 10 # Dummy embedding
    
#     class DummyMongoDBClient(MongoDBClient):
#         def __init__(self, connection_string: str, database_name: str):
#             self.collections = {}
#             self.logger = logging.getLogger(__name__)

#         def get_collection(self, collection_name: str) -> Any: # Returns a dummy collection
#             class DummyCollection:
#                 async def insert_one(self, doc): self.logger.info(f"Dummy insert: {doc}"); return None
#                 async def find_one(self, query, projection): self.logger.info(f"Dummy find_one: {query}"); return {"experience_id": query.get("experience_id"), **query}
#                 async def update_one(self, query, update): self.logger.info(f"Dummy update: {query}"); return type('obj', (object,), {'modified_count':1, 'matched_count': 1})
#                 async def delete_one(self, query): self.logger.info(f"Dummy delete: {query}"); return type('obj', (object,), {'deleted_count': 1})
#                 async def aggregate(self, pipeline): self.logger.info(f"Dummy aggregate: {pipeline}"); return self # Make it awaitable
#                 async def to_list(self, length): return [{"experience_id": "dummy_exp", "text": "some experience"}]


#             if collection_name not in self.collections:
#                 self.collections[collection_name] = DummyCollection()
#             return self.collections[collection_name]

#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)
#     logger.info("Starting MemoryManagerAgent example.")

#     try:
#         llm_service = DummyLLMService(api_key="dummy", default_model="dummy")
#         mongodb_client = DummyMongoDBClient(connection_string="dummy_string", database_name="dummy_db")
#         logger.info("Dummy services initialized.")

#         memory_manager = MemoryManagerAgent(llm_service=llm_service, mongodb_client=mongodb_client)
#         logger.info("MemoryManagerAgent instantiated.")

#         # 1. Store an experience
#         experience_to_store = {
#             "experience_id": "exp123", # In real tool, this is generated
#             "primary_goal_description": "Develop a new login system.",
#             "sub_task_description": "Implement OAuth 2.0.",
#             "input_context_summary": "User needs secure login.",
#             "output_summary": "OAuth 2.0 flow implemented.",
#             "status": "success",
#             "agent_version": "1.0"
#         }
#         store_result = await memory_manager.store_agent_experience(experience_data=experience_to_store)
#         logger.info(f"Store result: {store_result}")

#         # 2. Retrieve relevant experiences
#         retrieve_result = await memory_manager.retrieve_relevant_experiences(
#             goal_description="Develop a new login system.",
#             sub_task_description="Implement OAuth 2.0."
#         )
#         logger.info(f"Retrieve result: {retrieve_result}")

#         # 3. Summarize message history
#         messages_to_summarize = [
#             {"sender_id": "user1", "content": "We need a new login page."},
#             {"sender_id": "dev1", "content": "Okay, what features should it have?"},
#             {"sender_id": "user1", "content": "SSO with Google and password reset."},
#         ]
#         summary_result = await memory_manager.summarize_message_history(
#             messages=messages_to_summarize,
#             target_goal="Finalize login page requirements."
#         )
#         logger.info(f"Summary result: {summary_result}")
        
#         # Example of how ReAct 'run' might be invoked if it directly maps to capabilities
#         # This is highly dependent on the actual ReActAgent base class implementation
#         # and how SmartAgentBus interacts with it.
#         # Typically, the prompt would be a more structured request (e.g., JSON)
#         # if 'run' is the entry point for capability calls.
        
#         # react_prompt_store = '{"capability": "store_agent_experience", "args": {"experience_data": {"experience_id": "exp456", "primary_goal_description": "Test feature"}}}'
#         # react_run_result_store = await memory_manager.run(react_prompt_store) # This is a placeholder call
#         # logger.info(f"ReAct run (store) result: {react_run_result_store}")


#     except Exception as e:
#         logger.error(f"Error in example usage: {e}", exc_info=True)

# if __name__ == "__main__":
#     import asyncio
#     # To run this example, you'd need to ensure the placeholder ReActAgent and AgentMeta
#     # are sufficiently functional or replace them with the actual beeai_framework classes.
#     # Also, the DummyLLMService and DummyMongoDBClient would need to be more robust
#     # or replaced with actual instances for real testing.
#     # asyncio.run(main())
#     pass
