import uuid
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

from evolving_agents.core.base import BaseTool # Assuming BaseTool is not async by default
from evolving_agents.core.llm_service import LLMService
from evolving_agents.core.mongodb_client import MongoDBClient

# Configure logging
logger = logging.getLogger(__name__)

class MongoExperienceStoreTool(BaseTool):
    """
    A tool to store, retrieve, update, and delete agent experiences in a MongoDB collection.
    It also handles generating and updating embeddings for specified textual fields.
    """
    name: str = "MongoExperienceStoreTool"
    description: str = (
        "Manages agent experiences in MongoDB, including CRUD operations and "
        "generation of text embeddings for semantic search."
    )

    # Fields that are designated for embedding generation
    EMBEDDABLE_FIELDS = [
        "primary_goal_description",
        "sub_task_description",
        "input_context_summary",
        "output_summary",
    ]

    def __init__(
        self,
        mongodb_client: MongoDBClient,
        llm_service: LLMService,
        collection_name: str = "eat_agent_experiences",
    ):
        """
        Initializes the MongoExperienceStoreTool.

        Args:
            mongodb_client: An instance of MongoDBClient for database interaction.
            llm_service: An instance of LLMService for generating embeddings.
            collection_name: The name of the MongoDB collection to use.
                             Defaults to "eat_agent_experiences".
        """
        super().__init__() # Initialize BaseTool if it has an __init__
        self.mongodb_client = mongodb_client
        self.llm_service = llm_service
        self.collection_name = collection_name
        # Assuming MongoDBClient provides a method to get a collection,
        # and that it returns a MotorCollection or similar async-compatible collection object.
        self.collection = self.mongodb_client.get_collection(self.collection_name)

    async def _generate_embeddings(self, experience_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Generates embeddings for specified fields in the experience data.

        Args:
            experience_data: The dictionary containing experience details.

        Returns:
            A dictionary where keys are embedding field names (e.g., "field_embedding")
            and values are the embedding vectors.
        """
        embeddings: Dict[str, List[float]] = {}
        for field_name in self.EMBEDDABLE_FIELDS:
            if field_name in experience_data and experience_data[field_name]:
                text_to_embed = experience_data[field_name]
                try:
                    # Assuming llm_service.embed() is an async method
                    embedding_vector = await self.llm_service.embed(text_to_embed)
                    if embedding_vector:
                        embeddings[f"{field_name}_embedding"] = embedding_vector
                    else:
                        logger.warning(f"Received empty embedding for field: {field_name}")
                except Exception as e:
                    logger.error(f"Error generating embedding for field {field_name}: {e}", exc_info=True)
        return embeddings

    async def store_experience(self, experience_data: Dict[str, Any]) -> str:
        """
        Stores a new agent experience document in MongoDB.

        Args:
            experience_data: A dictionary containing the experience details.

        Returns:
            The unique ID of the stored experience.
        """
        if not isinstance(experience_data, dict):
            raise ValueError("experience_data must be a dictionary.")

        if "experience_id" not in experience_data or not experience_data["experience_id"]:
            experience_data["experience_id"] = uuid.uuid4().hex
        
        if "timestamp" not in experience_data or not experience_data["timestamp"]:
            experience_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        if "version" not in experience_data:
            experience_data["version"] = 1 # Default version

        # Generate and add embeddings
        generated_embeddings = await self._generate_embeddings(experience_data)
        if "embeddings" not in experience_data:
            experience_data["embeddings"] = {}
        experience_data["embeddings"].update(generated_embeddings)

        try:
            await self.collection.insert_one(experience_data)
            logger.info(f"Successfully stored experience with ID: {experience_data['experience_id']}")
            return experience_data["experience_id"]
        except Exception as e:
            logger.error(f"Error storing experience with ID {experience_data.get('experience_id')}: {e}", exc_info=True)
            # Depending on desired behavior, re-raise or return an indicator of failure
            raise

    async def get_experience(self, experience_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves an experience document from MongoDB by its experience_id.

        Args:
            experience_id: The unique ID of the experience to retrieve.

        Returns:
            The experience document (dictionary) if found, otherwise None.
        """
        if not experience_id:
            logger.warning("get_experience called with empty experience_id.")
            return None
        try:
            document = await self.collection.find_one({"experience_id": experience_id})
            if document:
                logger.debug(f"Successfully retrieved experience with ID: {experience_id}")
            else:
                logger.info(f"No experience found with ID: {experience_id}")
            return document
        except Exception as e:
            logger.error(f"Error retrieving experience with ID {experience_id}: {e}", exc_info=True)
            return None # Or re-raise

    async def update_experience(self, experience_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Updates an existing experience document in MongoDB.

        Args:
            experience_id: The unique ID of the experience to update.
            update_data: A dictionary containing the fields to update.

        Returns:
            True if the update was successful, False otherwise.
        """
        if not experience_id:
            logger.warning("update_experience called with empty experience_id.")
            return False
        if not isinstance(update_data, dict) or not update_data:
            logger.warning("update_experience called with empty or invalid update_data.")
            return False

        # Check if any embeddable fields are being updated
        needs_embedding_update = any(field in update_data for field in self.EMBEDDABLE_FIELDS)
        
        update_payload = {"$set": update_data}

        if needs_embedding_update:
            # To regenerate embeddings, we might need the full context of embeddable fields.
            # If only partial data for embeddable fields is in update_data,
            # we should fetch the existing document first to ensure embeddings are correct.
            # For simplicity here, we assume update_data contains the full new text for any embeddable field being changed.
            # A more robust implementation might merge update_data with existing data before embedding.
            
            # Create a temporary structure for embedding generation that includes the updated fields
            # and potentially existing fields if not all are being updated.
            # This is a simplification; a real scenario might need to fetch the doc.
            potential_full_data_for_embedding = update_data.copy() # Simplified: assumes update_data has all necessary text
            
            new_embeddings = await self._generate_embeddings(potential_full_data_for_embedding)
            if new_embeddings:
                # MongoDB syntax to update nested fields in an object
                for key, value in new_embeddings.items():
                    update_payload["$set"][f"embeddings.{key}"] = value
        
        # Ensure timestamp of update is recorded, if desired (e.g. a "last_modified_at" field)
        # update_payload["$set"]["last_modified_at"] = datetime.now(timezone.utc).isoformat()


        try:
            result = await self.collection.update_one(
                {"experience_id": experience_id},
                update_payload
            )
            if result.matched_count > 0:
                logger.info(f"Successfully updated experience with ID: {experience_id}. Modified count: {result.modified_count}")
                return True
            else:
                logger.warning(f"No experience found with ID: {experience_id} to update.")
                return False
        except Exception as e:
            logger.error(f"Error updating experience with ID {experience_id}: {e}", exc_info=True)
            return False

    async def delete_experience(self, experience_id: str) -> bool:
        """
        Deletes an experience document from MongoDB by its experience_id.

        Args:
            experience_id: The unique ID of the experience to delete.

        Returns:
            True if deletion was successful, False otherwise.
        """
        if not experience_id:
            logger.warning("delete_experience called with empty experience_id.")
            return False
        try:
            result = await self.collection.delete_one({"experience_id": experience_id})
            if result.deleted_count > 0:
                logger.info(f"Successfully deleted experience with ID: {experience_id}")
                return True
            else:
                logger.warning(f"No experience found with ID: {experience_id} to delete.")
                return False
        except Exception as e:
            logger.error(f"Error deleting experience with ID {experience_id}: {e}", exc_info=True)
            return False

    # Example of how a tool might be invoked (if BaseTool defines an execute method)
    # async def execute(self, action: str, params: Dict[str, Any]) -> Any:
    #     if action == "store":
    #         return await self.store_experience(params.get("experience_data"))
    #     elif action == "get":
    #         return await self.get_experience(params.get("experience_id"))
    #     elif action == "update":
    #         return await self.update_experience(params.get("experience_id"), params.get("update_data"))
    #     elif action == "delete":
    #         return await self.delete_experience(params.get("experience_id"))
    #     else:
    #         logger.error(f"Unsupported action: {action}")
    #         raise ValueError(f"Unsupported action: {action}")

# Example Usage (Illustrative - requires async environment and mock/real services)
# async def main():
#     # Mock MongoDBClient and LLMService
#     class MockMongoDBClient:
#         async def get_collection(self, name):
#             print(f"Getting collection: {name}")
#             # Return a mock collection object that supports async methods
#             class MockCollection:
#                 async def insert_one(self, data): print(f"Inserting: {data}"); return type('obj', (object,), {'inserted_id': data['experience_id']})
#                 async def find_one(self, query): print(f"Finding: {query}"); return {"experience_id": query["experience_id"], "test": "data"} if query["experience_id"] == "123" else None
#                 async def update_one(self, query, update): print(f"Updating: {query} with {update}"); return type('obj', (object,), {'matched_count': 1, 'modified_count': 1})
#                 async def delete_one(self, query): print(f"Deleting: {query}"); return type('obj', (object,), {'deleted_count': 1})
#             return MockCollection()

#     class MockLLMService:
#         async def embed(self, text: str) -> List[float]:
#             print(f"Embedding text: '{text[:30]}...'")
#             return [len(text) * 0.01, len(text.split()) * 0.1] # Dummy embedding

#     # Instantiate the tool
#     mongo_client = MockMongoDBClient()
#     llm_service = MockLLMService()
#     experience_tool = MongoExperienceStoreTool(mongodb_client=mongo_client, llm_service=llm_service)

#     # Test store_experience
#     print("\n--- Testing store_experience ---")
#     experience_id = await experience_tool.store_experience({
#         "primary_goal_description": "Test goal",
#         "sub_task_description": "Test sub-task",
#         "input_context_summary": "Some input context.",
#         "final_outcome": "success"
#     })
#     print(f"Stored experience with ID: {experience_id}")

#     # Test get_experience
#     print("\n--- Testing get_experience ---")
#     retrieved_exp = await experience_tool.get_experience(experience_id)
#     print(f"Retrieved experience: {retrieved_exp}")
    
#     retrieved_exp_fake = await experience_tool.get_experience("fake_id")
#     print(f"Retrieved fake experience: {retrieved_exp_fake}")


#     # Test update_experience
#     print("\n--- Testing update_experience ---")
#     update_success = await experience_tool.update_experience(
#         experience_id,
#         {"final_outcome": "failure", "output_summary": "It failed unexpectedly."}
#     )
#     print(f"Update successful: {update_success}")
#     updated_exp = await experience_tool.get_experience(experience_id)
#     print(f"Experience after update: {updated_exp}")

#     # Test delete_experience
#     print("\n--- Testing delete_experience ---")
#     delete_success = await experience_tool.delete_experience(experience_id)
#     print(f"Deletion successful: {delete_success}")
#     deleted_exp = await experience_tool.get_experience(experience_id)
#     print(f"Experience after deletion: {deleted_exp}")

# if __name__ == "__main__":
#     # To run the example main, you'd need an async event loop
#     # import asyncio
#     # asyncio.run(main())
#     pass
