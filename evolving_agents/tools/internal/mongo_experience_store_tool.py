import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from evolving_agents.core.llm_service import LLMService
from evolving_agents.core.mongodb_client import MongoDBClient


class MongoExperienceStoreTool:
    """
    Manages storage and retrieval of agent experiences in MongoDB,
    including embedding generation for specified text fields.
    """
    name = "mongo_experience_store_tool"
    description = (
        "Manages storage and retrieval of agent experiences in MongoDB, "
        "including embedding generation."
    )

    def __init__(self, mongodb_client: MongoDBClient, llm_service: LLMService):
        """
        Initializes the MongoExperienceStoreTool.

        Args:
            mongodb_client: An instance of MongoDBClient for database interaction.
            llm_service: An instance of LLMService for generating embeddings.
        """
        self.mongodb_client = mongodb_client
        self.llm_service = llm_service
        self.collection = self.mongodb_client.get_collection("eat_agent_experiences")
        self.embed_fields: List[str] = [
            "primary_goal_description",
            "sub_task_description",
            "input_context_summary",
            "output_summary",
            "user_feedback_summary",
            "error_summary"
        ]
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def _generate_embeddings(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Helper to generate embeddings for fields in data."""
        embeddings_to_add = {}
        for field in self.embed_fields:
            if field in data and data[field]:
                try:
                    embedding = await self.llm_service.embed(data[field])
                    embeddings_to_add[f"{field}_embedding"] = embedding
                except Exception as e:
                    self.logger.error(f"Error generating embedding for field '{field}': {e}")
                    # Store empty embedding or handle as per specific requirement
                    embeddings_to_add[f"{field}_embedding"] = []
            elif field in data and not data[field]:
                 # Handle empty text field: store empty embedding
                embeddings_to_add[f"{field}_embedding"] = []
        return embeddings_to_add

    async def store_experience(self, experience_data: Dict[str, Any]) -> str:
        """
        Stores a new agent experience, generating embeddings for relevant fields.

        Args:
            experience_data: A dictionary conforming to the eat_agent_experiences
                             schema (without embedding fields).

        Returns:
            The unique experience_id of the stored document.
        """
        experience_id = uuid.uuid4().hex
        timestamp = datetime.now(timezone.utc)

        doc_to_store = experience_data.copy()
        doc_to_store["experience_id"] = experience_id
        doc_to_store["timestamp"] = timestamp
        
        try:
            embeddings = await self._generate_embeddings(doc_to_store)
            doc_to_store.update(embeddings)

            await self.collection.insert_one(doc_to_store)
            self.logger.info(f"Stored experience with ID: {experience_id}")
            return experience_id
        except Exception as e:
            self.logger.error(f"Error storing experience {experience_id}: {e}")
            # Potentially re-raise or return a specific error indicator
            raise

    async def get_experience(self, experience_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single experience document by its experience_id.

        Args:
            experience_id: The unique ID of the experience to retrieve.

        Returns:
            The experience document as a dictionary, or None if not found.
            The MongoDB internal '_id' field is excluded.
        """
        try:
            document = await self.collection.find_one(
                {"experience_id": experience_id},
                {"_id": 0}  # Exclude the MongoDB internal _id
            )
            if document:
                self.logger.info(f"Retrieved experience with ID: {experience_id}")
            else:
                self.logger.info(f"No experience found with ID: {experience_id}")
            return document
        except Exception as e:
            self.logger.error(f"Error retrieving experience {experience_id}: {e}")
            return None

    async def update_experience(self, experience_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Updates an existing experience document. If text fields requiring
        embeddings are updated, their embeddings are regenerated.

        Args:
            experience_id: The ID of the experience to update.
            update_data: A dictionary containing fields to update.

        Returns:
            True if the update was successful (at least one document matched),
            False otherwise.
        """
        updates_to_apply = update_data.copy()
        updates_to_apply["timestamp"] = datetime.now(timezone.utc)

        try:
            # Check if any of the updated fields require re-embedding
            fields_to_re_embed = {k: v for k, v in update_data.items() if k in self.embed_fields}
            if fields_to_re_embed:
                new_embeddings = await self._generate_embeddings(fields_to_re_embed)
                updates_to_apply.update(new_embeddings)

            result = await self.collection.update_one(
                {"experience_id": experience_id},
                {"$set": updates_to_apply}
            )
            if result.modified_count > 0:
                self.logger.info(f"Successfully updated experience with ID: {experience_id}")
                return True
            elif result.matched_count > 0:
                self.logger.info(f"No changes applied, but experience with ID: {experience_id} was matched.")
                return True # Or False, depending on desired behavior for no-op updates
            else:
                self.logger.warning(f"No experience found with ID {experience_id} to update.")
                return False
        except Exception as e:
            self.logger.error(f"Error updating experience {experience_id}: {e}")
            return False

    async def delete_experience(self, experience_id: str) -> bool:
        """
        Deletes an experience document by its experience_id.

        Args:
            experience_id: The ID of the experience to delete.

        Returns:
            True if a document was deleted, False otherwise.
        """
        try:
            result = await self.collection.delete_one({"experience_id": experience_id})
            if result.deleted_count > 0:
                self.logger.info(f"Successfully deleted experience with ID: {experience_id}")
                return True
            else:
                self.logger.warning(f"No experience found with ID {experience_id} to delete.")
                return False
        except Exception as e:
            self.logger.error(f"Error deleting experience {experience_id}: {e}")
            return False

    async def find_similar_experiences(
        self,
        text_to_match: str,
        text_field_to_search: str,
        limit: int = 5,
        additional_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Finds experiences with text fields semantically similar to the given text,
        using vector search on pre-computed embeddings.

        Args:
            text_to_match: The text to find similar experiences for.
            text_field_to_search: The original text field name whose corresponding
                                  embedding field should be searched (e.g., "sub_task_description").
            limit: The maximum number of similar experiences to return.
            additional_filters: Optional MongoDB query filters to apply before
                                  the vector search stage.

        Returns:
            A list of matching experience documents, sorted by similarity.
            Excludes MongoDB internal '_id' and embedding fields from the results by default.
        """
        if not text_to_match or not text_field_to_search:
            self.logger.warning("Text to match or field to search cannot be empty.")
            return []
        if text_field_to_search not in self.embed_fields:
            self.logger.error(f"Field '{text_field_to_search}' is not configured for embedding search.")
            return []

        embedding_field_to_search = f"{text_field_to_search}_embedding"

        try:
            query_embedding = await self.llm_service.embed(text_to_match)
            if not query_embedding:
                self.logger.error("Could not generate embedding for the query text.")
                return []

            # Construct the aggregation pipeline for vector search
            # This assumes a MongoDB Atlas Search index is configured on the embedding field
            # The index should be a vector index, e.g., "vector_index"
            # The path is the field name, e.g. "sub_task_description_embedding"
            # numCandidates should be higher than limit for ANN search
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index", # Replace with your Atlas Search vector index name
                        "path": embedding_field_to_search,
                        "queryVector": query_embedding,
                        "numCandidates": limit * 10, # Number of candidates to consider
                        "limit": limit,              # Number of results to return
                    }
                },
                { # Optional: Add a match stage for additional filters if provided
                  # This stage is better placed *before* $vectorSearch if it significantly reduces documents
                  # However, $vectorSearch must be the first stage if used.
                  # Consider pre-filtering logic if 'additional_filters' are common.
                },
                {
                    "$project": {
                        "_id": 0,  # Exclude MongoDB internal ID
                        "score": {"$meta": "vectorSearchScore"}, # Include the similarity score
                        # Add other fields you want to return
                        "experience_id": 1,
                        "timestamp": 1,
                        "primary_goal_description": 1,
                        "sub_task_description": 1,
                        "input_context_summary": 1,
                        "output_summary": 1,
                        "status": 1,
                        "user_feedback_summary": 1,
                        "error_summary": 1,
                        "agent_version": 1,
                        "tool_versions": 1,
                        "session_id": 1,
                        "run_id": 1,
                    }
                }
            ]
            
            # Add $match stage if additional_filters are provided
            # Note: For $vectorSearch, any $match stage not part of $vectorSearch's `filter` 
            # parameter usually comes *after* $vectorSearch or it has to be a pre-filtering
            # that is compatible with how Atlas Search executes.
            # If additional_filters are on non-indexed fields or complex, they run post-search.
            if additional_filters:
                pipeline.insert(1, {"$match": additional_filters})


            self.logger.info(f"Executing vector search on '{embedding_field_to_search}' with limit {limit}.")
            results = await self.collection.aggregate(pipeline).to_list(length=limit)
            
            self.logger.info(f"Found {len(results)} similar experiences for text in '{text_field_to_search}'.")
            return results

        except Exception as e:
            self.logger.error(f"Error finding similar experiences for field '{text_field_to_search}': {e}")
            return []
