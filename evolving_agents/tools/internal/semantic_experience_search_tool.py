import logging
from typing import List, Dict, Optional, Any

from evolving_agents.core.base import BaseTool
from evolving_agents.core.llm_service import LLMService
from evolving_agents.core.mongodb_client import MongoDBClient

# Configure logging
logger = logging.getLogger(__name__)

# Default fields to search against if not specified by the user.
# These are the *embedding* field names.
DEFAULT_EMBEDDING_SEARCH_FIELDS = [
    "primary_goal_description_embedding",
    "sub_task_description_embedding",
    "input_context_summary_embedding",
    "output_summary_embedding",
]

# Assumed name for the MongoDB Atlas Vector Search index.
# This index should be configured on the collection for the fields
# listed in DEFAULT_EMBEDDING_SEARCH_FIELDS or any custom fields used.
DEFAULT_VECTOR_INDEX_NAME = "vector_index_experiences_default"


class SemanticExperienceSearchTool(BaseTool):
    """
    A tool to search for relevant agent experiences in MongoDB using semantic vector search.

    This tool requires a MongoDB Atlas Vector Search index to be configured on the
    target collection. The index should cover the embedding fields that will be searched.
    For example, an index named 'vector_index_experiences_default' might be configured
    on the 'eat_agent_experiences' collection for fields like
    'primary_goal_description_embedding', 'sub_task_description_embedding', etc.
    Each of these indexed fields should be of the 'vector' type in Atlas Search.
    """
    name: str = "SemanticExperienceSearchTool"
    description: str = (
        "Searches for agent experiences in MongoDB based on semantic similarity "
        "to a query string, using vector embeddings."
    )

    def __init__(
        self,
        mongodb_client: MongoDBClient,
        llm_service: LLMService,
        collection_name: str = "eat_agent_experiences",
        default_search_fields: Optional[List[str]] = None,
        vector_index_name: str = DEFAULT_VECTOR_INDEX_NAME,
    ):
        """
        Initializes the SemanticExperienceSearchTool.

        Args:
            mongodb_client: An instance of MongoDBClient for database interaction.
            llm_service: An instance of LLMService for generating query embeddings.
            collection_name: Name of the MongoDB collection. Defaults to "eat_agent_experiences".
            default_search_fields: Default list of embedding field names to search against.
                                   If None, uses a predefined list. These are fields within
                                   the 'embeddings' sub-document (e.g., "primary_goal_description_embedding").
            vector_index_name: The name of the MongoDB Atlas Vector Search index.
        """
        super().__init__()
        self.mongodb_client = mongodb_client
        self.llm_service = llm_service
        self.collection_name = collection_name
        self.collection = self.mongodb_client.get_collection(self.collection_name)
        self.default_search_fields = default_search_fields or list(DEFAULT_EMBEDDING_SEARCH_FIELDS)
        self.vector_index_name = vector_index_name

        if not self.default_search_fields:
            logger.warning(
                "SemanticExperienceSearchTool initialized with no default_search_fields. "
                "Searches will require 'search_fields' to be explicitly provided."
            )

    async def search_relevant_experiences(
        self,
        query_string: str,
        top_k: int = 5,
        search_fields: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Searches for relevant experiences using vector similarity.

        Note: This implementation currently supports searching against a single embedding
        field per call for simplicity. If multiple `search_fields` are provided,
        only the first one will be used. Future enhancements could involve multiple
        $vectorSearch stages or client-side merging for multi-field search.

        Args:
            query_string: The natural language query string to search for.
            top_k: The number of top relevant experiences to return.
            search_fields: A list of embedding field names to search against (e.g.,
                           ["embeddings.primary_goal_description_embedding"]).
                           If None or empty, uses the tool's default_search_fields.
                           Currently, only the first field in the list is used.
            metadata_filter: A MongoDB query document to filter experiences before
                             vector search (e.g., {"tags": "important"}).

        Returns:
            A list of experience documents, each augmented with a 'similarity_score'.
            Returns an empty list if an error occurs or no results are found.
        """
        if not query_string:
            logger.warning("search_relevant_experiences called with empty query_string.")
            return []

        try:
            query_embedding = await self.llm_service.embed(query_string)
            if not query_embedding:
                logger.error("Failed to generate embedding for query_string.")
                return []
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}", exc_info=True)
            return []

        target_search_fields = search_fields or self.default_search_fields
        if not target_search_fields:
            logger.error(
                "No search fields specified or configured for semantic search."
            )
            return []

        # Simplified approach: Use the first specified search field.
        # MongoDB's $vectorSearch path is typically a single field.
        # To search across multiple, one might need to:
        # 1. Have an index that combines text from multiple fields into one vector.
        # 2. Run multiple $vectorSearch queries (one for each field) and merge results.
        #    This adds complexity in ranking and de-duplication.
        # 3. Use a more complex index definition if Atlas Search supports multi-path directly
        #    in a way that's easily usable here (e.g. wildcard paths on an object of embeddings).
        # For now, we pick the first field.
        
        embedding_field_path = target_search_fields[0]
        if len(target_search_fields) > 1:
            logger.warning(
                f"Multiple search_fields provided: {target_search_fields}. "
                f"Currently using only the first field: '{embedding_field_path}'. "
                "Multi-field search in a single query requires specific index setup or multiple queries."
            )
        
        # Ensure the path points to the embedding within the 'embeddings' sub-document.
        if not embedding_field_path.startswith("embeddings."):
            embedding_field_path = f"embeddings.{embedding_field_path}"


        # Construct the $vectorSearch stage
        # numCandidates should be higher than top_k for better accuracy.
        num_candidates = top_k * 10

        vector_search_stage = {
            "$vectorSearch": {
                "index": self.vector_index_name,
                "path": embedding_field_path,
                "queryVector": query_embedding,
                "numCandidates": num_candidates,
                "limit": top_k,
            }
        }
        
        if metadata_filter:
            vector_search_stage["$vectorSearch"]["filter"] = metadata_filter

        # Projection stage to include original document fields and the search score
        project_stage = {
            "$project": {
                "_id": 0,  # Exclude the MongoDB default _id
                "similarity_score": {"$meta": "vectorSearchScore"},
                # Include all other fields from the original document
                # This requires knowing the document structure or using a more dynamic way
                # For now, we assume the document is stored at the root.
                # If '$$ROOT' is not what we want, specific fields need to be listed.
                "document": "$$ROOT"
            }
        }
        
        # Refined projection: Merge existing document fields with similarity_score
        # and remove the 'embeddings' field from the top-level of the returned document
        # as the raw embeddings are usually not needed by the caller.
        add_fields_stage = {
             "$addFields": {
                "similarity_score": {"$meta": "vectorSearchScore"}
            }
        }
        unset_stage = {
            "$unset": ["embeddings"] # Optionally remove embeddings from final result
        }


        pipeline = [vector_search_stage, add_fields_stage, unset_stage]
        
        # If there's a metadata_filter that should apply *before* vector search (e.g. on non-indexed fields for vector search)
        # one might add a $match stage here. However, $vectorSearch's "filter" is generally preferred for efficiency
        # if the filter criteria are on fields indexed by Atlas Search.
        # If metadata_filter is complex and needs to run on the full collection first,
        # it would be: pipeline = [{"$match": metadata_filter}, vector_search_stage, project_stage]
        # But this is less efficient if $vectorSearch filter can be used.

        logger.debug(f"Executing MongoDB aggregation pipeline for semantic search: {pipeline}")

        try:
            cursor = self.collection.aggregate(pipeline)
            results = await cursor.to_list(length=top_k)
            
            # Post-process to move 'document' fields to top level if using "$project": {"document": "$$ROOT"}
            # If using $addFields, the structure is already flat.
            # The current pipeline with $addFields and $unset should produce a good structure.

            logger.info(f"Found {len(results)} relevant experiences for query: '{query_string}'")
            return results
        except Exception as e:
            # Specific error handling for common MongoDB errors could be added here
            # For example, checking for OperationFailure and error codes.
            logger.error(f"Error during MongoDB aggregation for semantic search: {e}", exc_info=True)
            return []

    # Example of how BaseTool's execute might be structured (if needed)
    # async def execute(self, query_string: str, top_k: int = 5, search_fields: Optional[List[str]] = None, metadata_filter: Optional[Dict[str, Any]] = None) -> Any:
    #     return await self.search_relevant_experiences(
    #         query_string=query_string,
    #         top_k=top_k,
    #         search_fields=search_fields,
    #         metadata_filter=metadata_filter
    #     )

# Example Usage (Illustrative - requires async environment and mock/real services)
# async def main():
#     class MockMongoDBClient:
#         def get_collection(self, name):
#             class MockCollection:
#                 async def aggregate(self, pipeline):
#                     print(f"Aggregating with pipeline: {pipeline}")
#                     # Simulate results based on queryVector
#                     # This is highly simplified
#                     if pipeline[0]['$vectorSearch']['queryVector'] == [0.1, 0.2]: # some_embedding
#                         return MockAsyncCursor([
#                             {"similarity_score": 0.95, "primary_goal_description": "Goal related to query", "embeddings": {}},
#                             {"similarity_score": 0.88, "primary_goal_description": "Another goal", "embeddings": {}},
#                         ])
#                     return MockAsyncCursor([])
#             return MockCollection()

#     class MockAsyncCursor:
#         def __init__(self, items):
#             self.items = items
#         async def to_list(self, length):
#             return self.items[:length]

#     class MockLLMService:
#         async def embed(self, text: str) -> List[float]:
#             print(f"Embedding text for search: '{text[:30]}...'")
#             if text == "find tasks about coding":
#                 return [0.1, 0.2] # Example embedding
#             return [0.5, 0.5] # Default fallback

#     mongo_client = MockMongoDBClient()
#     llm_service = MockLLMService()
#     search_tool = SemanticExperienceSearchTool(mongodb_client=mongo_client, llm_service=llm_service)

#     print("\n--- Testing search_relevant_experiences ---")
#     experiences = await search_tool.search_relevant_experiences(
#         query_string="find tasks about coding",
#         top_k=3,
#         # search_fields=["embeddings.primary_goal_description_embedding"], # Full path
#         search_fields=["primary_goal_description_embedding"], # Tool will prefix with "embeddings."
#         metadata_filter={"tags": "python"}
#     )
#     if experiences:
#         for exp in experiences:
#             print(f"Score: {exp['similarity_score']:.2f}, Goal: {exp.get('primary_goal_description', exp.get('document', {}).get('primary_goal_description'))}")
#     else:
#         print("No experiences found.")

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
#     pass
