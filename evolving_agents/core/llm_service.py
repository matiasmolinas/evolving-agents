# evolving_agents/core/llm_service.py

import logging
import os
import json
import hashlib
import time
import asyncio # Added for async operations
from datetime import datetime, timezone # Added for BSON Date compatibility
from typing import Dict, Any, List, Optional, Union

import pymongo # For pymongo constants like ASCENDING

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import UserMessage
from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend.constants import ProviderName

from evolving_agents.core.mongodb_client import MongoDBClient # Assuming this exists

logger = logging.getLogger(__name__)

class LLMCache:
    """
    Cache for LLM completions and embeddings using MongoDB with TTL.
    """
    def __init__(
        self,
        mongodb_client: MongoDBClient, # Changed: Expect MongoDBClient instance
        ttl: int = 86400 * 7, # 7 days default TTL
        collection_name: str = "eat_llm_cache"
    ):
        """
        Initialize the LLM cache with MongoDB.

        Args:
            mongodb_client: Instance of MongoDBClient for database operations.
            ttl: Time-to-live for cache entries in seconds.
            collection_name: Name of the MongoDB collection for caching.
        """
        self.mongodb_client = mongodb_client
        self.ttl = ttl
        self.cache_collection_name = collection_name
        self.cache_collection = self.mongodb_client.get_collection(self.cache_collection_name)

        # Ensure TTL index is created (non-blocking)
        asyncio.create_task(self._ensure_indexes())

        logger.info(f"Initialized LLM cache with MongoDB collection '{self.cache_collection_name}' and TTL {self.ttl}s.")

    async def _ensure_indexes(self):
        """Ensure necessary indexes on the cache collection, especially TTL."""
        try:
            # TTL index on 'created_at' field
            # Note: MongoDB automatically converts datetime to UTC if no timezone info is present upon insertion.
            # It's good practice to store them as timezone-aware UTC datetimes.
            await self.cache_collection.create_index(
                [("created_at", pymongo.ASCENDING)],
                expireAfterSeconds=self.ttl,
                name="ttl_created_at_index" # Optional: give index a name
            )
            # Index for the _id field (which will be our hash key)
            # MongoDB automatically creates an index on _id, but if we use custom _id as hash, ensure it's efficient.
            # If self._generate_key() is used as _id, this default index is fine.
            # If we create a separate 'key' field, we might index that:
            # await self.cache_collection.create_index([("key", pymongo.ASCENDING)], unique=True, background=True)
            logger.info(f"Ensured TTL index on '{self.cache_collection_name}' for 'created_at' field.")
        except Exception as e:
            logger.error(f"Error creating TTL index for {self.cache_collection_name}: {e}", exc_info=True)


    def _generate_key(self, data: Any, model_id: str, cache_type: str) -> str:
        """Generate a unique key for the given input data, model, and type."""
        # Convert UserMessage to a consistent serializable format
        if isinstance(data, list) and all(isinstance(item, UserMessage) for item in data):
            serializable_data = []
            for msg in data:
                content = msg.content.text if hasattr(msg.content, 'text') else str(msg.content)
                serializable_data.append({"content": content, "role": msg.role})
        else:
            serializable_data = data # Assuming data is already suitable (e.g., string for embeddings)

        # Include model_id and cache_type in the key to ensure uniqueness across models/types
        key_structure = {
            "data": serializable_data,
            "model_id": model_id,
            "cache_type": cache_type
        }
        data_str = json.dumps(key_structure, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()

    async def get_completion(self, messages: List[UserMessage], model_id: str) -> Optional[str]:
        """Get a cached completion result from MongoDB."""
        key = self._generate_key(messages, model_id, "completion")
        try:
            # Using motor syntax for async find_one
            cached_doc = await self.cache_collection.find_one({"_id": key})
            if cached_doc:
                # MongoDB TTL index handles expiration automatically.
                # No need for manual timestamp check here if TTL index is working.
                logger.info(f"Cache hit for completion (key: {key[:8]}...).")
                return cached_doc.get("cached_data") # 'cached_data' stores the response string
        except Exception as e:
            logger.warning(f"Error reading completion cache from MongoDB (key: {key}): {e}", exc_info=True)
        return None

    async def save_completion(self, messages: List[UserMessage], model_id: str, response: str) -> None:
        """Save a completion result to MongoDB cache."""
        key = self._generate_key(messages, model_id, "completion")
        cache_entry = {
            "_id": key, # Use the generated key as MongoDB's _id
            "model_id": model_id,
            "cache_type": "completion",
            "cached_data": response,
            "created_at": datetime.now(timezone.utc) # Store as BSON Date for TTL
        }
        try:
            # Using motor syntax for async replace_one (upsert)
            await self.cache_collection.replace_one({"_id": key}, cache_entry, upsert=True)
            logger.info(f"Cached completion (key: {key[:8]}...).")
        except Exception as e:
            logger.warning(f"Error writing completion cache to MongoDB (key: {key}): {e}", exc_info=True)

    async def get_embedding(self, text: str, model_id: str) -> Optional[List[float]]:
        """Get a cached embedding from MongoDB."""
        key = self._generate_key(text, model_id, "embedding")
        try:
            cached_doc = await self.cache_collection.find_one({"_id": key})
            if cached_doc:
                logger.info(f"Cache hit for embedding (key: {key[:8]}...).")
                return cached_doc.get("cached_data") # 'cached_data' stores the embedding list
        except Exception as e:
            logger.warning(f"Error reading embedding cache from MongoDB (key: {key}): {e}", exc_info=True)
        return None

    async def save_embedding(self, text: str, model_id: str, embedding: List[float]) -> None:
        """Save an embedding to MongoDB cache."""
        key = self._generate_key(text, model_id, "embedding")
        cache_entry = {
            "_id": key,
            "model_id": model_id,
            "cache_type": "embedding",
            "cached_data": embedding, # Store the list of floats directly
            "created_at": datetime.now(timezone.utc)
        }
        try:
            await self.cache_collection.replace_one({"_id": key}, cache_entry, upsert=True)
            logger.info(f"Cached embedding (key: {key[:8]}...).")
        except Exception as e:
            logger.warning(f"Error writing embedding cache to MongoDB (key: {key}): {e}", exc_info=True)

    async def get_batch_embeddings(self, texts: List[str], model_id: str) -> Optional[List[List[float]]]:
        """Get cached batch embeddings if all are available from MongoDB."""
        # MongoDB doesn't have a direct "get all if present" for multiple keys efficiently like some caches.
        # We can fetch them individually or use $in operator. Fetching individually is simpler for now.
        results = []
        all_found = True
        for text in texts:
            embedding = await self.get_embedding(text, model_id)
            if embedding is None:
                all_found = False
                break
            results.append(embedding)
        return results if all_found else None

    async def clear_cache(self, older_than_seconds: Optional[int] = None) -> int:
        """
        Clear the MongoDB cache.
        If older_than_seconds is specified, removes entries older than that.
        Otherwise, removes all entries. Relies on TTL index for time-based clearing primarily.
        This method provides an explicit way to clear.

        Args:
            older_than_seconds: Clear entries older than this many seconds. If None, clear all.

        Returns:
            Number of entries removed.
        """
        query_filter = {}
        if older_than_seconds is not None:
            cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=older_than_seconds)
            query_filter = {"created_at": {"$lt": cutoff_time}}

        try:
            # Using motor syntax for async delete_many
            result = await self.cache_collection.delete_many(query_filter)
            deleted_count = result.deleted_count
            logger.info(f"Cleared {deleted_count} cache entries from MongoDB based on filter: {query_filter}.")
            return deleted_count
        except Exception as e:
            logger.error(f"Error clearing cache from MongoDB: {e}", exc_info=True)
            return 0

# ... (OpenAIChatModel, EmbeddingModel, OpenAIEmbeddingModel, OllamaEmbeddingModel remain unchanged) ...
class OpenAIChatModel(LiteLLMChatModel): # Unchanged
    @property
    def provider_id(self) -> ProviderName:
        return "openai"
    def __init__(self, model_id: str | None = None, settings: dict | None = None) -> None:
        _settings = settings.copy() if settings is not None else {}
        super().__init__(
            model_id if model_id else os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            provider_id="openai",
            settings=_settings,
        )

class EmbeddingModel: # Unchanged
    def __init__(self, model_id: str, provider_id: str, settings: Optional[Dict[str, Any]] = None):
        self.model_id = model_id
        self.provider_id = provider_id
        self.settings = settings or {}
    async def create(self, input_text: str) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement this method")
    async def create_batch(self, input_texts: List[str]) -> List[Dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement this method")

class OpenAIEmbeddingModel(EmbeddingModel): # Unchanged
    def __init__(self, model_id: str | None = None, settings: dict | None = None):
        _settings = settings.copy() if settings else {}
        _model_id = model_id if model_id else os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        super().__init__(model_id=_model_id, provider_id="openai", settings=_settings)
        import openai
        self.client = openai.OpenAI(
            api_key=_settings.get("api_key") or os.getenv("OPENAI_API_KEY"),
            base_url=_settings.get("base_url") or os.getenv("OPENAI_API_ENDPOINT"),
        )
    async def create(self, input_text: str) -> Dict[str, Any]:
        try:
            response = self.client.embeddings.create(model=self.model_id, input=input_text, encoding_format="float")
            return {"data": response.data[0].embedding, "model": response.model, "usage": response.usage.model_dump()} # Use model_dump for Pydantic v2
        except Exception as e: logger.error(f"Error creating OpenAI embedding: {e}"); return {"data": [0.0] * 1536, "error": str(e)} # Ensure consistent dim
    async def create_batch(self, input_texts: List[str]) -> List[Dict[str, Any]]:
        try:
            response = self.client.embeddings.create(model=self.model_id, input=input_texts, encoding_format="float")
            return [{"data": item.embedding, "model": response.model, "usage": response.usage.model_dump()} for item in response.data]
        except Exception as e: logger.error(f"Error creating OpenAI batch embeddings: {e}"); return [{"data": [0.0] * 1536, "error": str(e)} for _ in input_texts]

class OllamaEmbeddingModel(EmbeddingModel): # Unchanged
    def __init__(self, model_id: str | None = None, settings: dict | None = None):
        _settings = settings.copy() if settings else {}
        _model_id = model_id if model_id else "nomic-embed-text"
        super().__init__(model_id=_model_id, provider_id="ollama", settings=_settings)
        import httpx
        self.client = httpx.AsyncClient(base_url=_settings.get("base_url") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    async def create(self, input_text: str) -> Dict[str, Any]:
        try:
            response = await self.client.post("/api/embeddings", json={"model": self.model_id, "prompt": input_text})
            response.raise_for_status(); data = response.json()
            return {"data": data["embedding"], "model": self.model_id, "usage": {"total_tokens": data.get("total_tokens", 0)}}
        except Exception as e: logger.error(f"Error creating Ollama embedding: {e}"); return {"data": [0.0] * 768, "error": str(e)} # nomic-embed-text is often 768
    async def create_batch(self, input_texts: List[str]) -> List[Dict[str, Any]]:
        results = []; [results.append(await self.create(text)) for text in input_texts]; return results


class LLMService:
    """
    LLM service that interfaces with chat and embedding models.
    Now uses MongoDB-backed LLMCache if caching is enabled.
    """
    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None, # Made Optional for flexibility
        embedding_model: Optional[str] = None, # Made Optional
        use_cache: bool = True,
        # Removed cache_dir, cache config now via MongoDBClient and LLMCache constructor
        mongodb_client: Optional[MongoDBClient] = None, # Added for explicit passing
        container: Optional[Any] = None # For resolving MongoDBClient if not passed
    ):
        self.provider = provider
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") # OPENAI_API_KEY is common
        self.container = container

        # Resolve MongoDBClient
        if mongodb_client:
            self.mongodb_client = mongodb_client
        elif self.container and self.container.has('mongodb_client'):
            self.mongodb_client = self.container.get('mongodb_client')
        else:
            # Attempt to create a default MongoDBClient. This assumes MONGODB_URI is in .env
            try:
                self.mongodb_client = MongoDBClient()
                if self.container and not self.container.has('mongodb_client'):
                    self.container.register('mongodb_client', self.mongodb_client)
            except ValueError as e: # MongoDB URI missing
                 logger.warning(f"MongoDBClient could not be initialized for LLMService cache: {e}. Cache will be disabled.")
                 self.mongodb_client = None # Explicitly set to None
                 use_cache = False # Disable cache if DB client fails

        # Initialize cache if enabled AND mongodb_client is available
        self.use_cache = use_cache and self.mongodb_client is not None
        if self.use_cache and self.mongodb_client:
            self.cache = LLMCache(mongodb_client=self.mongodb_client)
        else:
            self.cache = None
            if use_cache and not self.mongodb_client: # Log if cache was desired but disabled
                logger.warning("LLM Cache was requested but is disabled due to missing MongoDB client.")


        # Save API key to environment if provided (for OpenAI)
        if self.api_key and provider == "openai":
            os.environ["OPENAI_API_KEY"] = self.api_key

        # Initialize models based on provider
        # Default model IDs are now more robustly handled within specific model classes
        if provider == "openai":
            self.chat_model = OpenAIChatModel(
                model_id=model, # Pass None if not specified, class will use default
                settings={"api_key": self.api_key} if self.api_key else None
            )
            self.embedding_model = OpenAIEmbeddingModel(
                model_id=embedding_model,
                settings={"api_key": self.api_key} if self.api_key else None
            )
        elif provider == "ollama":
            self.chat_model = LiteLLMChatModel(
                model_id=model or "llama3", # Ollama typically needs explicit model
                provider_id="ollama",
                settings={}
            )
            self.embedding_model = OllamaEmbeddingModel(
                model_id=embedding_model, # Pass None, class uses default
                settings={}
            )
        else: # Default to OpenAI if provider is unknown or not set
            logger.warning(f"Unknown LLM provider '{provider}'. Defaulting to OpenAI.")
            self.provider = "openai" # Correct the provider if defaulting
            self.chat_model = OpenAIChatModel(
                model_id=model,
                settings={"api_key": self.api_key} if self.api_key else None
            )
            self.embedding_model = OpenAIEmbeddingModel(
                model_id=embedding_model,
                settings={"api_key": self.api_key} if self.api_key else None
            )

        cache_status = "enabled (MongoDB)" if self.cache else "disabled"
        logger.info(f"Initialized LLM service with provider: {self.provider}, "
                    f"chat model: {self.chat_model.model_id if self.chat_model else 'N/A'}, "
                    f"embedding model: {self.embedding_model.model_id if self.embedding_model else 'N/A'}, "
                    f"cache: {cache_status}")

    async def generate(self, prompt: str) -> str:
        """Generate text based on a prompt."""
        if not self.chat_model:
            logger.error("Chat model not initialized. Cannot generate text.")
            return "Error: Chat model not available."

        logger.debug(f"Generating response for prompt: {prompt[:50]}...")
        messages = [UserMessage(prompt)]

        if self.cache: # Check if cache object exists
            cached_response = await self.cache.get_completion(messages, self.chat_model.model_id)
            if cached_response is not None:
                return cached_response
        try:
            response = await self.chat_model.create(messages=messages)
            response_text = response.get_text_content()
            if self.cache:
                await self.cache.save_completion(messages, self.chat_model.model_id, response_text)
            return response_text
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return f"Error generating response: {str(e)}"

    async def embed(self, text: str) -> List[float]:
        """Generate embeddings for text."""
        if not self.embedding_model:
            logger.error("Embedding model not initialized. Cannot generate embeddings.")
            # Return a list of floats of a default dimension
            return [0.0] * 1536 # Assuming a common default like OpenAI's

        logger.debug(f"Generating embedding for text: {text[:50]}...")
        if self.cache:
            cached_embedding = await self.cache.get_embedding(text, self.embedding_model.model_id)
            if cached_embedding is not None:
                return cached_embedding
        try:
            response = await self.embedding_model.create(text)
            embedding = response.get("data", []) # Default to empty list if 'data' is missing
            if not embedding and "error" not in response: # Log if data is empty but no error reported
                 logger.warning(f"Embedding model returned no data for text: {text[:50]}")
            if self.cache and embedding: # Only cache if embedding was successful
                await self.cache.save_embedding(text, self.embedding_model.model_id, embedding)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}", exc_info=True)
            # Fallback vector dimension should match your primary embedding model
            # e.g. 1536 for OpenAI, 768 for nomic-embed-text (Ollama)
            dim = 1536 if self.provider == "openai" else 768
            return [0.0] * dim


    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not self.embedding_model:
            logger.error("Embedding model not initialized. Cannot generate batch embeddings.")
            dim = 1536 if self.provider == "openai" else 768
            return [[0.0] * dim for _ in texts]

        logger.debug(f"Generating batch embeddings for {len(texts)} texts...")
        if self.cache:
            cached_embeddings = await self.cache.get_batch_embeddings(texts, self.embedding_model.model_id)
            if cached_embeddings is not None:
                return cached_embeddings
        try:
            responses = await self.embedding_model.create_batch(texts)
            embeddings = []
            for i, response in enumerate(responses):
                current_embedding = response.get("data", [])
                if not current_embedding and "error" not in response:
                     logger.warning(f"Embedding model returned no data for batch text item {i}: {texts[i][:50]}")
                embeddings.append(current_embedding)
                if self.cache and current_embedding: # Only cache successful ones
                    await self.cache.save_embedding(texts[i], self.embedding_model.model_id, current_embedding)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}", exc_info=True)
            dim = 1536 if self.provider == "openai" else 768
            return [[0.0] * dim for _ in range(len(texts))]

    async def clear_cache(self, older_than_seconds: Optional[int] = None) -> int:
        """Clear the LLM cache. Delegates to LLMCache instance."""
        if not self.cache:
            logger.info("Cache is disabled, nothing to clear.")
            return 0
        # The method in LLMCache is already async
        return await self.cache.clear_cache(older_than_seconds)

    async def generate_applicability_text(self, text_chunk: str, component_type: str = "", component_name: str = "") -> str:
        """Generate applicability text (T_raz). Logic remains the same."""
        # ... (prompt and call to self.generate as before) ...
        prompt = f"""
        Analyze the following text chunk from a component ({component_type or "unknown"}: {component_name or "unknown"}):
        '''
        {text_chunk}
        '''
        Generate a concise description (T_raz) focusing ONLY on its potential applicability and relevance for different tasks. Describe:
        1. **Relevant Tasks:** What specific developer/agent tasks might this chunk be useful for (e.g., 'code generation', 'API documentation', 'testing', 'requirements analysis', 'debugging', 'cost estimation', 'security review')?
        2. **Key Concepts/Implications:** What are the non-obvious technical or functional implications derived from this text? (e.g., 'dependency on X', 'requires async handling', 'critical for user authentication flow').
        3. **Target Audience/Context:** Who would find this most useful or in what situation? (e.g., 'backend developer implementing feature Y', 'project manager estimating effort', 'security auditor reviewing access control').

        Be concise and focus on applicability *beyond* just restating the content. Output ONLY the generated description (T_raz).
        """
        applicability_text = await self.generate(prompt)
        return applicability_text.strip()