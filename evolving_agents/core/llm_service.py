# evolving_agents/core/llm_service.py

import logging
import os
import json
import hashlib
import time
import asyncio 
from datetime import datetime, timezone, timedelta  # Added timedelta for clear_cache
from typing import Dict, Any, List, Optional, Union

import pymongo

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.embedding import EmbeddingModelOutput # Added import
from beeai_framework.backend.message import UserMessage
from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend.constants import ProviderName

from evolving_agents.core.mongodb_client import MongoDBClient

logger = logging.getLogger(__name__)

# Utility function for handling non-serializable types in JSON operations
# This might be removable if LLMCache is fully replaced by beeai-framework's native cache
# which should handle its own serialization for keys.
# For now, keeping it as LLMCache key generation might still be referenced indirectly.
def json_serializable(obj):
    """Convert an object to a JSON-serializable format."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, set):
        return list(obj)
    try:
        # Try using the default JSON encoder
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        # For objects with their own serialization method
        if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
            return obj.to_dict()
        # Last resort: convert to string
        return str(obj)

# LLMCache class is being removed as per migration plan.
# Native caching will be passed directly to models.

# Custom EmbeddingModel, OpenAIEmbeddingModel, and OllamaEmbeddingModel classes will be removed.
# Their functionality will be replaced by beeai_framework.backend.EmbeddingModel.from_name()

class LLMService:
    """
    LLM service that interfaces with chat and embedding models.
    Now uses MongoDB-backed LLMCache if caching is enabled.
    """
    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        use_cache: bool = True,
        mongodb_client: Optional[MongoDBClient] = None,
        container: Optional[Any] = None
    ):
        self.provider = provider
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.container = container
        self.chat_model = None
        self.embedding_model = None
        self.cache = None # This will hold the native beeai-framework cache instance
        # self.mongodb_client = None # No longer needed directly for LLMCache here

        # Resolve MongoDBClient (still needed if other parts of LLMService use it, but not for cache init)
        # For now, let's assume it's not strictly needed for this refactoring step if only cache is considered.
        # If other parts of LLMService (not generate/embed) use mongodb_client directly, this logic might need to stay.
        # Based on the current scope, we are focusing on model and cache.
        # The original code used mongodb_client to enable/disable custom LLMCache.
        # With native cache, this check might change or be removed.

        from beeai_framework.cache import UnconstrainedCache # Import native cache

        # Initialize native cache if use_cache is true.
        # The condition `and self.mongodb_client is not None` is removed as native cache might not depend on EAT's MongoDBClient.
        self.use_cache = use_cache
        if self.use_cache:
            self.cache = UnconstrainedCache()
            logger.info("LLMService: Native UnconstrainedCache initialized.")
        else:
            self.cache = None
            logger.info("LLMService: Native caching is disabled.")

        # Save API key to environment if provided (for OpenAI)
        if self.api_key and provider == "openai":
            os.environ["OPENAI_API_KEY"] = self.api_key

        # Initialize models based on provider
        # Import native EmbeddingModel from beeai_framework
        from beeai_framework.backend import EmbeddingModel as NativeEmbeddingModel

        try:
            chat_model_name_for_factory = None
            chat_settings_for_factory = {}
            embedding_model_name_for_factory = None
            embedding_settings_for_factory = {}

            if provider == "openai":
                actual_chat_model_id = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
                chat_model_name_for_factory = f"openai:{actual_chat_model_id}"
                if self.api_key:
                    chat_settings_for_factory["api_key"] = self.api_key

                actual_embedding_model_id = embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
                embedding_model_name_for_factory = f"openai:{actual_embedding_model_id}"
                if self.api_key:
                    embedding_settings_for_factory["api_key"] = self.api_key

            elif provider == "ollama":
                actual_chat_model_id = model or "llama3"
                chat_model_name_for_factory = f"ollama:{actual_chat_model_id}"
                # LiteLLM typically picks up OLLAMA_BASE_URL from env for settings

                actual_embedding_model_id = embedding_model or "nomic-embed-text"
                embedding_model_name_for_factory = f"ollama:{actual_embedding_model_id}"
                # LiteLLM typically picks up OLLAMA_BASE_URL from env for settings

            else: # Defaulting to OpenAI
                logger.warning(f"Unknown LLM provider '{provider}'. Defaulting to OpenAI.")
                self.provider = "openai"
                actual_chat_model_id = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
                chat_model_name_for_factory = f"openai:{actual_chat_model_id}"
                if self.api_key:
                    chat_settings_for_factory["api_key"] = self.api_key

                actual_embedding_model_id = embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
                embedding_model_name_for_factory = f"openai:{actual_embedding_model_id}"
                if self.api_key:
                    embedding_settings_for_factory["api_key"] = self.api_key

            # Instantiate ChatModel using the factory
            if chat_model_name_for_factory:
                chat_model_kwargs = {
                    "settings": chat_settings_for_factory
                }
                if self.cache:
                    chat_model_kwargs["cache"] = self.cache
                self.chat_model = ChatModel.from_name(
                    chat_model_name_for_factory,
                    **chat_model_kwargs
                )
            else:
                logger.error(f"Could not determine chat_model_name_for_factory for provider {self.provider}. Chat model not initialized.")
                self.chat_model = None

            # Instantiate the native EmbeddingModel
            if embedding_model_name_for_factory:
                embedding_model_kwargs = {
                    "settings": embedding_settings_for_factory
                }
                if self.cache:
                    embedding_model_kwargs["cache"] = self.cache
                self.embedding_model = NativeEmbeddingModel.from_name(
                    embedding_model_name_for_factory,
                    **embedding_model_kwargs
                )
            else:
                logger.error(f"Could not determine embedding_model_name_for_factory for provider {self.provider}. Embedding model not initialized.")
                self.embedding_model = None

        except Exception as e:
            logger.error(f"Error initializing LLM models: {e}", exc_info=True)
            # Ensure models are None if initialization failed
            if not hasattr(self, 'chat_model'): self.chat_model = None
            if not hasattr(self, 'embedding_model'): self.embedding_model = None
            
        cache_status = "enabled (MongoDB)" if self.cache else "disabled"
        # Ensure model_id and embedding_id are accessed safely
        chat_model_id_log = self.chat_model.model_id if hasattr(self.chat_model, 'model_id') and self.chat_model else "N/A"
        embedding_model_id_log = self.embedding_model.model_id if hasattr(self.embedding_model, 'model_id') and self.embedding_model else "N/A"
        
        logger.info(f"Initialized LLM service with provider: {self.provider}, "
                    f"chat model: {chat_model_id_log}, "
                    f"embedding model: {embedding_model_id_log}, "
                    f"cache: {cache_status}")

    async def generate(self, prompt: str) -> str:
        """Generate text based on a prompt."""
        if not self.chat_model:
            logger.error("Chat model not initialized. Cannot generate text.")
            return "Error: Chat model not available."

        logger.debug(f"Generating response for prompt: {prompt[:50]}...")
        messages = [UserMessage(prompt)]

        try:
            response_output = await self.chat_model.create(messages=messages) # Returns ChatModelOutput
            response_text = response_output.get_text_content()
            
            return response_text
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return f"Error generating response: {str(e)}"

    async def embed(self, text: str) -> List[float]:
        """Generate embeddings for text using the native EmbeddingModel."""
        if not self.embedding_model:
            logger.error("Embedding model not initialized. Cannot generate embeddings.")
            return [0.0] * 1536 # Default dimension for OpenAI, adjust if provider differs

        logger.debug(f"Generating embedding for text: {text[:50]}...")
        
        try:
            # Native EmbeddingModel.create expects a list of documents
            output = await self.embedding_model.create(documents=[text]) # Returns EmbeddingModelOutput
            
            if output.embeddings and output.embeddings[0]:
                embedding = output.embeddings[0]
                return embedding
            else:
                logger.warning(f"Embedding model returned no data for text: {text[:50]}")
                # Fallback based on provider or a generic dimension
                # For simplicity, using OpenAI's default. This might need refinement
                # if other providers have different default dimensions.
                default_dim = getattr(self.embedding_model, 'dimension', 1536) # Attempt to get dimension
                return [0.0] * default_dim
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}", exc_info=True)
            default_dim = getattr(self.embedding_model, 'dimension', 1536)
            return [0.0] * default_dim

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts using the native EmbeddingModel."""
        if not self.embedding_model:
            logger.error("Embedding model not initialized. Cannot generate batch embeddings.")
            default_dim = getattr(self.embedding_model, 'dimension', 1536)
            return [[0.0] * default_dim for _ in texts]

        logger.debug(f"Generating batch embeddings for {len(texts)} texts...")
        
        try:
            output = await self.embedding_model.create(documents=texts) # Returns EmbeddingModelOutput
            
            if output.embeddings and len(output.embeddings) == len(texts):
                return output.embeddings
            else:
                logger.warning(f"Embedding model returned mismatched or no data for batch texts. Expected {len(texts)}, got {len(output.embeddings) if output.embeddings else 0}.")
                default_dim = getattr(self.embedding_model, 'dimension', 1536)
                return [[0.0] * default_dim for _ in texts]
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}", exc_info=True)
            default_dim = getattr(self.embedding_model, 'dimension', 1536)
            return [[0.0] * default_dim for _ in texts]

    async def clear_cache(self, older_than_seconds: Optional[int] = None) -> int:
        """
        Clear the LLM cache.
        With UnconstrainedCache, only full clear is supported.
        The `older_than_seconds` parameter is no longer effective with UnconstrainedCache.
        """
        if not self.cache or not hasattr(self.cache, 'clear'):
            logger.info("LLMService: Cache is disabled or does not support clear(), nothing to clear.")
            return 0

        if older_than_seconds is not None:
            logger.warning("LLMService: `clear_cache` called with `older_than_seconds`. "
                           "UnconstrainedCache only supports full cache clear. This parameter will be ignored.")

        try:
            # UnconstrainedCache.clear() is synchronous.
            self.cache.clear()
            logger.info("LLMService: Native cache cleared successfully.")
            # UnconstrainedCache.clear() doesn't return a count.
            # Returning 0 as a placeholder.
            return 0
        except Exception as e:
            logger.error(f"LLMService: Error clearing native cache: {e}", exc_info=True)
            return 0

    async def generate_applicability_text(self, text_chunk: str, component_type: str = "", component_name: str = "") -> str:
        """Generate applicability text (T_raz)."""
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
        try:
            applicability_text = await self.generate(prompt)
            return applicability_text.strip()
        except Exception as e:
            logger.error(f"Error generating applicability text: {e}", exc_info=True)
            return f"Error generating applicability: {str(e)}"