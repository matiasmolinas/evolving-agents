# evolving_agents/core/llm_service.py

import logging
import os
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import UserMessage
from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend.constants import ProviderName

logger = logging.getLogger(__name__)

class LLMCache:
    """Cache for LLM completions and embeddings to reduce API usage during development."""
    
    def __init__(self, cache_dir: str = ".llm_cache", ttl: int = 86400 * 7):
        """
        Initialize the LLM cache.
        
        Args:
            cache_dir: Directory to store cache files
            ttl: Time-to-live for cache entries in seconds (default: 7 days)
        """
        self.cache_dir = Path(cache_dir)
        self.completion_cache_dir = self.cache_dir / "completions"
        self.embedding_cache_dir = self.cache_dir / "embeddings"
        self.ttl = ttl
        
        # Create cache directories if they don't exist
        self.completion_cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized LLM cache at {self.cache_dir}")
    
    def _generate_key(self, data: Any) -> str:
        """Generate a unique key for the given input data."""
        # Convert to JSON-serializable format
        if isinstance(data, list) and all(isinstance(item, UserMessage) for item in data):
            # Handle UserMessage objects
            serializable_data = []
            for msg in data:
                # Check if content is a complex object with text attribute
                if hasattr(msg.content, 'text'):
                    content = msg.content.text
                else:
                    content = str(msg.content)
                
                serializable_data.append({"content": content, "role": msg.role})
        else:
            serializable_data = data
            
        # Create a deterministic string representation
        data_str = json.dumps(serializable_data, sort_keys=True)
        
        # Generate hash
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get_completion(self, messages: List[UserMessage], model_id: str) -> Optional[str]:
        """Get a cached completion result if available."""
        key = self._generate_key(messages)
        cache_file = self.completion_cache_dir / f"{model_id}_{key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                # Check if cache entry has expired
                if time.time() - data["timestamp"] <= self.ttl:
                    logger.info(f"Cache hit for completion: {key[:8]}...")
                    return data["response"]
                else:
                    logger.info(f"Cache expired for completion: {key[:8]}...")
            except Exception as e:
                logger.warning(f"Error reading cache file: {str(e)}")
        
        return None
    
    def save_completion(self, messages: List[UserMessage], model_id: str, response: str) -> None:
        """Save a completion result to cache."""
        key = self._generate_key(messages)
        cache_file = self.completion_cache_dir / f"{model_id}_{key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    "timestamp": time.time(),
                    "model_id": model_id,
                    "response": response
                }, f)
            logger.info(f"Cached completion: {key[:8]}...")
        except Exception as e:
            logger.warning(f"Error writing cache file: {str(e)}")
    
    def get_embedding(self, text: str, model_id: str) -> Optional[List[float]]:
        """Get a cached embedding if available."""
        key = self._generate_key(text)
        cache_file = self.embedding_cache_dir / f"{model_id}_{key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                # Check if cache entry has expired
                if time.time() - data["timestamp"] <= self.ttl:
                    logger.info(f"Cache hit for embedding: {key[:8]}...")
                    return data["embedding"]
                else:
                    logger.info(f"Cache expired for embedding: {key[:8]}...")
            except Exception as e:
                logger.warning(f"Error reading cache file: {str(e)}")
        
        return None
    
    def save_embedding(self, text: str, model_id: str, embedding: List[float]) -> None:
        """Save an embedding to cache."""
        key = self._generate_key(text)
        cache_file = self.embedding_cache_dir / f"{model_id}_{key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    "timestamp": time.time(),
                    "model_id": model_id,
                    "embedding": embedding
                }, f)
            logger.info(f"Cached embedding: {key[:8]}...")
        except Exception as e:
            logger.warning(f"Error writing cache file: {str(e)}")
    
    def get_batch_embeddings(self, texts: List[str], model_id: str) -> Optional[List[List[float]]]:
        """Get cached batch embeddings if all are available."""
        results = []
        
        # Check if all texts are in cache
        for text in texts:
            embedding = self.get_embedding(text, model_id)
            if embedding is None:
                return None
            results.append(embedding)
        
        return results
    
    def clear_cache(self, older_than: Optional[int] = None) -> int:
        """
        Clear the cache.
        
        Args:
            older_than: Clear entries older than this many seconds. If None, clear all.
            
        Returns:
            Number of entries removed
        """
        count = 0
        now = time.time()
        
        for cache_dir in [self.completion_cache_dir, self.embedding_cache_dir]:
            for cache_file in cache_dir.glob("*.json"):
                try:
                    if older_than is not None:
                        with open(cache_file, 'r') as f:
                            data = json.load(f)
                        if now - data["timestamp"] > older_than:
                            cache_file.unlink()
                            count += 1
                    else:
                        cache_file.unlink()
                        count += 1
                except Exception as e:
                    logger.warning(f"Error clearing cache file {cache_file}: {str(e)}")
        
        return count

class OpenAIChatModel(LiteLLMChatModel):
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

class EmbeddingModel:
    """Base embedding model class for generating vector embeddings."""
    
    def __init__(self, model_id: str, provider_id: str, settings: Optional[Dict[str, Any]] = None):
        self.model_id = model_id
        self.provider_id = provider_id
        self.settings = settings or {}
        
    async def create(self, input_text: str) -> Dict[str, Any]:
        """Create embedding for a single text."""
        raise NotImplementedError("Subclasses must implement this method")
        
    async def create_batch(self, input_texts: List[str]) -> List[Dict[str, Any]]:
        """Create embeddings for multiple texts."""
        raise NotImplementedError("Subclasses must implement this method")

class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI-specific implementation of embedding model."""
    
    def __init__(self, model_id: str | None = None, settings: dict | None = None):
        _settings = settings.copy() if settings else {}
        _model_id = model_id if model_id else os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        
        super().__init__(
            model_id=_model_id,
            provider_id="openai",
            settings=_settings
        )
        
        # Import inside method to avoid circular imports
        import openai
        
        # Configure the client
        self.client = openai.OpenAI(
            api_key=_settings.get("api_key") or os.getenv("OPENAI_API_KEY"),
            base_url=_settings.get("base_url") or os.getenv("OPENAI_API_ENDPOINT"),
        )
    
    async def create(self, input_text: str) -> Dict[str, Any]:
        """Create an embedding for a single text using OpenAI."""
        try:
            response = self.client.embeddings.create(
                model=self.model_id,
                input=input_text,
                encoding_format="float"
            )
            return {
                "data": response.data[0].embedding,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            # Return a placeholder in case of error
            return {"data": [0.0] * 1536, "error": str(e)}
    
    async def create_batch(self, input_texts: List[str]) -> List[Dict[str, Any]]:
        """Create embeddings for multiple texts using OpenAI."""
        try:
            response = self.client.embeddings.create(
                model=self.model_id,
                input=input_texts,
                encoding_format="float"
            )
            return [
                {
                    "data": item.embedding,
                    "model": response.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
                for item in response.data
            ]
        except Exception as e:
            logger.error(f"Error creating batch embeddings: {str(e)}")
            # Return placeholders in case of error
            return [{"data": [0.0] * 1536, "error": str(e)} for _ in input_texts]

class OllamaEmbeddingModel(EmbeddingModel):
    """Ollama-specific implementation of embedding model."""
    
    def __init__(self, model_id: str | None = None, settings: dict | None = None):
        _settings = settings.copy() if settings else {}
        _model_id = model_id if model_id else "nomic-embed-text"
        
        super().__init__(
            model_id=_model_id,
            provider_id="ollama",
            settings=_settings
        )
        
        # Import ollama client
        import httpx
        self.client = httpx.AsyncClient(
            base_url=_settings.get("base_url") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
    
    async def create(self, input_text: str) -> Dict[str, Any]:
        """Create an embedding for a single text using Ollama."""
        try:
            response = await self.client.post(
                "/api/embeddings",
                json={"model": self.model_id, "prompt": input_text}
            )
            response.raise_for_status()
            data = response.json()
            return {
                "data": data["embedding"],
                "model": self.model_id,
                "usage": {"total_tokens": data.get("total_tokens", 0)}
            }
        except Exception as e:
            logger.error(f"Error creating Ollama embedding: {str(e)}")
            # Return a placeholder in case of error (typically 384 dimensions for Ollama)
            return {"data": [0.0] * 384, "error": str(e)}
    
    async def create_batch(self, input_texts: List[str]) -> List[Dict[str, Any]]:
        """Create embeddings for multiple texts using Ollama (one by one)."""
        results = []
        for text in input_texts:
            results.append(await self.create(text))
        return results

class LLMService:
    """
    LLM service that interfaces with chat and embedding models.
    """
    def __init__(
        self, 
        provider: str = "openai", 
        api_key: Optional[str] = None, 
        model: str = None,
        embedding_model: str = None,
        use_cache: bool = True,
        cache_dir: str = ".llm_cache"
    ):
        self.provider = provider
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        # Initialize cache if enabled
        self.use_cache = use_cache and os.environ.get("LLM_DISABLE_CACHE", "").lower() != "true"
        self.cache = LLMCache(cache_dir=cache_dir) if self.use_cache else None
        
        # Save API key to environment if provided
        if self.api_key and provider == "openai":
            os.environ["OPENAI_API_KEY"] = self.api_key
        
        # Initialize models based on provider
        if provider == "openai":
            self.chat_model = OpenAIChatModel(
                model_id=model or "gpt-4o-mini",
                settings={"api_key": self.api_key} if self.api_key else None
            )
            self.embedding_model = OpenAIEmbeddingModel(
                model_id=embedding_model or "text-embedding-3-small",
                settings={"api_key": self.api_key} if self.api_key else None
            )
        elif provider == "ollama":
            # Use LiteLLM for Ollama chat
            self.chat_model = LiteLLMChatModel(
                model_id=model or "llama3",
                provider_id="ollama",
                settings={}
            )
            # Use custom Ollama embedding implementation
            self.embedding_model = OllamaEmbeddingModel(
                model_id=embedding_model or "nomic-embed-text",
                settings={}
            )
        else:
            # Default to OpenAI
            self.chat_model = OpenAIChatModel(
                model_id=model or "gpt-4o-mini",
                settings={"api_key": self.api_key} if self.api_key else None
            )
            self.embedding_model = OpenAIEmbeddingModel(
                model_id=embedding_model or "text-embedding-3-small",
                settings={"api_key": self.api_key} if self.api_key else None
            )
        
        cache_status = "enabled" if self.use_cache else "disabled"
        logger.info(f"Initialized LLM service with provider: {provider}, "
                    f"chat model: {self.chat_model.model_id}, "
                    f"embedding model: {self.embedding_model.model_id}, "
                    f"cache: {cache_status}")
    
    async def generate(self, prompt: str) -> str:
        """
        Generate text based on a prompt.
        
        Args:
            prompt: The input prompt
            
        Returns:
            Generated text
        """
        logger.info(f"Generating response for prompt: {prompt[:50]}...")
        
        # Create a new message
        message = UserMessage(prompt)
        messages = [message]
        
        # Check cache first if enabled
        if self.use_cache:
            cached_response = self.cache.get_completion(messages, self.chat_model.model_id)
            if cached_response is not None:
                return cached_response
        
        try:
            # Generate response
            response = await self.chat_model.create(messages=messages)
            response_text = response.get_text_content()
            
            # Cache the response if caching is enabled
            if self.use_cache:
                self.cache.save_completion(messages, self.chat_model.model_id, response_text)
                
            return response_text
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            # Return a fallback response in case of error
            return f"Error generating response: {str(e)}"
    
    async def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding
        """
        logger.info(f"Generating embedding for text: {text[:50]}...")
        
        # Check cache first if enabled
        if self.use_cache:
            cached_embedding = self.cache.get_embedding(text, self.embedding_model.model_id)
            if cached_embedding is not None:
                return cached_embedding
        
        try:
            # Generate embedding
            response = await self.embedding_model.create(text)
            embedding = response["data"]
            
            # Cache the embedding if caching is enabled
            if self.use_cache:
                self.cache.save_embedding(text, self.embedding_model.model_id, embedding)
                
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return a fallback vector in case of error
            dim = 1536 if self.provider == "openai" else 384
            return [0.0] * dim
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of vector embeddings
        """
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Check cache first if enabled
        if self.use_cache:
            cached_embeddings = self.cache.get_batch_embeddings(texts, self.embedding_model.model_id)
            if cached_embeddings is not None:
                return cached_embeddings
        
        try:
            # Generate embeddings for multiple texts
            responses = await self.embedding_model.create_batch(texts)
            embeddings = [response["data"] for response in responses]
            
            # Cache each embedding if caching is enabled
            if self.use_cache:
                for i, text in enumerate(texts):
                    self.cache.save_embedding(text, self.embedding_model.model_id, embeddings[i])
                
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            # Return fallback vectors in case of error
            dim = 1536 if self.provider == "openai" else 384
            return [[0.0] * dim for _ in range(len(texts))]
    
    def clear_cache(self, older_than: Optional[int] = None) -> int:
        """
        Clear the LLM cache.
        
        Args:
            older_than: Clear entries older than this many seconds. If None, clear all.
            
        Returns:
            Number of entries removed
        """
        if not self.use_cache:
            return 0
            
        return self.cache.clear_cache(older_than)