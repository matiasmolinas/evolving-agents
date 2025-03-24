# tests/test_llm_cache.py

import os
import shutil
import pytest
import asyncio
from evolving_agents.core.llm_service import LLMService, LLMCache

@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    cache_dir = ".test_llm_cache"
    
    # Clean up any existing test cache
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    
    os.makedirs(cache_dir)
    yield cache_dir
    
    # Clean up after test
    shutil.rmtree(cache_dir)

def test_cache_key_generation():
    """Test that the cache key generation creates consistent keys."""
    cache = LLMCache(cache_dir=".test_llm_cache")
    
    # Same input should generate same key
    key1 = cache._generate_key("test input")
    key2 = cache._generate_key("test input")
    assert key1 == key2
    
    # Different inputs should generate different keys
    key3 = cache._generate_key("different input")
    assert key1 != key3

@pytest.mark.asyncio
async def test_embedding_cache(temp_cache_dir, monkeypatch):
    """Test that embeddings are properly cached."""
    # Mock the actual embedding calls
    call_count = 0
    
    class MockEmbeddingModel:
        async def create(self, input_text):
            nonlocal call_count
            call_count += 1
            # Return a simple vector
            return {"data": [0.1, 0.2, 0.3, 0.4, 0.5]}
        
        async def create_batch(self, input_texts):
            nonlocal call_count
            call_count += len(input_texts)
            # Return a list of vectors
            return [{"data": [0.1, 0.2, 0.3, 0.4, 0.5]} for _ in input_texts]
    
    # Create LLM service with mocked model and caching
    service = LLMService(provider="openai", use_cache=True, cache_dir=temp_cache_dir)
    service.embedding_model = MockEmbeddingModel()
    
    # First call should hit the model
    embedding1 = await service.embed("test text")
    assert call_count == 1
    
    # Second call with same text should use cache
    embedding2 = await service.embed("test text")
    assert call_count == 1  # Count should not increase
    
    # Call with different text should hit the model
    embedding3 = await service.embed("different text")
    assert call_count == 2
    
    # Test batch embedding
    batch_result = await service.embed_batch(["text1", "text2", "text3"])
    assert call_count == 5  # +3 for the batch
    
    # Second batch with same texts should use cache
    batch_result2 = await service.embed_batch(["text1", "text2", "text3"])
    assert call_count == 5  # Count should not increase

@pytest.mark.asyncio
async def test_completion_cache(temp_cache_dir, monkeypatch):
    """Test that completions are properly cached."""
    from beeai_framework.backend.message import UserMessage
    
    # Mock the actual completion calls
    call_count = 0
    
    class MockResponse:
        def get_text_content(self):
            return "Test response"
    
    class MockChatModel:
        model_id = "gpt-4o-mini"
        
        async def create(self, messages):
            nonlocal call_count
            call_count += 1
            return MockResponse()
    
    # Create LLM service with mocked model and caching
    service = LLMService(provider="openai", use_cache=True, cache_dir=temp_cache_dir)
    service.chat_model = MockChatModel()
    
    # First call should hit the model
    response1 = await service.generate("test prompt")
    assert call_count == 1
    
    # Second call with same prompt should use cache
    response2 = await service.generate("test prompt")
    assert call_count == 1  # Count should not increase
    
    # Call with different prompt should hit the model
    response3 = await service.generate("different prompt")
    assert call_count == 2
    
    # Test cache clearing
    count = service.clear_cache()
    
    # After clearing, calls should hit the model again
    response4 = await service.generate("test prompt")
    assert call_count == 3