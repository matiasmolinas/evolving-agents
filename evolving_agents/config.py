# evolving_agents/config.py

import os

# LLM settings
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_EMBEDDING_MODEL = os.environ.get("LLM_EMBEDDING_MODEL", "text-embedding-3-small")
LLM_USE_CACHE = os.environ.get("LLM_USE_CACHE", "true").lower() == "true"
LLM_CACHE_DIR = os.environ.get("LLM_CACHE_DIR", ".llm_cache")