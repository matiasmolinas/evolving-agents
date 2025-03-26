import os
import json
import logging
import asyncio
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path

import chromadb
import numpy as np
from evolving_agents.core.llm_service import LLMService
from evolving_agents.smart_library.smart_library import SmartLibrary
from beeai_framework.agents.react import ReActAgent

logger = logging.getLogger(__name__)

class SmartAgentBus:
    """
    A unified Agent Bus with semantic capability discovery, provider management,
    and execution orchestration.
    
    Features:
    - Semantic capability matching using ChromaDB
    - Circuit breaker pattern for fault tolerance
    - Integrated with SmartLibrary for capability discovery
    - Comprehensive logging and monitoring
    - Async operations throughout
    """

    def __init__(
        self,
        smart_library: SmartLibrary,
        system_agent: Optional[ReActAgent] = None,
        llm_service: Optional[LLMService] = None,
        storage_path: str = "smart_agent_bus.json",
        chroma_path: str = "./capability_db",
        log_path: str = "agent_bus_logs.json"
    ):
        """
        Initialize the SmartAgentBus.
        
        Args:
            smart_library: Initialized SmartLibrary instance
            system_agent: Optional ReActAgent for service execution
            llm_service: Optional LLMService for embeddings
            storage_path: Path for provider storage
            chroma_path: Path for ChromaDB storage
            log_path: Path for execution logs
        """
        self.smart_library = smart_library
        self.system_agent = system_agent
        self.llm_service = llm_service or LLMService()
        self.log_path = log_path

        # Initialize components
        self._init_chroma_db(chroma_path)
        self._init_provider_system(storage_path)
        self._load_capabilities()

        logger.info(f"SmartAgentBus initialized with logging at {log_path}")

    def _init_chroma_db(self, chroma_path: str) -> None:
        """Initialize the ChromaDB capability registry."""
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.capability_collection = self.chroma_client.get_or_create_collection(
            name="capability_registry",
            metadata={"hnsw:space": "cosine"}  # Using cosine similarity
        )

    def _init_provider_system(self, storage_path: str) -> None:
        """Initialize the provider management system."""
        self.provider_storage_path = storage_path
        self.providers: Dict[str, Dict[str, Any]] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self._load_providers()

    def _load_providers(self) -> None:
        """Load providers from persistent storage."""
        if Path(self.provider_storage_path).exists():
            try:
                with open(self.provider_storage_path, 'r') as f:
                    data = json.load(f)
                    self.providers = data.get("providers", {})
                    self.circuit_breakers = data.get("circuit_breakers", {})
                logger.info(f"Loaded {len(self.providers)} providers from storage")
            except Exception as e:
                logger.error(f"Error loading providers: {str(e)}")

    async def _save_providers(self) -> None:
        """Persist providers to storage."""
        data = {
            "providers": self.providers,
            "circuit_breakers": self.circuit_breakers,
            "updated_at": datetime.utcnow().isoformat()
        }
        with open(self.provider_storage_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved {len(self.providers)} providers to storage")

    def _load_capabilities(self) -> None:
        """Load capabilities from SmartLibrary records."""
        self.capabilities = {}
        for record in self.smart_library.records:
            if record.get("status") != "active":
                continue
                
            for cap in record.get("metadata", {}).get("capabilities", []):
                if self._validate_capability(cap):
                    self.capabilities[cap["id"]] = cap
        logger.info(f"Loaded {len(self.capabilities)} capabilities from library")

    def _validate_capability(self, capability: Dict[str, Any]) -> bool:
        """Validate capability structure."""
        required = {"id", "name", "description"}
        return all(field in capability for field in required)

    async def _register_capability(self, capability: Dict[str, Any]) -> None:
        """Register a capability in ChromaDB with semantic indexing."""
        semantic_text = (
            f"Capability: {capability['name']}\n"
            f"Description: {capability.get('description', '')}\n"
            f"Input: {capability.get('input_requirements', '')}\n"
            f"Output: {capability.get('output_guarantees', '')}"
        )
        
        embedding = await self.llm_service.embed(semantic_text)
        
        self.capability_collection.add(
            ids=[capability["id"]],
            embeddings=[embedding],
            documents=[semantic_text],
            metadatas=[{
                "name": capability["name"],
                "description": capability.get("description", ""),
                "created_at": datetime.utcnow().isoformat()
            }]
        )
        logger.debug(f"Registered capability: {capability['name']}")

    async def _log_execution(
        self,
        capability_id: str,
        input_data: Dict[str, Any],
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        provider_id: Optional[str] = None
    ) -> None:
        """Log execution details to file."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "capability": capability_id,
            "provider": provider_id,
            "input": input_data,
            "result": result,
            "error": error,
            "system_state": {
                "active_providers": len(self.providers),
                "tripped_circuits": sum(
                    1 for cb in self.circuit_breakers.values()
                    if cb["failures"] >= 3 and (time.time() - cb["last_failure"]) < 300
                )
            }
        }

        try:
            logs = []
            if Path(self.log_path).exists():
                with open(self.log_path, 'r') as f:
                    logs = json.load(f)
            
            logs.append(log_entry)
            
            with open(self.log_path, 'w') as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write execution log: {str(e)}")

    def _check_circuit_breaker(self, provider_id: str) -> bool:
        """Check if provider is in circuit breaker state."""
        if provider_id not in self.circuit_breakers:
            return False
        
        cb = self.circuit_breakers[provider_id]
        return cb["failures"] >= 3 and (time.time() - cb["last_failure"]) < 300

    def _record_failure(self, provider_id: str) -> None:
        """Record a provider failure."""
        if provider_id not in self.circuit_breakers:
            self.circuit_breakers[provider_id] = {
                "failures": 1,
                "last_failure": time.time()
            }
        else:
            self.circuit_breakers[provider_id]["failures"] += 1
            self.circuit_breakers[provider_id]["last_failure"] = time.time()

    def _reset_circuit_breaker(self, provider_id: str) -> None:
        """Reset circuit breaker after successful operation."""
        if provider_id in self.circuit_breakers:
            del self.circuit_breakers[provider_id]

    async def register_provider(
        self,
        name: str,
        capabilities: List[Dict[str, Any]],
        provider_type: str = "AGENT",
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a new capability provider.
        
        Args:
            name: Provider name
            capabilities: List of capabilities provided
            provider_type: Type of provider (AGENT, TOOL, etc.)
            description: Provider description
            metadata: Additional metadata
            
        Returns:
            Provider ID
        """
        provider_id = f"{provider_type.lower()}_{uuid.uuid4().hex[:8]}"
        
        valid_caps = []
        for cap in capabilities:
            if not self._validate_capability(cap):
                logger.warning(f"Skipping invalid capability: {cap}")
                continue
                
            valid_caps.append(cap)
            
            if cap["id"] not in self.capabilities:
                self.capabilities[cap["id"]] = cap
                await self._register_capability(cap)
        
        provider = {
            "id": provider_id,
            "name": name,
            "type": provider_type,
            "description": description,
            "capabilities": valid_caps,
            "metadata": metadata or {},
            "status": "active",
            "registered_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat()
        }
        
        self.providers[provider_id] = provider
        await self._save_providers()
        
        logger.info(f"Registered provider {name} with {len(valid_caps)} capabilities")
        return provider_id

    async def request_service(
        self,
        capability_query: Union[str, Dict[str, Any]],
        input_data: Dict[str, Any],
        provider_id: Optional[str] = None,
        min_confidence: float = 0.7,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Request a service by capability.
        
        Args:
            capability_query: Capability ID or natural language description
            input_data: Input data for the service
            provider_id: Optional specific provider
            min_confidence: Minimum semantic match confidence
            timeout: Operation timeout in seconds
            
        Returns:
            Service response with metadata
            
        Raises:
            ValueError: If capability or provider not found
            RuntimeError: If execution fails
        """
        start_time = time.time()
        
        # Resolve capability
        if isinstance(capability_query, str):
            if capability_query in self.capabilities:
                capability_id = capability_query
            else:
                # Semantic search
                query_embedding = await self.llm_service.embed(capability_query)
                results = self.capability_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=1,
                    include=["distances"]
                )
                
                if not results["ids"][0]:
                    raise ValueError(f"No matching capability found for: {capability_query}")
                
                distance = results["distances"][0][0]
                similarity = 1.0 - distance
                
                if similarity < min_confidence:
                    raise ValueError(
                        f"Best match similarity {similarity:.2f} below threshold {min_confidence}"
                    )
                
                capability_id = results["ids"][0][0]
                logger.info(f"Matched query to capability {capability_id} (similarity: {similarity:.2f})")
        else:
            capability_id = capability_query["id"]
        
        # Get provider
        if provider_id:
            if self._check_circuit_breaker(provider_id):
                raise RuntimeError(f"Provider {provider_id} is temporarily unavailable")
            
            provider = self.providers.get(provider_id)
            if not provider:
                raise ValueError(f"Provider not found: {provider_id}")
                
            if not any(cap["id"] == capability_id for cap in provider["capabilities"]):
                raise ValueError(f"Provider does not have capability: {capability_id}")
        else:
            # Find best available provider
            candidates = []
            for p in self.providers.values():
                if self._check_circuit_breaker(p["id"]):
                    continue
                    
                for cap in p["capabilities"]:
                    if cap["id"] == capability_id:
                        candidates.append((p, cap.get("confidence", 0.8)))
                        break
            
            if not candidates:
                raise ValueError(f"No available providers for capability: {capability_id}")
                
            provider, confidence = max(candidates, key=lambda x: x[1])
            logger.info(f"Selected provider {provider['name']} (confidence: {confidence:.2f})")
        
        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                self._execute_service(provider, capability_id, input_data),
                timeout=timeout
            )
            
            # Record success
            self._reset_circuit_breaker(provider["id"])
            
            response = {
                "result": result,
                "metadata": {
                    "provider": provider["id"],
                    "capability": capability_id,
                    "processing_time": time.time() - start_time,
                    "success": True
                }
            }
            
            await self._log_execution(
                capability_id,
                input_data,
                response,
                provider_id=provider["id"]
            )
            
            return response
            
        except Exception as e:
            # Record failure
            self._record_failure(provider["id"])
            
            await self._log_execution(
                capability_id,
                input_data,
                error=str(e),
                provider_id=provider["id"]
            )
            
            logger.error(f"Service execution failed: {str(e)}")
            raise RuntimeError(f"Service execution failed: {str(e)}")

    async def _execute_service(
        self,
        provider: Dict[str, Any],
        capability_id: str,
        input_data: Dict[str, Any]
    ) -> Any:
        """Execute a service using the provider."""
        # Check for direct execution function
        execution_func = provider.get("metadata", {}).get("execution_function")
        if execution_func and callable(execution_func):
            try:
                if asyncio.iscoroutinefunction(execution_func):
                    return await execution_func(input_data)
                return execution_func(input_data)
            except Exception as e:
                raise RuntimeError(f"Provider execution failed: {str(e)}")
        
        # Fall back to system agent
        if not self.system_agent:
            raise RuntimeError("No execution method available for provider")
        
        prompt = self._create_execution_prompt(provider, capability_id, input_data)
        return await self.system_agent.run(prompt)

    def _create_execution_prompt(
        self,
        provider: Dict[str, Any],
        capability_id: str,
        input_data: Dict[str, Any]
    ) -> str:
        """Create execution prompt for system agent."""
        capability = self.capabilities[capability_id]
        
        return (
            f"Execute capability: {capability['name']}\n"
            f"Description: {capability.get('description', '')}\n"
            f"Using provider: {provider['name']}\n"
            f"Provider type: {provider['type']}\n\n"
            f"Input Data:\n{json.dumps(input_data, indent=2)}\n\n"
            "Process this request according to the capability requirements."
        )

    async def discover_capabilities(
        self,
        query: str,
        min_confidence: float = 0.6,
        provider_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Discover capabilities using semantic search.
        
        Args:
            query: Natural language query
            min_confidence: Minimum similarity threshold
            provider_type: Optional provider type filter
            
        Returns:
            List of matching capabilities with providers
        """
        query_embedding = await self.llm_service.embed(query)
        results = self.capability_collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["distances", "metadatas"]
        )
        
        matches = []
        for i, cap_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            similarity = 1.0 - distance
            
            if similarity >= min_confidence and cap_id in self.capabilities:
                capability = self.capabilities[cap_id]
                
                # Find matching providers
                matching_providers = []
                for p in self.providers.values():
                    if provider_type and p["type"] != provider_type:
                        continue
                        
                    if any(c["id"] == cap_id for c in p["capabilities"]):
                        if not self._check_circuit_breaker(p["id"]):
                            matching_providers.append({
                                "id": p["id"],
                                "name": p["name"],
                                "type": p["type"]
                            })
                
                if matching_providers:
                    matches.append({
                        "capability": capability,
                        "similarity": similarity,
                        "providers": matching_providers
                    })
        
        return matches

    async def get_provider_status(self, provider_id: str) -> Dict[str, Any]:
        """
        Get detailed provider status.
        
        Args:
            provider_id: ID of the provider
            
        Returns:
            Provider status information
            
        Raises:
            ValueError: If provider not found
        """
        provider = self.providers.get(provider_id)
        if not provider:
            raise ValueError(f"Provider not found: {provider_id}")
        
        status = {
            **provider,
            "circuit_breaker": "tripped" if self._check_circuit_breaker(provider_id) else "normal"
        }
        
        if provider_id in self.circuit_breakers:
            cb = self.circuit_breakers[provider_id]
            status.update({
                "failure_count": cb["failures"],
                "last_failure": cb["last_failure"],
                "time_since_last_failure": time.time() - cb["last_failure"]
            })
        
        return status

    async def initialize_from_library(self) -> None:
        """Initialize providers from SmartLibrary records."""
        for record in self.smart_library.records:
            if record.get("status") != "active":
                continue
                
            # Skip if already registered
            if any(p["name"] == record["name"] for p in self.providers.values()):
                continue
                
            capabilities = record.get("metadata", {}).get("capabilities", [])
            if not capabilities:
                # Create default capability
                capabilities = [{
                    "id": f"{record['name'].lower().replace(' ', '_')}_default",
                    "name": record["name"],
                    "description": record.get("description", ""),
                    "confidence": 0.8
                }]
            
            await self.register_provider(
                name=record["name"],
                capabilities=capabilities,
                provider_type=record["record_type"],
                description=record.get("description", ""),
                metadata={"source": "SmartLibrary"}
            )
        
        logger.info(f"Initialized {len(self.providers)} providers from library")

    async def health_check(self) -> Dict[str, Any]:
        """
        Get system health status.
        
        Returns:
            Dictionary with health metrics
        """
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "stats": {
                "providers": len(self.providers),
                "capabilities": len(self.capabilities),
                "tripped_circuits": sum(
                    1 for cb in self.circuit_breakers.values()
                    if cb["failures"] >= 3 and (time.time() - cb["last_failure"]) < 300
                )
            },
            "storage": {
                "providers": self.provider_storage_path,
                "chromadb": self.chroma_client._path,
                "logs": self.log_path
            }
        }

    async def clear_logs(self) -> None:
        """Clear all execution logs."""
        try:
            if Path(self.log_path).exists():
                Path(self.log_path).unlink()
                logger.info("Cleared all execution logs")
        except Exception as e:
            logger.error(f"Failed to clear logs: {str(e)}")