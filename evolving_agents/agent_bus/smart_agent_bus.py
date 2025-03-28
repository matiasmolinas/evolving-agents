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
from evolving_agents.core.dependency_container import DependencyContainer
from beeai_framework.agents.react import ReActAgent

logger = logging.getLogger(__name__)

class SmartAgentBus:
    """
    An agent-centric service bus for discovering, managing, and orchestrating AI agents.
    
    Features:
    - Semantic agent discovery using vector embeddings
    - Agent health monitoring with circuit breakers
    - Agent-to-agent communication
    - Comprehensive execution logging
    - Persistent agent registry
    """

    def __init__(
        self,
        smart_library: Optional[SmartLibrary] = None,
        system_agent: Optional[ReActAgent] = None,
        llm_service: Optional[LLMService] = None,
        storage_path: str = "agent_registry.json",
        chroma_path: str = "./agent_db",
        log_path: str = "agent_execution_logs.json",
        container: Optional[DependencyContainer] = None
    ):
        """
        Initialize the Agent Bus.
        
        Args:
            smart_library: Initialized SmartLibrary instance
            system_agent: Optional ReActAgent for fallback execution
            llm_service: Optional LLMService for embeddings
            storage_path: Path for agent registry storage
            chroma_path: Path for ChromaDB storage
            log_path: Path for execution logs
            container: Optional dependency container for managing component dependencies
        """
        # Resolve dependencies from container if provided
        if container:
            self.smart_library = smart_library or container.get('smart_library')
            self.system_agent = system_agent
            self.llm_service = llm_service or container.get('llm_service')
        else:
            self.smart_library = smart_library
            self.system_agent = system_agent
            self.llm_service = llm_service or LLMService()
        
        self.log_path = log_path
        self._initialized = False

        # Register with container if provided
        if container and not container.has('agent_bus'):
            container.register('agent_bus', self)

        # Initialize components
        self._init_agent_database(chroma_path)
        self._init_agent_registry(storage_path)
        
        # Don't load agents immediately - will do this during initialize_from_library
        logger.info(f"AgentBus initialized with registry at {storage_path}")

    def _init_agent_database(self, chroma_path: str) -> None:
        """Initialize the ChromaDB agent database."""
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.agent_collection = self.chroma_client.get_or_create_collection(
            name="agent_registry",
            metadata={"hnsw:space": "cosine"}
        )

    def _init_agent_registry(self, storage_path: str) -> None:
        """Initialize the agent registry system."""
        self.registry_path = storage_path
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self._load_registry()

    async def initialize_from_library(self) -> None:
        """Initialize agents from SmartLibrary records."""
        if self._initialized:
            return
            
        if not self.smart_library:
            logger.warning("Cannot initialize from library: SmartLibrary not provided")
            return
            
        for record in self.smart_library.records:
            if record.get("status") != "active":
                continue
                
            # Skip if already registered
            if any(a["name"] == record["name"] for a in self.agents.values()):
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
            
            await self.register_agent(
                name=record["name"],
                description=record.get("description", ""),
                capabilities=capabilities,
                agent_type=record.get("record_type", "GENERIC"),
                metadata={"source": "SmartLibrary"}
            )
        
        self._initialized = True
        logger.info(f"Initialized {len(self.agents)} agents from library")

    def _load_registry(self) -> None:
        """Load agents from persistent storage."""
        if Path(self.registry_path).exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                    self.agents = data.get("agents", {})
                    self.circuit_breakers = data.get("circuit_breakers", {})
                logger.info(f"Loaded {len(self.agents)} agents from registry")
            except Exception as e:
                logger.error(f"Error loading agent registry: {str(e)}")

    async def _save_registry(self) -> None:
        """Persist agents to storage."""
        data = {
            "agents": self.agents,
            "circuit_breakers": self.circuit_breakers,
            "updated_at": datetime.utcnow().isoformat()
        }
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved {len(self.agents)} agents to registry")

    async def _index_agent(self, agent: Dict[str, Any]) -> None:
        """Index an agent for semantic search."""
        semantic_text = (
            f"Agent: {agent['name']}\n"
            f"Description: {agent.get('description', '')}\n"
            f"Type: {agent.get('type', '')}\n"
            f"Capabilities: {', '.join(c['name'] for c in agent.get('capabilities', []))}"
        )
        
        embedding = await self.llm_service.embed(semantic_text)
        
        # Convert capabilities list to a string to avoid ChromaDB error
        capabilities_str = ",".join([c["id"] for c in agent.get("capabilities", [])])
        
        self.agent_collection.add(
            ids=[agent["id"]],
            embeddings=[embedding],
            documents=[semantic_text],
            metadatas=[{
                "name": agent["name"],
                "type": agent.get("type", ""),
                "capabilities": capabilities_str,  # Store as string instead of list
                "created_at": datetime.utcnow().isoformat()
            }]
        )

    async def _log_agent_execution(
        self,
        agent_id: str,
        task: Dict[str, Any],
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """Log agent execution details."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": agent_id,
            "agent_name": self.agents.get(agent_id, {}).get("name"),
            "task": task,
            "result": result,
            "error": error,
            "system_state": {
                "total_agents": len(self.agents),
                "active_agents": sum(1 for a in self.agents.values() if a["status"] == "active"),
                "unhealthy_agents": sum(
                    1 for agent_id in self.agents 
                    if self._check_circuit_breaker(agent_id)
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

    def _check_circuit_breaker(self, agent_id: str) -> bool:
        """Check if agent is in circuit breaker state."""
        if agent_id not in self.circuit_breakers:
            return False
        
        cb = self.circuit_breakers[agent_id]
        return cb["failures"] >= 3 and (time.time() - cb["last_failure"]) < 300

    def _record_failure(self, agent_id: str) -> None:
        """Record an agent failure."""
        if agent_id not in self.circuit_breakers:
            self.circuit_breakers[agent_id] = {
                "failures": 1,
                "last_failure": time.time()
            }
        else:
            self.circuit_breakers[agent_id]["failures"] += 1
            self.circuit_breakers[agent_id]["last_failure"] = time.time()

    def _reset_circuit_breaker(self, agent_id: str) -> None:
        """Reset circuit breaker after successful operation."""
        if agent_id in self.circuit_breakers:
            del self.circuit_breakers[agent_id]

    async def register_agent(
        self,
        name: str,
        description: str,
        capabilities: List[Dict[str, Any]],
        agent_type: str = "GENERIC",
        metadata: Optional[Dict[str, Any]] = None,
        agent_instance = None  # New parameter, not stored in JSON
    ) -> str:
        """
        Register a new agent with the bus.
        
        Args:
            name: Agent name
            description: What this agent does
            capabilities: List of capabilities the agent provides
            agent_type: Type of agent (GENERIC, SPECIALIZED, etc.)
            metadata: Additional agent configuration (must be JSON serializable)
            agent_instance: Actual agent instance (not serialized, kept in memory)
                
        Returns:
            Agent ID
        """
        agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        
        # Ensure metadata is serializable by creating a clean copy
        clean_metadata = {}
        if metadata:
            for k, v in metadata.items():
                # Only keep serializable values
                if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                    clean_metadata[k] = v
        
        # Create the agent record (serializable)
        agent = {
            "id": agent_id,
            "name": name,
            "description": description,
            "type": agent_type,
            "capabilities": capabilities,
            "metadata": clean_metadata,
            "status": "active",
            "registered_at": datetime.utcnow().isoformat(),
            "last_seen": datetime.utcnow().isoformat()
        }
        
        # Store the serializable record
        self.agents[agent_id] = agent
        
        # If agent instance is provided, store it in a separate in-memory dictionary
        if agent_instance is not None:
            if not hasattr(self, '_agent_instances'):
                self._agent_instances = {}
            self._agent_instances[agent_id] = agent_instance
        
        await self._index_agent(agent)
        await self._save_registry()
        
        logger.info(f"Registered new agent: {name} ({agent_id})")
        return agent_id

    async def discover_agents(
        self,
        task_description: str,
        min_confidence: float = 0.6,
        agent_type: Optional[str] = None,
        required_capabilities: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Discover agents suitable for a given task.
        
        Args:
            task_description: Natural language description of the task
            min_confidence: Minimum match confidence threshold
            agent_type: Filter by agent type
            required_capabilities: List of specific capability IDs needed
            
        Returns:
            List of matching agents with their confidence scores
        """
        query_embedding = await self.llm_service.embed(task_description)
        results = self.agent_collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["distances", "metadatas"]
        )
        
        matches = []
        for i, agent_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            similarity = 1.0 - distance
            
            if similarity >= min_confidence and agent_id in self.agents:
                agent = self.agents[agent_id]
                
                # Apply filters
                if agent_type and agent["type"] != agent_type:
                    continue
                    
                if required_capabilities:
                    # Get capabilities from the stored agent data (not from metadata)
                    agent_caps = {c["id"] for c in agent["capabilities"]}
                    if not all(req_cap in agent_caps for req_cap in required_capabilities):
                        continue
                
                # Check agent health
                is_healthy = not self._check_circuit_breaker(agent_id)
                
                matches.append({
                    "agent": agent,
                    "similarity": similarity,
                    "is_healthy": is_healthy
                })
        
        return sorted(matches, key=lambda x: x["similarity"], reverse=True)

    async def execute_with_agent(
        self,
        agent_id: str,
        task: Dict[str, Any],
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Execute a task using a specific agent.
        
        Args:
            agent_id: ID of the agent to use
            task: Dictionary containing task details and input data
            timeout: Maximum execution time in seconds
            
        Returns:
            Dictionary with execution results and metadata
            
        Raises:
            ValueError: If agent not found
            RuntimeError: If execution fails
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent not found: {agent_id}")
            
        if self._check_circuit_breaker(agent_id):
            raise RuntimeError(f"Agent {agent_id} is temporarily unavailable")
        
        agent = self.agents[agent_id]
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                self._execute_agent_task(agent, task),
                timeout=timeout
            )
            
            # Record successful execution
            self._reset_circuit_breaker(agent_id)
            agent["last_seen"] = datetime.utcnow().isoformat()
            
            response = {
                "result": result,
                "metadata": {
                    "agent_id": agent_id,
                    "agent_name": agent["name"],
                    "processing_time": time.time() - start_time,
                    "success": True
                }
            }
            
            await self._log_agent_execution(
                agent_id,
                task,
                response
            )
            
            return response
            
        except Exception as e:
            self._record_failure(agent_id)
            await self._log_agent_execution(
                agent_id,
                task,
                error=str(e)
            )
            raise RuntimeError(f"Agent execution failed: {str(e)}")

    async def _execute_agent_task(
        self,
        agent: Dict[str, Any],
        task: Dict[str, Any]
    ) -> Any:
        """Execute a task using the specified agent."""
        agent_id = agent["id"]
        
        # Check if we have a live instance of this agent
        if hasattr(self, '_agent_instances') and agent_id in self._agent_instances:
            agent_instance = self._agent_instances[agent_id]
            
            # If the agent is a ReActAgent with run method
            if hasattr(agent_instance, 'run') and callable(agent_instance.run):
                try:
                    task_text = task.get("text", json.dumps(task))
                    return await agent_instance.run(task_text)
                except Exception as e:
                    raise RuntimeError(f"Agent execution with instance failed: {str(e)}")
        
        # Check for direct execution function
        execute_func = agent.get("metadata", {}).get("execute")
        if execute_func and callable(execute_func):
            try:
                if asyncio.iscoroutinefunction(execute_func):
                    return await execute_func(task)
                return execute_func(task)
            except Exception as e:
                raise RuntimeError(f"Agent execution failed: {str(e)}")
        
        # Fall back to system agent
        if not self.system_agent:
            raise RuntimeError("No execution method available for agent")
        
        prompt = self._create_agent_prompt(agent, task)
        return await self.system_agent.run(prompt)

    def _create_agent_prompt(
        self,
        agent: Dict[str, Any],
        task: Dict[str, Any]
    ) -> str:
        """Create execution prompt for system agent."""
        return (
            f"Act as {agent['name']} - {agent['description']}\n"
            f"Agent Type: {agent['type']}\n"
            f"Capabilities: {', '.join(c['name'] for c in agent['capabilities'])}\n\n"
            f"Task Request:\n{json.dumps(task, indent=2)}\n\n"
            "Perform this task to the best of your ability according to your capabilities."
        )

    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """
        Get detailed status of an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dictionary with agent status information
            
        Raises:
            ValueError: If agent not found
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent not found: {agent_id}")
        
        status = {
            **self.agents[agent_id],
            "circuit_breaker": "tripped" if self._check_circuit_breaker(agent_id) else "normal"
        }
        
        if agent_id in self.circuit_breakers:
            cb = self.circuit_breakers[agent_id]
            status.update({
                "failure_count": cb["failures"],
                "last_failure": cb["last_failure"],
                "time_since_last_failure": time.time() - cb["last_failure"]
            })
        
        return status

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
                "total_agents": len(self.agents),
                "active_agents": sum(1 for a in self.agents.values() if a["status"] == "active"),
                "unhealthy_agents": sum(
                    1 for agent_id in self.agents 
                    if self._check_circuit_breaker(agent_id))
            },
            "storage": {
                "registry": self.registry_path,
                "chromadb": str(Path(self.chroma_client._path).resolve()),
                "logs": str(Path(self.log_path).resolve())
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