# evolving_agents/agent_bus/smart_agent_bus.py

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
# Avoid importing specific agent types like ReActAgent directly if possible
from evolving_agents.core.base import IAgent # Use the protocol

logger = logging.getLogger(__name__)

class SmartAgentBus:
    """
    An agent-centric service bus for discovering, managing, and orchestrating AI agents.
    Implements a logical Dual Bus:
    - System Bus: Handles registration, discovery, health monitoring (e.g., register_agent, discover_agents).
    - Data Bus: Handles agent-to-agent capability requests and data exchange (e.g., request_capability).
    """

    def __init__(
        self,
        smart_library: Optional[SmartLibrary] = None,
        # system_agent: Optional[ReActAgent] = None, # Type hint using the protocol instead
        system_agent: Optional[IAgent] = None,
        llm_service: Optional[LLMService] = None,
        storage_path: str = "agent_registry.json",
        chroma_path: str = "./agent_db",
        log_path: str = "agent_execution_logs.json",
        container: Optional[DependencyContainer] = None
    ):
        """
        Initialize the Agent Bus.
        """
        # Resolve dependencies from container if provided
        if container:
            self.smart_library = smart_library or container.get('smart_library')
            # system_agent is resolved later if needed
            self.llm_service = llm_service or container.get('llm_service')
        else:
            self.smart_library = smart_library
            self.llm_service = llm_service or LLMService()

        # System agent can be set later, allows circular dependency resolution
        self._system_agent_instance = system_agent
        self.container = container # Store container reference

        self.log_path = log_path
        self._initialized = False
        self._agent_instances: Dict[str, Any] = {} # Store live agent instances

        # Register with container if provided
        if container and not container.has('agent_bus'):
            container.register('agent_bus', self)

        # Initialize components
        self._init_agent_database(chroma_path)
        self._init_agent_registry(storage_path)

        logger.info(f"AgentBus initialized with registry at {storage_path}")

    @property
    def system_agent(self) -> Optional[IAgent]:
        """Lazy load system agent from container if not set."""
        if not self._system_agent_instance and self.container and self.container.has('system_agent'):
             self._system_agent_instance = self.container.get('system_agent')
        return self._system_agent_instance


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
        """Initialize agents from SmartLibrary records (System Bus Operation)."""
        if self._initialized:
            return

        if not self.smart_library:
            logger.warning("Cannot initialize from library: SmartLibrary not provided")
            return

        library_records = await self.smart_library.export_records(record_type="AGENT") # Fetch only AGENT records
        library_records.extend(await self.smart_library.export_records(record_type="TOOL")) # Fetch TOOL records

        agents_initialized = 0
        for record in library_records:
            if record.get("status") != "active":
                continue

            # Check if already registered by ID
            if record["id"] in self.agents:
                 continue
            # Check if registered by name (less reliable, but useful during transition)
            if any(a["name"] == record["name"] for a in self.agents.values()):
                continue

            capabilities = record.get("metadata", {}).get("capabilities", [])
            # Generate default capability if none provided
            if not capabilities:
                cap_id = f"{record['name'].lower().replace(' ', '_')}_default_capability"
                capabilities = [{
                    "id": cap_id,
                    "name": record["name"],
                    "description": record.get("description", f"Default capability for {record['name']}"),
                    "confidence": 0.7 # Lower confidence for default
                }]

            # Attempt to create agent/tool instance using AgentFactory
            agent_instance = None
            if self.container and self.container.has('agent_factory'):
                 agent_factory = self.container.get('agent_factory')
                 try:
                      if record["record_type"] == "AGENT":
                           agent_instance = await agent_factory.create_agent(record)
                      elif record["record_type"] == "TOOL" and self.container.has('tool_factory'):
                           tool_factory = self.container.get('tool_factory')
                           agent_instance = await tool_factory.create_tool(record) # Tools are also 'agents' on the bus
                      else:
                           logger.warning(f"Cannot create instance for record type {record['record_type']} or missing factory.")

                 except Exception as e:
                      logger.error(f"Failed to create instance for {record['name']} ({record['id']}): {e}", exc_info=True)


            # Register with the agent instance if created
            await self.register_agent(
                agent_id=record["id"], # Use the library ID
                name=record["name"],
                description=record.get("description", ""),
                capabilities=capabilities,
                agent_type=record.get("record_type", "GENERIC"),
                metadata={"source": "SmartLibrary", **record.get("metadata", {})},
                agent_instance=agent_instance
            )
            agents_initialized += 1

        self._initialized = True
        logger.info(f"AgentBus initialized/synced {agents_initialized} components from SmartLibrary.")


    def _load_registry(self) -> None:
        """Load agents from persistent storage."""
        if Path(self.registry_path).exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                    self.agents = data.get("agents", {})
                    self.circuit_breakers = data.get("circuit_breakers", {})
                logger.info(f"Loaded {len(self.agents)} agents from registry {self.registry_path}")
            except Exception as e:
                logger.error(f"Error loading agent registry from {self.registry_path}: {str(e)}")
                self.agents = {} # Start fresh on error
                self.circuit_breakers = {}


    async def _save_registry(self) -> None:
        """Persist agents to storage."""
        data = {
            "agents": self.agents,
            "circuit_breakers": self.circuit_breakers,
            "updated_at": datetime.utcnow().isoformat()
        }
        try:
             with open(self.registry_path, 'w') as f:
                 json.dump(data, f, indent=2)
             logger.debug(f"Saved {len(self.agents)} agents to registry {self.registry_path}")
        except Exception as e:
             logger.error(f"Error saving agent registry to {self.registry_path}: {str(e)}")


    async def _index_agent(self, agent: Dict[str, Any]) -> None:
        """Index an agent for semantic search (System Bus Operation)."""
        semantic_text = (
            f"Agent Name: {agent['name']}\n"
            f"Type: {agent.get('type', 'UNKNOWN')}\n"
            f"Description: {agent.get('description', '')}\n"
            f"Capabilities: {', '.join(c['name'] for c in agent.get('capabilities', []))}"
        )

        try:
             embedding = await self.llm_service.embed(semantic_text)

             # Convert capabilities list to a string to avoid ChromaDB error
             capabilities_str = ",".join([c["id"] for c in agent.get("capabilities", [])])
             # Ensure metadata values are simple types supported by ChromaDB
             chroma_metadata = {
                "name": agent["name"],
                "type": agent.get("type", ""),
                "capabilities": capabilities_str,
                "source": agent.get("metadata", {}).get("source", "unknown") # Example: only store source
             }
             # Add other simple metadata if needed, avoiding nested dicts/lists

             self.agent_collection.upsert( # Use upsert for easier sync
                 ids=[agent["id"]],
                 embeddings=[embedding],
                 documents=[semantic_text],
                 metadatas=[chroma_metadata]
             )
             logger.debug(f"Indexed/Updated agent {agent['id']} in vector DB.")
        except Exception as e:
             logger.error(f"Failed to index agent {agent['id']}: {e}", exc_info=True)


    async def _log_agent_execution(
        self,
        bus_type: str, # Added: 'system' or 'data'
        agent_id: str,
        task: Dict[str, Any],
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        duration: Optional[float] = None
    ) -> None:
        """Log agent execution details, indicating bus type."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "bus_type": bus_type, # Indicate System or Data Bus
            "agent_id": agent_id,
            "agent_name": self.agents.get(agent_id, {}).get("name", "N/A"),
            "task_description": task.get("capability") or task.get("operation", "direct_execution"), # More specific task
            "task_details": task, # Keep full task info
            "result": result,
            "error": error,
            "duration_ms": int(duration * 1000) if duration is not None else None,
            "system_state": { # Optional: Keep system state snapshot
                "total_agents": len(self.agents),
                "active_agents": sum(1 for a in self.agents.values() if a["status"] == "active"),
                "unhealthy_agents": sum(
                    1 for ag_id in self.agents
                    if self._check_circuit_breaker(ag_id)
                )
            }
        }

        try:
            logs = []
            if Path(self.log_path).exists():
                with open(self.log_path, 'r') as f:
                    # Handle potential empty file or invalid JSON
                    try:
                        content = f.read()
                        if content.strip():
                             logs = json.loads(content)
                             if not isinstance(logs, list): logs = [] # Ensure it's a list
                        else:
                             logs = []
                    except json.JSONDecodeError:
                        logger.warning(f"Log file {self.log_path} contains invalid JSON. Starting fresh log.")
                        logs = []

            logs.append(log_entry)

            with open(self.log_path, 'w') as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write execution log to {self.log_path}: {str(e)}")


    def _check_circuit_breaker(self, agent_id: str) -> bool:
        """Check if agent is in circuit breaker state (System Bus Health Check)."""
        if agent_id not in self.circuit_breakers:
            return False

        cb = self.circuit_breakers[agent_id]
        if cb["failures"] >= 3 and (time.time() - cb["last_failure"]) < 300:
            logger.warning(f"Circuit breaker tripped for agent {agent_id}")
            return True
        return False


    def _record_failure(self, agent_id: str) -> None:
        """Record an agent failure (System Bus Health Update)."""
        if agent_id not in self.circuit_breakers:
            self.circuit_breakers[agent_id] = {
                "failures": 1,
                "last_failure": time.time()
            }
        else:
            self.circuit_breakers[agent_id]["failures"] += 1
            self.circuit_breakers[agent_id]["last_failure"] = time.time()
        logger.warning(f"Failure recorded for agent {agent_id}. Count: {self.circuit_breakers[agent_id]['failures']}")


    def _reset_circuit_breaker(self, agent_id: str) -> None:
        """Reset circuit breaker after successful operation (System Bus Health Update)."""
        if agent_id in self.circuit_breakers:
            logger.info(f"Resetting circuit breaker for agent {agent_id}")
            del self.circuit_breakers[agent_id]


    async def register_agent(
        self,
        name: str,
        description: str,
        capabilities: List[Dict[str, Any]],
        agent_type: str = "GENERIC",
        metadata: Optional[Dict[str, Any]] = None,
        agent_instance = None,
        agent_id: Optional[str] = None # Allow providing ID (e.g., from library)
    ) -> str:
        """
        Register a new agent or update an existing one (System Bus Operation).

        Args:
            name: Agent name
            description: What this agent does
            capabilities: List of capabilities the agent provides
            agent_type: Type of agent (GENERIC, SPECIALIZED, etc.)
            metadata: Additional agent configuration (must be JSON serializable)
            agent_instance: Actual agent instance (not serialized, kept in memory)
            agent_id: Optional existing agent ID to update or use

        Returns:
            Agent ID
        """
        op_type = "Registered new"
        if not agent_id:
            agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        elif agent_id in self.agents:
             op_type = "Updated"


        # Ensure metadata is serializable
        clean_metadata = {}
        if metadata:
            for k, v in metadata.items():
                try:
                    json.dumps({k: v}) # Test serializability
                    clean_metadata[k] = v
                except TypeError:
                    logger.warning(f"Metadata key '{k}' with non-serializable value skipped for agent '{name}'")

        # Create or update the agent record (serializable)
        agent_record = {
            "id": agent_id,
            "name": name,
            "description": description,
            "type": agent_type,
            "capabilities": capabilities,
            "metadata": clean_metadata,
            "status": "active", # Default to active on register/update
            "registered_at": self.agents.get(agent_id, {}).get("registered_at", datetime.utcnow().isoformat()),
            "last_updated": datetime.utcnow().isoformat(),
            "last_seen": datetime.utcnow().isoformat() # Update last seen on registration
        }

        # Store the serializable record
        self.agents[agent_id] = agent_record

        # Store/Update the live instance
        if agent_instance is not None:
            self._agent_instances[agent_id] = agent_instance
            logger.debug(f"Stored live instance for agent {agent_id} ({name})")
        elif agent_id in self._agent_instances:
            # If no instance provided on update, keep the existing one? Or remove?
            # Let's keep it for now, assuming it might be managed elsewhere.
             logger.debug(f"Agent {agent_id} updated without new instance, keeping existing.")


        await self._index_agent(agent_record)
        await self._save_registry()

        # Log system event
        await self._log_agent_execution(
             bus_type='system',
             agent_id=agent_id,
             task={'operation': 'register', 'details': {'name': name, 'type': agent_type}},
             result={'status': 'success'}
        )

        logger.info(f"{op_type} agent: {name} ({agent_id}) with {len(capabilities)} capabilities.")
        return agent_id


    async def discover_agents(
        self,
        task_description: Optional[str] = None,
        capability_id: Optional[str] = None,
        min_confidence: float = 0.6,
        agent_type: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Discover agents suitable for a task or capability (System Bus Operation).

        Args:
            task_description: Natural language description of the task (for semantic search)
            capability_id: Specific capability ID to filter by (exact match)
            min_confidence: Minimum match confidence threshold (for semantic search)
            agent_type: Filter by agent type
            limit: Max number of agents to return

        Returns:
            List of matching agents with details (including similarity if semantic search used)
        """
        if not task_description and not capability_id:
             raise ValueError("Either task_description or capability_id must be provided for discovery.")

        matches = []
        # --- Semantic Search Path ---
        if task_description:
            logger.debug(f"Discovering agents via semantic search for: '{task_description[:50]}...'")
            query_embedding = await self.llm_service.embed(task_description)

            # Build ChromaDB filter
            where_filter = {"status": {"$eq": "active"}} # Always filter active
            if agent_type:
                where_filter["type"] = {"$eq": agent_type}
            # Note: Capability filtering is harder with ChromaDB metadata limitations on lists.
            # We'll do post-filtering for capability_id if provided along with task_description.

            try:
                 results = self.agent_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=limit * 2, # Get more results initially for filtering
                    where=where_filter,
                    include=["distances", "metadatas"]
                 )

                 if not results or not results["ids"] or not results["ids"][0]:
                      logger.info("Semantic discovery returned no results.")
                      return []

                 for i, result_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    similarity = 1.0 - distance

                    if similarity >= min_confidence and result_id in self.agents:
                        agent = self.agents[result_id]

                        # Post-filter by capability_id if provided
                        if capability_id:
                            agent_caps = {c["id"] for c in agent["capabilities"]}
                            if capability_id not in agent_caps:
                                continue # Skip if required capability not present

                        # Check agent health
                        is_healthy = not self._check_circuit_breaker(result_id)

                        matches.append({
                            "id": agent["id"],
                            "name": agent["name"],
                            "description": agent.get("description", ""),
                            "type": agent.get("type", ""),
                            "capabilities": agent.get("capabilities", []),
                            "similarity": similarity, # Include similarity from semantic search
                            "is_healthy": is_healthy
                        })

            except Exception as e:
                 logger.error(f"Error during semantic discovery: {e}", exc_info=True)
                 # Potentially fall back to direct filtering if Chroma fails? For now, return empty.
                 return []

        # --- Direct Capability Filter Path ---
        elif capability_id:
            logger.debug(f"Discovering agents directly for capability: '{capability_id}'")
            for agent_id, agent in self.agents.items():
                 if agent["status"] != "active":
                      continue
                 if agent_type and agent["type"] != agent_type:
                      continue

                 # Check if agent provides the capability
                 provides_capability = False
                 for cap in agent["capabilities"]:
                      if cap["id"] == capability_id:
                           provides_capability = True
                           break
                 if not provides_capability:
                      continue

                 is_healthy = not self._check_circuit_breaker(agent_id)
                 matches.append({
                    "id": agent["id"],
                    "name": agent["name"],
                    "description": agent.get("description", ""),
                    "type": agent.get("type", ""),
                    "capabilities": agent.get("capabilities", []),
                    "similarity": 1.0, # Exact capability match implies high similarity conceptually
                    "is_healthy": is_healthy
                 })

        # Sort final matches (primarily relevant for semantic search) and limit
        return sorted(matches, key=lambda x: x.get("similarity", 0.0), reverse=True)[:limit]


    async def request_capability(
        self,
        capability: str,
        content: Dict[str, Any],
        min_confidence: float = 0.6,
        specific_agent_id: Optional[str] = None,
        timeout: float = 60.0
    ) -> Dict[str, Any]:
        """
        Request a capability from a suitable agent (Data Bus Operation).

        Finds an agent providing the capability and executes the task.

        Args:
            capability: The ID of the capability being requested.
            content: The input data/payload for the capability request.
            min_confidence: Minimum confidence if discovery is needed.
            specific_agent_id: Optional ID of a specific agent to target.
            timeout: Execution timeout in seconds.

        Returns:
            Dictionary with execution results.

        Raises:
            ValueError: If no suitable agent is found.
            RuntimeError: If execution fails or agent is unhealthy.
        """
        logger.info(f"Received data bus request for capability '{capability}'")
        start_time = time.time()
        agent_to_use = None
        agent_id_to_use = None

        task_details = {"capability": capability, **content} # Combine for logging

        # --- Find the Agent ---
        if specific_agent_id:
            if specific_agent_id in self.agents:
                if self.agents[specific_agent_id]["status"] == "active":
                    agent_to_use = self.agents[specific_agent_id]
                    agent_id_to_use = specific_agent_id
                    logger.debug(f"Targeting specific agent: {agent_id_to_use}")
                else:
                     raise ValueError(f"Targeted agent {specific_agent_id} is not active.")
            else:
                raise ValueError(f"Targeted agent {specific_agent_id} not found in registry.")
        else:
            # Discover agents based on capability ID
            suitable_agents = await self.discover_agents(
                capability_id=capability,
                min_confidence=min_confidence # Confidence applies if discovery uses semantic search fallback
            )
            # Filter for healthy agents
            healthy_agents = [a for a in suitable_agents if a["is_healthy"]]

            if not healthy_agents:
                # Log discovery failure details
                await self._log_agent_execution(
                    bus_type='data',
                    agent_id='N/A',
                    task=task_details,
                    error=f"No healthy agent found for capability '{capability}'",
                    duration=time.time() - start_time
                )
                unhealthy_count = len(suitable_agents) - len(healthy_agents)
                raise ValueError(f"No available agent found for capability '{capability}'. "
                                 f"(Found {len(suitable_agents)}, {unhealthy_count} unhealthy).")

            # Select the best healthy agent (highest confidence/similarity)
            # Confidence score might not be directly available if only capability_id was used
            # For now, just pick the first healthy one found by discovery (already sorted)
            best_agent_info = healthy_agents[0]
            agent_id_to_use = best_agent_info["id"]
            agent_to_use = self.agents[agent_id_to_use]
            logger.debug(f"Discovered agent {agent_id_to_use} for capability '{capability}'.")


        # --- Execute the Task ---
        if not agent_to_use: # Should not happen if logic above is correct
             raise RuntimeError("Internal error: Agent selected but not found.")

        if self._check_circuit_breaker(agent_id_to_use):
             error_msg = f"Agent {agent_id_to_use} ({agent_to_use['name']}) is temporarily unavailable (circuit breaker tripped)."
             await self._log_agent_execution('data', agent_id_to_use, task_details, error=error_msg, duration=time.time()-start_time)
             raise RuntimeError(error_msg)

        try:
            result_data = await asyncio.wait_for(
                self._execute_agent_task(agent_to_use, task=content), # Pass content as task
                timeout=timeout
            )

            # Record successful execution
            self._reset_circuit_breaker(agent_id_to_use)
            self.agents[agent_id_to_use]["last_seen"] = datetime.utcnow().isoformat()
            duration = time.time() - start_time

            # Prepare response, ensuring result_data is included
            response = {
                "status": "success",
                "agent_id": agent_id_to_use,
                "agent_name": agent_to_use["name"],
                "capability_executed": capability,
                "content": result_data, # Include the actual result here
                "metadata": {
                    "processing_time_ms": int(duration * 1000)
                }
            }

            await self._log_agent_execution(
                 bus_type='data',
                 agent_id=agent_id_to_use,
                 task=task_details,
                 result=response, # Log the structured response
                 duration=duration
            )
            logger.info(f"Capability '{capability}' executed successfully by {agent_id_to_use}.")
            return response

        except asyncio.TimeoutError:
            self._record_failure(agent_id_to_use)
            duration = time.time() - start_time
            error_msg = f"Agent {agent_id_to_use} timed out executing capability '{capability}' after {timeout}s."
            await self._log_agent_execution('data', agent_id_to_use, task_details, error=error_msg, duration=duration)
            raise RuntimeError(error_msg)
        except Exception as e:
            self._record_failure(agent_id_to_use)
            duration = time.time() - start_time
            error_msg = f"Agent {agent_id_to_use} failed executing capability '{capability}': {str(e)}"
            await self._log_agent_execution('data', agent_id_to_use, task_details, error=error_msg, duration=duration)
            logger.error(f"Error during capability execution: {e}", exc_info=True)
            raise RuntimeError(error_msg)


    async def execute_with_agent(
        self,
        agent_id: str,
        task: Dict[str, Any],
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Execute a task *directly* using a specific agent (System Bus/Debug Operation).
        Use request_capability for standard inter-agent communication.

        Args:
            agent_id: ID of the agent to use.
            task: Dictionary containing task details and input data.
            timeout: Maximum execution time in seconds.

        Returns:
            Dictionary with execution results and metadata.

        Raises:
            ValueError: If agent not found.
            RuntimeError: If execution fails or agent is unhealthy.
        """
        logger.info(f"Received direct execution request for agent {agent_id}")
        start_time = time.time()
        if agent_id not in self.agents:
            raise ValueError(f"Agent not found: {agent_id}")

        agent = self.agents[agent_id]
        if agent["status"] != "active":
             raise ValueError(f"Agent {agent_id} is not active.")

        if self._check_circuit_breaker(agent_id):
            error_msg = f"Agent {agent_id} is temporarily unavailable (circuit breaker tripped)."
            # Log this as a system-level failure attempt
            await self._log_agent_execution('system', agent_id, task, error=error_msg, duration=time.time()-start_time)
            raise RuntimeError(error_msg)

        try:
            result_data = await asyncio.wait_for(
                self._execute_agent_task(agent, task),
                timeout=timeout
            )

            # Record successful execution
            self._reset_circuit_breaker(agent_id)
            agent["last_seen"] = datetime.utcnow().isoformat()
            duration = time.time() - start_time

            response = {
                "result": result_data, # Main result data
                "metadata": {
                    "agent_id": agent_id,
                    "agent_name": agent["name"],
                    "processing_time_ms": int(duration * 1000),
                    "success": True
                }
            }

            # Log as a system-level execution
            await self._log_agent_execution(
                bus_type='system', # Log direct execution as system bus activity
                agent_id=agent_id,
                task=task,
                result=response,
                duration=duration
            )

            return response

        except asyncio.TimeoutError:
             self._record_failure(agent_id)
             duration = time.time() - start_time
             error_msg = f"Agent {agent_id} timed out during direct execution after {timeout}s."
             await self._log_agent_execution('system', agent_id, task, error=error_msg, duration=duration)
             raise RuntimeError(error_msg)
        except Exception as e:
            self._record_failure(agent_id)
            duration = time.time() - start_time
            error_msg = f"Agent {agent_id} failed during direct execution: {str(e)}"
            await self._log_agent_execution('system', agent_id, task, error=error_msg, duration=duration)
            logger.error(f"Error during direct agent execution: {e}", exc_info=True)
            raise RuntimeError(error_msg)


    async def _execute_agent_task(
        self,
        agent_record: Dict[str, Any],
        task: Dict[str, Any]
    ) -> Any:
        """
        Internal helper to execute a task using the specified agent's instance.
        This can be called by both data bus (request_capability) and system bus (execute_with_agent).

        Args:
             agent_record: The registry record of the agent.
             task: The task payload (e.g., content from request_capability or direct task).

        Returns:
             The result from the agent's execution.

        Raises:
             RuntimeError: If no execution method is available or execution fails.
        """
        agent_id = agent_record["id"]
        agent_instance = self._agent_instances.get(agent_id)

        if not agent_instance:
            # Try to dynamically create instance if not already loaded
            logger.warning(f"Live instance for agent {agent_id} not found, attempting dynamic creation...")
            if self.container and self.container.has('agent_factory'):
                 agent_factory = self.container.get('agent_factory')
                 try:
                      if agent_record["record_type"] == "AGENT":
                           agent_instance = await agent_factory.create_agent(agent_record)
                      elif agent_record["record_type"] == "TOOL" and self.container.has('tool_factory'):
                           tool_factory = self.container.get('tool_factory')
                           agent_instance = await tool_factory.create_tool(agent_record) # Tools are also 'agents' on the bus

                      if agent_instance:
                           self._agent_instances[agent_id] = agent_instance # Store for future use
                           logger.info(f"Dynamically created and stored instance for {agent_id}")
                      else:
                           raise RuntimeError(f"Agent factory could not create instance for {agent_id}.")

                 except Exception as e:
                      logger.error(f"Dynamic instance creation failed for {agent_id}: {e}", exc_info=True)
                      raise RuntimeError(f"Could not get or create live instance for agent {agent_id}.")
            else:
                 raise RuntimeError(f"No live instance found for agent {agent_id} and no AgentFactory available.")

        # --- Execute based on instance type ---
        # Check if it's an Evolving Agents/BeeAI style agent/tool with a run method
        if hasattr(agent_instance, 'run') and callable(agent_instance.run):
            try:
                # Prepare input for the run method
                # If task has 'text', use it as prompt, otherwise serialize task dict
                input_prompt = task.get("text", json.dumps(task))
                run_result = await agent_instance.run(input_prompt) # Assuming run takes a single prompt string

                # Extract meaningful result (adapt based on actual return type of run)
                if hasattr(run_result, 'result') and hasattr(run_result.result, 'text'):
                     return run_result.result.text
                elif hasattr(run_result, 'text'): # Simpler result object
                     return run_result.text
                elif isinstance(run_result, str):
                     return run_result
                else:
                     # Attempt to serialize if it's a complex object
                     try: return json.loads(json.dumps(run_result)) # Convert complex objects if possible
                     except: return str(run_result) # Fallback to string representation

            except Exception as e:
                logger.error(f"Execution failed for agent {agent_id} using run method: {e}", exc_info=True)
                raise RuntimeError(f"Agent execution via run method failed: {str(e)}")

        # Check for OpenAI Agent SDK style execution (requires provider)
        elif agent_record.get("metadata", {}).get("framework") == "openai-agents":
             if self.container and self.container.has('agent_factory'):
                  agent_factory = self.container.get('agent_factory')
                  provider = agent_factory.provider_registry.get_provider_for_framework("openai-agents")
                  if provider:
                       try:
                            input_text = task.get("text", json.dumps(task))
                            exec_result = await provider.execute_agent(agent_instance, input_text)
                            if exec_result.get("status") == "success":
                                 return exec_result.get("result")
                            else:
                                 raise RuntimeError(f"OpenAI Agent execution failed: {exec_result.get('message')}")
                       except Exception as e:
                            logger.error(f"Execution failed for OpenAI agent {agent_id}: {e}", exc_info=True)
                            raise RuntimeError(f"OpenAI Agent execution failed: {str(e)}")
                  else:
                       raise RuntimeError(f"OpenAI provider not found for agent {agent_id}.")
             else:
                  raise RuntimeError(f"AgentFactory not available to execute OpenAI agent {agent_id}.")

        # Fallback to System Agent if no other method works? (Risky - might lead to loops)
        # Let's disable this for now to enforce explicit execution methods.
        # if self.system_agent:
        #     logger.warning(f"No standard execution method for agent {agent_id}, falling back to SystemAgent.")
        #     prompt = self._create_agent_prompt(agent_record, task)
        #     # Make sure the system_agent.run call returns something useful
        #     sys_agent_result = await self.system_agent.run(prompt)
        #     # Extract result from system agent's response (this might need specific logic)
        #     if hasattr(sys_agent_result, 'result') and hasattr(sys_agent_result.result, 'text'):
        #         return sys_agent_result.result.text
        #     return str(sys_agent_result) # Fallback

        raise RuntimeError(f"No execution method available for agent {agent_id} (type: {type(agent_instance)})")


    def _create_agent_prompt(
        self,
        agent: Dict[str, Any],
        task: Dict[str, Any]
    ) -> str:
        """Create execution prompt for fallback system agent execution."""
        # This is less likely to be used now with direct instance execution
        return (
            f"Act as the agent '{agent['name']}' ({agent['type']}) described as: '{agent['description']}'.\n"
            f"Capabilities: {', '.join(c['name'] for c in agent['capabilities'])}\n\n"
            f"Perform the following task based on your capabilities:\n"
            f"Task Details: {json.dumps(task, indent=2)}\n\n"
            "Provide the result of performing this task."
        )

    # --- System Bus Helper Methods ---

    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """
        Get detailed status of an agent (System Bus Operation).
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent not found: {agent_id}")

        agent_data = self.agents[agent_id]
        status = {
            **agent_data,
            "health_status": "unhealthy" if self._check_circuit_breaker(agent_id) else "healthy",
            "is_instance_loaded": agent_id in self._agent_instances
        }

        if agent_id in self.circuit_breakers:
            cb = self.circuit_breakers[agent_id]
            status.update({
                "failure_count": cb["failures"],
                "last_failure_timestamp": cb["last_failure"],
                "seconds_since_last_failure": time.time() - cb["last_failure"]
            })
        else:
             status.update({
                "failure_count": 0,
                "last_failure_timestamp": None,
             })


        return status

    async def health_check(self) -> Dict[str, Any]:
        """
        Get system health status (System Bus Operation).
        """
        unhealthy_count = sum(1 for agent_id in self.agents if self._check_circuit_breaker(agent_id))
        overall_status = "unhealthy" if unhealthy_count > 0 else "healthy"

        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "stats": {
                "total_agents": len(self.agents),
                "active_agents": sum(1 for a in self.agents.values() if a["status"] == "active"),
                "loaded_instances": len(self._agent_instances),
                "unhealthy_agents": unhealthy_count,
            },
            "storage": {
                "registry_path": self.registry_path,
                "chromadb_path": str(Path(self.chroma_client._path).resolve()), # Use _path attribute
                "log_path": self.log_path
            },
            "dependencies": { # Basic check for core dependencies
                 "llm_service": "available" if self.llm_service else "missing",
                 "smart_library": "available" if self.smart_library else "missing",
                 "system_agent": "available" if self.system_agent else "not_set_or_missing",
                 "agent_factory": "available" if self.container and self.container.has('agent_factory') else "missing_or_not_in_container"
            }
        }

    async def clear_logs(self) -> None:
        """Clear all execution logs (System Bus Operation)."""
        try:
            if Path(self.log_path).exists():
                Path(self.log_path).unlink()
                logger.info(f"Cleared all execution logs from {self.log_path}")
        except Exception as e:
            logger.error(f"Failed to clear logs: {str(e)}")

    async def list_all_agents(self, agent_type: Optional[str] = None) -> List[Dict[str, Any]]:
         """List all registered agents, optionally filtered by type (System Bus Operation)."""
         agents_list = list(self.agents.values())
         if agent_type:
              agents_list = [a for a in agents_list if a.get("type") == agent_type]
         # Return essential info, not the full record necessarily
         return [
              {"id": a["id"], "name": a["name"], "type": a["type"], "status": a["status"], "description": a.get("description", "")[:100]+"..."}
              for a in agents_list
         ]