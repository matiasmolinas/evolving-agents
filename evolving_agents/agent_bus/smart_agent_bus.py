# evolving_agents/agent_bus/smart_agent_bus.py

import os
import json
import logging
import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Union

import pymongo
import motor.motor_asyncio

from evolving_agents.core.llm_service import LLMService
from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.core.dependency_container import DependencyContainer
from evolving_agents.core.base import IAgent
from evolving_agents.core.mongodb_client import MongoDBClient

logger = logging.getLogger(__name__)

DEFAULT_AGENT_EMBEDDING_DIM = 1536

class SmartAgentBus:
    def __init__(
        self,
        smart_library: Optional[SmartLibrary] = None,
        system_agent: Optional[IAgent] = None,
        llm_service: Optional[LLMService] = None,
        container: Optional[DependencyContainer] = None,
        mongodb_client: Optional[MongoDBClient] = None, # Added mongodb_client as an explicit parameter
        mongodb_uri: Optional[str] = None, # Kept for fallback if mongodb_client and container are None
        mongodb_db_name: Optional[str] = None, # Kept for fallback
        registry_collection_name: str = "eat_agent_registry",
        logs_collection_name: str = "eat_agent_bus_logs",
        circuit_breaker_path: str = "agent_bus_circuit_breakers.json"
    ):
        self.container = container
        self._system_agent_instance = system_agent
        self._initialized = False
        self._agent_instances: Dict[str, Any] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self.circuit_breaker_path = circuit_breaker_path

        # Resolve core dependencies: SmartLibrary and LLMService
        if container:
            self.smart_library = smart_library or container.get('smart_library', None)
            self.llm_service = llm_service or container.get('llm_service', None)
            if not self.llm_service: # If still None, create a default one
                self.llm_service = LLMService(container=container) # LLMService can use container to get MongoDBClient for its cache
                container.register('llm_service', self.llm_service)
        else: # No container
            self.smart_library = smart_library
            self.llm_service = llm_service or LLMService() # Create default if not passed

        # Resolve MongoDBClient:
        # Priority: 1. Direct parameter, 2. From container, 3. Create new
        if mongodb_client is not None:
            self.mongodb_client = mongodb_client
            logger.debug("SmartAgentBus: Using directly passed MongoDBClient.")
        elif container and container.has('mongodb_client'):
            self.mongodb_client = container.get('mongodb_client')
            logger.debug("SmartAgentBus: Using MongoDBClient from container.")
        else:
            logger.warning("SmartAgentBus: MongoDBClient not passed directly or found in container. Creating new instance.")
            # MongoDBClient's __init__ will use os.getenv for URI/DB if not provided
            self.mongodb_client = MongoDBClient(uri=mongodb_uri, db_name=mongodb_db_name)
            if container: # If container exists, register the new client
                container.register('mongodb_client', self.mongodb_client)
        
        if not isinstance(self.mongodb_client.client, motor.motor_asyncio.AsyncIOMotorClient):
            logger.critical("SmartAgentBus: MongoDBClient is NOT using an AsyncIOMotorClient (Motor). Async DB ops will fail.")
            # Consider raising TypeError here to stop initialization if Motor is a hard requirement.

        self.registry_collection_name = registry_collection_name
        self.logs_collection_name = logs_collection_name
        self.registry_collection: motor.motor_asyncio.AsyncIOMotorCollection = self.mongodb_client.get_collection(self.registry_collection_name)
        self.logs_collection: motor.motor_asyncio.AsyncIOMotorCollection = self.mongodb_client.get_collection(self.logs_collection_name)

        self.agents: Dict[str, Dict[str, Any]] = {} # In-memory cache
        
        asyncio.create_task(self._ensure_registry_indexes_and_load())
        self._load_circuit_breakers()

        if container and not container.has('agent_bus'):
            container.register('agent_bus', self)

        logger.info(f"SmartAgentBus initialized with MongoDB collections: '{registry_collection_name}', '{logs_collection_name}'")

    # ... (rest of the SmartAgentBus class remains the same as your last provided version)
    @property
    def system_agent(self) -> Optional[IAgent]:
        if not self._system_agent_instance and self.container and self.container.has('system_agent'):
             self._system_agent_instance = self.container.get('system_agent')
        return self._system_agent_instance

    async def _ensure_registry_indexes_and_load(self):
        """Ensure MongoDB indexes for the agent registry and load data into memory."""
        if self.registry_collection is None:
            logger.error(f"Cannot ensure indexes: registry_collection '{self.registry_collection_name}' is None.")
            return
        try:
            await self.registry_collection.create_index([("id", pymongo.ASCENDING)], unique=True, background=True)
            await self.registry_collection.create_index([("name", pymongo.ASCENDING)], background=True)
            await self.registry_collection.create_index([("status", pymongo.ASCENDING)], background=True)
            await self.registry_collection.create_index([("type", pymongo.ASCENDING)], background=True)
            await self.registry_collection.create_index([("capabilities.id", pymongo.ASCENDING)], background=True)
            await self.registry_collection.create_index([("description", pymongo.TEXT)], name="description_text_index", background=True)

            logger.info(f"Ensured standard indexes on '{self.registry_collection_name}'.")
            logger.info("For vector search on agent/capability descriptions, ensure corresponding Atlas Vector Search indexes "
                        "('vector_index_agent_description', 'vector_index_capability_description') are configured.")
        except Exception as e:
            logger.error(f"Error creating MongoDB indexes for {self.registry_collection_name}: {e}", exc_info=True)
        await self._load_registry_from_db()


    async def _load_registry_from_db(self) -> None:
        """Load all agent records from MongoDB into the in-memory self.agents dictionary."""
        logger.info(f"Loading agent registry from MongoDB collection '{self.registry_collection_name}'...")
        self.agents = {}
        if self.registry_collection is None:
            logger.error("Registry collection is not initialized. Cannot load from DB.")
            return
        try:
            projection = {"_id": 0, "description_embedding": 0, "capabilities.description_embedding": 0}
            cursor = self.registry_collection.find({}, projection)
            self.agents = {agent_doc["id"]: agent_doc async for agent_doc in cursor}
            logger.info(f"Loaded {len(self.agents)} agents from MongoDB into in-memory cache.")
        except Exception as e:
            logger.error(f"Error loading agent registry from MongoDB: {e}", exc_info=True)


    def _load_circuit_breakers(self) -> None:
        if os.path.exists(self.circuit_breaker_path):
            try:
                with open(self.circuit_breaker_path, 'r') as f: self.circuit_breakers = json.load(f)
                logger.info(f"Loaded {len(self.circuit_breakers)} circuit breaker states from {self.circuit_breaker_path}")
            except Exception as e:
                logger.error(f"Error loading circuit breakers: {e}"); self.circuit_breakers = {}

    async def _save_circuit_breakers(self) -> None:
        try:
            with open(self.circuit_breaker_path, 'w') as f: json.dump(self.circuit_breakers, f, indent=2)
        except Exception as e: logger.error(f"Error saving circuit breakers: {e}")


    async def initialize_from_library(self) -> None:
        if self._initialized: return
        if not self.smart_library:
            logger.warning("SmartLibrary not provided; cannot initialize AgentBus from library."); return

        agent_records = await self.smart_library.export_records(record_type="AGENT")
        tool_records = await self.smart_library.export_records(record_type="TOOL")
        library_records = agent_records + tool_records

        initialized_count = 0
        for record in library_records:
            if record.get("status") != "active": continue
            if record["id"] in self.agents: continue 

            capabilities = record.get("metadata", {}).get("capabilities", [])
            if not capabilities and record.get("name"):
                cap_id = f"{record['name'].lower().replace(' ', '_')}_default_capability"
                capabilities = [{"id": cap_id, "name": record["name"], "description": record.get("description", f"Default for {record['name']}"), "confidence": 0.7}]

            agent_instance = None
            if self.container:
                agent_factory = self.container.get('agent_factory', None)
                tool_factory = self.container.get('tool_factory', None)
                try:
                    if record["record_type"] == "AGENT" and agent_factory:
                        agent_instance = await agent_factory.create_agent(record)
                    elif record["record_type"] == "TOOL" and tool_factory:
                        agent_instance = await tool_factory.create_tool(record)
                except Exception as e: logger.error(f"Instance creation failed for {record.get('name')}: {e}", exc_info=True)

            await self.register_agent(
                agent_id=record["id"], name=record["name"], description=record.get("description", ""),
                capabilities=capabilities, agent_type=record.get("record_type", "GENERIC"),
                metadata={"source": "SmartLibrary", **record.get("metadata", {})},
                agent_instance=agent_instance,
                embed_capabilities=True 
            )
            initialized_count += 1
        self._initialized = True
        logger.info(f"AgentBus synced {initialized_count} new components from SmartLibrary into MongoDB registry.")

    async def _log_agent_execution(
        self, bus_type: str, agent_id: str, task: Dict[str, Any],
        result: Optional[Dict[str, Any]] = None, error: Optional[str] = None,
        duration: Optional[float] = None
    ) -> None:
        if self.logs_collection is None: logger.warning("Logs collection not available. Skipping log."); return
        agent_name = self.agents.get(agent_id, {}).get("name", "N/A") if agent_id != "N/A" else "N/A"

        log_entry = {
            "log_id": f"log_{uuid.uuid4().hex[:12]}",
            "timestamp": datetime.now(timezone.utc), 
            "bus_type": bus_type, "agent_id": agent_id, "agent_name": agent_name,
            "task_description": task.get("capability") or task.get("operation", "direct_execution"),
            "task_details": task, "result": result, "error": error,
            "duration_ms": int(duration * 1000) if duration is not None else None,
        }
        try:
            await self.logs_collection.insert_one(log_entry)
        except Exception as e: logger.error(f"Failed to write execution log to MongoDB: {e}", exc_info=True)

    def _check_circuit_breaker(self, agent_id: str) -> bool:
        if agent_id not in self.circuit_breakers: return False
        cb = self.circuit_breakers[agent_id]
        if cb.get("failures", 0) >= 3 and (time.time() - cb.get("last_failure", 0)) < 300:
            logger.warning(f"Circuit breaker tripped for agent {agent_id}")
            return True
        return False

    async def _record_failure(self, agent_id: str) -> None:
        if agent_id not in self.circuit_breakers: self.circuit_breakers[agent_id] = {"failures": 0, "last_failure": 0.0}
        self.circuit_breakers[agent_id]["failures"] = self.circuit_breakers[agent_id].get("failures", 0) + 1
        self.circuit_breakers[agent_id]["last_failure"] = time.time()
        await self._save_circuit_breakers()
        logger.warning(f"Failure recorded for agent {agent_id}. Count: {self.circuit_breakers[agent_id]['failures']}")

    async def _reset_circuit_breaker(self, agent_id: str) -> None:
        if agent_id in self.circuit_breakers:
            logger.info(f"Resetting circuit breaker for agent {agent_id}")
            del self.circuit_breakers[agent_id]
            await self._save_circuit_breakers()

    async def register_agent(
        self, name: str, description: str, capabilities: List[Dict[str, Any]],
        agent_type: str = "GENERIC", metadata: Optional[Dict[str, Any]] = None,
        agent_instance=None, agent_id: Optional[str] = None,
        embed_capabilities: bool = False
    ) -> str:
        if self.registry_collection is None:
            logger.error("Registry collection not available. Cannot register agent.")
            raise RuntimeError("Agent registry (MongoDB) is not available.")

        if not agent_id: agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        
        agent_desc_for_embedding = f"Agent Name: {name}\nType: {agent_type}\nDescription: {description}\nCapabilities: {', '.join(c.get('name', c.get('id','')) for c in capabilities)}"
        try:
            description_embedding = await self.llm_service.embed(agent_desc_for_embedding)
        except Exception as e:
            logger.error(f"Embedding failed for agent {name}: {e}"); description_embedding = [0.0] * DEFAULT_AGENT_EMBEDDING_DIM

        processed_capabilities = []
        for cap_data in capabilities:
            cap_id = cap_data.get("id", f"{cap_data.get('name', 'cap').lower().replace(' ', '_')}_{uuid.uuid4().hex[:4]}")
            cap = {"id": cap_id, "name": cap_data.get("name", cap_id), 
                   "description": cap_data.get("description", ""), "confidence": cap_data.get("confidence", 0.8)}
            if embed_capabilities and cap["description"]:
                try: cap["description_embedding"] = await self.llm_service.embed(cap["description"])
                except Exception as e: logger.error(f"Embedding cap '{cap['name']}' failed: {e}"); cap["description_embedding"] = [0.0]*DEFAULT_AGENT_EMBEDDING_DIM
            processed_capabilities.append(cap)
        
        current_time = datetime.now(timezone.utc)
        existing_doc = await self.registry_collection.find_one({"id": agent_id}, {"registered_at": 1})
        registered_at = existing_doc.get("registered_at", current_time) if existing_doc else current_time

        agent_doc = {
            "id": agent_id, "name": name, "description": description, "type": agent_type,
            "capabilities": processed_capabilities, "metadata": metadata or {}, "status": "active",
            "description_embedding": description_embedding,
            "registered_at": registered_at, "last_updated": current_time, "last_seen": current_time
        }
        try:
            result = await self.registry_collection.replace_one({"id": agent_id}, agent_doc, upsert=True)
            op_type = "Registered new" if result.upserted_id else "Updated" if result.modified_count > 0 else "Refreshed"
            
            cached_doc = agent_doc.copy()
            cached_doc.pop("description_embedding", None)
            for cap in cached_doc.get("capabilities", []): cap.pop("description_embedding", None)
            self.agents[agent_id] = cached_doc

            if agent_instance: self._agent_instances[agent_id] = agent_instance
            
            await self._log_agent_execution('system', agent_id, {'operation': 'register', 'details': {'name': name, 'type': agent_type}}, {'status': 'success', 'type': op_type})
            logger.info(f"{op_type} agent: {name} ({agent_id}) in MongoDB registry.")
            return agent_id
        except Exception as e: logger.error(f"MongoDB registration error for {name}: {e}", exc_info=True); raise

    async def discover_agents(
        self, task_description: Optional[str] = None, capability_id: Optional[str] = None,
        min_confidence: float = 0.6, agent_type: Optional[str] = None, limit: int = 5
    ) -> List[Dict[str, Any]]:
        if self.registry_collection is None: logger.error("Registry collection unavailable."); return []
        if not task_description and not capability_id:
             raise ValueError("Either task_description or capability_id must be provided.")

        matches: List[Dict[str, Any]] = []
        query_filter: Dict[str, Any] = {"status": "active"}
        if agent_type: query_filter["type"] = agent_type
        
        projection = {"_id": 0, "description_embedding": 0} 
        for cap_field_to_exclude in ["capabilities.description_embedding"]:
            projection[cap_field_to_exclude] = 0

        if capability_id:
            logger.debug(f"Discovering by capability_id '{capability_id}' via metadata query.")
            query_filter["capabilities.id"] = capability_id
            cursor = self.registry_collection.find(query_filter, projection).limit(limit * 2)
            async for agent_doc in cursor:
                cap_conf = next((c.get("confidence",0.0) for c in agent_doc.get("capabilities",[]) if c.get("id")==capability_id), 0.0)
                if cap_conf >= min_confidence:
                    is_healthy = not self._check_circuit_breaker(agent_doc["id"])
                    if is_healthy:
                        matches.append({**agent_doc, "similarity_score": cap_conf, "is_healthy": True})
            matches.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)

        elif task_description:
            logger.debug(f"Discovering by task_description '{task_description[:50]}...' using vector_index_agent_description.")
            try:
                query_embedding = await self.llm_service.embed(task_description)
                search_filter_conditions = [{"text": {"path": "status", "query": "active"}}]
                if agent_type: search_filter_conditions.append({"text": {"path": "type", "query": agent_type}})

                pipeline = [ { "$search": {
                                "index": "vector_index_agent_description",
                                "vectorSearch": { "path": "description_embedding", "queryVector": query_embedding,
                                                  "numCandidates": limit * 15, "limit": limit * 3 },
                                "filter": {"compound": {"must": search_filter_conditions}} if search_filter_conditions else {}
                              }},
                            {"$addFields": {"similarity_score": {"$meta": "searchScore"}}},
                            {"$project": projection} ]
                cursor = self.registry_collection.aggregate(pipeline)
                async for agent_doc in cursor:
                    if agent_doc.get("similarity_score", 0.0) >= min_confidence:
                        is_healthy = not self._check_circuit_breaker(agent_doc["id"])
                        if is_healthy: matches.append({**agent_doc, "is_healthy": True})
                matches.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
            except Exception as e:
                logger.error(f"Vector search failed: {e}. Falling back to text search.", exc_info=True)
                text_search_filter = query_filter.copy()
                text_search_filter["$text"] = {"$search": task_description}
                cursor = self.registry_collection.find(text_search_filter, projection).limit(limit)
                async for agent_doc in cursor:
                    is_healthy = not self._check_circuit_breaker(agent_doc["id"])
                    if is_healthy: matches.append({**agent_doc, "similarity_score": 0.5, "is_healthy": True}) # Neutral score
                matches.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True) # Still sort even for fallback
            
        return matches[:limit]

    async def request_capability(
        self, capability: str, content: Dict[str, Any], min_confidence: float = 0.6,
        specific_agent_id: Optional[str] = None, timeout: float = 60.0
    ) -> Dict[str, Any]:
        logger.info(f"Received data bus request for capability '{capability}'")
        start_time = time.time()
        agent_to_use_doc = None
        agent_id_to_use = None
        task_details = {"capability": capability, **content}

        if specific_agent_id:
            agent_to_use_doc = self.agents.get(specific_agent_id)
            if not (agent_to_use_doc and agent_to_use_doc.get("status") == "active"):
                # Try DB if not in active cache
                db_doc = await self.registry_collection.find_one({"id": specific_agent_id, "status":"active"}, 
                                                                 {"_id": 0, "description_embedding": 0})
                if db_doc:
                    agent_to_use_doc = db_doc
                    self.agents[specific_agent_id] = db_doc # Cache it
                else:
                    error_msg = f"Targeted agent {specific_agent_id} not found or not active."
                    await self._log_agent_execution('data', specific_agent_id or 'N/A', task_details, error=error_msg, duration=time.time()-start_time)
                    raise ValueError(error_msg)
            agent_id_to_use = specific_agent_id
            logger.debug(f"Targeting specific agent: {agent_id_to_use}")
        else:
            suitable_agents_info = await self.discover_agents(capability_id=capability, min_confidence=min_confidence)
            if not suitable_agents_info: # discover_agents already filters for healthy
                error_msg = f"No healthy agent found for capability '{capability}'."
                await self._log_agent_execution('data', 'N/A', task_details, error=error_msg, duration=time.time()-start_time)
                raise ValueError(error_msg)
            
            best_agent_info = suitable_agents_info[0]
            agent_id_to_use = best_agent_info["id"]
            agent_to_use_doc = self.agents.get(agent_id_to_use) 
            if not agent_to_use_doc: # Fallback if cache is stale
                 agent_to_use_doc = await self.registry_collection.find_one({"id": agent_id_to_use, "status": "active"},
                                                                            {"_id": 0, "description_embedding": 0})
                 if not agent_to_use_doc: raise RuntimeError(f"Discovered agent {agent_id_to_use} could not be retrieved.")
                 self.agents[agent_id_to_use] = agent_to_use_doc
            logger.debug(f"Discovered agent {agent_id_to_use} for capability '{capability}'.")

        if not agent_to_use_doc: raise RuntimeError("Internal error: Agent selected but document not found.")

        if self._check_circuit_breaker(agent_id_to_use):
             error_msg = f"Agent {agent_id_to_use} ({agent_to_use_doc.get('name')}) is temporarily unavailable (circuit breaker tripped)."
             await self._log_agent_execution('data', agent_id_to_use, task_details, error=error_msg, duration=time.time()-start_time)
             raise RuntimeError(error_msg)

        try:
            result_data = await asyncio.wait_for(
                self._execute_agent_task(agent_to_use_doc, task=content), timeout=timeout
            )
            await self._reset_circuit_breaker(agent_id_to_use)
            
            current_time = datetime.now(timezone.utc)
            await self.registry_collection.update_one({"id": agent_id_to_use}, {"$set": {"last_seen": current_time}})
            if agent_id_to_use in self.agents: self.agents[agent_id_to_use]["last_seen"] = current_time.isoformat()

            duration = time.time() - start_time
            response = {"status": "success", "agent_id": agent_id_to_use, "agent_name": agent_to_use_doc.get("name"),
                        "capability_executed": capability, "content": result_data,
                        "metadata": {"processing_time_ms": int(duration * 1000)}}
            await self._log_agent_execution('data', agent_id_to_use, task_details, result=response, duration=duration)
            logger.info(f"Capability '{capability}' executed by {agent_id_to_use} successfully.")
            return response
        except asyncio.TimeoutError:
            await self._record_failure(agent_id_to_use)
            error_msg = f"Agent {agent_id_to_use} timed out for capability '{capability}' after {timeout}s."
            await self._log_agent_execution('data', agent_id_to_use, task_details, error=error_msg, duration=timeout)
            raise RuntimeError(error_msg)
        except Exception as e:
            await self._record_failure(agent_id_to_use)
            error_msg = f"Agent {agent_id_to_use} failed capability '{capability}': {str(e)}"
            await self._log_agent_execution('data', agent_id_to_use, task_details, error=error_msg, duration=time.time()-start_time)
            logger.error(f"Capability execution error for {agent_id_to_use}: {e}", exc_info=True)
            raise RuntimeError(error_msg)


    async def execute_with_agent(
        self, agent_id: str, task: Dict[str, Any], timeout: float = 30.0
    ) -> Dict[str, Any]:
        logger.info(f"Received direct execution request for agent {agent_id}")
        start_time = time.time()
        
        agent_doc = self.agents.get(agent_id)
        if not agent_doc:
            agent_doc_db = await self.registry_collection.find_one({"id": agent_id}, {"_id":0, "description_embedding":0})
            if agent_doc_db: self.agents[agent_id] = agent_doc_db; agent_doc = agent_doc_db
        
        if not agent_doc: raise ValueError(f"Agent not found: {agent_id}")
        if agent_doc.get("status") != "active": raise ValueError(f"Agent {agent_id} is not active.")

        if self._check_circuit_breaker(agent_id):
            error_msg = f"Agent {agent_id} ({agent_doc.get('name')}) temporarily unavailable (circuit breaker)."
            await self._log_agent_execution('system', agent_id, task, error=error_msg, duration=time.time()-start_time)
            raise RuntimeError(error_msg)
        try:
            result_data = await asyncio.wait_for(self._execute_agent_task(agent_doc, task), timeout=timeout)
            await self._reset_circuit_breaker(agent_id)
            
            current_time = datetime.now(timezone.utc)
            await self.registry_collection.update_one({"id": agent_id}, {"$set": {"last_seen": current_time}})
            if agent_id in self.agents: self.agents[agent_id]["last_seen"] = current_time.isoformat()
            
            duration = time.time() - start_time
            response = {"result": result_data, "metadata": {"agent_id": agent_id, "agent_name": agent_doc.get("name"),
                                                           "processing_time_ms": int(duration*1000), "success":True}}
            await self._log_agent_execution('system', agent_id, task, result=response, duration=duration)
            return response
        except asyncio.TimeoutError:
            await self._record_failure(agent_id)
            error_msg = f"Agent {agent_id} timed out (direct execution) after {timeout}s."
            await self._log_agent_execution('system', agent_id, task, error=error_msg, duration=timeout)
            raise RuntimeError(error_msg)
        except Exception as e:
            await self._record_failure(agent_id)
            error_msg = f"Agent {agent_id} failed (direct execution): {str(e)}"
            await self._log_agent_execution('system', agent_id, task, error=error_msg, duration=time.time()-start_time)
            logger.error(f"Direct agent execution error for {agent_id}: {e}", exc_info=True)
            raise RuntimeError(error_msg)


    async def _execute_agent_task(self, agent_record_doc: Dict[str, Any], task: Dict[str, Any]) -> Any:
        agent_id = agent_record_doc["id"]
        agent_instance = self._agent_instances.get(agent_id)

        if not agent_instance:
            logger.warning(f"Live instance for agent {agent_id} not found, attempting dynamic creation...")
            if self.container:
                agent_factory = self.container.get('agent_factory', None)
                tool_factory = self.container.get('tool_factory', None)
                created = False
                if agent_factory and agent_record_doc.get("type") == "AGENT":
                    try: agent_instance = await agent_factory.create_agent(agent_record_doc); created=True
                    except Exception as e: logger.error(f"Dynamic AGENT creation failed for {agent_id}: {e}", exc_info=True)
                elif tool_factory and agent_record_doc.get("type") == "TOOL":
                    try: agent_instance = await tool_factory.create_tool(agent_record_doc); created=True
                    except Exception as e: logger.error(f"Dynamic TOOL creation failed for {agent_id}: {e}", exc_info=True)
                
                if created and agent_instance:
                    self._agent_instances[agent_id] = agent_instance
                    logger.info(f"Dynamically created and stored instance for {agent_id}")
                elif not created: # If no factory was applicable
                     raise RuntimeError(f"No suitable factory (agent or tool) found for agent {agent_id} type {agent_record_doc.get('type')}")
                else: # Factory was applicable but creation failed
                    raise RuntimeError(f"Could not create live instance for agent {agent_id} (type: {agent_record_doc.get('type')}).")
            else:
                raise RuntimeError(f"No live instance for agent {agent_id} and no DependencyContainer available for factories.")

        if hasattr(agent_instance, 'run') and callable(agent_instance.run):
            try:
                input_prompt = task.get("text", json.dumps(task) if isinstance(task, dict) else str(task))
                run_result = await agent_instance.run(input_prompt)
                if hasattr(run_result, 'result') and hasattr(run_result.result, 'text'): return run_result.result.text
                elif hasattr(run_result, 'text'): return run_result.text
                elif isinstance(run_result, str): return run_result
                try: return json.loads(json.dumps(run_result))
                except: return str(run_result)
            except Exception as e: logger.error(f"Execution failed for agent {agent_id} via run: {e}", exc_info=True); raise RuntimeError(f"Run method failed: {e}")
        elif agent_record_doc.get("metadata", {}).get("framework") == "openai-agents":
            if self.container and self.container.has('agent_factory'):
                provider = self.container.get('agent_factory').provider_registry.get_provider_for_framework("openai-agents")
                if provider:
                    try:
                        input_text = task.get("text", json.dumps(task) if isinstance(task, dict) else str(task))
                        exec_result = await provider.execute_agent(agent_instance, input_text)
                        if exec_result.get("status") == "success": return exec_result.get("result")
                        else: raise RuntimeError(f"OpenAI Agent execution failed: {exec_result.get('message')}")
                    except Exception as e: logger.error(f"OpenAI agent {agent_id} exec failed: {e}", exc_info=True); raise RuntimeError(f"OpenAI exec failed: {e}")
                else: raise RuntimeError(f"OpenAI provider not found for {agent_id}.")
            else: raise RuntimeError(f"AgentFactory not available for OpenAI agent {agent_id}.")
        raise RuntimeError(f"No execution method for agent {agent_id} (type: {type(agent_instance)})")

    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        agent_data = self.agents.get(agent_id)
        if not agent_data:
            agent_data_db = await self.registry_collection.find_one({"id": agent_id}, {"_id": 0, "description_embedding": 0})
            if not agent_data_db: raise ValueError(f"Agent {agent_id} not found.")
            self.agents[agent_id] = agent_data_db 
            agent_data = agent_data_db
        
        status_dict = { **agent_data, "health_status": "healthy" if not self._check_circuit_breaker(agent_id) else "unhealthy",
                       "is_instance_loaded": agent_id in self._agent_instances }
        if agent_id in self.circuit_breakers: status_dict.update(self.circuit_breakers[agent_id])
        else: status_dict.update({"failure_count": 0, "last_failure_timestamp": None})
        return status_dict

    async def health_check(self) -> Dict[str, Any]:
        db_ping_ok = False
        if self.mongodb_client: db_ping_ok = await self.mongodb_client.ping_server()
        total_agents_in_cache = len(self.agents)
        active_agents_in_cache = sum(1 for a in self.agents.values() if a.get("status") == "active")
        unhealthy_agents_count = sum(1 for agent_id in self.agents if self._check_circuit_breaker(agent_id))
        
        return {"status": "healthy" if unhealthy_agents_count == 0 and db_ping_ok else "unhealthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "mongodb_connection": "ok" if db_ping_ok else "failed_ping",
                "stats": { "total_agents_in_registry_cache": total_agents_in_cache,
                           "active_agents_in_registry_cache": active_agents_in_cache,
                           "loaded_live_instances": len(self._agent_instances),
                           "agents_in_circuit_breaker": unhealthy_agents_count, },
                "storage_config": { "registry_collection": self.registry_collection_name,
                                   "logs_collection": self.logs_collection_name,
                                   "mongodb_database": self.mongodb_client.db_name if self.mongodb_client else "N/A",
                                   "circuit_breaker_file": self.circuit_breaker_path },
                "dependencies": {"llm_service": "ok" if self.llm_service else "N/A",
                                 "smart_library": "ok" if self.smart_library else "N/A",
                                 "system_agent": "ok" if self.system_agent else "N/A",
                                 "agent_factory": "ok" if self.container and self.container.has('agent_factory') else "N/A"}}

    async def clear_logs(self) -> None:
        if self.logs_collection is None: logger.warning("Logs collection unavailable."); return
        try:
            result = await self.logs_collection.delete_many({})
            logger.info(f"Cleared {result.deleted_count} execution logs from MongoDB '{self.logs_collection_name}'.")
        except Exception as e: logger.error(f"Failed to clear logs from MongoDB: {e}", exc_info=True)

    async def list_all_agents(self, agent_type: Optional[str] = None) -> List[Dict[str, Any]]:
         agents_list = list(self.agents.values())
         if agent_type: agents_list = [a for a in agents_list if a.get("type") == agent_type]
         return [{"id": a["id"], "name": a.get("name"), "type": a.get("type"), "status": a.get("status"),
                  "description_snippet": (a.get("description","")[:70]+"...") if a.get("description") else ""}
                 for a in agents_list]