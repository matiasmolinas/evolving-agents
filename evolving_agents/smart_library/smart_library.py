# evolving_agents/smart_library/smart_library.py

import os
import json # Still used for logging/debug, not primary storage
import logging
import uuid
import re
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pymongo # For index creation constants
import motor.motor_asyncio # Import Motor for async MongoDB operations

from evolving_agents.core.llm_service import LLMService
from evolving_agents.core.mongodb_client import MongoDBClient # Assuming this is created

# Import DependencyContainer using TYPE_CHECKING to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from evolving_agents.core.dependency_container import DependencyContainer

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_DIMENSION = 1536


class SmartLibrary:
    """
    Unified library that stores all agents, tools, and firmware as dictionary records
    in MongoDB, with MongoDB Atlas Vector Search for semantic search using dual embeddings.
    """
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        container: Optional["DependencyContainer"] = None,
        mongodb_uri: Optional[str] = None,
        mongodb_db_name: Optional[str] = None,
        components_collection_name: str = "eat_components"
    ):
        self.container = container
        self._initialized = False

        # Initialize LLM service
        if container and container.has('llm_service'):
            self.llm_service = llm_service or container.get('llm_service')
        elif llm_service:
            self.llm_service = llm_service
        else:
            # LLMService constructor needs to handle resolving MongoDBClient if its cache uses it
            # and container might or might not have 'mongodb_client' yet.
            # This implies LLMService's default creation of MongoDBClient (if needed for cache)
            # must be robust or we ensure mongodb_client is always available before LLMService.
            # For now, assuming LLMService or container handles this.
            self.llm_service = LLMService(container=container) # LLMService will resolve its own deps
            if container: container.register('llm_service', self.llm_service)


        # Initialize MongoDB client
        if container and container.has('mongodb_client'):
            self.mongodb_client: MongoDBClient = container.get('mongodb_client')
        elif mongodb_client:
             self.mongodb_client = mongodb_client
        else:
            self.mongodb_client = MongoDBClient(uri=mongodb_uri, db_name=mongodb_db_name)
            if container:
                container.register('mongodb_client', self.mongodb_client)
        
        # Type check for Motor client instance
        if not isinstance(self.mongodb_client.client, motor.motor_asyncio.AsyncIOMotorClient):
            logger.critical("MongoDBClient is NOT using an AsyncIOMotorClient (Motor). "
                           "SmartLibrary WILL FAIL with async database operations. Ensure MongoDBClient is correctly implemented with Motor.")
            # Potentially raise an error here or allow to proceed with warnings if some sync fallback exists (not ideal)
            # For now, it will likely fail later if not using Motor.

        self.components_collection_name = components_collection_name
        # Explicitly type hint for clarity with Motor
        self.components_collection: motor.motor_asyncio.AsyncIOMotorCollection = self.mongodb_client.get_collection(self.components_collection_name)

        logger.info(f"SmartLibrary initialized with MongoDB collection: '{self.components_collection_name}'")
        asyncio.create_task(self._ensure_indexes())

        if container and not container.has('smart_library'):
            container.register('smart_library', self)

    async def _ensure_indexes(self):
        """Ensure standard MongoDB indexes are created. Vector indexes are managed in Atlas."""
        # CORRECTED CHECK:
        if self.components_collection is None:
            logger.error(f"Cannot ensure indexes: components_collection '{self.components_collection_name}' is None. MongoDBClient might have failed to initialize or get the collection.")
            return
        try:
            await self.components_collection.create_index([("id", pymongo.ASCENDING)], unique=True, background=True)
            await self.components_collection.create_index([("name", pymongo.ASCENDING)], background=True)
            await self.components_collection.create_index([("record_type", pymongo.ASCENDING)], background=True)
            await self.components_collection.create_index([("domain", pymongo.ASCENDING)], background=True)
            await self.components_collection.create_index([("tags", pymongo.ASCENDING)], background=True) # Indexing array field for $in queries
            await self.components_collection.create_index([("status", pymongo.ASCENDING)], background=True)
            await self.components_collection.create_index([("last_updated", pymongo.DESCENDING)], background=True)
            logger.info(f"Ensured standard indexes on '{self.components_collection_name}' collection.")
            logger.info("Reminder: Vector Search Indexes for 'content_embedding' and 'applicability_embedding' must be configured manually in MongoDB Atlas or via its API.")
        except Exception as e:
            logger.error(f"Error creating MongoDB indexes for {self.components_collection_name}: {e}", exc_info=True)

    async def initialize(self):
        if self._initialized: return
        if self.container and self.container.has('agent_bus'):
            agent_bus = self.container.get('agent_bus')
            if hasattr(agent_bus, 'initialize_from_library'):
                await agent_bus.initialize_from_library()
        self._initialized = True
        logger.info("SmartLibrary fully initialized and integrated.")

    def _get_record_vector_text(self, record: Dict[str, Any]) -> str:
        name = record.get("name", "")
        description = record.get("description", "")
        record_type = record.get("record_type", "")
        domain = record.get("domain", "")
        tags = " ".join(record.get("tags", []))
        code_snippet = record.get("code_snippet", "")
        usage_count = record.get("usage_count", 0)
        success_count = record.get("success_count", 0)
        usage_stats = f"Used {usage_count} times with {((success_count / usage_count) if usage_count > 0 else 0):.0%} success." if usage_count > 0 else ""
        
        interface_info = ""
        if record_type == "TOOL":
            # Simplified regex, assuming standard Python function/class defs
            signatures = re.findall(r'def\s+(\w+\([^)]*\))', code_snippet)
            if signatures: interface_info += "Functions: " + ", ".join(signatures) + ". "
            
            api_endpoints = re.findall(r'@\w+\.(?:route|get|post|put|delete)\([\'"]([^\'"]+)[\'"]', code_snippet) # More generic for web frameworks
            if api_endpoints: interface_info += "API endpoints: " + ", ".join(api_endpoints) + ". "
            
            class_defs = re.findall(r'class\s+(\w+)(?:\([^)]*\))?:', code_snippet)
            if class_defs: interface_info += "Classes: " + ", ".join(class_defs) + ". "
        
        return (
            f"Component Name: {name}. Type: {record_type}. Domain: {domain}. "
            f"Functional Purpose: {description}. {interface_info}"
            f"{('Relevant Tags: ' + tags + '. ') if tags else ''}"
            f"{('Usage Statistics: ' + usage_stats + '. ') if usage_stats else ''}"
            f"Core Implementation Snippet (summary):\n```\n{code_snippet[:1000]}\n```" # Limit snippet for embedding
        )

    async def generate_applicability_text(self, record_data: Dict[str, Any]) -> str:
        name = record_data.get("name", "Unknown Component")
        description = record_data.get("description", "No description")
        record_type = record_data.get("record_type", "UNKNOWN")
        domain = record_data.get("domain", "general")
        code_snippet = record_data.get("code_snippet", "")
        code_summary = code_snippet[:500] + ("..." if len(code_snippet) > 500 else "")
        
        prompt_template = """
        Analyze the following component specification:
        Name: {name}
        Type: {record_type}
        Domain: {domain}
        Description: {description}
        Code Summary:
        ```
        {code_summary}
        ```
        Generate a concise applicability text (T_raz). This text should clearly and succinctly describe:
        1. **Primary Use Cases:** What specific problems, tasks, or scenarios is this component designed to address or solve effectively?
        2. **Target User/System:** Who or what (e.g., backend developer, data pipeline, another AI agent, end-user via an application) would typically use or benefit from this component?
        3. **Key Benefits/Strengths:** What are the 1-3 most significant advantages or standout features of this component for its intended purpose?
        4. **Contextual Relevance:** In what types of projects, development phases, or operational situations would this component be most valuable or appropriate?
        5. **Integration Points/Synergies:** How might this component typically interact with, depend on, or complement other types of tools, agents, or systems?
        Focus on the functional applicability and intended operational context. The output should be a single, coherent paragraph. Output ONLY the generated T_raz description.
        """
        prompt = prompt_template.format(
            name=name, record_type=record_type, domain=domain,
            description=description, code_summary=code_summary
        )
        applicability_text = await self.llm_service.generate(prompt)
        logger.debug(f"Generated T_raz for {name}: {applicability_text[:100]}...")
        return applicability_text.strip()

    async def save_record(self, record: Dict[str, Any]) -> str:
        record_id = record.get("id")
        if not record_id: raise ValueError("Record must have an 'id' to be saved.")
        # Ensure embeddings are Python lists of floats
        for key in ["content_embedding", "applicability_embedding"]:
            if key in record and record[key] is not None:
                # Convert numpy arrays or other iterables to list of floats
                record[key] = [float(x) for x in record[key]]
        try:
            result = await self.components_collection.replace_one({"id": record_id}, record, upsert=True)
            log_action = "Inserted new" if result.upserted_id else "Updated" if result.modified_count > 0 else "Refreshed (no change to)"
            logger.info(f"{log_action} record {record_id} ({record.get('name')}) in MongoDB.")
            return record_id
        except Exception as e:
            logger.error(f"Error saving record {record_id} to MongoDB: {e}", exc_info=True); raise

    async def find_record_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        if self.components_collection is None: return None
        return await self.components_collection.find_one({"id": record_id}, {"_id": 0})

    async def find_record_by_name(self, name: str, record_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if self.components_collection is None: return None
        query = {"name": name}; _ = query.update({"record_type": record_type}) if record_type else None
        return await self.components_collection.find_one(query, {"_id": 0})

    async def find_records_by_domain(self, domain: str, record_type: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.components_collection is None: return []
        query = {"domain": domain}; _ = query.update({"record_type": record_type}) if record_type else None
        cursor = self.components_collection.find(query, {"_id": 0})
        return await cursor.to_list(length=None) # Motor specific

    async def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        if not embedding1 or not embedding2: return 0.0
        # Ensure embeddings are numpy arrays for dot product and norm calculation
        vec1, vec2 = np.array(embedding1, dtype=np.float32), np.array(embedding2, dtype=np.float32)
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0: return 0.0
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return float(np.clip(similarity, -1.0, 1.0)) # Clip to handle potential floating point inaccuracies


    async def semantic_search(
        self, query: str, task_context: Optional[str] = None,
        record_type: Optional[str] = None, domain: Optional[str] = None,
        limit: int = 5, threshold: float = 0.0, task_weight: Optional[float] = 0.7
    ) -> List[Tuple[Dict[str, Any], float, float, float]]:
        if self.components_collection is None:
            logger.error("Semantic search unavailable: components_collection is None.")
            return []
        if query is None: raise ValueError("Query string cannot be None")

        effective_task_weight = task_weight if task_context and task_weight is not None else (0.7 if task_context else 0.0)
        query_embedding_orig = await self.llm_service.embed(query)
        search_pipeline: List[Dict[str, Any]] = []

        vector_search_params: Dict[str, Any] = {
            "queryVector": [], # Will be set below
            "path": "",       # Will be set below
            "numCandidates": limit * 20, # Fetch more for re-ranking, adjust as needed
            "limit": limit * 3           # Limit for the $vectorSearch stage itself
            # "filter": {} # Atlas Search filter, applied during vector search
        }
        
        # Atlas Search $search stage (which contains $vectorSearch) filter
        search_stage_filter_conditions = []
        if record_type: search_stage_filter_conditions.append({"text": {"path": "record_type", "query": record_type}})
        if domain: search_stage_filter_conditions.append({"text": {"path": "domain", "query": domain}})
        # Always filter for active status within the $search stage if possible
        search_stage_filter_conditions.append({"text": {"path": "status", "query": "active"}})


        if task_context:
            query_embedding_raz = await self.llm_service.embed(task_context)
            vector_search_params["index"] = "idx_components_applicability_embedding"
            vector_search_params["path"] = "applicability_embedding"
            vector_search_params["queryVector"] = query_embedding_raz
            primary_score_field = "task_score_raw"
        else:
            vector_search_params["index"] = "idx_components_content_embedding"
            vector_search_params["path"] = "content_embedding"
            vector_search_params["queryVector"] = query_embedding_orig
            primary_score_field = "content_score_raw"

        # Construct the $search stage with $vectorSearch and pre-filtering
        search_stage: Dict[str, Any] = { "$search": { **vector_search_params }}
        if search_stage_filter_conditions:
            search_stage["$search"]["filter"] = {"compound": {"must": search_stage_filter_conditions}}
        
        search_pipeline.append(search_stage)
        # Add score directly from $search, as $vectorSearch is nested
        search_pipeline.append({"$addFields": {primary_score_field: {"$meta": "searchScore"}}})


        # Standard $match filters (applied after $search if needed, but better to filter in $search if possible)
        # If some filters cannot be expressed in Atlas $search's "filter", use $match here.
        # For now, assuming 'status:active' is handled by $search filter.

        fields_to_project = {"_id": 0, "id": 1, "name": 1, "record_type": 1, "domain": 1, "description": 1,
                             "version": 1, "usage_count": 1, "success_count": 1, "tags": 1, "metadata": 1,
                             primary_score_field: 1, "content_embedding": 1, "applicability_embedding": 1} # Always project both embeddings
        search_pipeline.append({"$project": fields_to_project})

        logger.debug(f"MongoDB Aggregation Pipeline for semantic search: {json.dumps(search_pipeline, indent=2)}")
        try:
            candidate_docs_cursor = self.components_collection.aggregate(search_pipeline)
            candidate_docs = await candidate_docs_cursor.to_list(length=None)
        except Exception as e:
            logger.error(f"MongoDB aggregation failed: {e}", exc_info=True)
            if "index not found" in str(e).lower() or "Unknown $vectorSearch index" in str(e):
                 logger.error(f"CRITICAL: Atlas Vector Search index '{vector_search_params['index']}' likely missing or misconfigured on collection '{self.components_collection_name}'. Please create it in Atlas with correct dimensions and path.")
            return []

        search_results_tuples = []
        for doc in candidate_docs:
            # Atlas $meta: "searchScore" for $vectorSearch is typically 0 to 1 (cosine) or higher (euclidean/dotProduct)
            # Assuming cosine, where higher is better. If it's distance, it needs inversion.
            # Let's assume it's already a similarity-like score from Atlas (0 to 1 for cosine).
            
            raw_score = doc.get(primary_score_field, 0.0)

            if task_context: # Primary search was on applicability (E_raz)
                task_score = raw_score
                content_score = await self.compute_similarity(query_embedding_orig, doc.get("content_embedding", []))
            else: # Primary search was on content (E_orig)
                content_score = raw_score
                # If no task_context, T_raz is not directly queried. We can either:
                # 1. Calculate similarity with T_raz of doc and a generic/null task query (less meaningful)
                # 2. Assign a neutral task_score or 0.
                # For this iteration, if applicability_embedding is present, we can compare with query_embedding_orig
                # as a proxy, or assign a neutral value. Let's assign neutral for simplicity when no task_context.
                task_score = 0.5 # Neutral placeholder if no task context

            final_score = (effective_task_weight * task_score) + ((1.0 - effective_task_weight) * content_score)
            usage_count = doc.get("usage_count", 0); success_rate = doc.get("success_count", 0) / max(usage_count, 1)
            boost = min(0.05, (usage_count / 200.0) * success_rate)
            adjusted_score = min(1.0, final_score + boost)

            if adjusted_score >= threshold:
                doc_copy = doc.copy() # Work with a copy to pop fields
                doc_copy.pop("content_embedding", None); doc_copy.pop("applicability_embedding", None)
                doc_copy.pop(primary_score_field, None)
                search_results_tuples.append((doc_copy, adjusted_score, content_score, task_score))
        
        search_results_tuples.sort(key=lambda x: x[1], reverse=True)
        return search_results_tuples[:limit]

    async def create_record(
        self, name: str, record_type: str, domain: str, description: str,
        code_snippet: str, version: str = "1.0.0", status: str = "active",
        tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        record_id = str(uuid.uuid4())
        current_time_iso = datetime.now(timezone.utc).isoformat() # Use timezone-aware UTC
        traz_input = {"name": name, "record_type": record_type, "domain": domain, "description": description, "code_snippet": code_snippet}
        applicability_text = await self.generate_applicability_text(traz_input)
        t_orig_input = {**traz_input, "tags": tags or [], "usage_count":0, "success_count":0}
        content_for_E_orig = self._get_record_vector_text(t_orig_input)
        try:
            content_embedding = await self.llm_service.embed(content_for_E_orig)
            applicability_embedding = await self.llm_service.embed(applicability_text)
        except Exception as e:
            logger.error(f"Embedding generation failed for {name}: {e}", exc_info=True)
            content_embedding = [0.0] * DEFAULT_EMBEDDING_DIMENSION
            applicability_embedding = [0.0] * DEFAULT_EMBEDDING_DIMENSION
        record = {"id": record_id, "name": name, "record_type": record_type, "domain": domain,
                  "description": description, "code_snippet": code_snippet, "version": version,
                  "usage_count": 0, "success_count": 0, "fail_count": 0, "status": status,
                  "created_at": datetime.fromisoformat(current_time_iso), # Store as BSON datetime
                  "last_updated": datetime.fromisoformat(current_time_iso), # Store as BSON datetime
                  "tags": tags or [], "metadata": {**(metadata or {}), "applicability_text": applicability_text},
                  "content_embedding": content_embedding, "applicability_embedding": applicability_embedding}
        await self.save_record(record)
        logger.info(f"Created new {record_type} record '{name}' (ID: {record_id}) in MongoDB.")
        return record

    async def update_usage_metrics(self, record_id: str, success: bool = True) -> None:
        if self.components_collection is None: return
        update_result = await self.components_collection.update_one(
            {"id": record_id},
            {"$inc": {"usage_count": 1, "success_count": 1 if success else 0, "fail_count": 0 if success else 1},
             "$set": {"last_updated": datetime.now(timezone.utc)}}) # Store as BSON datetime
        if update_result.matched_count == 0: logger.warning(f"Update metrics: record {record_id} not found.")
        else: logger.info(f"Updated usage metrics for record {record_id} (success={success}).")

    async def get_firmware(self, domain: str) -> Optional[Dict[str, Any]]:
        if self.components_collection is None: return None
        query = {"record_type": "FIRMWARE", "status": "active"}
        firmware = await self.components_collection.find_one({**query, "domain": domain}, {"_id": 0})
        return firmware if firmware else await self.components_collection.find_one({**query, "domain": "general"}, {"_id": 0})

    async def evolve_record(
        self, parent_id: str, new_code_snippet: str, description: Optional[str] = None,
        new_version: Optional[str] = None, status: str = "active"
    ) -> Dict[str, Any]:
        parent = await self.find_record_by_id(parent_id)
        if not parent: raise ValueError(f"Parent record {parent_id} not found")
        evolved_version_str = new_version or self._increment_version(parent["version"])
        current_time_iso = datetime.now(timezone.utc).isoformat()
        evolved_desc = description or parent["description"]
        
        # Prepare data for generating new embeddings
        traz_input = {"name": parent["name"], "record_type": parent["record_type"], 
                      "domain": parent["domain"], "description": evolved_desc, 
                      "code_snippet": new_code_snippet}
        new_applicability_text = await self.generate_applicability_text(traz_input)
        
        t_orig_input = {**traz_input, "tags": parent.get("tags", []), "usage_count":0, "success_count":0}
        new_content_for_E_orig = self._get_record_vector_text(t_orig_input)
        
        try:
            new_content_embedding = await self.llm_service.embed(new_content_for_E_orig)
            new_applicability_embedding = await self.llm_service.embed(new_applicability_text)
        except Exception as e:
            logger.error(f"Embedding generation failed for evolved {parent['name']}: {e}", exc_info=True)
            new_content_embedding = [0.0] * DEFAULT_EMBEDDING_DIMENSION
            new_applicability_embedding = [0.0] * DEFAULT_EMBEDDING_DIMENSION
        
        evolved_metadata = parent.get("metadata", {}).copy()
        evolved_metadata.update({ # Update existing metadata dict
            "applicability_text": new_applicability_text, 
            "evolved_at": current_time_iso,
            "evolved_from": parent_id, 
            "previous_version": parent["version"]
        })
        # Ensure applicability_text is set correctly, not duplicated if key existed
        evolved_metadata["applicability_text"] = new_applicability_text


        evolved_record = {
            "id": str(uuid.uuid4()), "name": parent["name"], "record_type": parent["record_type"],
            "domain": parent["domain"], "description": evolved_desc, "code_snippet": new_code_snippet,
            "version": evolved_version_str, "usage_count": 0, "success_count": 0, "fail_count": 0, "status": status,
            "created_at": datetime.fromisoformat(current_time_iso), # BSON datetime
            "last_updated": datetime.fromisoformat(current_time_iso), # BSON datetime
            "parent_id": parent_id, "tags": parent.get("tags", []).copy(),
            "metadata": evolved_metadata, 
            "content_embedding": new_content_embedding,
            "applicability_embedding": new_applicability_embedding
        }
        await self.save_record(evolved_record)
        logger.info(f"Evolved record {parent['name']} to {evolved_version_str} (ID: {evolved_record['id']}).")
        return evolved_record

    def _increment_version(self, version: str) -> str:
        parts = version.split("."); parts.extend(["0"] * (3 - len(parts))) # Ensure 3 parts
        try: parts[2] = str(int(parts[2]) + 1); return ".".join(parts[:3])
        except (ValueError, IndexError): logger.warning(f"Invalid version format '{version}', appending '.1'"); return f"{version}.1"


    async def search_by_tag(self, tags: List[str], record_type: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.components_collection is None: return []
        # MongoDB $in is case-sensitive by default. If tags are not consistently cased,
        # this might miss matches. Solution: store tags as lowercase or use regex with $options: "i".
        # For simplicity, assuming tags are stored consistently or case sensitivity is acceptable.
        query: Dict[str, Any] = {"tags": {"$in": [t.lower() for t in tags]}, "status": "active"}
        if record_type: query["record_type"] = record_type
        cursor = self.components_collection.find(query, {"_id": 0})
        return await cursor.to_list(length=None)

    async def bulk_import(self, records_data: List[Dict[str, Any]]) -> int:
        if not records_data or self.components_collection is None: return 0
        operations = []
        for r_dict in records_data:
            record_id = r_dict.get("id", str(uuid.uuid4()))
            now_iso = datetime.now(timezone.utc).isoformat()
            now_dt = datetime.fromisoformat(now_iso) # BSON datetime

            # Prepare data for T_raz and T_orig
            # Ensure all necessary fields are present for text generation, provide defaults if missing
            temp_rec_data = {
                "name": r_dict.get("name", "Unnamed Component"),
                "record_type": r_dict.get("record_type", "UNKNOWN"),
                "domain": r_dict.get("domain", "general"),
                "description": r_dict.get("description", ""),
                "code_snippet": r_dict.get("code_snippet", ""),
                "tags": r_dict.get("tags", []),
                "usage_count": r_dict.get("usage_count", 0), # For _get_record_vector_text
                "success_count": r_dict.get("success_count", 0) # For _get_record_vector_text
            }
            applicability_text = await self.generate_applicability_text(temp_rec_data)
            content_for_E_orig = self._get_record_vector_text(temp_rec_data)
            try:
                content_embedding = await self.llm_service.embed(content_for_E_orig)
                applicability_embedding = await self.llm_service.embed(applicability_text)
            except Exception as e:
                logger.error(f"Embedding failed for bulk import {r_dict.get('name', record_id)}: {e}")
                content_embedding = [0.0] * DEFAULT_EMBEDDING_DIMENSION
                applicability_embedding = [0.0] * DEFAULT_EMBEDDING_DIMENSION
            
            full_rec = {
                **r_dict, # Start with provided data
                "id":record_id,
                "created_at": r_dict.get("created_at", now_dt), # Use BSON datetime
                "last_updated": now_dt, # Use BSON datetime
                "status": r_dict.get("status", "active"),
                "metadata": {**(r_dict.get("metadata", {})), "applicability_text": applicability_text},
                "content_embedding": content_embedding,
                "applicability_embedding": applicability_embedding
            }
            # Ensure all fields are present even if not in r_dict initially
            for key, default_val in [("usage_count",0),("success_count",0),("fail_count",0),("tags",[])]:
                if key not in full_rec: full_rec[key] = default_val
            
            operations.append(pymongo.ReplaceOne({"id": record_id}, full_rec, upsert=True))
        
        if not operations: return 0
        try:
            result = await self.components_collection.bulk_write(operations, ordered=False)
            success_count = (result.upserted_count or 0) + (result.modified_count or 0)
            logger.info(f"Bulk imported/updated {success_count} records.")
            return success_count
        except pymongo.errors.BulkWriteError as bwe:
            logger.error(f"Error during bulk import: {bwe.details}", exc_info=True)
            return (bwe.details.get('nInserted',0) +
                    bwe.details.get('nUpserted',0) +
                    bwe.details.get('nModified',0))
        except Exception as e:
            logger.error(f"Unexpected error during bulk import: {e}", exc_info=True); return 0

    async def export_records(self, record_type: Optional[str] = None, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.components_collection is None: return []
        query: Dict[str, Any] = {}; 
        if record_type: query["record_type"] = record_type
        if domain: query["domain"] = domain
        cursor = self.components_collection.find(query, {"_id": 0, "content_embedding": 0, "applicability_embedding": 0})
        return await cursor.to_list(length=None)

    async def clear_cache(self, older_than: Optional[int] = None) -> int:
        if not self.llm_service.cache:
            logger.info("LLM Cache is disabled in LLMService, nothing to clear from SmartLibrary perspective.")
            return 0
        # LLMService's cache is now MongoDB backed, so this delegates to its clear method
        return await self.llm_service.clear_cache(older_than_seconds=older_than)

    async def get_status_summary(self) -> Dict[str, Any]:
        if self.components_collection is None:
            logger.error("Cannot get status summary: components_collection is None.")
            return {"error": "components_collection not initialized", "total_records": 0}

        # Aggregations
        type_pipeline = [{"$group": {"_id": "$record_type", "count": {"$sum": 1}}}]
        status_pipeline = [{"$group": {"_id": "$status", "count": {"$sum": 1}}}]
        
        type_counts_raw = await self.components_collection.aggregate(type_pipeline).to_list(length=None)
        status_counts_raw = await self.components_collection.aggregate(status_pipeline).to_list(length=None)
        
        type_counts = {item["_id"]: item["count"] for item in type_counts_raw if item.get("_id")}
        status_counts = {item["_id"]: item["count"] for item in status_counts_raw if item.get("_id")}

        distinct_domains_list = await self.components_collection.distinct("domain")
        distinct_domains = [d for d in distinct_domains_list if d] # Filter out None/empty domains

        # Queries for specific lists
        projection = {"_id":0, "name":1, "id":1, "usage_count":1, "success_count":1, "last_updated":1}
        most_used = await self.components_collection.find({"status":"active"}, projection).sort("usage_count", pymongo.DESCENDING).limit(5).to_list(length=None)
        
        successful_candidates_cursor = self.components_collection.find(
            {"status":"active", "usage_count": {"$gte": 5}}, projection).limit(50) # Fetch more to sort client-side
        successful_candidates = await successful_candidates_cursor.to_list(length=None)
        for doc in successful_candidates:
            doc["success_rate"] = doc.get("success_count",0) / doc.get("usage_count",1) if doc.get("usage_count",0) > 0 else 0.0
        most_successful = sorted(successful_candidates, key=lambda x: x["success_rate"], reverse=True)[:5]

        recently_updated = await self.components_collection.find({"status":"active"}, projection).sort("last_updated", pymongo.DESCENDING).limit(5).to_list(length=None)
        total_records = await self.components_collection.count_documents({})

        return {
            "total_records": total_records,
            "by_type": { "AGENT": type_counts.get("AGENT", 0), "TOOL": type_counts.get("TOOL", 0),
                         "FIRMWARE": type_counts.get("FIRMWARE", 0),
                         **{k:v for k,v in type_counts.items() if k not in ["AGENT", "TOOL", "FIRMWARE"] and k}},
            "by_status": status_counts,
            "domains": distinct_domains,
            "most_used": [{"id": r.get("id"), "name": r.get("name"), "usage_count": r.get("usage_count", 0)} for r in most_used],
            "most_successful": [{"id": r.get("id"), "name": r.get("name"), "success_rate": r.get("success_rate", 0.0)} for r in most_successful],
            "recently_updated": [{"id": r.get("id"), "name": r.get("name"), "last_updated": r.get("last_updated", "")} for r in recently_updated]
        }