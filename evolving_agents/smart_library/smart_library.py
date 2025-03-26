import os
import json
import logging
import uuid
import re
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import chromadb
from evolving_agents.core.llm_service import LLMService

# Import DependencyContainer using TYPE_CHECKING to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from evolving_agents.core.dependency_container import DependencyContainer

logger = logging.getLogger(__name__)

class SmartLibrary:
    """
    Unified library that stores all agents, tools, and firmware as simple dictionary records
    with vector database-powered semantic search.
    """
    def __init__(
        self, 
        storage_path: str = "smart_library.json", 
        vector_db_path: str = "./vector_db",
        llm_service: Optional[LLMService] = None,
        use_cache: bool = True,
        cache_dir: str = ".llm_cache",
        container: Optional["DependencyContainer"] = None
    ):
        """
        Initialize the SmartLibrary.
        
        Args:
            storage_path: Path to the JSON file storing the library records
            vector_db_path: Directory for the vector database
            llm_service: Optional pre-configured LLM service 
            use_cache: Whether to use caching for LLM operations
            cache_dir: Directory for the LLM cache
            container: Optional dependency container for managing component dependencies
        """
        self.storage_path = storage_path
        self.records = []
        self.container = container
        self._initialized = False
        
        # Initialize LLM service for embeddings if not provided
        if container and container.has('llm_service'):
            self.llm_service = llm_service or container.get('llm_service')
        else:
            self.llm_service = llm_service or LLMService(use_cache=use_cache, cache_dir=cache_dir)
        
        # Register with container if provided
        if container and not container.has('smart_library'):
            container.register('smart_library', self)
        
        # Initialize vector DB and load library on creation
        self._init_vector_db(vector_db_path)
        self._load_library()
        
    def _init_vector_db(self, vector_db_path: str = "./vector_db"):
        """Initialize the vector database connection."""
        try:
            self.chroma_client = chromadb.PersistentClient(path=vector_db_path)
            
            # Create our collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="smart_library_records"
            )
            logger.info(f"Connected to vector database at {vector_db_path}")
        except Exception as e:
            logger.error(f"Error initializing vector database: {str(e)}")
            self.collection = None
    
    def _load_library(self):
        """Load the library from storage."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    self.records = json.load(f)
                logger.info(f"Loaded {len(self.records)} records from {self.storage_path}")
                
                # Ensure all records are in the vector database
                if self.collection:
                    asyncio.create_task(self._sync_vector_db())
            except Exception as e:
                logger.error(f"Error loading library: {str(e)}")
                self.records = []
        else:
            logger.info(f"No existing library found at {self.storage_path}. Creating new library.")
            self.records = []
    
    async def initialize(self):
        """Complete library initialization and integration with other components."""
        if self._initialized:
            return
            
        # Initialize agent bus from library if available
        if self.container and self.container.has('agent_bus'):
            agent_bus = self.container.get('agent_bus')
            if hasattr(agent_bus, 'initialize_from_library'):
                await agent_bus.initialize_from_library()
        
        self._initialized = True
    
    def _get_record_vector_text(self, record: Dict[str, Any]) -> str:
        """
        Create a functionally-focused text representation of a record for embedding.
        
        Args:
            record: The record to represent as text
            
        Returns:
            Text representation optimized for vector search
        """
        name = record.get("name", "")
        description = record.get("description", "")
        record_type = record.get("record_type", "")
        domain = record.get("domain", "")
        tags = " ".join(record.get("tags", []))
        
        # Include usage statistics if available
        usage_stats = ""
        usage_count = record.get("usage_count", 0)
        success_count = record.get("success_count", 0)
        if usage_count > 0:
            success_rate = success_count / usage_count if usage_count > 0 else 0
            usage_stats = f"Used {usage_count} times with {success_rate:.0%} success rate."
        
        # For TOOL records, include function signatures or interfaces 
        interface_info = ""
        if record_type == "TOOL":
            code_snippet = record.get("code_snippet", "")
            
            # Extract function signature or API info if available
            # Try to find function signatures
            signatures = re.findall(r'def\s+(\w+\([^)]*\))', code_snippet)
            if signatures:
                interface_info = "Functions: " + " ".join(signatures)
            
            # Look for API endpoint definitions
            api_endpoints = re.findall(r'@\w+\.route\([\'"]([^\'"]+)[\'"]', code_snippet)
            if api_endpoints:
                interface_info += " API endpoints: " + " ".join(api_endpoints)
        
        # Create a weighted representation focusing on functional aspects
        return (
            f"Name: {name}. "
            f"Type: {record_type}. "
            f"Domain: {domain}. "
            f"Purpose: {description}. "
            f"{interface_info} "
            f"Tags: {tags}. "
            f"{usage_stats}"
        )
    
    async def _sync_vector_db(self):
        """Synchronize the vector database with the current records."""
        if not self.collection:
            logger.warning("Vector database not available. Skipping sync.")
            return
            
        # Get existing IDs in the collection
        existing_ids = set()
        try:
            results = self.collection.get(include=[])
            if results and "ids" in results:
                existing_ids = set(results["ids"])
        except Exception as e:
            logger.error(f"Error getting IDs from vector database: {str(e)}")
            # Collection might be empty
            pass
        
        # Find records to add or update
        current_ids = set(r["id"] for r in self.records)
        
        # IDs to add to vector DB
        ids_to_add = current_ids - existing_ids
        records_to_add = [r for r in self.records if r["id"] in ids_to_add]
        
        # IDs to remove from vector DB
        ids_to_remove = existing_ids - current_ids
        
        # Add new records to vector DB
        if records_to_add:
            try:
                # Prepare data for batch addition
                ids = [r["id"] for r in records_to_add]
                texts = [self._get_record_vector_text(r) for r in records_to_add]
                metadatas = [{
                    "name": r.get("name", ""),
                    "record_type": r.get("record_type", ""),
                    "domain": r.get("domain", ""),
                    "status": r.get("status", "active"),
                    "version": r.get("version", "")
                } for r in records_to_add]
                
                # Generate embeddings for the texts
                embeddings = await self.llm_service.embed_batch(texts)
                
                # Add to collection
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas
                )
                logger.info(f"Added {len(records_to_add)} records to vector database")
            except Exception as e:
                logger.error(f"Error adding records to vector database: {str(e)}")
        
        # Remove deleted records from vector DB
        if ids_to_remove:
            try:
                for id_to_remove in ids_to_remove:
                    self.collection.delete(ids=[id_to_remove])
                logger.info(f"Removed {len(ids_to_remove)} records from vector database")
            except Exception as e:
                logger.error(f"Error removing records from vector database: {str(e)}")
    
    def _save_library(self):
        """Save the library to storage and update vector database."""
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, indent=2)
        
        # Update vector database asynchronously
        try:
            asyncio.create_task(self._sync_vector_db())
        except Exception as e:
            logger.error(f"Error creating sync task for vector database: {str(e)}")
        
        logger.info(f"Saved {len(self.records)} records to {self.storage_path}")
    
    async def save_record(self, record: Dict[str, Any]) -> str:
        """
        Save a record to the library.
        
        Args:
            record: Dictionary containing record data
            
        Returns:
            ID of the saved record
        """
        # Update existing or add new
        idx = next((i for i, r in enumerate(self.records) if r["id"] == record["id"]), -1)
        if idx >= 0:
            self.records[idx] = record
            logger.info(f"Updated record {record['id']} ({record['name']})")
        else:
            self.records.append(record)
            logger.info(f"Added new record {record['id']} ({record['name']})")
        
        self._save_library()
        return record["id"]
    
    async def find_record_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Find a record by ID.
        
        Args:
            record_id: The record ID to find
            
        Returns:
            The record if found, None otherwise
        """
        return next((r for r in self.records if r["id"] == record_id), None)
    
    async def find_record_by_name(self, name: str, record_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Find a record by name.
        
        Args:
            name: Record name to find
            record_type: Optional record type filter (AGENT, TOOL, FIRMWARE)
            
        Returns:
            The record if found, None otherwise
        """
        if record_type:
            return next((r for r in self.records if r["name"] == name and r["record_type"] == record_type), None)
        return next((r for r in self.records if r["name"] == name), None)
    
    async def find_records_by_domain(self, domain: str, record_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find records by domain.
        
        Args:
            domain: Domain to search for
            record_type: Optional record type filter
            
        Returns:
            List of matching records
        """
        if record_type:
            return [r for r in self.records if r.get("domain") == domain and r["record_type"] == record_type]
        return [r for r in self.records if r.get("domain") == domain]
    
    async def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        # Convert to numpy arrays for efficient calculation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    async def semantic_search(
        self, 
        query: str, 
        record_type: Optional[str] = None,
        domain: Optional[str] = None,
        limit: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for records semantically similar to the query using the vector database.
        
        Args:
            query: The search query (can be a functional requirement or description)
            record_type: Optional record type filter (AGENT, TOOL, FIRMWARE)
            domain: Optional domain filter
            limit: Maximum number of results
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            List of (record, similarity) tuples sorted by similarity
        """
        # Check if vector DB is available, otherwise fall back to direct search
        if not self.collection:
            return await self._semantic_search_direct(query, record_type, domain, limit, threshold)
        
        # Prepare filters for the query
        where_filter = {}
        
        if record_type:
            where_filter["record_type"] = record_type
        
        if domain:
            where_filter["domain"] = domain
        
        # Always filter for active records
        where_filter["status"] = "active"
        
        try:
            # Generate embedding for the query
            query_embedding = await self.llm_service.embed(query)
            
            # Search the vector database
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit * 2,  # Query more results than needed to account for filtering
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Transform results to the expected format
            search_results = []
            
            if not results["ids"] or not results["ids"][0]:
                return []
            
            # Process results
            for i, result_id in enumerate(results["ids"][0]):
                # Distance is cosine distance, convert to similarity
                distance = results["distances"][0][i]
                similarity = 1.0 - distance  # Convert distance to similarity
                
                if similarity >= threshold:
                    # Find the full record by ID
                    record = await self.find_record_by_id(result_id)
                    if record:
                        search_results.append((record, similarity))
            
            # No need to sort again as the vector DB already returns sorted results
            return search_results[:limit]
            
        except Exception as e:
            logger.error(f"Error using vector database for search: {str(e)}. Falling back to direct search.")
            # Fall back to direct search if vector DB fails
            return await self._semantic_search_direct(query, record_type, domain, limit, threshold)
    
    async def _semantic_search_direct(
        self, 
        query: str, 
        record_type: Optional[str] = None,
        domain: Optional[str] = None,
        limit: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Fallback direct semantic search without vector database.
        
        Args:
            query: The search query
            record_type: Optional record type filter
            domain: Optional domain filter
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            
        Returns:
            List of (record, similarity) tuples sorted by similarity
        """
        # Filter records by type and domain if specified
        filtered_records = self.records
        if record_type:
            filtered_records = [r for r in filtered_records if r["record_type"] == record_type]
        if domain:
            filtered_records = [r for r in filtered_records if r.get("domain") == domain]
            
        # Filter only active records
        active_records = [r for r in filtered_records if r.get("status", "active") == "active"]
        
        if not active_records:
            logger.info(f"No active records found for search: {query}")
            return []
        
        # Get embedding for the query
        query_embedding = await self.llm_service.embed(query)
        
        # Create functional texts for batch embedding
        record_texts = [self._get_record_vector_text(record) for record in active_records]
        
        # Generate embeddings for all records in one batch
        record_embeddings = await self.llm_service.embed_batch(record_texts)
        
        # Compute similarities
        results = []
        for i, record in enumerate(active_records):
            # Compute similarity
            similarity = await self.compute_similarity(query_embedding, record_embeddings[i])
            
            # Apply additional weighting based on usage metrics if available
            usage_count = record.get("usage_count", 0)
            success_rate = record.get("success_count", 0) / usage_count if usage_count > 0 else 0
            
            # Boost score for frequently used and successful records (small boost of max 10%)
            boost = min(0.1, (usage_count / 100) * success_rate)
            adjusted_similarity = min(1.0, similarity + boost)
            
            if adjusted_similarity >= threshold:
                results.append((record, adjusted_similarity))
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    async def create_record(
        self,
        name: str,
        record_type: str,  # "AGENT", "TOOL", or "FIRMWARE"
        domain: str,
        description: str,
        code_snippet: str,
        version: str = "1.0.0",
        status: str = "active",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new record in the library.
        
        Args:
            name: Record name
            record_type: Record type (AGENT, TOOL, FIRMWARE)
            domain: Domain of the record
            description: Description of the record
            code_snippet: Code snippet or content
            version: Version string
            status: Status of the record
            tags: Optional tags
            metadata: Optional metadata
            
        Returns:
            The created record
        """
        record_id = str(uuid.uuid4())
        
        record = {
            "id": record_id,
            "name": name,
            "record_type": record_type,
            "domain": domain,
            "description": description,
            "code_snippet": code_snippet,
            "version": version,
            "usage_count": 0,
            "success_count": 0,
            "fail_count": 0,
            "status": status,
            "created_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat(),
            "tags": tags or [],
            "metadata": metadata or {}
        }
        
        await self.save_record(record)
        logger.info(f"Created new {record_type} record: {name}")
        return record
    
    async def update_usage_metrics(self, record_id: str, success: bool = True) -> None:
        """
        Update usage metrics for a record.
        
        Args:
            record_id: ID of the record to update
            success: Whether the usage was successful
        """
        record = await self.find_record_by_id(record_id)
        if record:
            record["usage_count"] = record.get("usage_count", 0) + 1
            if success:
                record["success_count"] = record.get("success_count", 0) + 1
            else:
                record["fail_count"] = record.get("fail_count", 0) + 1
            
            record["last_updated"] = datetime.utcnow().isoformat()
            await self.save_record(record)
            logger.info(f"Updated usage metrics for {record['name']} (success={success})")
        else:
            logger.warning(f"Attempted to update metrics for non-existent record: {record_id}")
    
    async def get_firmware(self, domain: str) -> Optional[Dict[str, Any]]:
        """
        Get the active firmware for a domain.
        
        Args:
            domain: Domain to get firmware for
            
        Returns:
            Firmware record if found, None otherwise
        """
        # Find active firmware for the domain, or general if not found
        firmware = next(
            (r for r in self.records 
             if r["record_type"] == "FIRMWARE" 
             and r.get("domain") == domain 
             and r.get("status") == "active"),
            None
        )
        
        if not firmware:
            # Fall back to general firmware
            firmware = next(
                (r for r in self.records 
                 if r["record_type"] == "FIRMWARE" 
                 and r.get("domain") == "general" 
                 and r.get("status") == "active"),
                None
            )
            
        return firmware
    
    async def evolve_record(
        self,
        parent_id: str,
        new_code_snippet: str,
        description: Optional[str] = None,
        new_version: Optional[str] = None,
        status: str = "active"
    ) -> Dict[str, Any]:
        """
        Create an evolved version of an existing record.
        
        Args:
            parent_id: ID of the parent record
            new_code_snippet: New code snippet for the evolved record
            description: Optional new description
            new_version: Optional version override (otherwise incremented)
            status: Status for the new record
            
        Returns:
            The newly created record
        """
        parent = await self.find_record_by_id(parent_id)
        if not parent:
            raise ValueError(f"Parent record not found: {parent_id}")
            
        # Increment version if not specified
        if new_version is None:
            new_version = self._increment_version(parent["version"])
        
        # Create new record with parent's metadata
        new_record = {
            "id": str(uuid.uuid4()),
            "name": parent["name"],
            "record_type": parent["record_type"],
            "domain": parent["domain"],
            "description": description or parent["description"],
            "code_snippet": new_code_snippet,
            "version": new_version,
            "usage_count": 0,
            "success_count": 0,
            "fail_count": 0,
            "status": status,
            "created_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat(),
            "parent_id": parent_id,
            "tags": parent.get("tags", []).copy(),
            "metadata": {
                **(parent.get("metadata", {}).copy()),
                "evolved_at": datetime.utcnow().isoformat(),
                "evolved_from": parent_id,
                "previous_version": parent["version"]
            }
        }
        
        # Save and return
        await self.save_record(new_record)
        logger.info(f"Evolved record {parent['name']} from {parent['version']} to {new_version}")
        
        return new_record
    
    def _increment_version(self, version: str) -> str:
        """
        Increment the version number.
        
        Args:
            version: Current version string (e.g., "1.0.0")
            
        Returns:
            Incremented version string
        """
        parts = version.split(".")
        if len(parts) < 3:
            parts += ["0"] * (3 - len(parts))
            
        # Increment patch version
        try:
            patch = int(parts[2]) + 1
            return f"{parts[0]}.{parts[1]}.{patch}"
        except (ValueError, IndexError):
            # If version format is invalid, just append .1
            return f"{version}.1"
            
    async def search_by_tag(self, tags: List[str], record_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for records by tags.
        
        Args:
            tags: List of tags to search for (records must have at least one)
            record_type: Optional record type filter
            
        Returns:
            List of matching records
        """
        # Lower case all tags for case-insensitive matching
        search_tags = [tag.lower() for tag in tags]
        
        results = []
        for record in self.records:
            # Skip if record type doesn't match
            if record_type and record["record_type"] != record_type:
                continue
                
            # Skip inactive records
            if record.get("status", "active") != "active":
                continue
                
            # Check if any of the record's tags match any search tag
            record_tags = [tag.lower() for tag in record.get("tags", [])]
            if any(tag in record_tags for tag in search_tags):
                results.append(record)
                
        return results
    
    async def bulk_import(self, records: List[Dict[str, Any]]) -> int:
        """
        Import multiple records at once.
        
        Args:
            records: List of record dictionaries to import
            
        Returns:
            Number of records imported
        """
        count = 0
        for record in records:
            # Ensure each record has an ID
            if "id" not in record:
                record["id"] = str(uuid.uuid4())
                
            # Set created_at and last_updated if not present
            if "created_at" not in record:
                record["created_at"] = datetime.utcnow().isoformat()
                
            if "last_updated" not in record:
                record["last_updated"] = datetime.utcnow().isoformat()
                
            # Save the record
            await self.save_record(record)
            count += 1
            
        return count
    
    async def export_records(self, record_type: Optional[str] = None, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Export records as a JSON-serializable list.
        
        Args:
            record_type: Optional record type filter
            domain: Optional domain filter
            
        Returns:
            List of records matching the filters
        """
        filtered_records = self.records
        
        if record_type:
            filtered_records = [r for r in filtered_records if r["record_type"] == record_type]
            
        if domain:
            filtered_records = [r for r in filtered_records if r.get("domain") == domain]
            
        return filtered_records
    
    def clear_cache(self, older_than: Optional[int] = None) -> int:
        """
        Clear the LLM cache.
        
        Args:
            older_than: Clear entries older than this many seconds. If None, clear all.
            
        Returns:
            Number of entries removed
        """
        return self.llm_service.clear_cache(older_than)
    
    async def get_status_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the library status.
        
        Returns:
            Dictionary with status information
        """
        # Count by record type
        agent_count = len([r for r in self.records if r["record_type"] == "AGENT"])
        tool_count = len([r for r in self.records if r["record_type"] == "TOOL"])
        firmware_count = len([r for r in self.records if r["record_type"] == "FIRMWARE"])
        
        # Count by status
        active_count = len([r for r in self.records if r.get("status", "active") == "active"])
        inactive_count = len(self.records) - active_count
        
        # Get domains
        domains = set(r.get("domain") for r in self.records if "domain" in r)
        
        # Most used and successful records
        sorted_by_usage = sorted(self.records, key=lambda r: r.get("usage_count", 0), reverse=True)
        most_used = sorted_by_usage[:5] if sorted_by_usage else []
        
        # Records with high success rate (min 5 usages)
        success_rated = [
            r for r in self.records 
            if r.get("usage_count", 0) >= 5
        ]
        success_rated.sort(
            key=lambda r: r.get("success_count", 0) / r.get("usage_count", 1), 
            reverse=True
        )
        most_successful = success_rated[:5] if success_rated else []
        
        # Recently updated
        sorted_by_updated = sorted(
            self.records, 
            key=lambda r: r.get("last_updated", ""), 
            reverse=True
        )
        recently_updated = sorted_by_updated[:5] if sorted_by_updated else []
        
        return {
            "total_records": len(self.records),
            "by_type": {
                "AGENT": agent_count,
                "TOOL": tool_count,
                "FIRMWARE": firmware_count
            },
            "by_status": {
                "active": active_count,
                "inactive": inactive_count
            },
            "domains": list(domains),
            "most_used": [
                {"id": r["id"], "name": r["name"], "usage_count": r.get("usage_count", 0)}
                for r in most_used
            ],
            "most_successful": [
                {
                    "id": r["id"], 
                    "name": r["name"], 
                    "success_rate": r.get("success_count", 0) / r.get("usage_count", 1)
                }
                for r in most_successful
            ],
            "recently_updated": [
                {"id": r["id"], "name": r["name"], "last_updated": r.get("last_updated", "")}
                for r in recently_updated
            ]
        }