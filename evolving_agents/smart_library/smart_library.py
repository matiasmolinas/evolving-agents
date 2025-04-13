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
    with vector database-powered semantic search using dual embeddings.
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
                
            # Extract class definitions
            class_defs = re.findall(r'class\s+(\w+)(?:\([^)]*\))?:', code_snippet)
            if class_defs:
                interface_info += " Classes: " + " ".join(class_defs)
                
            # Extract method names for more context
            method_names = re.findall(r'def\s+(\w+)\(', code_snippet)
            if method_names:
                interface_info += " Methods: " + " ".join(method_names)
        
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
                
                # Generate text representation for each record
                texts = []
                for r in records_to_add:
                    text = self._get_record_vector_text(r)
                    if text is None:
                        logger.warning(f"Record {r.get('id', 'unknown')} generated None text, using default")
                        text = f"Name: {r.get('name', 'Unknown')}. Type: {r.get('record_type', 'Unknown')}."
                    texts.append(text)
                
                metadatas = [{
                    "name": r.get("name", ""),
                    "record_type": r.get("record_type", ""),
                    "domain": r.get("domain", ""),
                    "status": r.get("status", "active"),
                    "version": r.get("version", ""),
                    # Include applicability text if available for task-aware search
                    "applicability_text": r.get("metadata", {}).get("applicability_text", "")
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
    
    async def generate_applicability_text(self, record: Dict[str, Any]) -> str:
        """
        Generate rich applicability text for a component to enhance task relevance matching.
        
        Args:
            record: The component record
        
        Returns:
            Rich applicability text describing when and how to use the component
        """
        # Prepare component information
        name = record.get("name", "")
        description = record.get("description", "")
        record_type = record.get("record_type", "")
        domain = record.get("domain", "")
        code_snippet = record.get("code_snippet", "")
        
        # Extract code summary if code snippet is too long
        code_summary = code_snippet[:500] + "..." if len(code_snippet) > 500 else code_snippet
        
        # Use specialized prompts based on the component type and domain
        if record_type == "TOOL" and ("doc" in name.lower() or domain.lower() == "documentation"):
            # Specialized prompt for documentation tools
            prompt = f"""
            Based on this documentation tool:
            
            Name: {name}
            Type: {record_type}
            Domain: {domain}
            Description: {description}
            Code Summary: 
            ```
            {code_summary}
            ```
            
            Generate a comprehensive applicability text focused specifically on 
            documentation generation scenarios, such as API docs, reference manuals,
            and developer guides. Include detailed information about:
            
            1. DOCUMENTATION TASKS: Types of documentation this tool excels at generating
            2. INPUT FORMATS: What inputs the tool can process (annotations, spec files, etc.)
            3. OUTPUT FORMATS: What documentation formats can be produced
            4. INTEGRATION: How the tool integrates with development workflows
            5. AUDIENCE: Which types of documentation users (developers, tech writers) benefit most
            
            Format as a cohesive paragraph focusing specifically on documentation generation use cases.
            """
        elif record_type == "TOOL" and ("test" in name.lower() or domain.lower() == "testing"):
            # Specialized prompt for testing tools
            prompt = f"""
            Based on this testing tool:
            
            Name: {name}
            Type: {record_type}
            Domain: {domain}
            Description: {description}
            Code Summary: 
            ```
            {code_summary}
            ```
            
            Generate a comprehensive applicability text focused specifically on 
            testing scenarios and quality assurance. Include detailed information about:
            
            1. TESTING SCENARIOS: What types of tests this tool is best suited for
            2. TEST APPROACHES: What testing methodologies it supports
            3. COVERAGE: What aspects of the system can be validated
            4. INTEGRATION: How it fits into CI/CD pipelines
            5. EDGE CASES: Special conditions it can test for
            
            Format as a cohesive paragraph focusing specifically on testing and validation use cases.
            """
        elif record_type == "TOOL" and ("auth" in name.lower() or domain.lower() == "authentication"):
            # Specialized prompt for authentication tools
            prompt = f"""
            Based on this authentication tool:
            
            Name: {name}
            Type: {record_type}
            Domain: {domain}
            Description: {description}
            Code Summary: 
            ```
            {code_summary}
            ```
            
            Generate a comprehensive applicability text focused specifically on 
            authentication implementation scenarios. Include detailed information about:
            
            1. AUTH MECHANISMS: What authentication protocols it supports
            2. SECURITY ASPECTS: How it addresses security concerns
            3. IMPLEMENTATION: How developers would use it in their applications
            4. INTEGRATION: How it connects with identity providers or services
            5. USE CASES: Specific authentication scenarios it excels at
            
            Format as a cohesive paragraph focusing specifically on authentication implementation use cases.
            """
        else:
            # Default prompt for all other components
            prompt = f"""
            Based on this component information:
            
            Name: {name}
            Type: {record_type}
            Domain: {domain}
            Description: {description}
            Code Summary: 
            ```
            {code_summary}
            ```
            
            Generate a comprehensive applicability text that describes:
            
            1. RELEVANT TASKS: Specific tasks and use cases where this component is most applicable
            2. USER PERSONAS: Who would benefit most from using this component (developers, testers, etc.)
            3. IDEAL SCENARIOS: Scenarios where this component shines compared to alternatives
            4. INTEGRATION PATTERNS: How this component typically integrates with other systems
            5. TECHNICAL REQUIREMENTS: Prerequisites or dependencies needed to use this component
            6. LIMITATIONS: Any notable limitations or cases when NOT to use this component
            
            Format as a cohesive paragraph focusing on when and how this component should be used.
            """
        
        # Generate the applicability text
        applicability_text = await self.llm_service.generate(prompt)
        logger.info(f"Generated applicability text for {name}")
        return applicability_text.strip()
    
    async def semantic_search(
        self,
        query: str,
        task_context: Optional[str] = None,
        record_type: Optional[str] = None,
        domain: Optional[str] = None,
        limit: int = 5,
        threshold: float = 0.0,
        task_weight: Optional[float] = None
    ) -> List[Tuple[Dict[str, Any], float, float, float]]:
        """
        Search for records semantically similar to the query using dual embeddings.
        
        Args:
            query: Content query string
            task_context: Optional task context for relevance-based search
            record_type: Optional record type filter
            domain: Optional domain filter
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            task_weight: Weight for task relevance score (0.0-1.0), or None to auto-determine
            
        Returns:
            List of (record, final_score, content_score, task_score) tuples
        """
        if not self.collection:
            logger.warning("Vector database not available for semantic search.")
            raise ValueError("Vector database is not available for search.")

        # Check for None input values
        if query is None:
            raise ValueError("Query string cannot be None")

        # Dynamically adjust task_weight based on task context if not explicitly provided
        if task_weight is None:
            if task_context:
                # Auto-determine weight based on task context content
                task_context_lower = task_context.lower()
                if "implement" in task_context_lower or "develop" in task_context_lower:
                    # Implementation tasks - heavier weight on task context
                    task_weight = 0.8
                elif "test" in task_context_lower or "validate" in task_context_lower:
                    # Testing tasks - significant weight on task context
                    task_weight = 0.75
                elif "document" in task_context_lower or "write" in task_context_lower:
                    # Documentation tasks - balanced weight
                    task_weight = 0.65
                else:
                    # Default for other task contexts
                    task_weight = 0.7
            else:
                # No task context, rely only on content
                task_weight = 0.0
        
        # --- Prepare filters for ChromaDB ---
        filters = []
        if record_type:
            filters.append({"record_type": {"$eq": record_type}})
        if domain:
            filters.append({
                "$or": [
                    {"domain": {"$eq": domain}},
                    {"domain": {"$eq": "general"}}
                ]
            })
        
        # Always filter for active records
        filters.append({"status": {"$eq": "active"}})

        # Combine filters using $and
        where_filter = None
        if filters:
            if len(filters) == 1:
                where_filter = filters[0]
            else:
                where_filter = {"$and": filters}

        try:
            # --- Generate embedding for the query (always needed) ---
            query_embedding = await self.llm_service.embed(query)
            
            # Specifically for dual embedding search, we need two different embeddings
            
            # --- PHASE 1: Get candidates based on task relevance (if available) ---
            if task_context:
                # Check for None task_context
                if task_context is None:
                    raise ValueError("Task context cannot be None when task-aware search is requested")
                    
                # Generate embedding for the task context
                task_embedding = await self.llm_service.embed(task_context)
                
                # First search phase: Get potential candidates based on task context
                # Use the applicability_text field for matching against task context
                phase1_results = self.collection.query(
                    query_embeddings=[task_embedding],
                    n_results=limit * 3,  # Get more candidates for re-ranking
                    where=where_filter,
                    include=["documents", "metadatas", "distances"]
                )
                
                if not phase1_results["ids"] or not phase1_results["ids"][0]:
                    return []
                    
                # Prepare task scores for candidates
                candidates = []
                for i, result_id in enumerate(phase1_results["ids"][0]):
                    # Convert distance to similarity (1.0 - distance), handle edge cases
                    distance = phase1_results["distances"][0][i]
                    # Normalize the distance to ensure it's at most 1.0
                    normalized_distance = min(distance, 1.0)
                    task_score = 1.0 - normalized_distance
                    
                    # Boost scores that are very close - help with low similarity issue
                    if task_score > 0.8:
                        task_score = min(1.0, task_score * 1.15)  # 15% boost for high matches
                    elif task_score > 0.6:
                        task_score = min(1.0, task_score * 1.1)   # 10% boost for good matches
                    
                    record = await self.find_record_by_id(result_id)
                    if record:
                        candidates.append((record, task_score))
                
                # --- PHASE 2: Content relevance scoring ---
                
                # Re-rank candidates based on content relevance
                search_results = []
                
                for record, task_score in candidates:
                    # Get functional text representation focused on content
                    record_text = self._get_record_vector_text(record)
                    
                    if record_text is None:
                        logger.warning(f"Record {record.get('id', 'unknown')} generated None text, skipping")
                        continue
                        
                    # Create content embedding for this specific record
                    content_embedding = await self.llm_service.embed(record_text)
                    
                    # Calculate content similarity
                    content_score = await self.compute_similarity(query_embedding, content_embedding)
                    
                    # Boost scores that are very close - help with low similarity issue
                    if content_score > 0.8:
                        content_score = min(1.0, content_score * 1.15)  # 15% boost for high matches
                    elif content_score > 0.6:
                        content_score = min(1.0, content_score * 1.1)   # 10% boost for good matches
                    
                    # Apply both signals with specified weighting
                    final_score = (task_weight * task_score) + ((1.0 - task_weight) * content_score)
                    
                    # Apply usage-based boosting for frequently used and successful components
                    usage_count = record.get("usage_count", 0)
                    success_rate = record.get("success_count", 0) / max(usage_count, 1)
                    
                    # Small boost of max 5% for frequently used successful components
                    boost = min(0.05, (usage_count / 200) * success_rate)
                    adjusted_score = min(1.0, final_score + boost)
                    
                    if adjusted_score >= threshold:
                        search_results.append((record, adjusted_score, content_score, task_score))
            else:
                # Standard content-only search when no task context provided
                
                # First try direct search in the vector database
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=limit * 2,
                    where=where_filter,
                    include=["documents", "metadatas", "distances", "embeddings"]
                )
                
                search_results = []
                if results["ids"] and results["ids"][0]:
                    # Process each result
                    for i, result_id in enumerate(results["ids"][0]):
                        # Get the record
                        record = await self.find_record_by_id(result_id)
                        if not record:
                            continue
                            
                        # Get the distance from the query
                        distance = results["distances"][0][i]
                        
                        # FIX: For standard search, make sure we get meaningful similarity scores
                        # The issue was that Chroma distance wasn't being properly converted to a similarity score
                        
                        # Convert distance to similarity (ensure a reasonable value)
                        # Chroma uses cosine distance, which ranges from 0 (identical) to 2 (completely opposite)
                        # We need to convert this to a similarity score from 0 to 1
                        
                        # Proper calculation for cosine similarity from distance
                        content_score = max(0.0, 1.0 - (distance / 2.0))
                        
                        # Apply score boosting
                        if content_score > 0.8:
                            content_score = min(1.0, content_score * 1.15)  # 15% boost for high matches
                        elif content_score > 0.6:
                            content_score = min(1.0, content_score * 1.1)   # 10% boost for good matches
                            
                        # Add usage-based boosting
                        usage_count = record.get("usage_count", 0)
                        success_rate = record.get("success_count", 0) / max(usage_count, 1)
                        boost = min(0.05, (usage_count / 200) * success_rate)
                        
                        final_score = min(1.0, content_score + boost)
                        
                        if final_score >= threshold:
                            # For standard search, use a neutral task score of 0.5
                            search_results.append((record, final_score, content_score, 0.5))
                
                # If we didn't get any good results, try a secondary approach
                if not search_results:
                    logger.debug("No results from primary search, trying secondary approach")
                    
                    # Get all records
                    active_records = [r for r in self.records if r.get("status", "active") == "active"]
                    
                    if record_type:
                        active_records = [r for r in active_records if r["record_type"] == record_type]
                        
                    if domain:
                        active_records = [r for r in active_records if r.get("domain") in [domain, "general"]]
                    
                    # Directly compute similarity for each record
                    for record in active_records:
                        record_text = self._get_record_vector_text(record)
                        if record_text is None:
                            continue
                            
                        record_embedding = await self.llm_service.embed(record_text)
                        content_score = await self.compute_similarity(query_embedding, record_embedding)
                        
                        # Apply score boosting
                        if content_score > 0.8:
                            content_score = min(1.0, content_score * 1.15)
                        elif content_score > 0.6:
                            content_score = min(1.0, content_score * 1.1)
                        
                        final_score = content_score  # No task weighting for standard search
                        
                        if final_score >= threshold:
                            search_results.append((record, final_score, content_score, 0.5))
            
            # Log the number of results before sorting/limiting
            logger.debug(f"Found {len(search_results)} search results before sorting and limiting")
            
            # Sort by final score and return top results
            search_results.sort(key=lambda x: x[1], reverse=True)
            return search_results[:limit]
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}", exc_info=True)
            raise
    
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
        
        # Check if query is None
        if query is None:
            raise ValueError("Query string cannot be None")
            
        # Get embedding for the query
        query_embedding = await self.llm_service.embed(query)
        
        # Create functional texts for batch embedding
        record_texts = []
        valid_records = []
        for record in active_records:
            record_text = self._get_record_vector_text(record)
            # Check for None before adding
            if record_text is None:
                logger.warning(f"Record {record.get('id', 'unknown')} generated None text, skipping")
                continue
            record_texts.append(record_text)
            valid_records.append(record)
        
        # Ensure we have records to process
        if not record_texts:
            logger.warning("No valid record texts found for embedding")
            return []
            
        # Generate embeddings for all records in one batch
        record_embeddings = await self.llm_service.embed_batch(record_texts)
        
        # Compute similarities
        results = []
        for i, record in enumerate(valid_records):
            # Skip records that might have been filtered out due to None text
            if i >= len(record_embeddings):
                continue
                
            # Compute similarity
            similarity = await self.compute_similarity(query_embedding, record_embeddings[i])
            
            # Apply additional weighting based on usage metrics if available
            usage_count = record.get("usage_count", 0)
            success_rate = record.get("success_count", 0) / max(usage_count, 1)
            
            # Boost score for frequently used and successful records (small boost of max 5%)
            boost = min(0.05, (usage_count / 200) * success_rate)
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
        
        # Generate applicability text for task-aware search
        try:
            applicability_text = await self.generate_applicability_text(record)
            record["metadata"]["applicability_text"] = applicability_text
            logger.info(f"Generated applicability text for {name}")
        except Exception as e:
            logger.warning(f"Failed to generate applicability text for {name}: {e}")
        
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
        
        # Generate updated applicability text for the evolved record
        try:
            applicability_text = await self.generate_applicability_text(new_record)
            new_record["metadata"]["applicability_text"] = applicability_text
            logger.info(f"Generated updated applicability text for evolved {new_record['name']}")
        except Exception as e:
            logger.warning(f"Failed to generate applicability text for evolved {new_record['name']}: {e}")
            # Inherit parent's applicability text if available
            if "applicability_text" in parent.get("metadata", {}):
                new_record["metadata"]["applicability_text"] = parent["metadata"]["applicability_text"]
        
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
                
            # Generate applicability text for task-aware search if not present
            if "applicability_text" not in record.get("metadata", {}):
                try:
                    record["metadata"] = record.get("metadata", {})
                    record["metadata"]["applicability_text"] = await self.generate_applicability_text(record)
                except Exception as e:
                    logger.warning(f"Failed to generate applicability text for {record.get('name', 'unknown')}: {e}")
                
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
            key=lambda r: r.get("success_count", 0) / max(r.get("usage_count", 1), 1), 
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
                    "success_rate": r.get("success_count", 0) / max(r.get("usage_count", 1), 1)
                }
                for r in most_successful
            ],
            "recently_updated": [
                {"id": r["id"], "name": r["name"], "last_updated": r.get("last_updated", "")}
                for r in recently_updated
            ]
        }