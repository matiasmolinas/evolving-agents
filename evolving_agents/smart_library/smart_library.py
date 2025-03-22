# evolving_agents/smart_library/smart_library.py

import json
import os
import logging
import uuid
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union, Set

# Import our capability contract model
from evolving_agents.core.capability_contract import (
    CapabilityContract, dict_to_capability_contract, capability_contract_to_dict
)

from evolving_agents.utils.embeddings import embedding_service

logger = logging.getLogger(__name__)

class SmartLibrary:
    """
    Unified library that stores all agents, tools, and firmware as simple dictionary records.
    Includes performance metrics for informed component selection and capability-based search.
    Leverages LLMs for intelligent matching and component selection.
    """
    def __init__(
        self, 
        storage_path: str = "smart_library.json", 
        llm_service = None
    ):
        self.storage_path = storage_path
        self.records = []
        self.llm_service = llm_service  # LLM service for intelligent component selection
        self._load_library()
    
    def set_llm_service(self, llm_service):
        """Set the LLM service to use for intelligent component selection."""
        self.llm_service = llm_service
        logger.info("LLM service set for SmartLibrary")
    
    def _load_library(self):
        """Load the library from storage."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    self.records = json.load(f)
                logger.info(f"Loaded {len(self.records)} records from {self.storage_path}")
            except Exception as e:
                logger.error(f"Error loading library: {str(e)}")
                self.records = []
        else:
            logger.info(f"No existing library found at {self.storage_path}. Creating new library.")
            self.records = []
    
    def _save_library(self):
        """Save the library to storage."""
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, indent=2)
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
        """Find a record by ID."""
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
    
    async def semantic_search(
        self, 
        query: str, 
        record_type: Optional[str] = None,
        domain: Optional[str] = None,
        limit: int = 5,
        threshold: float = 0.0,
        capability_focus: bool = True,
        use_llm: bool = True
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for records semantically similar to the query using embeddings and optionally LLM.
        
        Args:
            query: The search query
            record_type: Optional record type filter
            domain: Optional domain filter
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            capability_focus: Whether to prioritize capability matching
            use_llm: Whether to use LLM for enhancing search (if available)
            
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
        
        # Use LLM to enhance query if available and requested
        enhanced_query = query
        if self.llm_service and use_llm and capability_focus:
            enhanced_query = await self._enhance_query_with_llm(query, domain)
            logger.debug(f"LLM enhanced query: {enhanced_query}")
        elif capability_focus:
            # Fall back to heuristic enhancement if no LLM
            enhanced_query = self._enhance_query_with_capability_focus(query)
            logger.debug(f"Heuristic enhanced query: {enhanced_query}")
        
        # Get embedding for the query
        query_embedding = await embedding_service.generate_embedding(enhanced_query)
        
        # For each record, calculate similarity
        results = []
        for record in active_records:
            # Extract record information for embedding comparison
            record_text = self._prepare_record_text_for_embedding(record, capability_focus)
            
            # Generate embedding for the record
            record_embedding = await embedding_service.generate_embedding(record_text)
            
            # Compute similarity
            similarity = embedding_service.compute_similarity(query_embedding, record_embedding)
            
            # Use LLM to evaluate capability match if available
            if self.llm_service and use_llm and capability_focus:
                llm_boost = await self._calculate_capability_match_with_llm(query, record)
                similarity = min(1.0, similarity + llm_boost)
            elif capability_focus and record.get("capabilities"):
                # Fall back to heuristic boosting if no LLM
                capability_boost = self._calculate_capability_boost(query, record.get("capabilities", []))
                similarity = min(1.0, similarity + capability_boost)
            
            if similarity >= threshold:
                results.append((record, similarity))
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    async def _enhance_query_with_llm(self, query: str, domain: Optional[str] = None) -> str:
        """Use LLM to enhance a query for better capability matching."""
        if not self.llm_service:
            return query
            
        try:
            prompt = f"""
            You are a specialized AI assistant that translates user requirements into precise capability queries.
            
            USER REQUIREMENT:
            "{query}"
            
            {f"DOMAIN CONTEXT: {domain}" if domain else ""}
            
            First, identify the core capabilities needed to satisfy this requirement. 
            Then, formulate a search query that would find components with these capabilities.
            
            Your query should:
            1. Be specific about what the component should DO
            2. Include key technical terms related to the capabilities
            3. Focus on the functional requirements, not implementation details
            
            FORMAT: Return only the enhanced search query without explanation or additional text.
            """
            
            result = await self.llm_service.generate(prompt)
            
            # Clean up the response
            result = result.strip()
            if result.startswith('"') and result.endswith('"'):
                result = result[1:-1].strip()
                
            # Fall back to original if something went wrong
            if not result or len(result) < 10:
                return query
                
            return result
            
        except Exception as e:
            logger.warning(f"Error enhancing query with LLM: {str(e)}. Using original query.")
            return query
    
    def _enhance_query_with_capability_focus(self, query: str) -> str:
        """Enhance a query to better focus on capabilities using heuristics."""
        # List of terms that suggest the query is about a capability
        capability_indicators = [
            "can", "able to", "capability", "function", "ability", 
            "perform", "handle", "process", "analyze", "extract",
            "identify", "detect", "recognize", "understand"
        ]
        
        # Check if query already contains capability indicators
        has_indicator = any(indicator in query.lower() for indicator in capability_indicators)
        
        if has_indicator:
            # Already capability-focused
            return query
        else:
            # Add capability focus if it's likely describing a capability
            if "that" in query or "which" in query:
                return f"component with capability to {query}"
            else:
                return f"component that can {query}"
    
    def _prepare_record_text_for_embedding(self, record: Dict[str, Any], capability_focus: bool) -> str:
        """Prepare a record's text representation for embedding comparison."""
        name = record.get("name", "")
        description = record.get("description", "")
        record_type = record.get("record_type", "UNKNOWN")
        capabilities = record.get("capabilities", [])
        
        if capabilities and capability_focus:
            # Create capability-focused text representation
            capability_texts = []
            for cap in capabilities:
                cap_id = cap.get("id", "")
                cap_name = cap.get("name", "")
                cap_desc = cap.get("description", "")
                
                # Build a rich capability description
                cap_text = f"{cap_id} {cap_name} {cap_desc}"
                
                # Add context requirements if available
                if "context" in cap:
                    context = cap["context"]
                    required = context.get("required_fields", [])
                    produced = context.get("produced_fields", [])
                    if required:
                        cap_text += f" requires {', '.join(required if isinstance(required, list) else [required])}"
                    if produced:
                        cap_text += f" produces {', '.join(produced if isinstance(produced, list) else [produced])}"
                capability_texts.append(cap_text)
            
            # Type is important for matching AGENT vs TOOL
            type_text = f"record_type:{record_type}"
                
            return f"{name} {description} {type_text} Capabilities: {' '.join(capability_texts)}"
        else:
            # Create a general text representation
            code_snippet = record.get("code_snippet", "")
            # Limit code snippet length to avoid embedding token limits
            if len(code_snippet) > 500:
                code_snippet = code_snippet[:500] + "..."
            
            # Type is important for matching AGENT vs TOOL
            type_text = f"record_type:{record_type}"
            
        return f"{name} {description} {type_text} {code_snippet}"
    
    async def _calculate_capability_match_with_llm(self, query: str, record: Dict[str, Any]) -> float:
        """Use LLM to evaluate how well a record's capabilities match a query."""
        if not self.llm_service:
            return 0.0
            
        try:
            # Extract capabilities information
            capabilities = record.get("capabilities", [])
            if not capabilities:
                return 0.0
                
            # Format capabilities information
            capabilities_text = []
            for i, cap in enumerate(capabilities[:5]):  # Limit to 5 capabilities to avoid token limits
                cap_text = f"Capability {i+1}: {cap.get('name', '')}\n"
                cap_text += f"Description: {cap.get('description', '')}\n"
                
                if "context" in cap:
                    context = cap["context"]
                    required = context.get("required_fields", [])
                    produced = context.get("produced_fields", [])
                    if required:
                        req_fields = required if isinstance(required, list) else [required]
                        cap_text += f"Requires: {', '.join(req_fields)}\n"
                    if produced:
                        prod_fields = produced if isinstance(produced, list) else [produced]
                        cap_text += f"Produces: {', '.join(prod_fields)}\n"
                        
                capabilities_text.append(cap_text)
            
            capabilities_str = "\n".join(capabilities_text)
            
            prompt = f"""
            I need to evaluate how well a component's capabilities match a user's request.
            
            USER REQUEST:
            "{query}"
            
            COMPONENT NAME: {record.get('name', '')}
            COMPONENT DESCRIPTION: {record.get('description', '')}
            
            COMPONENT CAPABILITIES:
            {capabilities_str}
            
            On a scale from 0.0 to 0.3, where 0.0 means "no match" and 0.3 means "perfect match",
            how well do the component's capabilities match the user's request?
            
            Return ONLY a single float number between 0.0 and 0.3, nothing else.
            """
            
            result = await self.llm_service.generate(prompt)
            
            # Try to parse the result as a float
            result = result.strip()
            try:
                boost = float(result)
                # Ensure it's within the allowed range
                boost = max(0.0, min(0.3, boost))
                return boost
            except ValueError:
                logger.warning(f"Failed to parse LLM capability match result: {result}")
                return 0.0
                
        except Exception as e:
            logger.warning(f"Error calculating capability match with LLM: {str(e)}")
            return 0.0
    
    def _calculate_capability_boost(self, query: str, capabilities: List[Dict[str, Any]]) -> float:
        """Calculate a similarity boost based on specific capability matches using heuristics."""
        boost = 0.0
        query_lower = query.lower()
        
        # Extract key terms from the query
        words = set([w.lower() for w in query_lower.split() if len(w) > 3])
        
        for capability in capabilities:
            cap_name = capability.get("name", "").lower()
            cap_desc = capability.get("description", "").lower()
            
            # Calculate term overlap with capability name and description
            name_overlap = sum(1 for word in words if word in cap_name) / max(1, len(cap_name.split()))
            desc_overlap = sum(1 for word in words if word in cap_desc) / max(1, len(cap_desc.split()))
            
            # Calculate boost based on overlap
            cap_boost = (name_overlap * 0.2) + (desc_overlap * 0.1)
            boost = max(boost, cap_boost)  # Take the highest boost from any capability
        
        return min(0.3, boost)  # Cap at 0.3 to avoid overwhelming the embedding similarity
    
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
        metadata: Optional[Dict[str, Any]] = None,
        capabilities: Optional[List[Dict[str, Any]]] = None
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
            capabilities: Optional list of capabilities this component provides
            
        Returns:
            The created record
        """
        record_id = str(uuid.uuid4())
        
        # Initialize empty performance metrics structure
        performance_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "average_execution_time": 0.0,
            "capabilities": {},
            "domains": {},
            "last_execution": None
        }
        
        # Process capabilities - ensure they have proper structure
        processed_capabilities = []
        if capabilities:
            for cap in capabilities:
                # Ensure capability has an ID
                if "id" not in cap:
                    cap["id"] = str(uuid.uuid4())
                
                # Ensure it has a name if not provided
                if "name" not in cap:
                    # Extract name from id or description
                    if "description" in cap:
                        words = cap["description"].split()
                        cap["name"] = " ".join(words[:3]) + "..."
                    else:
                        cap["name"] = f"Capability {cap['id'][-6:]}"
                
                # Ensure it has context structure if using capability contracts
                if "context" not in cap:
                    cap["context"] = {
                        "required_fields": [],
                        "optional_fields": [],
                        "produced_fields": []
                    }
                
                processed_capabilities.append(cap)
        
        # If using LLM and no capabilities provided, try to infer them
        if self.llm_service and not capabilities:
            inferred_capabilities = await self._infer_capabilities(name, description, record_type, domain, code_snippet)
            if inferred_capabilities:
                processed_capabilities.extend(inferred_capabilities)
        
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
            "metadata": metadata or {},
            "performance_metrics": performance_metrics,
            "capabilities": processed_capabilities
        }
        
        await self.save_record(record)
        logger.info(f"Created new {record_type} record: {name}")
        return record
    
    async def _infer_capabilities(
        self, 
        name: str, 
        description: str, 
        record_type: str,
        domain: str,
        code_snippet: str
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to infer capabilities from a component's description and code.
        
        Args:
            name: Component name
            description: Component description
            record_type: Component type (AGENT or TOOL)
            domain: Component domain
            code_snippet: Component code
            
        Returns:
            List of inferred capabilities
        """
        try:
            # Truncate code snippet if it's too long
            if len(code_snippet) > 2000:
                code_snippet = code_snippet[:2000] + "...(truncated)"
                
            prompt = f"""
            Analyze this {record_type.lower()} and identify its capabilities:
            
            NAME: {name}
            DESCRIPTION: {description}
            DOMAIN: {domain}
            
            CODE:
            {code_snippet}
            
            Generate a list of capabilities this component provides. Each capability should include:
            1. id: A unique identifier (lowercase with underscores)
            2. name: A human-readable name
            3. description: What the capability does
            4. context: Required inputs and produced outputs
            
            Format your response as a JSON array of capability objects:
            [
              {{
                "id": "unique_capability_id",
                "name": "Capability Name",
                "description": "Detailed description of what this capability does",
                "context": {{
                  "required_fields": ["field1", "field2"],
                  "produced_fields": ["field3", "field4"]
                }}
              }}
            ]
            
            Return ONLY the JSON array, nothing else.
            """
            
            result = await self.llm_service.generate(prompt)
            
            # Clean up the response and parse JSON
            try:
                # Extract JSON if in markdown format
                if "```json" in result:
                    result = result.split("```json")[1].split("```")[0].strip()
                elif "```" in result:
                    result = result.split("```")[1].split("```")[0].strip()
                    
                capabilities = json.loads(result)
                if isinstance(capabilities, list):
                    return capabilities
                return []
            except Exception as e:
                logger.warning(f"Error parsing inferred capabilities: {str(e)}")
                return []
        except Exception as e:
            logger.warning(f"Error inferring capabilities: {str(e)}")
            return []
    
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
            },
            # Initialize empty performance metrics
            "performance_metrics": {
                "total_executions": 0,
                "successful_executions": 0,
                "average_execution_time": 0.0,
                "capabilities": {},
                "domains": {},
                "last_execution": None
            },
            # Copy capabilities from parent
            "capabilities": parent.get("capabilities", []).copy()
        }
        
        # If using LLM, check if capabilities should be evolved
        if self.llm_service and "capabilities" in parent:
            evolved_capabilities = await self._evolve_capabilities(parent, new_code_snippet, description)
            if evolved_capabilities:
                new_record["capabilities"] = evolved_capabilities
        
        # Save and return
        await self.save_record(new_record)
        logger.info(f"Evolved record {parent['name']} from {parent['version']} to {new_version}")
        
        return new_record
    
    async def _evolve_capabilities(
        self, 
        parent: Dict[str, Any],
        new_code_snippet: str,
        new_description: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Evolve a component's capabilities based on changes.
        
        Args:
            parent: Parent component record
            new_code_snippet: Updated code snippet
            new_description: Updated description if available
            
        Returns:
            Updated list of capabilities
        """
        try:
            description = new_description or parent.get("description", "")
            parent_capabilities = parent.get("capabilities", [])
            
            # If no parent capabilities, infer from the new code
            if not parent_capabilities:
                return await self._infer_capabilities(
                    parent["name"], 
                    description, 
                    parent["record_type"],
                    parent["domain"],
                    new_code_snippet
                )
            
            # Format the parent capabilities
            capabilities_str = json.dumps(parent_capabilities, indent=2)
            
            prompt = f"""
            I need to evolve the capabilities of a component based on code changes.
            
            ORIGINAL COMPONENT:
            Name: {parent["name"]}
            Description: {parent["description"]}
            
            CURRENT CAPABILITIES:
            {capabilities_str}
            
            UPDATED COMPONENT:
            Description: {description}
            
            CODE CHANGES:
            {self._summarize_code_diff(parent["code_snippet"], new_code_snippet)}
            
            Analyze the changes and update the capabilities. You can:
            1. Keep existing capabilities unchanged
            2. Modify existing capabilities to reflect improvements
            3. Add new capabilities if the code now supports them
            4. Remove capabilities that are no longer supported
            
            Return a JSON array with the updated capabilities in the same format:
            [
              {{
                "id": "capability_id",
                "name": "Capability Name",
                "description": "Updated description",
                "context": {{
                  "required_fields": [...],
                  "produced_fields": [...]
                }}
              }}
            ]
            
            Return ONLY the JSON array, nothing else.
            """
            
            result = await self.llm_service.generate(prompt)
            
            # Clean up the response and parse JSON
            try:
                # Extract JSON if in markdown format
                if "```json" in result:
                    result = result.split("```json")[1].split("```")[0].strip()
                elif "```" in result:
                    result = result.split("```")[1].split("```")[0].strip()
                    
                capabilities = json.loads(result)
                if isinstance(capabilities, list):
                    return capabilities
                return parent_capabilities  # Fall back to parent capabilities if parsing fails
            except Exception as e:
                logger.warning(f"Error parsing evolved capabilities: {str(e)}")
                return parent_capabilities
        except Exception as e:
            logger.warning(f"Error evolving capabilities: {str(e)}")
            return parent.get("capabilities", []).copy()
    
    def _summarize_code_diff(self, old_code: str, new_code: str) -> str:
        """
        Create a simple summary of the differences between two code snippets.
        
        Args:
            old_code: Original code snippet
            new_code: New code snippet
            
        Returns:
            A summary of the differences
        """
        # Simple line-based diff - this is a basic implementation
        # In a real system, consider using a proper diff library
        old_lines = old_code.strip().split("\n")
        new_lines = new_code.strip().split("\n")
        
        # Count added, removed, and modified lines (simple approximation)
        removed = len(old_lines) - sum(1 for line in old_lines if line in new_lines)
        added = len(new_lines) - sum(1 for line in new_lines if line in old_lines)
        
        # Extract function/method names using regex
        def extract_functions(code):
            method_pattern = r'def\s+([a-zA-Z0-9_]+)'
            class_pattern = r'class\s+([a-zA-Z0-9_]+)'
            methods = re.findall(method_pattern, code)
            classes = re.findall(class_pattern, code)
            return {"methods": methods, "classes": classes}
        
        old_functions = extract_functions(old_code)
        new_functions = extract_functions(new_code)
        
        new_methods = [m for m in new_functions["methods"] if m not in old_functions["methods"]]
        removed_methods = [m for m in old_functions["methods"] if m not in new_functions["methods"]]
        
        # Create a summary
        summary = [
            f"Code changes summary:",
            f"- Added approximately {added} new lines",
            f"- Removed approximately {removed} lines"
        ]
        
        if new_methods:
            summary.append(f"- Added methods: {', '.join(new_methods)}")
        if removed_methods:
            summary.append(f"- Removed methods: {', '.join(removed_methods)}")
            
        return "\n".join(summary)
    
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
    
    # Enhanced capability-related methods
    
    async def update_component_metrics(
        self,
        component_id: str,
        execution_metrics: Dict[str, Any]
    ) -> bool:
        """
        Update performance metrics for a component based on instance execution.
        
        Args:
            component_id: ID of the component definition
            execution_metrics: Metrics from a runtime instance containing:
                - success: Whether execution was successful
                - execution_time: Time taken to execute
                - capability_id: ID of the executed capability
                - domain: Optional domain context
                - timestamp: When execution occurred
                
        Returns:
            True if metrics were updated, False otherwise
        """
        record = await self.find_record_by_id(component_id)
        if not record:
            logger.warning(f"Cannot update metrics for non-existent component: {component_id}")
            return False
        
        # Initialize metrics structure if it doesn't exist
        if "performance_metrics" not in record:
            record["performance_metrics"] = {
                "total_executions": 0,
                "successful_executions": 0,
                "average_execution_time": 0.0,
                "capabilities": {},
                "domains": {},
                "last_execution": None
            }
        
        metrics = record["performance_metrics"]
        
        # Update overall metrics
        metrics["total_executions"] = metrics.get("total_executions", 0) + 1
        if execution_metrics.get("success", False):
            metrics["successful_executions"] = metrics.get("successful_executions", 0) + 1
        
        # Update execution time rolling average
        prev_avg = metrics.get("average_execution_time", 0.0)
        n = metrics["total_executions"]
        execution_time = execution_metrics.get("execution_time", 0.0)
        if n > 1:  # Avoid division by zero
            metrics["average_execution_time"] = (prev_avg * (n-1) + execution_time) / n
        else:
            metrics["average_execution_time"] = execution_time
        
        # Update timestamp
        metrics["last_execution"] = execution_metrics.get("timestamp", datetime.utcnow().isoformat())
        
        # Update capability-specific metrics
        capability_id = execution_metrics.get("capability_id")
        if capability_id:
            if "capabilities" not in metrics:
                metrics["capabilities"] = {}
                
            if capability_id not in metrics["capabilities"]:
                metrics["capabilities"][capability_id] = {
                    "total_executions": 0,
                    "successful_executions": 0,
                    "average_execution_time": 0.0,
                    "last_execution": None
                }
                
            cap_metrics = metrics["capabilities"][capability_id]
            cap_metrics["total_executions"] = cap_metrics.get("total_executions", 0) + 1
            if execution_metrics.get("success", False):
                cap_metrics["successful_executions"] = cap_metrics.get("successful_executions", 0) + 1
                
            # Update capability execution time rolling average
            prev_cap_avg = cap_metrics.get("average_execution_time", 0.0)
            n_cap = cap_metrics["total_executions"]
            if n_cap > 1:  # Avoid division by zero
                cap_metrics["average_execution_time"] = (prev_cap_avg * (n_cap-1) + execution_time) / n_cap
            else:
                cap_metrics["average_execution_time"] = execution_time
                
            cap_metrics["last_execution"] = execution_metrics.get("timestamp", datetime.utcnow().isoformat())
        
        # Update domain-specific metrics
        domain = execution_metrics.get("domain")
        if domain:
            if "domains" not in metrics:
                metrics["domains"] = {}
                
            if domain not in metrics["domains"]:
                metrics["domains"][domain] = {
                    "total_executions": 0,
                    "successful_executions": 0
                }
                
            domain_metrics = metrics["domains"][domain]
            domain_metrics["total_executions"] = domain_metrics.get("total_executions", 0) + 1
            if execution_metrics.get("success", False):
                domain_metrics["successful_executions"] = domain_metrics.get("successful_executions", 0) + 1
        
        # Save updated record
        await self.save_record(record)
        logger.info(f"Updated performance metrics for component {record['name']} ({component_id})")
        
        return True
    
    async def find_component_by_capability(
        self,
        capability_id: str,
        domain: Optional[str] = None,
        min_success_rate: float = 0.5,
        prefer_performance: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Find the best component for implementing a capability based on performance history.
        This method is enhanced to better handle partial matches and semantic similarity.
        
        Args:
            capability_id: ID of the required capability
            domain: Optional domain context
            min_success_rate: Minimum acceptable success rate
            prefer_performance: Whether to prioritize performance history over semantic similarity
            
        Returns:
            The best component record or None if no suitable component found
        """
        # First try to find direct matches by capability ID
        direct_matches = []
        
        # Clean the capability ID for matching (remove common words)
        clean_capability = capability_id.lower().replace("agent", "").replace("tool", "").strip()
        
        for record in self.records:
            # Skip inactive records
            if record.get("status", "active") != "active":
                continue
            
            # Check for direct match on capability ID
            for cap in record.get("capabilities", []):
                cap_id = cap.get("id", "").lower()
                
                # Check for exact match or if the clean capability is contained in the capability ID
                if cap_id == capability_id.lower() or clean_capability in cap_id:
                    direct_matches.append(record)
                    break
        
        # If direct matches found, return the most appropriate one
        if direct_matches:
            if len(direct_matches) == 1:
                return direct_matches[0]
            
            # If multiple matches, prioritize by domain if specified
            if domain:
                domain_matches = [r for r in direct_matches if r.get("domain") == domain]
                if domain_matches:
                    return domain_matches[0]
            
            # Otherwise return first match
            return direct_matches[0]
        
        # If no direct matches, try semantic search
        search_query = f"component that can {capability_id}"
        if domain:
            search_query += f" for {domain} domain"
        
        search_results = await self.semantic_search(
            query=search_query,
            domain=domain,
            threshold=0.5  # Lower threshold to catch more potential matches
        )
        
        if search_results:
            return search_results[0][0]
        
        # No matches found
        return None
    
    async def find_components_by_capability_description(
        self, 
        capability_description: str,
        domain: Optional[str] = None,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find components that can provide a capability based on its description.
        
        Args:
            capability_description: Description of the capability needed
            domain: Optional domain context
            limit: Maximum number of results to return
            
        Returns:
            List of component records that can provide the capability
        """
        if not self.llm_service:
            # Fall back to semantic search without LLM
            return await self._fallback_capability_search(capability_description, domain, limit)
        
        try:
            # First, identify existing components that might match
            candidates = []
            for record in self.records:
                if record.get("status", "active") != "active":
                    continue
                    
                if domain and record.get("domain") != domain:
                    continue
                    
                # Add to candidates if it has capabilities defined
                if "capabilities" in record and record["capabilities"]:
                    candidates.append(record)
            
            if not candidates:
                return []
                
            # Use LLM to evaluate how well each component matches the capability
            matching_components = []
            for record in candidates:
                # Prepare a description of the component's capabilities
                capability_texts = []
                for cap in record.get("capabilities", []):
                    cap_text = f"{cap.get('name', '')}: {cap.get('description', '')}"
                    capability_texts.append(cap_text)
                    
                capabilities_str = "\n".join(capability_texts)
                
                # Ask LLM to evaluate the match
                prompt = f"""
                You need to determine if a component can provide the following capability:
                
                REQUIRED CAPABILITY: {capability_description}
                
                COMPONENT: {record['name']}
                COMPONENT DESCRIPTION: {record['description']}
                COMPONENT CAPABILITIES:
                {capabilities_str}
                
                Evaluate how well this component can provide the required capability on a scale from 0 to 100,
                where 0 means "cannot provide this capability at all" and 100 means "perfectly matches this capability".
                
                Return ONLY a single number from 0-100, nothing else.
                """
                
                result = await self.llm_service.generate(prompt)
                
                # Parse the result
                try:
                    match_score = float(result.strip())
                    if match_score >= 50:  # Only include if at least 50% match
                        matching_components.append((record, match_score))
                except ValueError:
                    # If parsing fails, estimate a score using string similarity
                    similarity = self._estimate_similarity(capability_description, record["description"])
                    if similarity >= 0.5:
                        matching_components.append((record, similarity * 100))
            
            # Sort by score and return top results
            matching_components.sort(key=lambda x: x[1], reverse=True)
            return [comp[0] for comp in matching_components[:limit]]
        
        except Exception as e:
            logger.warning(f"Error in capability-based search: {str(e)}")
            return await self._fallback_capability_search(capability_description, domain, limit)
    
    async def _fallback_capability_search(
        self, 
        capability_description: str, 
        domain: Optional[str] = None,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Fallback search method using basic semantic matching."""
        query = f"component that can {capability_description}"
        if domain:
            query += f" in {domain} domain"
            
        search_results = await self.semantic_search(
            query=query,
            domain=domain,
            limit=limit,
            threshold=0.5,
            capability_focus=True,
            use_llm=False  # Don't use LLM for enhancement in fallback
        )
        
        return [r[0] for r in search_results]
    
    def _estimate_similarity(self, text1: str, text2: str) -> float:
        """Estimate similarity between two texts using string comparison."""
        # Simple Jaccard similarity for word sets
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_component_success_rate(
        self,
        component_id: str,
        capability_id: Optional[str] = None,
        domain: Optional[str] = None
    ) -> float:
        """
        Get the success rate for a component, optionally for a specific capability and domain.
        
        Args:
            component_id: ID of the component
            capability_id: Optional specific capability to check
            domain: Optional domain context
            
        Returns:
            Success rate as a float between 0.0 and 1.0
        """
        record = None
        for r in self.records:
            if r["id"] == component_id:
                record = r
                break
                
        if not record:
            return 0.0
            
        metrics = record.get("performance_metrics", {})
        
        # If checking for specific capability
        if capability_id:
            cap_metrics = metrics.get("capabilities", {}).get(capability_id, {})
            total = cap_metrics.get("total_executions", 0)
            successful = cap_metrics.get("successful_executions", 0)
            
            # If also filtering by domain
            if domain and total > 0:
                domain_metrics = metrics.get("domains", {}).get(domain, {})
                domain_total = domain_metrics.get("total_executions", 0)
                domain_successful = domain_metrics.get("successful_executions", 0)
                
                # If we have domain data, use a weighted combination
                if domain_total > 0:
                    # Weight domain-specific data more heavily
                    capability_weight = 0.4
                    domain_weight = 0.6
                    capability_rate = successful / total
                    domain_rate = domain_successful / domain_total
                    return (capability_rate * capability_weight) + (domain_rate * domain_weight)
            
            # Just capability rate if no domain or no domain data
            return successful / total if total > 0 else 0.0
            
        # Overall success rate
        total = metrics.get("total_executions", 0)
        successful = metrics.get("successful_executions", 0)
        
        # If filtering by domain
        if domain:
            domain_metrics = metrics.get("domains", {}).get(domain, {})
            domain_total = domain_metrics.get("total_executions", 0)
            domain_successful = domain_metrics.get("successful_executions", 0)
            
            if domain_total > 0:
                return domain_successful / domain_total
        
        # Overall rate if no capability/domain specified or no domain data
        return successful / total if total > 0 else 0.0
    
    async def get_component_capabilities(
        self,
        component_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get the capabilities provided by a component.
        
        Args:
            component_id: ID of the component
            
        Returns:
            List of capability dictionaries
        """
        record = await self.find_record_by_id(component_id)
        if not record:
            return []
            
        return record.get("capabilities", [])
        
    async def find_components_for_workflow(
        self, 
        workflow_description: str,
        required_capabilities: List[str] = None,
        domain: str = None,
        use_llm: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find the best components for a workflow based on its description and required capabilities.
        
        Args:
            workflow_description: Description of the workflow
            required_capabilities: List of capability IDs needed for the workflow
            domain: Optional domain context
            use_llm: Whether to use LLM for capability analysis (if available)
            
        Returns:
            Dictionary mapping capability IDs to lists of potential components
        """
        # If LLM is available and requested, use it to extract capabilities from workflow
        extracted_capabilities = []
        if self.llm_service and use_llm and workflow_description:
            extracted_caps = await self._extract_capabilities_with_llm(workflow_description, domain)
            if extracted_caps:
                extracted_capabilities = extracted_caps
        
        # Combine explicit and extracted capabilities
        all_capabilities = set()
        if required_capabilities:
            all_capabilities.update(required_capabilities)
        if extracted_capabilities:
            all_capabilities.update(extracted_capabilities)
        
        result = {}
        
        # Find components for each capability
        for cap_id in all_capabilities:
            # Try to find the best component for this capability
            component = await self.find_component_by_capability(
                capability_id=cap_id,
                domain=domain,
                min_success_rate=0.6
            )
            
            if component:
                result[cap_id] = [component]
            else:
                # Use semantic search to find potential matches
                query = f"component that provides {cap_id} capability"
                if domain:
                    query += f" in {domain} domain"
                
                search_results = await self.semantic_search(
                    query=query,
                    threshold=0.6,
                    capability_focus=True,
                    use_llm=use_llm
                )
                
                if search_results:
                    result[cap_id] = [r[0] for r in search_results[:3]]
                else:
                    result[cap_id] = []
        
        # Use the workflow description to find additional relevant components
        search_results = await self.semantic_search(
            query=workflow_description,
            domain=domain,
            threshold=0.5,
            limit=10,
            capability_focus=True,
            use_llm=use_llm
        )
        
        if search_results:
            # Group by capability
            for record, score in search_results:
                for cap in record.get("capabilities", []):
                    cap_id = cap.get("id", "")
                    if cap_id:
                        if cap_id not in result:
                            result[cap_id] = []
                        if record not in result[cap_id]:
                            result[cap_id].append(record)
        
        return result
    
    async def _extract_capabilities_with_llm(self, workflow_description: str, domain: Optional[str] = None) -> List[str]:
        """
        Use LLM to extract required capabilities from a workflow description.
        
        Args:
            workflow_description: Description of the workflow
            domain: Optional domain context
            
        Returns:
            List of capability IDs
        """
        if not self.llm_service:
            return []
            
        try:
            prompt = f"""
            I need to analyze a workflow description and identify the key capabilities required.
            
            WORKFLOW DESCRIPTION:
            {workflow_description}
            
            {f"DOMAIN: {domain}" if domain else ""}
            
            Please identify the distinct capabilities required to implement this workflow.
            Each capability should be a concise, specific functionality (e.g., "text_extraction", "sentiment_analysis").
            Format capabilities as ID strings (lowercase with underscores).
            
            Return a JSON array containing only the capability IDs, nothing else.
            Example: ["document_analysis", "data_extraction", "sentiment_analysis"]
            """
            
            result = await self.llm_service.generate(prompt)
            
            # Try to parse the result as JSON
            try:
                # Clean up the result (in case the LLM added extra text)
                json_str = result.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].strip()
                
                capabilities = json.loads(json_str)
                if isinstance(capabilities, list):
                    return capabilities
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse LLM capability extraction result: {result}")
                
            return []
                
        except Exception as e:
            logger.warning(f"Error extracting capabilities with LLM: {str(e)}")
            return []
    
    async def analyze_workflow_component_compatibility(
        self,
        workflow_description: str,
        selected_components: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Use LLM to analyze if selected components will work well together in a workflow.
        
        Args:
            workflow_description: Description of the workflow
            selected_components: Dictionary mapping capability IDs to selected components
            
        Returns:
            Analysis results including compatibility score and potential issues
        """
        if not self.llm_service:
            return {"compatibility": "unknown", "message": "No LLM service available for analysis"}
            
        try:
            # Format components for LLM analysis
            components_text = []
            for cap_id, component in selected_components.items():
                component_text = f"Capability: {cap_id}\n"
                component_text += f"Component: {component.get('name', 'Unknown')}\n"
                component_text += f"Description: {component.get('description', 'No description')}\n"
                
                # Add capabilities details
                capabilities = component.get("capabilities", [])
                if capabilities:
                    cap_details = []
                    for cap in capabilities:
                        if cap.get("id") == cap_id:  # Focus on the relevant capability
                            cap_details.append(f"- {cap.get('description', 'No description')}")
                            # Add context info if available
                            if "context" in cap:
                                context = cap["context"]
                                required = context.get("required_fields", [])
                                produced = context.get("produced_fields", [])
                                if required:
                                    # Handle both list and string representations
                                    if isinstance(required, list):
                                        cap_details.append(f"  Requires: {', '.join(required)}")
                                    else:
                                        cap_details.append(f"  Requires: {required}")
                                if produced:
                                    # Handle both list and string representations
                                    if isinstance(produced, list):
                                        cap_details.append(f"  Produces: {', '.join(produced)}")
                                    else:
                                        cap_details.append(f"  Produces: {produced}")
                    
                    if cap_details:
                        component_text += "Details:\n" + "\n".join(cap_details) + "\n"
                
                components_text.append(component_text)
            
            components_str = "\n---\n".join(components_text)
            
            prompt = f"""
            I need to analyze if these components will work well together in a workflow.
            
            WORKFLOW DESCRIPTION:
            {workflow_description}
            
            SELECTED COMPONENTS:
            {components_str}
            
            Please analyze the compatibility of these components for this workflow. Consider:
            1. Do they cover all required capabilities?
            2. Will their inputs/outputs work together?
            3. Are there any missing capabilities or potential issues?
            
            Return your analysis as a JSON object with these fields:
            - compatibility_score: 0-100 score indicating how well these components will work together
            - missing_capabilities: list of any capabilities that seem to be missing
            - potential_issues: list of potential integration problems
            - recommendations: list of suggestions to improve the workflow
            """
            
            result = await self.llm_service.generate(prompt)
            
            # Try to parse the result as JSON
            try:
                # Clean up the result (in case the LLM added extra text)
                json_str = result.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].strip()
                
                analysis = json.loads(json_str)
                return analysis
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse LLM compatibility analysis result: {result}")
                return {
                    "compatibility_score": 50,
                    "message": "Analysis completed but result format was invalid",
                    "raw_response": result[:500]  # Include part of the raw response
                }
                
        except Exception as e:
            logger.warning(f"Error analyzing component compatibility with LLM: {str(e)}")
            return {
                "compatibility_score": 0,
                "error": str(e),
                "message": "Error occurred during compatibility analysis"
            }
    
    async def search_and_analyze_capabilities(
        self,
        query: str,
        domain: Optional[str] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Enhanced search that returns components with capability analysis.
        
        Args:
            query: Search query describing what you're looking for
            domain: Optional domain to search within
            limit: Maximum number of results
            
        Returns:
            Dictionary with results and capability analysis
        """
        if not self.llm_service:
            # Fall back to regular semantic search
            results = await self.semantic_search(query, domain=domain, limit=limit)
            return {
                "results": [r[0] for r in results],
                "capabilities_needed": [],
                "capability_coverage": {},
                "missing_capabilities": []
            }
        
        try:
            # First, identify the capabilities needed for this query
            required_capabilities = await self._extract_capabilities_with_llm(query, domain)
            
            # Get search results
            results = await self.semantic_search(query, domain=domain, limit=limit)
            
            # Analyze capability coverage
            capability_coverage = {}
            for cap_id in required_capabilities:
                capability_coverage[cap_id] = []
                
                for record, score in results:
                    # Check if this record provides the capability
                    for cap in record.get("capabilities", []):
                        if cap.get("id") == cap_id:
                            capability_coverage[cap_id].append({
                                "component_id": record["id"],
                                "component_name": record["name"],
                                "confidence": score
                            })
            
            # Identify missing capabilities
            missing_capabilities = [
                cap_id for cap_id in required_capabilities 
                if not capability_coverage.get(cap_id)
            ]
            
            return {
                "results": [(r[0], r[1]) for r in results],
                "capabilities_needed": required_capabilities,
                "capability_coverage": capability_coverage,
                "missing_capabilities": missing_capabilities
            }
        except Exception as e:
            logger.warning(f"Error in search and capability analysis: {str(e)}")
            # Fall back to regular search
            results = await self.semantic_search(query, domain=domain, limit=limit)
            return {
                "results": [r[0] for r in results],
                "error": str(e)
            }