import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

from evolving_agents.smart_library.smart_library import SmartLibrary
from beeai_framework.agents.react import ReActAgent

logger = logging.getLogger(__name__)

class SmartAgentBus:
    """
    A unified Agent Bus that integrates with SmartLibrary and supports an agent-centric,
    capability-based communication model. Capabilities are treated as descriptive criteria—
    if a record lacks explicit capability definitions, a default capability is derived from its description.
    
    Production strategy:
      - If no provider is found for a requested capability, the request fails.
      - If no system agent is configured to process the request, the request fails.
      - Execution errors from the system agent are propagated.
    """
    def __init__(
        self,
        smart_library: SmartLibrary,
        system_agent: Optional[ReActAgent] = None,
        storage_path: str = "smart_agent_bus.json",
        log_path: str = "agent_bus_logs.json"
    ):
        self.storage_path = storage_path
        self.log_path = log_path
        self.smart_library = smart_library
        self.system_agent = system_agent
        self.providers: Dict[str, Dict[str, Any]] = {}
        self.capabilities: Dict[str, Dict[str, Any]] = {}
        self._load_data()
    
    def _load_data(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.providers = data.get("providers", {})
                    self.capabilities = data.get("capabilities", {})
                logger.info(f"Loaded Agent Bus data from {self.storage_path}")
            except Exception as e:
                logger.error(f"Error loading Agent Bus data: {str(e)}")
                self.providers = {}
                self.capabilities = {}
        else:
            logger.info(f"No existing Agent Bus data found at {self.storage_path}, starting fresh.")
    
    def _save_data(self):
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump({
                "providers": self.providers,
                "capabilities": self.capabilities,
                "updated_at": datetime.utcnow().isoformat()
            }, f, indent=2)
        logger.info(f"Saved Agent Bus data to {self.storage_path}")
    
    def _validate_capability(self, capability: Dict[str, Any]) -> bool:
        required_keys = {"id", "name", "description"}
        return required_keys.issubset(capability.keys())
    
    def _derive_default_capability(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        If a record does not include explicit capabilities, derive one using its description.
        """
        return {
            "id": f"{record['name'].lower().replace(' ', '_')}_default",
            "name": record["name"],
            "description": record.get("description", ""),
            "confidence": 0.8
        }
    
    async def register_provider(
        self, 
        name: str, 
        capabilities: List[Dict[str, Any]], 
        provider_type: str = "AGENT",
        description: str = "",
        metadata: Dict[str, Any] = {}
    ) -> str:
        """
        Register a provider. If explicit capabilities are not provided, derive one from the description.
        """
        provider_id = f"{name.lower().replace(' ', '_')}_{provider_type.lower()}"
        if not capabilities:
            capabilities = [self._derive_default_capability({"name": name, "description": description})]
        
        valid_capabilities = []
        for cap in capabilities:
            if self._validate_capability(cap):
                valid_capabilities.append(cap)
            else:
                logger.warning(f"Skipping invalid capability: {cap}")
        
        self.providers[provider_id] = {
            "id": provider_id,
            "name": name,
            "provider_type": provider_type,
            "description": description,
            "capabilities": valid_capabilities,
            "metadata": metadata,
            "status": "active",
            "registered_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat()
        }
        
        # Update the capability registry.
        for capability in valid_capabilities:
            cap_id = capability["id"]
            if cap_id not in self.capabilities:
                self.capabilities[cap_id] = {
                    "id": cap_id,
                    "name": capability["name"],
                    "description": capability["description"],
                    "context_contract": capability.get("context_contract", {}),
                    "providers": []
                }
            self.capabilities[cap_id]["providers"].append({
                "id": provider_id,
                "name": name,
                "confidence": capability.get("confidence", 0.8)
            })
        
        self._save_data()
        return provider_id
    
    async def update_provider_capabilities(
        self, 
        provider_id: str, 
        capabilities: List[Dict[str, Any]]
    ) -> bool:
        if provider_id not in self.providers:
            return False
        
        # Remove provider from all capability listings.
        for cap_id in self.capabilities:
            self.capabilities[cap_id]["providers"] = [
                p for p in self.capabilities[cap_id]["providers"] if p["id"] != provider_id
            ]
        
        valid_capabilities = [cap for cap in capabilities if self._validate_capability(cap)]
        self.providers[provider_id]["capabilities"] = valid_capabilities
        self.providers[provider_id]["last_updated"] = datetime.utcnow().isoformat()
        
        for capability in valid_capabilities:
            cap_id = capability["id"]
            if cap_id not in self.capabilities:
                self.capabilities[cap_id] = {
                    "id": cap_id,
                    "name": capability["name"],
                    "description": capability["description"],
                    "context_contract": capability.get("context_contract", {}),
                    "providers": []
                }
            self.capabilities[cap_id]["providers"].append({
                "id": provider_id,
                "name": self.providers[provider_id]["name"],
                "confidence": capability.get("confidence", 0.8)
            })
        
        self._save_data()
        return True
    
    async def deregister_provider(self, provider_id: str) -> bool:
        if provider_id not in self.providers:
            return False
        
        for cap_id in self.capabilities:
            self.capabilities[cap_id]["providers"] = [
                p for p in self.capabilities[cap_id]["providers"] if p["id"] != provider_id
            ]
        del self.providers[provider_id]
        self._save_data()
        return True
    
    async def request_service(
        self,
        capability: str,
        content: Dict[str, Any],
        provider_id: Optional[str] = None,
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        Request a service by capability. The provided content may include context data under the "context" key.
        
        Production strategy: If no provider or no system agent is available to process the request,
        the method will fail.
        """
        if provider_id:
            if provider_id not in self.providers:
                raise ValueError(f"Provider '{provider_id}' not found")
            provider = self.providers[provider_id]
            has_capability = False
            capability_confidence = 0.0
            for cap in provider["capabilities"]:
                if cap["id"] == capability:
                    confidence = cap.get("confidence", 0.0)
                    if confidence >= min_confidence:
                        has_capability = True
                        capability_confidence = confidence
                        break
            if not has_capability:
                raise ValueError(f"Provider '{provider_id}' does not have capability '{capability}' with sufficient confidence")
            response = {
                "provider_id": provider_id,
                "provider_name": provider["name"],
                "content": self._process_content(capability, content, provider),
                "confidence": capability_confidence
            }
        else:
            if capability not in self.capabilities:
                raise ValueError(f"No providers found for capability '{capability}'")
            eligible_providers = [
                p for p in self.capabilities[capability]["providers"]
                if p.get("confidence", 0.0) >= min_confidence
            ]
            if not eligible_providers:
                raise ValueError(f"No providers with sufficient confidence for capability '{capability}'")
            best_provider = max(eligible_providers, key=lambda p: p.get("confidence", 0.0))
            provider_id = best_provider["id"]
            provider = self.providers[provider_id]
            response = {
                "provider_id": provider_id,
                "provider_name": provider["name"],
                "content": self._process_content(capability, content, provider),
                "confidence": best_provider.get("confidence", 0.8)
            }
        
        # In production, we require a system agent to process the request.
        if not self.system_agent:
            raise RuntimeError("No system agent configured to process service requests in production")
        
        prompt = (
            f"You are the system agent. Execute a request using provider '{response['provider_name']}' "
            f"for capability '{capability}'.\n\nInput Content:\n{json.dumps(content, indent=2)}\n\n"
            "Ensure that the context requirements are met."
        )
        try:
            result = await self.system_agent.run(prompt)
        except Exception as e:
            raise RuntimeError(f"System agent failed to process request: {e}")
        
        execution_result = {
            "result": result,
            "executed_by": response["provider_id"],
            "executed_at": datetime.utcnow().isoformat(),
            "success": True
        }
        
        await self._log_execution(capability, content, execution_result)
        response["execution"] = execution_result
        return response
    
    def _process_content(self, capability: str, content: Dict[str, Any], provider: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the content by invoking the provider's real execution function, if defined.
        If the provider has an "execution_function" (either as a direct attribute or in its metadata)
        and it is callable, that function is used to process the content.
        
        Otherwise, the function applies a default transformation (in production, you would expect
        providers to have real implementations).
        """
        execution_func = provider.get("execution_function") or provider.get("metadata", {}).get("execution_function")
        if execution_func and callable(execution_func):
            try:
                result = execution_func(content)
            except Exception as e:
                raise RuntimeError(f"Error during provider execution: {e}")
        else:
            # In production, we expect a provider to have its own execution logic.
            raise RuntimeError(f"Provider '{provider['name']}' does not implement an execution function.")
        
        return {
            "result": result,
            "provider_type": provider["provider_type"],
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "input_size": len(content.get("text", json.dumps(content))),
                "capability": capability
            }
        }
    
    async def find_providers_for_capability(
        self, 
        capability: str,
        min_confidence: float = 0.5,
        provider_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if capability not in self.capabilities:
            return []
        provider_ids = [
            p["id"] for p in self.capabilities[capability]["providers"]
            if p.get("confidence", 0.0) >= min_confidence
        ]
        return [
            self.providers[pid] for pid in provider_ids
            if pid in self.providers and (not provider_type or self.providers[pid]["provider_type"] == provider_type)
        ]
    
    async def find_provider_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        for provider in self.providers.values():
            if provider["name"] == name:
                return provider
        return None
    
    async def get_provider(self, provider_id: str) -> Optional[Dict[str, Any]]:
        return self.providers.get(provider_id)
    
    async def list_all_providers(self, provider_type: Optional[str] = None) -> List[Dict[str, Any]]:
        if provider_type:
            return [p for p in self.providers.values() if p["provider_type"] == provider_type]
        return list(self.providers.values())
    
    async def list_all_capabilities(
        self, 
        provider_type: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        result = []
        for cap_id, capability in self.capabilities.items():
            filtered_providers = capability["providers"]
            if provider_type or min_confidence > 0:
                filtered_providers = []
                for provider in capability["providers"]:
                    provider_record = self.providers.get(provider["id"])
                    if not provider_record:
                        continue
                    if provider_type and provider_record["provider_type"] != provider_type:
                        continue
                    if min_confidence > 0 and provider.get("confidence", 0) < min_confidence:
                        continue
                    filtered_providers.append(provider)
            if not filtered_providers:
                continue
            result.append({
                "id": cap_id,
                "name": capability["name"],
                "description": capability.get("description", ""),
                "providers": filtered_providers
            })
        if limit is not None and limit > 0:
            result = result[:limit]
        return result
    
    async def search_capabilities(
        self, 
        query: str,
        min_confidence: float = 0.5,
        provider_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        query = query.lower()
        matches = []
        for cap_id, capability in self.capabilities.items():
            if query in capability["name"].lower() or query in capability.get("description", "").lower():
                filtered_providers = []
                for provider in capability["providers"]:
                    provider_record = self.providers.get(provider["id"])
                    if not provider_record:
                        continue
                    if provider_type and provider_record["provider_type"] != provider_type:
                        continue
                    if min_confidence > 0 and provider.get("confidence", 0) < min_confidence:
                        continue
                    filtered_providers.append(provider)
                if not filtered_providers:
                    continue
                matches.append({
                    "id": cap_id,
                    "name": capability["name"],
                    "description": capability.get("description", ""),
                    "providers": filtered_providers
                })
        matches.sort(key=lambda c: len(c["providers"]), reverse=True)
        return matches[:limit]
    
    async def initialize_from_library(self):
        """
        Register all active records from SmartLibrary as providers.
        If a record does not include explicit capabilities, derive a default capability from its description.
        """
        for record in self.smart_library.records:
            if record.get("status", "active") != "active":
                continue
            if await self.find_provider_by_name(record["name"]):
                continue
            capabilities = record.get("metadata", {}).get("capabilities", [])
            if not capabilities:
                capabilities = [self._derive_default_capability(record)]
            await self.register_provider(
                name=record["name"],
                capabilities=capabilities,
                provider_type=record["record_type"],
                description=record.get("description", ""),
                metadata={"source": "SmartLibrary"}
            )
        logger.info("SmartAgentBus initialized from SmartLibrary.")
    
    async def _lazy_register_from_library(self, capability_id: str):
        """
        If a requested capability is missing, search SmartLibrary and register matching records.
        """
        logger.info(f"Capability '{capability_id}' not found in AgentBus. Searching SmartLibrary...")
        matches = await self.smart_library.search_capabilities(
            query=capability_id,
            min_confidence=0.5,
            limit=5
        )
        for record in matches:
            capabilities = record.get("metadata", {}).get("capabilities", [])
            if not capabilities:
                capabilities = [self._derive_default_capability(record)]
            await self.register_provider(
                name=record["name"],
                capabilities=capabilities,
                provider_type=record["record_type"],
                description=record.get("description", ""),
                metadata={"source": "SmartLibrary", "similarity": record.get("similarity")}
            )
        logger.info(f"Registered {len(matches)} records for capability '{capability_id}'.")
    
    async def _log_execution(self, capability: str, input_content: Dict[str, Any], result: Dict[str, Any]):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "capability": capability,
            "input": input_content,
            "execution_result": result
        }
        logs = []
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r") as f:
                    logs = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read previous logs: {e}")
        logs.append(log_entry)
        try:
            with open(self.log_path, "w") as f:
                json.dump(logs, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to write execution log: {e}")


