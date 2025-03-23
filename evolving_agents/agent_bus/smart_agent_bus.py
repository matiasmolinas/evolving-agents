import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

from evolving_agents.agent_bus.simple_agent_bus import SimpleAgentBus
from evolving_agents.smart_library.smart_library import SmartLibrary
from beeai_framework.agents.react import ReActAgent

logger = logging.getLogger(__name__)


class SmartAgentBus(SimpleAgentBus):
    def __init__(
        self,
        smart_library: SmartLibrary,
        system_agent: Optional[ReActAgent] = None,
        storage_path: str = "smart_agent_bus.json",
        log_path: str = "agent_bus_logs.json"
    ):
        super().__init__(storage_path=storage_path)
        self.smart_library = smart_library
        self.system_agent = system_agent
        self.log_path = log_path

    async def initialize_from_library(self):
        """Register all active records in SmartLibrary."""
        for record in self.smart_library.records:
            if record.get("status", "active") != "active":
                continue

            if await self.find_provider_by_name(record["name"]):
                continue

            capabilities = record.get("metadata", {}).get("capabilities", [])
            await self.register_provider(
                name=record["name"],
                capabilities=capabilities,
                provider_type=record["record_type"],
                description=record.get("description", ""),
                metadata={"source": "SmartLibrary"}
            )
        logger.info("SmartAgentBus initialized from SmartLibrary.")

    async def request_service(
        self,
        capability: str,
        content: Dict[str, Any],
        provider_id: Optional[str] = None,
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """Try to execute via SystemAgent. Fallback to mock if needed."""
        # Step 1: Ensure a provider is available
        if capability not in self.capabilities:
            await self._lazy_register_from_library(capability)

        try:
            base_response = await super().request_service(
                capability, content, provider_id, min_confidence
            )
        except Exception as e:
            return {"error": str(e)}

        execution_result = None

        if self.system_agent:
            prompt = (
                f"You are the system agent. Please execute a request using provider '{base_response['provider_name']}' "
                f"for capability '{capability}' with input:\n\n{json.dumps(content, indent=2)}"
            )
            try:
                result = await self.system_agent.run(prompt)
                execution_result = {
                    "result": result,
                    "executed_by": base_response["provider_id"],
                    "executed_at": datetime.utcnow().isoformat(),
                    "success": True
                }
            except Exception as e:
                execution_result = {
                    "error": str(e),
                    "executed_by": base_response["provider_id"],
                    "executed_at": datetime.utcnow().isoformat(),
                    "success": False
                }
        else:
            execution_result = {
                "result": base_response["content"],
                "executed_by": base_response["provider_id"],
                "executed_at": datetime.utcnow().isoformat(),
                "success": False,
                "fallback": True
            }

        # Step 2: Log result
        await self._log_execution(capability, content, execution_result)

        # Step 3: Return enriched response
        base_response["execution"] = execution_result
        return base_response

    async def _lazy_register_from_library(self, capability_id: str):
        """Search the SmartLibrary for components matching the capability and register them."""
        logger.info(f"Capability '{capability_id}' not found in AgentBus. Searching SmartLibrary...")

        matches = await self.smart_library.search_capabilities(
            query=capability_id,
            record_type="AGENT",
            threshold=0.5
        )

        for record, similarity in matches:
            capabilities = record.get("metadata", {}).get("capabilities", [])
            await self.register_provider(
                name=record["name"],
                capabilities=capabilities,
                provider_type=record["record_type"],
                description=record.get("description", ""),
                metadata={"source": "SmartLibrary", "similarity": similarity}
            )

        logger.info(f"Registered {len(matches)} records for capability '{capability_id}'.")

    async def _log_execution(self, capability: str, input_content: Dict[str, Any], result: Dict[str, Any]):
        """Append a log entry to the log file."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "capability": capability,
            "input": input_content,
            "execution_result": result
        }

        logs = []
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'r') as f:
                    logs = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read previous logs: {e}")

        logs.append(log_entry)

        try:
            with open(self.log_path, 'w') as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write execution log: {e}")
