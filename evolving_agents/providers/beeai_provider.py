# evolving_agents/providers/beeai_provider.py (improved version)

import logging
import importlib
import inspect
import json
from typing import Dict, Any, List, Optional, Union
import ast

# BeeAI framework imports
from beeai_framework.agents.tool_calling import ToolCallingAgent # Changed from ReActAgent
from beeai_framework.agents.types import AgentExecutionConfig, AgentMeta
from beeai_framework.memory import TokenMemory, UnconstrainedMemory
from beeai_framework.tools.tool import Tool
from beeai_framework.backend.message import UserMessage
from beeai_framework.emitter.emitter import Emitter

from evolving_agents.providers.base import FrameworkProvider
from evolving_agents.core.llm_service import LLMService

logger = logging.getLogger(__name__)

class BeeAIProvider(FrameworkProvider):
    """
    Provider for BeeAI framework integration.
    Handles creation and execution of BeeAgents.
    """
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        """
        Initialize the BeeAI provider.
        
        Args:
            llm_service: Optional LLM service to use for agents
        """
        self.llm_service = llm_service
        logger.info("BeeAI Provider initialized")
    
    async def create_agent(
    self, 
    record: Dict[str, Any],
    tools: Optional[List[Tool]] = None,
    firmware_content: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
    ) -> ToolCallingAgent: # Updated return type
        """
        Create a BeeAI ToolCallingAgent with the specified configuration.
        
        Args:
            record: Agent record from the Smart Library
            tools: Optional list of tools to provide to the agent
            firmware_content: Optional firmware content to inject
            config: Optional configuration parameters
            
        Returns:
            Instantiated ToolCallingAgent
        """
        logger.info(f"Creating BeeAI ToolCallingAgent '{record['name']}' with {len(tools) if tools else 0} tools")
        
        # Apply default config if none provided
        config = config or {}
        
        # Get the ChatModel - prefer provided llm_service
        chat_model = None
        if self.llm_service and self.llm_service.chat_model:
            chat_model = self.llm_service.chat_model
        elif "llm" in config:
            chat_model = config["llm"]
        
        if not chat_model:
            logger.error("No ChatModel available for BeeAgent")
            raise ValueError("ChatModel not available. Provide an LLMService or chat_model in config.")
        
        # Clean up the code snippet
        code_snippet = record["code_snippet"]
        
        # Remove markdown code block formatting if present
        if "```" in code_snippet:
            lines = code_snippet.split("\n")
            clean_lines = []
            inside_code_block = False
            for i, line in enumerate(lines):
                if line.strip().startswith("```"):
                    inside_code_block = not inside_code_block
                    continue
                if not inside_code_block or (inside_code_block and not line.strip().startswith("```")):
                    clean_lines.append(line)
            code_snippet = "\n".join(clean_lines)
        
        # Prepare description/instructions
        instructions = record["description"]
        if firmware_content:
            instructions = f"{firmware_content}\n\n{instructions}"
        
        # Create meta information
        meta = AgentMeta(
            name=record["name"],
            description=instructions,
            tools=tools or []
        )
        
        # Create memory - use TokenMemory by default, but allow configuration
        memory_type = config.get("memory_type", "token")
        if memory_type == "unconstrained":
            memory = UnconstrainedMemory()
        else:  # token memory by default
            memory = TokenMemory(chat_model)
        
        # Create the ToolCallingAgent with proper parameters
        # Assuming ToolCallingAgent can be initialized without an explicit output_schema,
        # or it defaults to a generic one (e.g., string or dict).
        # If output_schema is mandatory, this provider would need a way to determine it from 'record' or 'config'.
        agent = ToolCallingAgent(
            llm=chat_model,
            tools=tools or [],
            memory=memory,
            meta=meta
            # output_schema=config.get("output_schema_class") # Example if schema class is passed in config
        )
        
        return agent
    
    async def execute_agent(
        self, 
        agent_instance: Union[ToolCallingAgent, str], # Updated type hint
        input_text: str,
        execution_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a BeeAI ToolCallingAgent with input text.
        
        Args:
            agent_instance: The ToolCallingAgent instance or name
            input_text: Input text to process
            execution_config: Optional execution configuration parameters
            
        Returns:
            Execution result dictionary with status, message, and result
        """
        if isinstance(agent_instance, str):
            logger.warning(f"Expected agent instance but got name: {agent_instance}. "
                        "This may not work - fixes needed in SystemAgent.execute_item")
            return {
                "status": "error",
                "message": f"BeeAIProvider requires agent instance, not name: {agent_instance}",
                "result": f"Error: BeeAIProvider requires agent instance, not name"
            }
                
        logger.info(f"Executing BeeAI ToolCallingAgent with input: {input_text[:50]}...")
        
        # Apply default execution config if none provided
        execution_config = execution_config or {}
        
        # Create execution config with defaults - Changed BeeAgentExecutionConfig to AgentExecutionConfig
        bee_exec_config = AgentExecutionConfig(
            max_retries_per_step=execution_config.get("max_retries_per_step", 3),
            total_max_retries=execution_config.get("total_max_retries", 10),
            max_iterations=execution_config.get("max_iterations", 20)
        )
        
        try:
            # Run the agent with the input text.
            # The agent.run() method now returns a Run object. Execution is triggered by await-ing it.
            run_obj = await agent_instance.run( # Renamed variable to run_obj
                prompt=input_text,
                execution=bee_exec_config
            )
            
            # Assuming the actual output is on run_obj.final_output or similar
            # and this final_output is the Pydantic model instance (e.g., MemoryOperationOutput)
            actual_output = getattr(run_obj, 'final_output', None)
            if actual_output is None: # Fallback if final_output is not the attribute name
                 actual_output = getattr(run_obj, 'output', run_obj) # Try 'output' or the run_obj itself

            if hasattr(actual_output, 'model_dump_json'): # Pydantic model
                result_content = actual_output.model_dump_json(indent=2)
            elif isinstance(actual_output, dict) or isinstance(actual_output, list):
                result_content = json.dumps(actual_output, indent=2)
            # Remove old ReActAgent specific handling if ToolCallingAgent output is always structured
            # elif hasattr(actual_output, 'result') and hasattr(actual_output.result, 'text'):
            #     result_content = actual_output.result.text
            else:
                result_content = str(actual_output)

            return {
                "status": "success",
                "message": "ToolCallingAgent executed successfully",
                "result": result_content,
                "raw_result_type": str(type(actual_output)),
                "run_id": str(getattr(run_obj, 'id', 'N/A')) # Example of accessing Run object property
            }
            
        except Exception as e:
            logger.error(f"Error executing ToolCallingAgent: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                "status": "error",
                "message": f"Error executing ToolCallingAgent: {str(e)}",
                "result": f"Error: {str(e)}",
                "error": e
            }
    
    def supports_framework(self, framework_name: str) -> bool:
        """
        Check if this provider supports the specified framework.
        
        Args:
            framework_name: Name of the framework to check
            
        Returns:
            True if supported, False otherwise
        """
        return framework_name.lower() in ["beeai", "bee", "bee-framework", "bee_framework"]
    
    def get_supported_agent_types(self) -> List[str]:
        """
        Get the agent types supported by this provider.
        
        Returns:
            List of supported agent type names
        """
        return ["BeeAgent", "ReActAgent", "ToolCallingAgent", "BaseAgent"] # Added ToolCallingAgent and BaseAgent
    
    def get_configuration_schema(self) -> Dict[str, Any]:
        """
        Get the schema for provider configuration.
        
        Returns:
            Dictionary with configuration options and their schema
        """
        return {
            "memory_type": {
                "type": "string",
                "enum": ["token", "unconstrained"],
                "default": "token",
                "description": "Type of memory to use for the agent"
            },
            "execution": {
                "type": "object",
                "properties": {
                    "max_retries_per_step": {
                        "type": "integer",
                        "default": 3,
                        "description": "Maximum number of retries per step"
                    },
                    "total_max_retries": {
                        "type": "integer",
                        "default": 10,
                        "description": "Total maximum number of retries"
                    },
                    "max_iterations": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum number of iterations"
                    }
                }
            },
            "observability": {
                "type": "boolean",
                "default": False,
                "description": "Enable observability for agent execution"
            }
        }