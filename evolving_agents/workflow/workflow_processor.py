# evolving_agents/workflow/workflow_processor.py

import logging
import yaml
import uuid
from typing import Dict, Any, List, Optional, Tuple

from evolving_agents.core.system_agent import SystemAgent
from evolving_agents.acp.client import ACPClient

logger = logging.getLogger(__name__)

class WorkflowProcessor:
    """
    Processes YAML-based workflows with support for multiple agent frameworks.
    """
    def __init__(self, system_agent: SystemAgent):
        """
        Initialize the workflow processor.
        
        Args:
            system_agent: System agent for creating and executing components
        """
        self.system_agent = system_agent
        logger.info("Workflow Processor initialized")
    
    async def process_workflow(self, workflow_yaml: str) -> Dict[str, Any]:
        """
        Process a workflow from a YAML string.
        
        Args:
            workflow_yaml: YAML string defining the workflow
            
        Returns:
            Execution results
        """
        logger.info("Processing workflow")
        
        try:
            workflow = yaml.safe_load(workflow_yaml)
        except Exception as e:
            error = f"Error parsing workflow YAML: {str(e)}"
            logger.error(error)
            return {"status": "error", "message": error}
        
        # Extract metadata
        scenario_name = workflow.get("scenario_name", "Unnamed Scenario")
        domain = workflow.get("domain", "general")
        disclaimers = workflow.get("additional_disclaimers", [])
        
        logger.info(f"Executing scenario: {scenario_name} in domain: {domain}")
        
        # Get firmware for this domain
        firmware_content = self.system_agent.firmware.get_firmware_prompt(domain)
        
        # Execute steps
        results = {
            "scenario_name": scenario_name,
            "domain": domain,
            "steps": []
        }
        
        # Track created items and their dependencies
        created_items = {}
        
        for i, step in enumerate(workflow.get("steps", [])):
            step_type = step.get("type", "").upper()
            logger.info(f"Executing step {i+1}: {step_type}")
            
            try:
                if step_type == "DEFINE":
                    step_result = await self._process_define_step(
                        step, domain, firmware_content, disclaimers
                    )
                    
                elif step_type == "CREATE":
                    # Find required tools/dependencies for this item
                    tools = await self._resolve_dependencies(step, created_items)
                    
                    step_result = await self._process_create_step(
                        step, domain, firmware_content, tools
                    )
                    
                    # Track created item
                    if step_result.get("status") == "success":
                        item_name = step.get("name")
                        created_items[item_name] = {
                            "type": step.get("item_type", "").upper(),
                            "record_id": step_result.get("record_id")
                        }
                    
                elif step_type == "EXECUTE":
                    step_result = await self._process_execute_step(step)
                    
                else:
                    step_result = {
                        "status": "error", 
                        "message": f"Unknown step type: {step_type}"
                    }
                    
            except Exception as e:
                logger.error(f"Error executing step {i+1}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                step_result = {"status": "error", "message": f"Error: {str(e)}"}
            
            results["steps"].append(step_result)
            
            # Stop if step failed
            if step_result.get("status") == "error":
                logger.warning(f"Workflow execution stopped due to error in step {i+1}")
                results["status"] = "error"
                results["message"] = f"Failed at step {i+1}: {step_result.get('message')}"
                break
        
        if "status" not in results:
            results["status"] = "success"
            results["message"] = f"Successfully executed workflow '{scenario_name}'"
        
        return results
    
    async def _resolve_dependencies(
        self, 
        step: Dict[str, Any], 
        created_items: Dict[str, Dict[str, Any]]
    ) -> List[Any]:
        """
        Resolve dependencies for an item.
        
        Args:
            step: Step definition
            created_items: Dictionary of already created items
            
        Returns:
            List of tool instances required by this item
        """
        item_type = step.get("item_type", "").upper()
        name = step.get("name")
        
        # Only agents need tool dependencies
        if item_type != "AGENT":
            return []
        
        # Find the record to check for required tools
        record = await self.system_agent.library.find_record_by_name(name)
        if not record:
            logger.warning(f"Record not found for {item_type} '{name}'")
            return []
        
        # Get required tools from metadata
        metadata = record.get("metadata", {})
        required_tool_names = metadata.get("required_tools", [])
        
        # No dependencies
        if not required_tool_names:
            return []
        
        logger.info(f"Resolving dependencies for {item_type} '{name}': {required_tool_names}")
        
        # Collect tool instances
        tools = []
        
        for tool_name in required_tool_names:
            # Check if the tool has already been created in this workflow
            if tool_name in created_items and created_items[tool_name]["type"] == "TOOL":
                # Tool exists in active items
                if tool_name in self.system_agent.active_items:
                    tool_info = self.system_agent.active_items[tool_name]
                    tools.append(tool_info["instance"])
                    logger.info(f"Using active tool: {tool_name}")
                else:
                    # Tool is defined but not created yet in active items
                    logger.warning(f"Tool '{tool_name}' is defined but not active")
            else:
                # Tool doesn't exist yet, check if it's in the library
                tool_record = await self.system_agent.library.find_record_by_name(tool_name)
                if tool_record:
                    logger.info(f"Creating dependency tool: {tool_name}")
                    # Create the tool instance
                    try:
                        tool_instance = await self.system_agent.tool_factory.create_tool(tool_record)
                        tools.append(tool_instance)
                        
                        # Add to active items
                        self.system_agent.active_items[tool_name] = {
                            "record": tool_record,
                            "instance": tool_instance,
                            "type": "TOOL"
                        }
                        
                        # Add to created items
                        created_items[tool_name] = {
                            "type": "TOOL",
                            "record_id": tool_record["id"]
                        }
                    except Exception as e:
                        logger.error(f"Error creating dependency tool '{tool_name}': {str(e)}")
                else:
                    logger.warning(f"Required tool '{tool_name}' not found in library")
        
        return tools
    
    async def _process_define_step(
        self,
        step: Dict[str, Any],
        domain: str,
        firmware_content: str,
        disclaimers: List[str]
    ) -> Dict[str, Any]:
        """
        Process a DEFINE step to define an agent or tool.
        
        Args:
            step: Step definition
            domain: Domain for the step
            firmware_content: Firmware content
            disclaimers: Additional disclaimers
            
        Returns:
            Step result
        """
        item_type = step.get("item_type", "").upper()
        name = step.get("name")
        description = step.get("description", "")
        
        logger.info(f"Defining {item_type} '{name}'")
        
        # Check if we need to reuse or evolve an existing item
        if "from_existing_snippet" in step:
            source_name = step.get("from_existing_snippet")
            source_record = await self.system_agent.library.find_record_by_name(source_name)
            
            if not source_record:
                return {"status": "error", "message": f"Source item not found: {source_name}"}
            
            if step.get("reuse_as_is", False):
                # Simple reuse
                message = f"Reused {source_name} as {name}"
                logger.info(message)
                
                # Extract framework from source if present
                framework = None
                if item_type == "AGENT" and source_record.get("metadata", {}).get("framework"):
                    framework = source_record["metadata"]["framework"]
                
                # Just copy with a new name
                new_record = source_record.copy()
                new_record["id"] = str(uuid.uuid4())
                new_record["name"] = name
                new_record["description"] = description or source_record["description"]
                new_record["metadata"] = source_record.get("metadata", {}).copy()
                new_record["metadata"]["reused_from"] = source_record["id"]
                new_record["metadata"]["disclaimers"] = disclaimers
                
                # Preserve framework if specified
                if framework:
                    new_record["metadata"]["framework"] = framework
                
                # Add required_tools if specified for agents
                if item_type == "AGENT" and "required_tools" in step:
                    new_record["metadata"]["required_tools"] = step["required_tools"]
                
                await self.system_agent.library.save_record(new_record)
                
                return {
                    "status": "success",
                    "action": "reuse",
                    "message": message,
                    "record_id": new_record["id"]
                }
            else:
                # Evolve
                evolve_changes = step.get("evolve_changes", {})
                docstring_update = evolve_changes.get("docstring_update", "")
                
                message = f"Evolved {source_name} to {name}, injecting firmware for domain '{domain}'."
                logger.info(message)
                
                # Extract framework from source if present
                framework = None
                if item_type == "AGENT" and source_record.get("metadata", {}).get("framework"):
                    framework = source_record["metadata"]["framework"]
                
                # Generate evolved code
                evolve_prompt = f"""
                {firmware_content}
                
                ORIGINAL CODE:
                ```
                {source_record['code_snippet']}
                ```
                
                REQUIRED CHANGES:
                - New name: {name}
                - Description: {description or source_record['description']}
                - Docstring update: {docstring_update}
                
                DOMAIN REQUIREMENTS:
                - Domain: {domain}
                
                REQUIRED DISCLAIMERS:
                {chr(10).join(disclaimers)}
                
                Evolve the code to implement these changes while maintaining core functionality.
                Include all required disclaimers and follow domain guidelines.
                
                EVOLVED CODE:
                """
                
                evolved_code = await self.system_agent.llm.generate(evolve_prompt)
                
                # Create evolved record
                metadata = {
                    "evolved_from": source_record["id"],
                    "evolution_changes": evolve_changes,
                    "disclaimers": disclaimers
                }
                
                # Preserve framework if specified
                if framework:
                    metadata["framework"] = framework
                
                # Add required_tools if specified for agents
                if item_type == "AGENT" and "required_tools" in step:
                    metadata["required_tools"] = step["required_tools"]
                
                new_record = await self.system_agent.library.create_record(
                    name=name,
                    record_type=item_type,
                    domain=domain,
                    description=description or source_record["description"],
                    code_snippet=evolved_code,
                    metadata=metadata
                )
                
                return {
                    "status": "success",
                    "action": "evolve",
                    "message": message,
                    "record_id": new_record["id"]
                }
        else:
            # Create new from scratch
            message = f"Created new {item_type} '{name}' from scratch"
            logger.info(message)
            
            # Check for framework specification
            framework = step.get("framework")
            
            # Generate code
            create_prompt = f"""
            {firmware_content}
            
            CREATE A NEW {item_type}:
            Name: {name}
            Description: {description}
            
            DOMAIN: {domain}
            
            REQUIRED DISCLAIMERS:
            {chr(10).join(disclaimers)}
            
            Generate complete code for this {item_type.lower()} that follows all guidelines.
            Include proper docstrings and all required disclaimers.
            
            CODE:
            """
            
            new_code = await self.system_agent.llm.generate(create_prompt)
            
            # Prepare metadata
            metadata = {
                "created_from": "workflow",
                "disclaimers": disclaimers
            }
            
            # Add framework if specified
            if framework:
                metadata["framework"] = framework
            
            # Add required_tools if specified for agents
            if item_type == "AGENT" and "required_tools" in step:
                metadata["required_tools"] = step["required_tools"]
            
            # Create record
            new_record = await self.system_agent.library.create_record(
                name=name,
                record_type=item_type,
                domain=domain,
                description=description,
                code_snippet=new_code,
                metadata=metadata
            )
            
            return {
                "status": "success",
                "action": "create",
                "message": message,
                "record_id": new_record["id"]
            }
    
    async def _process_create_step(
        self, 
        step: Dict[str, Any], 
        domain: str, 
        firmware_content: str,
        tools: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a CREATE step to instantiate an agent or tool.
        
        Args:
            step: Step definition
            domain: Domain for the step
            firmware_content: Firmware content
            tools: Optional tools to provide to agents
            
        Returns:
            Step result
        """
        item_type = step.get("item_type", "").upper()
        name = step.get("name")
        
        message = f"Created {item_type} instance {name}"
        logger.info(message)
        
        # Find the record
        record = await self.system_agent.library.find_record_by_name(name)
        
        if not record:
            return {"status": "error", "message": f"Item not found: {name}"}
        
        if record["record_type"] != item_type:
            return {"status": "error", "message": f"Item {name} is not a {item_type}"}
        
        try:
            # Get framework and config from step or record
            config = step.get("config", {})
            
            # Create instance based on record type
            if item_type == "AGENT":
                # Create the agent with tools if available
                instance = await self.system_agent.agent_factory.create_agent(
                    record=record,
                    firmware_content=firmware_content,
                    tools=tools,
                    config=config
                )
            else:  # TOOL
                instance = await self.system_agent.tool_factory.create_tool(
                    record=record,
                    firmware_content=firmware_content
                )
            
            # Store in active items
            self.system_agent.active_items[name] = {
                "record": record,
                "instance": instance,
                "type": record["record_type"]
            }
            
            return {
                "status": "success",
                "message": message,
                "record_id": record["id"]
            }
        except Exception as e:
            logger.error(f"Error creating {item_type} {name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "message": f"Error creating {item_type} {name}: {str(e)}"
            }
    
    async def _process_execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an EXECUTE step to run an agent or tool.
        
        Args:
            step: Step definition
            
        Returns:
            Step result
        """
        item_type = step.get("item_type", "").upper()
        name = step.get("name")
        user_input = step.get("user_input", "")
        execution_config = step.get("execution_config", {})
        
        message = f"Executed {item_type} {name}"
        logger.info(message)
        
        # For agents, use the agent_factory for execution
        if item_type == "AGENT":
            # Execute with execution config if provided
            result = await self.system_agent.agent_factory.execute_agent(
                name, user_input, execution_config
            )
        else:
            # Execute using system agent's execute_item method for tools
            result = await self.system_agent.execute_item(name, user_input)
        
        if result["status"] == "success":
            # Include details about the runtime environment
            execution_details = {}
            
            if name in self.system_agent.active_items:
                item_info = self.system_agent.active_items[name]
                if "framework" in item_info:
                    execution_details["framework"] = item_info["framework"]
                if "provider_id" in item_info:
                    execution_details["provider"] = item_info["provider_id"]
            
            return {
                "status": "success",
                "message": message,
                "result": result["result"],
                "record_id": self.system_agent.active_items.get(name, {}).get("record", {}).get("id"),
                "execution_details": execution_details
            }
        else:
            return {
                "status": "error",
                "message": result["message"]
            }
        
    async def process_acp_workflow(self, workflow_yaml: str, acp_client: Optional[ACPClient] = None) -> Dict[str, Any]:
        """
        Process a workflow with ACP-enabled communication between agents.
        
        Args:
            workflow_yaml: YAML string defining the workflow
            acp_client: Optional ACP client to use
            
        Returns:
            Execution results
        """
        # Parse workflow 
        workflow = yaml.safe_load(workflow_yaml)
        
        # Extract metadata
        scenario_name = workflow.get("scenario_name", "Unnamed ACP Scenario")
        domain = workflow.get("domain", "general")
        
        # Use provided ACP client or create one
        acp_client = acp_client or ACPClient()
        
        # Execute steps with ACP awareness
        results = {
            "scenario_name": scenario_name,
            "domain": domain,
            "steps": []
        }
        
        # Track created agents and their ACP identifiers
        acp_agents = {}
        
        for i, step in enumerate(workflow.get("steps", [])):
            step_type = step.get("type", "").upper()
            
            try:
                if step_type == "ACP_REGISTER":
                    # Register an agent with the ACP registry
                    agent_name = step.get("name")
                    agent_info = self.system_agent.active_items.get(agent_name)
                    
                    if not agent_info:
                        raise ValueError(f"Agent '{agent_name}' not found in active items")
                    
                    agent_instance = agent_info["instance"]
                    agent_id = await acp_client.register_agent(agent_instance)
                    acp_agents[agent_name] = agent_id
                    
                    step_result = {
                        "status": "success",
                        "message": f"Registered agent '{agent_name}' with ACP (ID: {agent_id})"
                    }
                    
                elif step_type == "ACP_COMMUNICATE":
                    # Handle agent-to-agent communication via ACP
                    sender = step.get("sender")
                    recipient = step.get("recipient")
                    message = step.get("message", "")
                    
                    if sender not in acp_agents:
                        raise ValueError(f"Sender agent '{sender}' not registered with ACP")
                    
                    if recipient not in acp_agents:
                        raise ValueError(f"Recipient agent '{recipient}' not registered with ACP")
                    
                    # Execute communication
                    response = await acp_client.send_message(
                        sender_id=acp_agents[sender],
                        recipient_id=acp_agents[recipient],
                        message=message
                    )
                    
                    step_result = {
                        "status": "success",
                        "message": f"Communication from '{sender}' to '{recipient}' completed",
                        "result": response
                    }
                    
                else:
                    # Process other steps normally
                    step_result = await self._process_step(step, acp_client=acp_client)
                    
            except Exception as e:
                step_result = {
                    "status": "error",
                    "message": f"Error processing ACP step: {str(e)}"
                }
            
            results["steps"].append(step_result)
            
            # Stop if step failed
            if step_result["status"] == "error":
                results["status"] = "error"
                results["message"] = f"Failed at step {i+1}: {step_result['message']}"
                break
        
        if "status" not in results:
            results["status"] = "success"
            results["message"] = f"Successfully executed ACP workflow '{scenario_name}'"
        
        return results
