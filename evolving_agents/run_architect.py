# run_architect.py
import asyncio
import json
import os
import logging
import colorama
from colorama import Fore, Style
import time
import yaml
import re
from typing import Dict, Any, Optional, List

from evolving_agents.core.llm_service import LLMService
from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.agent_bus.simple_agent_bus import SimpleAgentBus
from evolving_agents.core.system_agent import SystemAgentFactory
from evolving_agents.agents.architect_zero import create_architect_zero

# Initialize colorama for cross-platform colored terminal output
colorama.init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Helper function for pretty printing
def print_step(title, content=None, step_type="INFO"):
    """Print a beautifully formatted step."""
    colors = {
        "INFO": Fore.BLUE,
        "AGENT": Fore.GREEN,
        "REASONING": Fore.YELLOW,
        "EXECUTION": Fore.CYAN,
        "SUCCESS": Fore.GREEN,
        "ERROR": Fore.RED,
        "COMPONENT": Fore.MAGENTA,
        "TASK": Fore.LIGHTBLUE_EX,  # New step type for task visibility
        "PROGRESS": Fore.LIGHTGREEN_EX  # New step type for progress updates
    }
    
    color = colors.get(step_type, Fore.WHITE)
    
    # Print header
    print(f"\n{color}{'=' * 80}")
    print(f"  {step_type}: {title}")
    print(f"{'=' * 80}{Style.RESET_ALL}")
    
    # Print content if provided
    if content:
        if isinstance(content, dict):
            for key, value in content.items():
                print(f"{Fore.CYAN}{key}{Style.RESET_ALL}: {value}")
        else:
            print(content)

async def initialize_system(library_path: str):
    """Initialize the system with a specific library."""
    print_step("INITIALIZING SYSTEM COMPONENTS", 
             "Setting up the LLM service, Smart Library, Agent Bus, and System Agent", 
             "TASK")
    
    # Load the library from the JSON file
    print_step("LOADING SMART LIBRARY", f"Loading component library from {library_path}", "PROGRESS")
    with open(library_path, "r") as f:
        library_data = json.load(f)
    
    # Analyze library contents for visibility
    num_agents = sum(1 for record in library_data.get("records", []) if record.get("record_type") == "AGENT")
    num_tools = sum(1 for record in library_data.get("records", []) if record.get("record_type") == "TOOL")
    domains = set(record.get("domain") for record in library_data.get("records", []) if "domain" in record)
    
    print_step("LIBRARY ANALYSIS", {
        "Total Components": len(library_data.get("records", [])),
        "Agents": num_agents,
        "Tools": num_tools,
        "Domains": ", ".join(domains)
    }, "INFO")
    
    # Create a temporary file for the library
    temp_library_path = f"temp_{os.path.basename(library_path)}"
    with open(temp_library_path, "w") as f:
        json.dump(library_data, f, indent=2)
    
    # Initialize LLM service
    print_step("INITIALIZING LLM SERVICE", "Setting up the language model for agents", "PROGRESS")
    llm_service = LLMService(provider="openai", model="gpt-4o")
    
    # Initialize SmartLibrary with the temporary file
    print_step("CONFIGURING SMART LIBRARY", "Loading components and preparing capabilities", "PROGRESS")
    smart_library = SmartLibrary(temp_library_path, llm_service)
    smart_library.records = library_data["records"]
    
    # Initialize agent bus
    print_step("SETTING UP AGENT BUS", "Creating communication channels between agents", "PROGRESS")
    agent_bus = SimpleAgentBus("agent_bus.json")
    agent_bus.set_llm_service(llm_service)
    
    # Create system agent
    print_step("CREATING SYSTEM AGENT", "Initializing the orchestration agent", "PROGRESS")
    system_agent = await SystemAgentFactory.create_agent(
        llm_service=llm_service,
        smart_library=smart_library,
        agent_bus=agent_bus
    )
    system_agent.workflow_processor.set_llm_service(llm_service)
    system_agent.workflow_generator.set_llm_service(llm_service)
    
    # Create Architect-Zero meta-agent
    print_step("CREATING ARCHITECT-ZERO AGENT", 
             "Initializing the high-level design agent that will analyze requirements", 
             "PROGRESS")
    architect_agent = await create_architect_zero(
        llm_service=llm_service,
        smart_library=smart_library,
        agent_bus=agent_bus,
        system_agent=system_agent
    )
    
    return {
        "llm_service": llm_service, 
        "smart_library": smart_library,
        "agent_bus": agent_bus,
        "system_agent": system_agent,
        "architect_agent": architect_agent,
        "temp_library_path": temp_library_path
    }

async def register_real_components(system_components, yaml_content):
    """Register real component instances with the Agent Bus."""
    print_step("REGISTERING REAL COMPONENTS", 
             "Creating and registering executable component instances", 
             "TASK")
    
    smart_library = system_components["smart_library"]
    agent_bus = system_components["agent_bus"]
    llm_service = system_components["llm_service"]
    
    # Create a tool factory for component instantiation
    from evolving_agents.tools.tool_factory import ToolFactory
    tool_factory = ToolFactory(smart_library, llm_service)
    
    # Create an agent factory for agent instantiation
    from evolving_agents.agents.agent_factory import AgentFactory
    agent_factory = AgentFactory(smart_library, llm_service)
    
    # Extract component names from the YAML workflow
    import re
    component_pattern = r'name:\s+(\w+).*?item_type:\s+(\w+)'
    component_matches = re.findall(component_pattern, yaml_content, re.DOTALL)
    
    # Convert to a set of (name, type) tuples to ensure uniqueness
    unique_components = set(component_matches)
    
    print(f"Found {len(unique_components)} unique components to register")
    
    # Register each component
    registered_components = []
    registration_errors = []
    
    for name, item_type in unique_components:
        print(f"Attempting to register {name} ({item_type})")
        
        # Find the component in the library
        component = await smart_library.find_record_by_name(name, item_type)
        if not component:
            error_msg = f"Component '{name}' ({item_type}) not found in library"
            print(error_msg)
            registration_errors.append(error_msg)
            continue
            
        try:
            if item_type.upper() == "TOOL":
                # Create real tool instance
                print(f"Creating tool instance for {name}")
                tool_instance = await tool_factory.create_tool(component)
                print(f"Tool instance created: {type(tool_instance).__name__}")
                
                # Extract capabilities
                capabilities = []
                for cap in component.get("capabilities", []):
                    capabilities.append({
                        "id": cap.get("id", "unknown_capability"),
                        "name": cap.get("name", "Unnamed Capability"),
                        "description": cap.get("description", ""),
                        "confidence": 0.9
                    })
                
                # Register with Agent Bus
                provider_id = await agent_bus.register_provider(
                    name=component["name"],
                    capabilities=capabilities,
                    provider_type="TOOL",
                    description=component["description"],
                    metadata=component.get("metadata", {}),
                    instance=tool_instance
                )
                
                print(f"Tool {name} registered with provider ID: {provider_id}")
                registered_components.append(f"{component['name']} (TOOL)")
                
            elif item_type.upper() == "AGENT":
                # Create real agent instance
                print(f"Creating agent instance for {name}")
                agent_instance = await agent_factory.create_agent(component)
                print(f"Agent instance created: {type(agent_instance).__name__}")
                
                # Extract capabilities
                capabilities = []
                for cap in component.get("capabilities", []):
                    capabilities.append({
                        "id": cap.get("id", "unknown_capability"),
                        "name": cap.get("name", "Unnamed Capability"),
                        "description": cap.get("description", ""),
                        "confidence": 0.9
                    })
                
                # Register with Agent Bus
                provider_id = await agent_bus.register_provider(
                    name=component["name"],
                    capabilities=capabilities,
                    provider_type="AGENT",
                    description=component["description"],
                    metadata=component.get("metadata", {}),
                    instance=agent_instance
                )
                
                print(f"Agent {name} registered with provider ID: {provider_id}")
                registered_components.append(f"{component['name']} (AGENT)")
        except Exception as e:
            import traceback
            error_msg = f"Error registering component '{name}': {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            registration_errors.append(error_msg)
    
    print_step("COMPONENTS REGISTERED", {
        "Successfully registered": len(registered_components),
        "Failed registrations": len(registration_errors),
        "Components": ", ".join(registered_components),
        "Errors": "\n".join(registration_errors) if registration_errors else "None"
    }, "INFO")
    
    return registered_components

async def extract_yaml_workflow(text):
    """Extract YAML workflow from the agent's response."""
    # Try to extract code between ```yaml and ``` markers
    yaml_match = re.search(r'```yaml\s*\n(.*?)\n\s*```', text, re.DOTALL)
    if yaml_match:
        yaml_content = yaml_match.group(1).strip()
    else:
        # Try with different syntax
        yaml_match2 = re.search(r'```\s*\n(scenario_name:.*?)\n\s*```', text, re.DOTALL)
        if yaml_match2:
            yaml_content = yaml_match2.group(1).strip()
        else:
            # Look for YAML content without a specific header
            lines = text.split('\n')
            yaml_lines = []
            collecting = False
            
            for line in lines:
                if not collecting and line.strip().startswith('scenario_name:'):
                    collecting = True
                    yaml_lines.append(line)
                elif collecting:
                    if line.strip().startswith('#') or line.strip().startswith('```'):
                        break
                    yaml_lines.append(line)
            
            if yaml_lines:
                yaml_content = '\n'.join(yaml_lines)
            else:
                return None
    
    return yaml_content

async def analyze_task_with_llm(llm_service, task_requirement):
    """Use LLM to analyze the task and provide detailed insights."""
    print_step("ANALYZING TASK REQUIREMENTS", 
             "Using LLM to break down the task into key components", 
             "TASK")
    
    prompt = f"""
    Analyze the following task requirement and break it down into:
    1. Main objective
    2. Key functional requirements (3-5 bullet points)
    3. Technical components likely needed
    4. Potential challenges in implementation
    5. Success criteria for the solution

    TASK REQUIREMENT:
    {task_requirement}

    Provide a clear, structured analysis that would help a system architect understand
    what needs to be built.
    """
    
    analysis = await llm_service.generate(prompt)
    print_step("TASK ANALYSIS RESULTS", analysis, "INFO")
    return analysis

async def enhance_workflow_processor(system_components):
    """Modify the workflow processor to use real component instances."""
    print_step("ENHANCING WORKFLOW PROCESSOR", 
             "Modifying the workflow processor to use real component instances", 
             "TASK")
    
    system_agent = system_components["system_agent"]
    agent_bus = system_components["agent_bus"]
    
    # Store the original process_workflow method
    original_process_workflow = system_agent.workflow_processor.process_workflow
    
    # Create an enhanced version that uses the agent bus
    async def enhanced_process_workflow(workflow_yaml, **kwargs):
        print_step("PROCESSING WORKFLOW WITH REAL COMPONENTS", 
                 "Using registered component instances instead of mock execution", 
                 "PROGRESS")
        
        # Parse the workflow YAML
        try:
            workflow = yaml.safe_load(workflow_yaml)
        except Exception as e:
            return {"status": "error", "message": f"Error parsing workflow YAML: {str(e)}"}
        
        # Extract metadata
        scenario_name = workflow.get("scenario_name", "Unnamed Scenario")
        domain = workflow.get("domain", "general")
        
        # Process steps with real component execution where possible
        results = []
        
        for i, step in enumerate(workflow.get("steps", [])):
            step_type = step.get("type", "").upper()
            item_type = step.get("item_type", "").upper()
            name = step.get("name", "")
            
            print_step(f"EXECUTING STEP {i+1}", 
                     f"Type: {step_type}, Item: {name} ({item_type})", 
                     "PROGRESS")
            
            if step_type == "EXECUTE":
                try:
                    # Construct input content
                    content = {}
                    
                    # Handle different input formats
                    if "user_input" in step:
                        content["text"] = step["user_input"]
                    
                    if "inputs" in step:
                        for key, value in step["inputs"].items():
                            content[key] = value
                    
                    # Find the appropriate capability for this component
                    capability = "process_input"  # Default capability
                    
                    # Find registered provider by name
                    provider_id = None
                    for pid, provider in agent_bus.providers.items():
                        if provider["name"] == name:
                            provider_id = pid
                            # Try to find a more specific capability
                            if provider["capabilities"]:
                                capability = provider["capabilities"][0]["id"]
                            break
                    
                    if provider_id:
                        print(f"Using registered instance of {name} with capability '{capability}'")
                        # Execute using the agent bus with real registered instance
                        result = await agent_bus.request_service(
                            capability=capability,
                            content=content,
                            provider_id=provider_id
                        )
                        results.append({
                            "step": i+1,
                            "name": name,
                            "result": result,
                            "using": "registered_instance"
                        })
                    else:
                        print(f"No registered instance found for {name}, falling back to workflow processing")
                        # Fall back to original processing with a YAML subset containing just this step
                        result = await original_process_workflow(
                            workflow_yaml=yaml.dump({"steps": [step]})
                        )
                        results.append({
                            "step": i+1,
                            "name": name,
                            "result": result,
                            "using": "fallback_processing"
                        })
                        
                except Exception as e:
                    print(f"Error executing {name}: {str(e)}")
                    results.append({
                        "step": i+1,
                        "name": name,
                        "error": str(e),
                        "using": "attempted_real_execution"
                    })
            else:
                # For non-EXECUTE steps, use the original processing
                try:
                    step_workflow = yaml.dump({"steps": [step]})
                    result = await original_process_workflow(workflow_yaml=step_workflow)
                    results.append({
                        "step": i+1,
                        "name": name if name else f"Step {i+1}",
                        "result": result,
                        "using": "original_processing"
                    })
                except Exception as e:
                    print(f"Error processing step {i+1}: {str(e)}")
                    results.append({
                        "step": i+1,
                        "name": name if name else f"Step {i+1}",
                        "error": str(e),
                        "using": "original_processing"
                    })
        
        # Create a comprehensive result
        execution_results = {
            "status": "success",
            "scenario_name": scenario_name,
            "domain": domain,
            "steps_executed": len(results),
            "results": results,
            "using_real_components": True
        }
        
        # Format the result for output
        result_output = f"Executed workflow '{scenario_name}' with {len(results)} steps.\n\n"
        
        for step_result in results:
            step_num = step_result["step"]
            name = step_result["name"]
            using = step_result["using"]
            
            result_output += f"Step {step_num}: **{name}**\n"
            
            if "error" in step_result:
                result_output += f"  - **Error**: {step_result['error']}\n"
            else:
                result_output += f"  - **Execution Mode**: {using}\n"
                
                if using == "registered_instance":
                    # Extract relevant information from agent bus result
                    component_result = step_result.get("result", {})
                    provider_name = component_result.get("provider_name", "Unknown")
                    content = component_result.get("content", {})
                    
                    result_output += f"  - **Provider**: {provider_name}\n"
                    result_output += f"  - **Result**: {json.dumps(content, indent=2)[:300]}...\n"
                else:
                    # Handle results from original processing
                    step_status = step_result.get("result", {}).get("status", "unknown")
                    step_message = step_result.get("result", {}).get("message", "No message")
                    
                    result_output += f"  - **Status**: {step_status}\n"
                    result_output += f"  - **Result**: {step_message}\n"
            
            result_output += "\n"
        
        return {
            "status": "success",
            "scenario_name": scenario_name,
            "domain": domain,
            "message": f"Successfully executed workflow '{scenario_name}' with real components where possible",
            "result": result_output,
            "detailed_results": execution_results
        }
    
    # Replace the workflow processor's process_workflow method
    system_agent.workflow_processor.process_workflow = enhanced_process_workflow
    
    print_step("WORKFLOW PROCESSOR ENHANCED", 
             "Workflow processor now attempts to use real registered components", 
             "SUCCESS")
    
async def analyze_library_components(system_components):
    """Analyze library components to verify their instantiability."""
    print_step("ANALYZING LIBRARY COMPONENTS", 
             "Checking if components can be instantiated", 
             "TASK")
    
    smart_library = system_components["smart_library"]
    llm_service = system_components["llm_service"]
    
    # Create factories
    from evolving_agents.tools.tool_factory import ToolFactory
    tool_factory = ToolFactory(smart_library, llm_service)
    
    from evolving_agents.agents.agent_factory import AgentFactory
    agent_factory = AgentFactory(smart_library, llm_service)
    
    # Get all components
    all_components = smart_library.records
    
    instantiation_results = []
    
    for component in all_components:
        component_name = component["name"]
        component_type = component["record_type"]
        
        try:
            if component_type == "TOOL":
                instance = await tool_factory.create_tool(component)
                instantiation_results.append({
                    "name": component_name,
                    "type": "TOOL",
                    "status": "success",
                    "instance_type": type(instance).__name__
                })
            elif component_type == "AGENT":
                instance = await agent_factory.create_agent(component)
                instantiation_results.append({
                    "name": component_name,
                    "type": "AGENT", 
                    "status": "success",
                    "instance_type": type(instance).__name__
                })
        except Exception as e:
            instantiation_results.append({
                "name": component_name,
                "type": component_type,
                "status": "error",
                "error": str(e)
            })
    
    # Summarize results
    success_count = sum(1 for r in instantiation_results if r["status"] == "success")
    error_count = sum(1 for r in instantiation_results if r["status"] == "error")
    
    print_step("LIBRARY COMPONENT ANALYSIS RESULTS", {
        "Total components": len(all_components),
        "Successfully instantiable": success_count,
        "Failed instantiation": error_count,
        "Success rate": f"{success_count/len(all_components)*100:.1f}%"
    }, "INFO")
    
    if error_count > 0:
        print("\nComponents with instantiation errors:")
        for result in instantiation_results:
            if result["status"] == "error":
                print(f"- {result['name']} ({result['type']}): {result['error']}")
    
    return instantiation_results

async def run_architect_agent(system_components, prompt_file, sample_data=None, output_prefix="result"):
    """Run the architect agent with a prompt from a file."""
    architect_agent = system_components["architect_agent"]
    system_agent = system_components["system_agent"]
    smart_library = system_components["smart_library"]
    llm_service = system_components["llm_service"]

    library_analysis = await analyze_library_components(system_components)
    
    # Load the prompt from the file
    print_step("LOADING TASK PROMPT", f"Reading task requirements from {prompt_file}", "TASK")
    with open(prompt_file, "r") as f:
        task_requirement = f.read()
    
    # Print the task
    print_step("TASK REQUIREMENTS", task_requirement, "INFO")
    
    # Analyze the task using LLM for better visibility
    task_analysis = await analyze_task_with_llm(llm_service, task_requirement)
    
    # Extract required capabilities
    print_step("EXTRACTING REQUIRED CAPABILITIES", 
             "Using the LLM to identify the specialized capabilities needed for this task", 
             "TASK")
    
    extracted_capabilities = await smart_library._extract_capabilities_with_llm(
        task_requirement,
        "general"
    )
    print_step("REQUIRED CAPABILITIES", {
        "Extracted capabilities": ", ".join(extracted_capabilities)
    }, "INFO")
    
    # Search for components that can fulfill these capabilities
    print_step("COMPONENT MATCHING", 
             "Searching for existing components that can fulfill the required capabilities", 
             "TASK")
    
    capability_components = {}
    for capability in extracted_capabilities:
        component = await smart_library.find_component_by_capability(capability)
        if component:
            capability_components[capability] = component["name"]
        else:
            capability_components[capability] = "No existing component found"
    
    print_step("CAPABILITY-COMPONENT MAPPING", capability_components, "INFO")
    
    # Execute Architect-Zero to design the solution
    print_step("DESIGNING SOLUTION", 
             "Architect-Zero is designing a complete solution with full reasoning transparency", 
             "TASK")
    
    try:
        # Execute the architect agent
        print(f"{Fore.GREEN}Starting agent reasoning process...{Style.RESET_ALL}")
        start_time = time.time()
        result = await architect_agent.run(task_requirement)
        design_time = time.time() - start_time
        
        # Save the full thought process
        with open(f"{output_prefix}_interaction.txt", "w") as f:
            f.write(f"TASK REQUIREMENT:\n\n{task_requirement}\n\n")
            f.write(f"AGENT THOUGHT PROCESS:\n{result.result.text}")
        
        # Use LLM to summarize the reasoning process for better visibility
        print_step("SUMMARIZING AGENT REASONING", 
                 "Using LLM to extract key points from the design process", 
                 "PROGRESS")
        
        reasoning_summary_prompt = f"""
        Summarize the key design decisions and reasoning steps in this agent's thought process.
        Focus on:
        1. The main components identified
        2. The architecture pattern chosen
        3. Key interactions between components
        4. Trade-offs considered
        5. Novel approaches proposed

        Provide a concise, structured summary of no more than 250 words.

        AGENT THOUGHT PROCESS:
        {result.result.text[:4000]}  # Take a portion to avoid token limits
        """
        
        reasoning_summary = await llm_service.generate(reasoning_summary_prompt)
        
        print_step("AGENT REASONING SUMMARY", 
                 reasoning_summary, 
                 "REASONING")
        
        # Extract workflow from the result
        yaml_content = await extract_yaml_workflow(result.result.text)
        if yaml_content:
            # Save the workflow to a file
            with open(f"{output_prefix}_workflow.yaml", "w") as f:
                f.write(yaml_content)
            
            print_step("WORKFLOW GENERATED", 
                     "Architect-Zero has created a complete workflow", 
                     "SUCCESS")
            
            # Analyze workflow structure with LLM
            print_step("ANALYZING WORKFLOW STRUCTURE", 
                     "Examining the components and execution flow", 
                     "TASK")
            
            workflow_analysis_prompt = f"""
            Analyze this YAML workflow and provide:
            1. A count and list of the main components (agents and tools)
            2. The execution sequence/flow between components
            3. Any interesting or novel aspects of the design
            4. Potential strengths and weaknesses of this approach

            YAML WORKFLOW:
            {yaml_content}
            
            Provide a clear, structured analysis.
            """
            
            workflow_analysis = await llm_service.generate(workflow_analysis_prompt)
            print_step("WORKFLOW ANALYSIS", workflow_analysis, "INFO")
            
            # Enhance the workflow processor
            await enhance_workflow_processor(system_components)
            
            # Register real component instances
            registered_components = await register_real_components(system_components, yaml_content)
            
            if registered_components:
                # Execute the workflow with real components
                print_step("EXECUTING WORKFLOW WITH REAL COMPONENTS", 
                        f"Running workflow with {len(registered_components)} registered components", 
                        "TASK")
                
                workflow_start_time = time.time()
                execution_result = await system_agent.workflow_processor.process_workflow(yaml_content)
                workflow_time = time.time() - workflow_start_time
                
                # Save execution result
                with open(f"{output_prefix}_result.json", "w") as f:
                    json.dump(execution_result, f, indent=2)
                
                # Show execution results
                if execution_result and execution_result.get("status") == "success":
                    print_step("WORKFLOW EXECUTION RESULTS", {
                        "Execution time": f"{workflow_time:.2f} seconds",
                        "Status": execution_result.get("status"),
                        "Using real components": "Yes",
                        "Components used": len(registered_components),
                        "Result": "See detailed output below"
                    }, "SUCCESS")
                    
                    # Show the complete result
                    print(execution_result.get("result", "No detailed result available"))
                else:
                    print_step("WORKFLOW EXECUTION ISSUE", 
                            f"Status: {execution_result.get('status', 'unknown')}, Message: {execution_result.get('message', 'Unknown error')}", 
                            "ERROR")
            else:
                print_step("WORKFLOW EXECUTION WITH MOCK COMPONENTS", 
                        "No real components could be registered, falling back to mock execution", 
                        "WARNING")
                
                workflow_start_time = time.time()
                execution_result = await system_agent.workflow_processor.process_workflow(yaml_content)
                workflow_time = time.time() - workflow_start_time
            
            # Save execution result
            with open(f"{output_prefix}_result.json", "w") as f:
                json.dump(execution_result, f, indent=2)
            
            # Show execution results
            if execution_result and execution_result.get("status") == "success":
                print_step("WORKFLOW EXECUTION RESULTS", {
                    "Execution time": f"{workflow_time:.2f} seconds",
                    "Status": execution_result.get("status"),
                    "Result": "See detailed output below"
                }, "SUCCESS")
                
                # Use LLM to provide a summary of the execution results
                print_step("ANALYZING EXECUTION RESULTS", 
                         "Extracting key insights from the execution output", 
                         "TASK")
                
                result_text = execution_result.get("result", "")
                results_summary_prompt = f"""
                Analyze these execution results and provide:
                1. A summary of what the workflow accomplished
                2. The key insights or findings
                3. Any issues or errors encountered
                4. The overall quality of the output

                EXECUTION RESULTS:
                {result_text[:4000]}  # Limit to avoid token issues
                
                Provide a clear, concise summary of the results.
                """
                
                results_summary = await llm_service.generate(results_summary_prompt)
                print_step("EXECUTION RESULTS SUMMARY", results_summary, "INFO")
                
                # Show the complete result
                print(execution_result.get("result", "No detailed result available"))
            else:
                print_step("WORKFLOW EXECUTION ISSUE", 
                         f"Status: {execution_result.get('status', 'unknown')}, Message: {execution_result.get('message', 'Unknown error')}", 
                         "ERROR")
        else:
            print_step("WORKFLOW GENERATION ISSUE", 
                     "No YAML workflow found in the agent's response.", 
                     "ERROR")
            
    except Exception as e:
        print_step("ERROR", str(e), "ERROR")
        import traceback
        print(traceback.format_exc())
    
    # Clean up temporary files
    if os.path.exists(system_components["temp_library_path"]):
        os.remove(system_components["temp_library_path"])
    
    return result