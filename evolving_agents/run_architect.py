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
from typing import Dict, Any, Optional

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
        "MEDICAL": Fore.LIGHTRED_EX
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
    # Load the library from the JSON file
    with open(library_path, "r") as f:
        library_data = json.load(f)
    
    # Create a temporary file for the library
    temp_library_path = f"temp_{os.path.basename(library_path)}"
    with open(temp_library_path, "w") as f:
        json.dump(library_data, f, indent=2)
    
    # Initialize LLM service
    llm_service = LLMService(provider="openai", model="gpt-4o")
    
    # Initialize SmartLibrary with the temporary file
    smart_library = SmartLibrary(temp_library_path, llm_service)
    smart_library.records = library_data["records"]
    
    # Initialize agent bus
    agent_bus = SimpleAgentBus("agent_bus.json")
    agent_bus.set_llm_service(llm_service)
    
    # Create system agent
    system_agent = await SystemAgentFactory.create_agent(
        llm_service=llm_service,
        smart_library=smart_library,
        agent_bus=agent_bus
    )
    system_agent.workflow_processor.set_llm_service(llm_service)
    system_agent.workflow_generator.set_llm_service(llm_service)
    
    # Create Architect-Zero meta-agent
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

async def run_architect_agent(system_components, prompt_file, sample_data=None, output_prefix="result"):
    """Run the architect agent with a prompt from a file."""
    architect_agent = system_components["architect_agent"]
    system_agent = system_components["system_agent"]
    smart_library = system_components["smart_library"]
    
    # Load the prompt from the file
    with open(prompt_file, "r") as f:
        task_requirement = f.read()
    
    # Print the task
    print_step("TASK REQUIREMENTS", task_requirement, "INFO")
    
    # Extract required capabilities
    print_step("CAPABILITY EXTRACTION WITH LLM", 
             "Using the LLM to identify the specialized capabilities needed for this task", 
             "REASONING")
    
    extracted_capabilities = await smart_library._extract_capabilities_with_llm(
        task_requirement,
        "general"
    )
    print_step("REQUIRED CAPABILITIES", {
        "Extracted capabilities": ", ".join(extracted_capabilities)
    }, "REASONING")
    
    # Execute Architect-Zero to design the solution
    print_step("DESIGNING SOLUTION", 
             "Architect-Zero is designing a solution with full reasoning transparency", 
             "AGENT")
    
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
        
        # Extract workflow from the result
        yaml_content = await extract_yaml_workflow(result.result.text)
        if yaml_content:
            # Save the workflow to a file
            with open(f"{output_prefix}_workflow.yaml", "w") as f:
                f.write(yaml_content)
            
            print_step("WORKFLOW GENERATED", 
                     "Architect-Zero has created a complete workflow", 
                     "SUCCESS")
            
            # Execute the workflow
            print_step("EXECUTING WORKFLOW", 
                     "Now the system will instantiate and execute components", 
                     "EXECUTION")
            
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