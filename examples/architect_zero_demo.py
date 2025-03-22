# architect_zero_demo.py

import asyncio
import logging
import json
import os
import re
import time
import argparse
import colorama
from colorama import Fore, Style

# Import core components from the Evolving Agents Toolkit
from evolving_agents.core.llm_service import LLMService
from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.agent_bus.simple_agent_bus import SimpleAgentBus
from evolving_agents.core.system_agent import SystemAgentFactory
from evolving_agents.agents.architect_zero import create_architect_zero
from evolving_agents.tools.tool_factory import ToolFactory

# Initialize colorama for cross-platform colored terminal output
colorama.init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
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
        "COMPONENT": Fore.MAGENTA  # For component creation
    }
    
    color = colors.get(step_type, Fore.WHITE)
    
    # Print header
    print(f"\n{color}{'=' * 80}")
    print(f"  {step_type}: {title}")
    print(f"{'=' * 80}{Style.RESET_ALL}")
    
    # Print content if provided
    if isinstance(content, dict):
        for key, value in content.items():
            print(f"{Fore.CYAN}{key}{Style.RESET_ALL}: {value}")
    else:
        print(content)

def clean_previous_files():
    """Remove previous files to start fresh."""
    files_to_remove = [
        "smart_library.json",
        "agent_bus.json",
        "architect_interaction.txt",
        "workflow.yaml",
        "workflow_execution_result.json",
    ]
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Removed previous file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {str(e)}")

def read_task_from_file(file_path):
    """Read the task requirements from a file."""
    try:
        with open(file_path, 'r') as f:
            task = f.read().strip()
        return task
    except Exception as e:
        logger.error(f"Error reading task file: {str(e)}")
        raise

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

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Architect Zero Demo with Flexible Task')
    parser.add_argument('--task_file', type=str, default='task.txt', 
                      help='Path to the file containing the task requirements')
    parser.add_argument('--model', type=str, default='gpt-4o',
                      help='LLM model to use (default: gpt-4o)')
    parser.add_argument('--clean', action='store_true',
                      help='Clean previous files before running')
    args = parser.parse_args()
    
    print_step("EVOLVING AGENTS TOOLKIT DEMONSTRATION", 
              "This demonstration shows how Architect-Zero designs multi-agent systems based on task requirements.", 
              "INFO")
    
    # Clean up previous files if requested
    if args.clean:
        clean_previous_files()
    
    # Read the task from file
    task_requirement = read_task_from_file(args.task_file)
    
    # Initialize LLM service
    llm_service = LLMService(provider="openai", model=args.model)
    
    # Initialize the library and agent bus
    print_step("INITIALIZING CORE COMPONENTS", 
              "Setting up the Smart Library and Agent Bus", 
              "COMPONENT")
    
    smart_library = SmartLibrary("smart_library.json", llm_service)
    agent_bus = SimpleAgentBus("agent_bus.json")
    agent_bus.set_llm_service(llm_service)
    
    # Create the system agent
    print_step("INITIALIZING SYSTEM AGENT", 
              "Creating the System Agent that manages the agent ecosystem", 
              "AGENT")
    
    system_agent = await SystemAgentFactory.create_agent(
        llm_service=llm_service,
        smart_library=smart_library,
        agent_bus=agent_bus
    )
    system_agent.workflow_processor.set_llm_service(llm_service)
    system_agent.workflow_generator.set_llm_service(llm_service)
    
    # Create the Architect-Zero meta-agent
    print_step("CREATING ARCHITECT-ZERO META-AGENT", 
              "This agent designs entire agent systems by analyzing requirements", 
              "AGENT")
    
    architect_agent = await create_architect_zero(
        llm_service=llm_service,
        smart_library=smart_library,
        agent_bus=agent_bus,
        system_agent=system_agent
    )
    
    # Print the task
    print_step("TASK REQUIREMENTS", task_requirement, "INFO")
    
    # Extract required capabilities from the task requirements
    print_step("CAPABILITY EXTRACTION WITH LLM", 
              "Using the LLM to identify the specialized capabilities needed for this task", 
              "REASONING")
    
    # Determine domain based on the task
    domain = await smart_library._extract_domain(task_requirement)
    print_step("DOMAIN IDENTIFICATION", {
        "Identified domain": domain
    }, "REASONING")
    
    # Extract capabilities
    extracted_capabilities = await smart_library._extract_capabilities_with_llm(task_requirement, domain)
    print_step("REQUIRED CAPABILITIES", {
        "Extracted capabilities": ", ".join(extracted_capabilities)
    }, "REASONING")
    
    # Execute Architect-Zero to design the solution
    print_step("DESIGNING SOLUTION", 
              "Architect-Zero is designing a multi-agent solution with full reasoning transparency", 
              "AGENT")
    
    try:
        # Execute the architect agent with full reasoning log
        print(f"{Fore.GREEN}Starting agent reasoning process...{Style.RESET_ALL}")
        start_time = time.time()
        result = await architect_agent.run(task_requirement)
        design_time = time.time() - start_time
        
        # Save the full thought process
        with open("architect_interaction.txt", "w") as f:
            f.write(f"TASK REQUIREMENT:\n\n    {task_requirement}\n\n")
            f.write(f"AGENT THOUGHT PROCESS:\n{result.result.text}")
        
        # Show the reasoning process (truncated)
        reasoning_preview = result.result.text[:500] + "..." if len(result.result.text) > 500 else result.result.text
        print_step("AGENT REASONING REVEALED", {
            "Design time": f"{design_time:.2f} seconds",
            "Reasoning preview": reasoning_preview,
            "Full reasoning": "Saved to 'architect_interaction.txt'"
        }, "REASONING")
        
        # Extract workflow from the result
        yaml_content = await extract_yaml_workflow(result.result.text)
        if yaml_content:
            # Save the workflow to a file
            with open("workflow.yaml", "w") as f:
                f.write(yaml_content)
            
            print_step("MULTI-AGENT WORKFLOW GENERATED", 
                      "Architect-Zero has created a complete workflow with specialized agents", 
                      "SUCCESS")
            
            # Show abbreviated workflow
            workflow_lines = yaml_content.split('\n')
            workflow_preview = '\n'.join(workflow_lines[:20])
            if len(workflow_lines) > 20:
                workflow_preview += f"\n{Fore.CYAN}... (see workflow.yaml for complete workflow){Style.RESET_ALL}"
            print(workflow_preview)
            
            # Extract information about the components in the workflow
            component_definitions = re.findall(r'type:\s+DEFINE.*?name:\s+(\w+).*?item_type:\s+(\w+)', yaml_content, re.DOTALL)
            component_executions = re.findall(r'type:\s+EXECUTE.*?name:\s+(\w+)', yaml_content, re.DOTALL)
            
            print_step("WORKFLOW COMPONENT ANALYSIS", {
                "Component definitions": len(component_definitions),
                "Component executions": len(component_executions),
                "Defined components": ", ".join([f"{name} ({type})" for name, type in component_definitions]) if component_definitions else "None found",
                "Execution sequence": " → ".join(component_executions) if component_executions else "None found"
            }, "REASONING")
            
            # Execute the workflow
            print_step("EXECUTING MULTI-AGENT WORKFLOW", 
                      "Now the system will instantiate and execute all components in the workflow", 
                      "EXECUTION")
            
            workflow_start_time = time.time()
            execution_result = await system_agent.workflow_processor.process_workflow(yaml_content)
            workflow_time = time.time() - workflow_start_time
            
            # Save execution result
            with open("workflow_execution_result.json", "w") as f:
                json.dump(execution_result, f, indent=2)
            
            # Show execution results
            if execution_result and execution_result.get("status") == "success":
                print_step("WORKFLOW EXECUTION RESULTS", {
                    "Execution time": f"{workflow_time:.2f} seconds",
                    "Status": execution_result.get("status"),
                    "Result": "See detailed output below"
                }, "SUCCESS")
                
                # Extract detailed results from execution output
                result_text = execution_result.get("result", "")
                
                # Extract step executions for better visibility
                step_pattern = r'Step (\d+): \*\*([^*]+)\*\*\s*\n\s*- \*\*([^*]+)\*\*: (.*?)(?=\n\s*(?:- \*\*|\n\d+\.|$))'
                steps = re.findall(step_pattern, result_text, re.DOTALL)
                
                # Display the step results in a more readable format
                if steps:
                    print(f"\n{Fore.CYAN}Workflow Execution Steps:{Style.RESET_ALL}")
                    for step_num, step_type, action_type, action_result in steps:
                        print(f"{Fore.GREEN}Step {step_num}{Style.RESET_ALL}: {step_type.strip()}")
                        print(f"  {Fore.YELLOW}{action_type.strip()}{Style.RESET_ALL}: {action_result.strip()}")
                        print()
                else:
                    # Fallback to showing the original text
                    print(result_text)
                
                # Show insights about the agent collaboration
                agent_count = len(re.findall(r'type:\s+DEFINE.*?item_type:\s+AGENT', yaml_content, re.DOTALL))
                tool_count = len(re.findall(r'type:\s+DEFINE.*?item_type:\s+TOOL', yaml_content, re.DOTALL))
                data_flows = len(re.findall(r'input_data:', yaml_content))
                
                print_step("EVOLVING AGENTS SYSTEM INSIGHTS", {
                    "Specialized agents": agent_count,
                    "Specialized tools": tool_count, 
                    "Data flows between components": data_flows,
                    "Domain": domain,
                    "System advantages": "Component specialization with transparent reasoning and evolution capability"
                }, "SUCCESS")
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
    
    print_step("DEMONSTRATION COMPLETE", 
              """
This demonstration showed the key capabilities of the Evolving Agents Toolkit:

1. Smart Library with LLM-powered capability matching for component selection
2. Architect-Zero design of multi-agent systems from high-level requirements
3. Execution of workflows with specialized components
4. Full transparency into agent reasoning and collaboration
5. Automatic orchestration of complex multi-agent systems

The toolkit enables the creation of systems where specialized agents collaborate with 
transparent reasoning, providing both performance and explainability.
              """, 
              "INFO")

if __name__ == "__main__":
    asyncio.run(main())