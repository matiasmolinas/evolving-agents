# evolving_agents/workflow/workflow_processor.py

import logging
import yaml
from typing import Dict, Any, Optional
import json
import sys

# Import the interface instead of the concrete class
from evolving_agents.core.base import IAgent

logger = logging.getLogger(__name__)

class WorkflowProcessor:
    """
    Processes YAML-based workflows with support for multiple agent frameworks.
    """
    def __init__(self, agent: Optional[IAgent] = None):
        """
        Initialize the workflow processor.
        
        Args:
            agent: Agent for processing workflows (can be set later)
        """
        self.agent = agent
        logger.info("Workflow Processor initialized")
    
    def set_agent(self, agent: IAgent) -> None:
        """Set the agent after initialization."""
        self.agent = agent
    
    async def process_workflow(self, workflow_yaml: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a workflow from a YAML string.
        
        Args:
            workflow_yaml: YAML string defining the workflow
            params: Optional parameters to pass to the workflow
                
        Returns:
            Execution results
        """
        if not self.agent:
            return {"status": "error", "message": "No agent set for workflow processing"}
            
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
        
        # Include params in the prompt if provided
        params_section = ""
        if params:
            params_section = f"\nParameters:\n{json.dumps(params, indent=2)}"
        
        # Run the workflow using the agent's run method
        prompt = f"""
        I need to process this workflow YAML for domain '{domain}'{params_section}:
        
        ```yaml
        {workflow_yaml}
        ```
        
        Please execute this workflow step by step and provide the results for each step.
        For each step, indicate the step type, what action was performed, and the outcome.
        
        When executing tools or agents, show the input and output for each execution.
        When you see placeholders like {{params.some_variable}}, replace them with the corresponding value from the parameters.
        """
        
        response = await self.agent.run(prompt)
        
        # Return structured result
        return {
            "status": "success",
            "scenario_name": scenario_name,
            "domain": domain,
            "result": response.result.text,
            "message": f"Successfully executed workflow '{scenario_name}' using agent"
        }