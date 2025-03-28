import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field

from beeai_framework.agents.react import ReActAgent
from beeai_framework.agents.types import AgentMeta
from beeai_framework.memory import TokenMemory
from beeai_framework.tools.tool import Tool, StringToolOutput
from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter

from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.core.llm_service import LLMService
from evolving_agents.agent_bus.smart_agent_bus import SmartAgentBus
from evolving_agents.workflow.workflow_generator import WorkflowGenerator
from evolving_agents.core.system_agent import SystemAgentFactory
from evolving_agents.core.dependency_container import DependencyContainer

logger = logging.getLogger(__name__)

class ArchitectZeroAgentInitializer:
    """
    Architect-Zero agent that analyzes task requirements and designs complete 
    agent-based solutions.
    
    This agent serves as a solution designer that:
    1. Analyzes task requirements in natural language
    2. Designs a workflow of agents and tools to complete the task
    3. Generates component designs with appropriate specifications
    4. Creates YAML workflows that can be executed by the System Agent
    5. Provides guidance on component reuse, evolution, and creation
    """
    
    @staticmethod
    async def create_agent(
        llm_service: Optional[LLMService] = None,
        smart_library: Optional[SmartLibrary] = None,
        agent_bus: Optional[SmartAgentBus] = None,
        container: Optional[DependencyContainer] = None
    ) -> ReActAgent:
        """
        Create and configure the Architect-Zero agent.
        
        Args:
            llm_service: LLM service for text generation
            smart_library: Smart library for component management
            agent_bus: Agent bus for component communication
            container: Optional dependency container for managing component dependencies
            
        Returns:
            Configured Architect-Zero agent
        """
        # Resolve dependencies from container if provided
        if container:
            llm_service = llm_service or container.get('llm_service')
            smart_library = smart_library or container.get('smart_library')
            agent_bus = agent_bus or container.get('agent_bus')
            
            # Get system agent if it exists in the container
            system_agent = container.get('system_agent') if container.has('system_agent') else None
        else:
            system_agent = None
            if not llm_service:
                raise ValueError("LLM service must be provided when not using a dependency container")
        
        # Get the chat model from LLM service
        chat_model = llm_service.chat_model
        
        # Initialize SmartLibrary if needed
        if smart_library and hasattr(smart_library, 'initialize'):
            await smart_library.initialize()
        
        # Get workflow generator - will use existing or create a new one
        if container and container.has('workflow_generator'):
            workflow_generator = container.get('workflow_generator')
        else:
            workflow_generator = WorkflowGenerator(llm_service, smart_library)
            if system_agent:
                workflow_generator.set_agent(system_agent)
            if container:
                container.register('workflow_generator', workflow_generator)
        
        # Create tools for the architect agent
        architect_tools = [
            AnalyzeRequirementsTool(llm_service, smart_library),
            DesignSolutionTool(llm_service, smart_library),
            ComponentSpecificationTool(llm_service, smart_library),
            GenerateWorkflowTool(workflow_generator)
        ]
        
        # Create agent metadata
        meta = AgentMeta(
            name="Architect-Zero",
            description=(
                "I am Architect-Zero, specialized in solution design and architecture. "
                "I analyze requirements, design component-based solutions, create specifications, "
                "and generate workflow designs to be executed by the System Agent. I focus on "
                "high-level architecture rather than implementation details."
            ),
            extra_description=(
                "I work closely with the System Agent, which handles execution and implementation. "
                "My role is to design and architect solutions that can be created and executed "
                "by other agents in the ecosystem."
            ),
            tools=architect_tools
        )
        
        # Create the Architect-Zero agent
        agent = ReActAgent(
            llm=chat_model,
            tools=architect_tools,
            memory=TokenMemory(chat_model),
            meta=meta
        )
        
        # Add tools dictionary for direct access
        tools_dict = {
            "AnalyzeRequirementsTool": architect_tools[0],
            "DesignSolutionTool": architect_tools[1],
            "ComponentSpecificationTool": architect_tools[2],
            "GenerateWorkflowTool": architect_tools[3]
        }
        agent.tools = tools_dict
        
        # Register with agent bus if provided
        if agent_bus:
            # Create serializable metadata (no agent instance)
            serializable_metadata = {
                "role": "architecture_designer",
                "creation_timestamp": datetime.now().isoformat(),
                "tool_count": len(architect_tools),
                "framework": "beeai"
            }
            
            # Register agent in the bus
            await agent_bus.register_agent(
                name="Architect-Zero",
                capabilities=[
                    {
                        "id": "solution_design",
                        "name": "Solution Design",
                        "description": "Design complete agent-based solutions from requirements",
                        "confidence": 0.95
                    },
                    {
                        "id": "requirements_analysis",
                        "name": "Requirements Analysis",
                        "description": "Analyze task requirements to identify needed components",
                        "confidence": 0.9
                    },
                    {
                        "id": "workflow_generation",
                        "name": "Workflow Generation",
                        "description": "Generate workflows for complex agent-based solutions",
                        "confidence": 0.85
                    },
                    {
                        "id": "component_specification",
                        "name": "Component Specification",
                        "description": "Create detailed specifications for components",
                        "confidence": 0.9
                    }
                ],
                agent_type="ARCHITECT",
                description="Solution architect agent that designs agent-based solutions",
                metadata=serializable_metadata,  # Only serializable metadata here
                agent_instance=agent  # Pass agent instance separately if supported
            )
        
        # Register with container if provided
        if container and not container.has('architect_agent'):
            container.register('architect_agent', agent)
        
        return agent


# Input schemas for the tools
class AnalyzeRequirementsInput(BaseModel):
    """Input schema for the AnalyzeRequirementsTool."""
    task_requirements: str = Field(description="Task requirements in natural language")
    domain: Optional[str] = Field(None, description="Optional domain hint for the analysis")
    constraints: Optional[str] = Field(None, description="Optional constraints to consider")

class DesignSolutionInput(BaseModel):
    """Input schema for the DesignSolutionTool."""
    requirements_analysis: Dict[str, Any] = Field(description="Analysis from AnalyzeRequirementsTool")
    solution_type: Optional[str] = Field("agent_based", description="Type of solution to design (agent_based, pipeline, etc.)")
    optimization_focus: Optional[str] = Field(None, description="What to optimize for (speed, accuracy, etc.)")

class ComponentSpecInput(BaseModel):
    """Input schema for the ComponentSpecificationTool."""
    solution_design: Dict[str, Any] = Field(description="Solution design from DesignSolutionTool")
    component_name: Optional[str] = Field(None, description="Specific component to generate a spec for (or all if None)")
    detail_level: Optional[str] = Field("standard", description="Level of detail in the specification (basic, standard, detailed)")

class GenerateWorkflowInput(BaseModel):
    """Input schema for the GenerateWorkflowTool."""
    solution_design: Dict[str, Any] = Field(description="Solution design from DesignSolutionTool")
    component_specs: Optional[Dict[str, Any]] = Field(None, description="Component specifications from ComponentSpecificationTool")
    workflow_name: Optional[str] = Field(None, description="Name for the workflow")
    include_comments: Optional[bool] = Field(True, description="Whether to include explanatory comments in the workflow")


class AnalyzeRequirementsTool(Tool[AnalyzeRequirementsInput, None, StringToolOutput]):
    """Tool for analyzing task requirements and identifying needed components."""
    
    name = "AnalyzeRequirementsTool"
    description = "Analyze task requirements to identify needed agents, tools, and capabilities"
    input_schema = AnalyzeRequirementsInput
    
    def __init__(self, llm_service: LLMService, smart_library: SmartLibrary):
        super().__init__()
        self.llm = llm_service
        self.library = smart_library
    
    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "architect", "analyze"],
            creator=self,
        )
    
    async def _run(
        self, 
        input: AnalyzeRequirementsInput, 
        options: Optional[Dict[str, Any]] = None, 
        context: Optional[RunContext] = None
    ) -> StringToolOutput:
        """
        Analyze task requirements and identify needed components.
        
        Args:
            input: Task requirements in natural language
            
        Returns:
            Analysis results including required components and capabilities
        """
        domain_hint = f"\nDOMAIN HINT: {input.domain}" if input.domain else ""
        constraints = f"\nCONSTRAINTS: {input.constraints}" if input.constraints else ""
        
        # Prompt the LLM to analyze the requirements
        analysis_prompt = f"""
        Analyze the following task requirements and identify the necessary components:

        TASK REQUIREMENTS:
        {input.task_requirements}{domain_hint}{constraints}

        Please provide:
        1. A clear summary of the task's objective
        2. The domain(s) this task falls under
        3. A list of required agents with their purpose
        4. A list of required tools with their purpose
        5. The key capabilities needed by these components
        6. Any constraints or special considerations
        7. Any data or resources that would be needed

        Format your response as a structured JSON object with these sections.
        """
        
        # Generate analysis
        analysis_response = await self.llm.generate(analysis_prompt)
        
        try:
            # Parse the JSON response
            analysis = json.loads(analysis_response)
            
            # Check for existing components in the library that match requirements
            existing_components = []
            
            # Check for agents
            if "required_agents" in analysis:
                for agent in analysis["required_agents"]:
                    agent_name = agent.get("name", "")
                    agent_purpose = agent.get("purpose", "")
                    
                    # Search for similar agents in the library
                    search_query = f"agent that can {agent_purpose}"
                    similar_agents = await self.library.semantic_search(
                        search_query,
                        record_type="AGENT",
                        limit=3,
                        threshold=0.6
                    )
                    
                    if similar_agents:
                        existing_components.append({
                            "type": "AGENT",
                            "name": agent_name,
                            "purpose": agent_purpose,
                            "similar_existing_components": [
                                {
                                    "id": sa[0]["id"],
                                    "name": sa[0]["name"],
                                    "similarity": sa[1],
                                    "description": sa[0].get("description", "")
                                }
                                for sa in similar_agents
                            ]
                        })
            
            # Check for tools
            if "required_tools" in analysis:
                for tool in analysis["required_tools"]:
                    tool_name = tool.get("name", "")
                    tool_purpose = tool.get("purpose", "")
                    
                    # Search for similar tools in the library
                    search_query = f"tool that can {tool_purpose}"
                    similar_tools = await self.library.semantic_search(
                        search_query,
                        record_type="TOOL",
                        limit=3,
                        threshold=0.6
                    )
                    
                    if similar_tools:
                        existing_components.append({
                            "type": "TOOL",
                            "name": tool_name,
                            "purpose": tool_purpose,
                            "similar_existing_components": [
                                {
                                    "id": st[0]["id"],
                                    "name": st[0]["name"],
                                    "similarity": st[1],
                                    "description": st[0].get("description", "")
                                }
                                for st in similar_tools
                            ]
                        })
            
            # Add the existing components to the analysis
            analysis["existing_components"] = existing_components
            
            return StringToolOutput(json.dumps(analysis, indent=2))
        
        except json.JSONDecodeError:
            # If parsing fails, return a structured error
            error_response = {
                "status": "error",
                "message": "Failed to parse analysis response as JSON",
                "raw_response": analysis_response
            }
            return StringToolOutput(json.dumps(error_response, indent=2))


class DesignSolutionTool(Tool[DesignSolutionInput, None, StringToolOutput]):
    """Tool for designing a solution based on the requirements analysis."""
    
    name = "DesignSolutionTool"
    description = "Design a complete solution including component architecture and interactions"
    input_schema = DesignSolutionInput
    
    def __init__(self, llm_service: LLMService, smart_library: SmartLibrary):
        super().__init__()
        self.llm = llm_service
        self.library = smart_library
    
    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "architect", "design"],
            creator=self,
        )
    
    async def _run(
        self, 
        input: DesignSolutionInput, 
        options: Optional[Dict[str, Any]] = None, 
        context: Optional[RunContext] = None
    ) -> StringToolOutput:
        """
        Design a complete solution based on requirements analysis.
        
        Args:
            input: Analysis from AnalyzeRequirementsTool and design parameters
            
        Returns:
            Comprehensive solution design
        """
        # Extract key information from the analysis
        task_objective = input.requirements_analysis.get("task_objective", "")
        domains = input.requirements_analysis.get("domains", [])
        required_agents = input.requirements_analysis.get("required_agents", [])
        required_tools = input.requirements_analysis.get("required_tools", [])
        existing_components = input.requirements_analysis.get("existing_components", [])
        
        # Determine optimization focus
        optimization_notes = ""
        if input.optimization_focus:
            optimization_notes = f"\nOPTIMIZATION FOCUS: Optimize the solution for {input.optimization_focus}."
        
        # Prompt the LLM to design the solution
        design_prompt = f"""
        Design a complete {input.solution_type} solution for the following task:

        TASK OBJECTIVE:
        {task_objective}

        DOMAINS:
        {json.dumps(domains, indent=2) if isinstance(domains, list) else domains}

        REQUIRED AGENTS:
        {json.dumps(required_agents, indent=2)}

        REQUIRED TOOLS:
        {json.dumps(required_tools, indent=2)}

        EXISTING COMPONENTS:
        {json.dumps(existing_components, indent=2)}{optimization_notes}

        Please create a comprehensive solution design with these sections:
        
        1. "overview": A high-level overview of the solution approach
        2. "architecture": The architectural pattern and component organization
        3. "components": A list of all components (both new and existing) with:
           - "name": Component name
           - "type": AGENT or TOOL
           - "purpose": What the component does
           - "action": "reuse" (use existing), "evolve" (modify existing), or "create" (new)
           - "existing_component_id": ID of existing component to reuse or evolve (if applicable)
           - "interfaces": Input/output interfaces
        4. "workflow": The sequence of operations and data flow between components
        5. "implementation_strategy": How to implement this solution (using SystemAgent)
        
        Format your response as a structured JSON object with these sections.
        """
        
        # Generate solution design
        design_response = await self.llm.generate(design_prompt)
        
        try:
            # Parse the JSON response
            solution_design = json.loads(design_response)
            
            # Return the solution design
            return StringToolOutput(json.dumps(solution_design, indent=2))
        
        except json.JSONDecodeError:
            # If parsing fails, return a structured error
            error_response = {
                "status": "error",
                "message": "Failed to parse solution design response as JSON",
                "raw_response": design_response
            }
            return StringToolOutput(json.dumps(error_response, indent=2))


class ComponentSpecificationTool(Tool[ComponentSpecInput, None, StringToolOutput]):
    """Tool for generating detailed component specifications."""
    
    name = "ComponentSpecificationTool"
    description = "Generate detailed specifications for components in the solution design"
    input_schema = ComponentSpecInput
    
    def __init__(self, llm_service: LLMService, smart_library: SmartLibrary):
        super().__init__()
        self.llm = llm_service
        self.library = smart_library
    
    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "architect", "specify"],
            creator=self,
        )
    
    async def _run(
        self, 
        input: ComponentSpecInput, 
        options: Optional[Dict[str, Any]] = None, 
        context: Optional[RunContext] = None
    ) -> StringToolOutput:
        """
        Generate detailed specifications for components.
        
        Args:
            input: Solution design and component to specify
            
        Returns:
            Detailed component specifications
        """
        components = input.solution_design.get("components", [])
        specifications = {}
        
        # Filter for a specific component if requested
        if input.component_name:
            components = [c for c in components if c.get("name") == input.component_name]
            if not components:
                return StringToolOutput(json.dumps({
                    "status": "error",
                    "message": f"Component '{input.component_name}' not found in the solution design"
                }, indent=2))
        
        # Process each component (or just the specified one)
        for component in components:
            component_name = component.get("name")
            component_type = component.get("type")
            component_purpose = component.get("purpose")
            component_action = component.get("action", "create")
            existing_id = component.get("existing_component_id")
            
            # Get existing component details if we're reusing or evolving
            existing_component = None
            if existing_id and component_action in ["reuse", "evolve"]:
                existing_component = await self.library.find_record_by_id(existing_id)
            
            # Generate the specification for this component
            spec = await self._generate_component_spec(
                component, 
                existing_component, 
                input.detail_level
            )
            
            specifications[component_name] = spec
        
        result = {
            "status": "success",
            "component_count": len(specifications),
            "specifications": specifications
        }
        
        return StringToolOutput(json.dumps(result, indent=2))
    
    async def _generate_component_spec(
        self, 
        component: Dict[str, Any], 
        existing_component: Optional[Dict[str, Any]], 
        detail_level: str
    ) -> Dict[str, Any]:
        """Generate a detailed specification for a component."""
        component_name = component.get("name")
        component_type = component.get("type")
        component_purpose = component.get("purpose")
        component_action = component.get("action", "create")
        
        # Build the prompt based on the component action
        if component_action == "reuse":
            prompt = f"""
            Create a specification for reusing this existing component:
            
            COMPONENT:
            Name: {component_name}
            Type: {component_type}
            Purpose: {component_purpose}
            
            EXISTING COMPONENT:
            {json.dumps(existing_component, indent=2) if existing_component else "Not available"}
            
            DETAIL LEVEL: {detail_level}
            
            Generate a specification for reusing this component that includes:
            1. A detailed description
            2. Input/output interfaces
            3. Required configurations
            4. Usage instructions
            5. Integration guidelines
            
            Format as JSON.
            """
        elif component_action == "evolve":
            prompt = f"""
            Create a specification for evolving this existing component:
            
            COMPONENT:
            Name: {component_name}
            Type: {component_type}
            Purpose: {component_purpose}
            
            EXISTING COMPONENT:
            {json.dumps(existing_component, indent=2) if existing_component else "Not available"}
            
            DETAIL LEVEL: {detail_level}
            
            Generate a specification for evolving this component that includes:
            1. A detailed description
            2. Required changes/additions
            3. Input/output interfaces
            4. Implementation guidelines
            5. Evolution strategy
            
            Format as JSON.
            """
        else:  # create
            prompt = f"""
            Create a specification for building this new component:
            
            COMPONENT:
            Name: {component_name}
            Type: {component_type}
            Purpose: {component_purpose}
            
            DETAIL LEVEL: {detail_level}
            
            Generate a detailed specification for creating this component that includes:
            1. A detailed description
            2. Required capabilities
            3. Input/output interfaces
            4. Implementation requirements
            5. Required tools or dependencies
            6. Recommended framework
            
            Format as JSON.
            """
        
        # Generate the specification
        spec_response = await self.llm.generate(prompt)
        
        try:
            # Parse the JSON response
            specification = json.loads(spec_response)
            return specification
        except json.JSONDecodeError:
            # If parsing fails, return a structured approximation
            return {
                "name": component_name,
                "type": component_type,
                "purpose": component_purpose,
                "action": component_action,
                "description": "Failed to parse specification as JSON",
                "raw_specification": spec_response
            }


class GenerateWorkflowTool(Tool[GenerateWorkflowInput, None, StringToolOutput]):
    """Tool for generating a YAML workflow from the solution design."""
    
    name = "GenerateWorkflowTool"
    description = "Generate a YAML workflow from the solution design that can be executed by the System Agent"
    input_schema = GenerateWorkflowInput
    
    def __init__(self, workflow_generator: WorkflowGenerator):
        super().__init__()
        self.workflow_generator = workflow_generator
    
    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "architect", "workflow"],
            creator=self,
        )
    
    async def _run(
        self, 
        input: GenerateWorkflowInput, 
        options: Optional[Dict[str, Any]] = None, 
        context: Optional[RunContext] = None
    ) -> StringToolOutput:
        """
        Generate a YAML workflow from the solution design.
        
        Args:
            input: Solution design and optional parameters
            
        Returns:
            Generated YAML workflow
        """
        workflow_name = input.workflow_name or "generated_workflow"
        solution_overview = input.solution_design.get("overview", "")
        
        # Extract workflow sequence from the solution design
        workflow_sequence = input.solution_design.get("workflow", [])
        components = input.solution_design.get("components", [])
        
        # Generate the YAML workflow
        yaml_prompt = f"""
        Convert this solution design to a YAML workflow representation:

        WORKFLOW NAME: {workflow_name}
        
        OVERVIEW:
        {solution_overview}

        COMPONENTS:
        {json.dumps(components, indent=2)}

        WORKFLOW SEQUENCE:
        {json.dumps(workflow_sequence, indent=2)}

        CREATE a YAML workflow that follows this structure:
        ```yaml
        scenario_name: "{workflow_name}"
        domain: "[domain from the solution design]"
        description: "[brief description from the solution design]"

        steps:
          - type: "[CREATE/EXECUTE/DEFINE]"
            item_type: "[AGENT/TOOL]"
            name: "[component name]"
            [additional parameters as needed]
            
          # Add steps for each component and workflow sequence item
          # Include component creation before component execution
        ```

        The workflow should:
        1. Create any new components needed
        2. Execute the components in the correct sequence
        3. Pass data between components as needed
        4. Include any required DEFINE steps for parameters
        5. {f"Include explanatory comments" if input.include_comments else "Minimize comments"}

        Return only the YAML content.
        """
        
        # Generate the YAML workflow content
        yaml_workflow = await self.workflow_generator.llm.generate(yaml_prompt)
        
        # Extract YAML content if wrapped in code blocks
        if "```yaml" in yaml_workflow and "```" in yaml_workflow:
            yaml_content = yaml_workflow.split("```yaml")[1].split("```")[0].strip()
        elif "```" in yaml_workflow:
            yaml_content = yaml_workflow.split("```")[1].strip()
        else:
            yaml_content = yaml_workflow.strip()
        
        # Format the result
        result = {
            "status": "success",
            "workflow_name": workflow_name,
            "yaml_workflow": yaml_content,
            "execution_guidance": [
                "This workflow can be executed using the System Agent's workflow processor",
                "Review and adjust the workflow before execution if needed",
                "Make sure all required components exist in the library",
                "You can execute this workflow using the System Agent's process_workflow method"
            ]
        }
        
        return StringToolOutput(json.dumps(result, indent=2))


# Main function to create the Architect-Zero agent
async def create_architect_zero(
    llm_service: Optional[LLMService] = None,
    smart_library: Optional[SmartLibrary] = None,
    agent_bus: Optional[SmartAgentBus] = None,
    container: Optional[DependencyContainer] = None
) -> ReActAgent:
    """
    Create and configure the Architect-Zero agent.
    
    Args:
        llm_service: LLM service for text generation
        smart_library: Smart library for component management
        agent_bus: Agent bus for component communication
        container: Optional dependency container for managing component dependencies
        
    Returns:
        Configured Architect-Zero agent
    """
    return await ArchitectZeroAgentInitializer.create_agent(
        llm_service=llm_service,
        smart_library=smart_library,
        agent_bus=agent_bus,
        container=container
    )