import logging
import json
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
from evolving_agents.agent_bus.simple_agent_bus import SimpleAgentBus
from evolving_agents.workflow.workflow_generator import WorkflowGenerator
from evolving_agents.core.system_agent import SystemAgentFactory

logger = logging.getLogger(__name__)

class ArchitectZeroAgentInitializer:
    """
    Architect-Zero agent that analyzes task requirements and designs complete 
    agent-based solutions.
    
    This agent can:
    1. Analyze task requirements in natural language
    2. Design a workflow of agents and tools to complete the task
    3. Generate agent/tool entries for the smart library
    4. Create YAML workflows to execute the solution
    5. Leverage existing components when possible through the smart library
    6. Create new agents and tools from scratch when necessary
    """
    
    
    @staticmethod
    async def create_agent(
        llm_service: LLMService,
        smart_library: SmartLibrary,
        agent_bus: SimpleAgentBus,
        system_agent_factory: Optional[callable] = None,
        system_agent: Optional[ReActAgent] = None,
        tools: Optional[List[Tool]] = None
    ) -> ReActAgent:
        """
        Create and configure the Architect-Zero agent.
        
        Args:
            llm_service: LLM service for text generation
            smart_library: Smart library for component management
            agent_bus: Agent bus for component communication
            system_agent_factory: Optional factory function for creating the system agent
            system_agent: Optional existing system agent instance (takes precedence over factory)
            tools: Optional additional tools to provide to the agent
            
        Returns:
            Configured Architect-Zero agent
        """
        # Get the chat model from LLM service
        chat_model = llm_service.chat_model
        
        # Use provided system agent or create one using factory
        if system_agent is None:
            if system_agent_factory:
                system_agent = await system_agent_factory(
                    llm_service=llm_service,
                    smart_library=smart_library,
                    agent_bus=agent_bus
                )
            else:
                # Create using the default factory
                system_agent = await SystemAgentFactory.create_agent(
                    llm_service=llm_service,
                    smart_library=smart_library,
                    agent_bus=agent_bus
                )
        
        # Create workflow generator
        workflow_generator = WorkflowGenerator(llm_service, smart_library)
        workflow_generator.set_agent(system_agent)
        
        # Create tools for the architect agent
        architect_tools = [
            AnalyzeRequirementsTool(llm_service, smart_library),
            DesignWorkflowTool(llm_service, workflow_generator, smart_library),
            GenerateLibraryEntriesHool(llm_service, smart_library),
            CreateWorkflowTool(workflow_generator),
            ExecuteWorkflowTool(system_agent)
        ]
        
        # Add any additional tools provided
        if tools:
            architect_tools.extend(tools)
        
        # Create agent metadata
        meta = AgentMeta(
            name="Architect-Zero",
            description=(
                "I am Architect-Zero, responsible for designing and creating agent-based solutions "
                "from natural language task requirements. I can analyze requirements, design workflows, "
                "create library entries, generate YAML workflows, and execute solutions using the "
                "evolving agents toolkit."
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
        
        return agent


# Custom tools for Architect-Zero

# Input schemas for the tools
class AnalyzeRequirementsInput(BaseModel):
    """Input schema for the AnalyzeRequirementsTool."""
    task_requirements: str = Field(description="Task requirements in natural language")

class DesignWorkflowInput(BaseModel):
    """Input schema for the DesignWorkflowTool."""
    requirements_analysis: Dict[str, Any] = Field(description="Analysis from AnalyzeRequirementsTool")

class GenerateLibraryEntriesInput(BaseModel):
    """Input schema for the GenerateLibraryEntriesHool."""
    workflow_design: Dict[str, Any] = Field(description="Workflow design from DesignWorkflowTool")

class CreateWorkflowInput(BaseModel):
    """Input schema for the CreateWorkflowTool."""
    workflow_design: Dict[str, Any] = Field(description="Workflow design from DesignWorkflowTool")
    library_entries: Dict[str, Any] = Field(description="Library entries from GenerateLibraryEntriesHool")

class ExecuteWorkflowInput(BaseModel):
    """Input schema for the ExecuteWorkflowTool."""
    yaml_workflow: str = Field(description="YAML workflow to execute")
    execution_params: Dict[str, Any] = Field(
        default={}, 
        description="Optional parameters for workflow execution"
    )


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
        # Extract domain information from requirements
        domain = await self._extract_domain(input.task_requirements)
        
        # Leverage SmartLibrary's capability analysis if available
        capabilities = []
        if hasattr(self.library, "_extract_capabilities_with_llm"):
            capabilities = await self.library._extract_capabilities_with_llm(
                input.task_requirements, 
                domain
            )
        
        # Prompt the LLM to analyze the requirements
        analysis_prompt = f"""
        Analyze the following task requirements and identify the necessary components:

        TASK REQUIREMENTS:
        {input.task_requirements}

        {f"IDENTIFIED CAPABILITIES: {', '.join(capabilities)}" if capabilities else ""}

        Please provide:
        1. A clear summary of the task's objective
        2. The domain(s) this task falls under (use: {domain})
        3. A list of required agents with their purpose and required capabilities
        4. A list of required tools with their purpose and required capabilities
        5. The key capabilities needed by these components
        6. Any constraints or special considerations

        Format your response as a structured JSON object with these sections.
        """
        
        # Generate analysis
        analysis_response = await self.llm.generate(analysis_prompt)
        
        try:
            # Parse the JSON response
            analysis = self._extract_json(analysis_response)
            
            # Add extracted capabilities if not already included
            if capabilities and "required_capabilities" not in analysis:
                analysis["required_capabilities"] = capabilities
            
            # Check for existing components in the library that match requirements
            existing_components = []
            
            # Check for components matching extracted capabilities
            if capabilities:
                for capability in capabilities:
                    component = await self.library.find_component_by_capability(
                        capability_id=capability,
                        domain=domain
                    )
                    
                    if component:
                        existing_components.append({
                            "type": component["record_type"],
                            "name": component["name"],
                            "id": component["id"],
                            "capability": capability,
                            "description": component["description"],
                            "match_type": "capability_exact"
                        })
            
            # Also check for components matching agent and tool purposes
            for agent in analysis.get("required_agents", []):
                agent_name = agent.get("name", "")
                agent_purpose = agent.get("purpose", "")
                
                # Search for similar agents in the library
                similar_agents = await self.library.semantic_search(
                    f"agent that can {agent_purpose}",
                    record_type="AGENT",
                    domain=domain,
                    limit=3
                )
                
                if similar_agents:
                    for sa, score in similar_agents:
                        existing_components.append({
                            "type": "AGENT",
                            "name": sa["name"],
                            "id": sa["id"],
                            "purpose": agent_purpose,
                            "proposed_name": agent_name,
                            "similarity": score,
                            "match_type": "semantic"
                        })
            
            # Similarly for tools
            for tool in analysis.get("required_tools", []):
                tool_name = tool.get("name", "")
                tool_purpose = tool.get("purpose", "")
                
                # Search for similar tools in the library
                similar_tools = await self.library.semantic_search(
                    f"tool that can {tool_purpose}",
                    record_type="TOOL",
                    domain=domain,
                    limit=3
                )
                
                if similar_tools:
                    for st, score in similar_tools:
                        existing_components.append({
                            "type": "TOOL",
                            "name": st["name"],
                            "id": st["id"],
                            "purpose": tool_purpose,
                            "proposed_name": tool_name,
                            "similarity": score,
                            "match_type": "semantic"
                        })
            
            # Add the existing components to the analysis
            analysis["existing_components"] = existing_components
            
            # Add capability coverage analysis
            if capabilities:
                coverage = {}
                for capability in capabilities:
                    matching_components = [c for c in existing_components if c.get("capability") == capability]
                    coverage[capability] = {
                        "covered": len(matching_components) > 0,
                        "components": matching_components
                    }
                analysis["capability_coverage"] = coverage
            
            return StringToolOutput(json.dumps(analysis, indent=2))
        
        except Exception as e:
            # If parsing fails, return a structured error
            import traceback
            error_response = {
                "status": "error",
                "message": f"Failed to analyze requirements: {str(e)}",
                "traceback": traceback.format_exc(),
                "raw_response": analysis_response
            }
            return StringToolOutput(json.dumps(error_response, indent=2))
    
    async def _extract_domain(self, task_requirements: str) -> str:
        """Extract the domain from task requirements."""
        if not self.llm:
            # Default to "general" if no LLM available
            return "general"
            
        domain_prompt = f"""
        From the following task requirements, identify the primary domain this task falls under.
        Choose the most specific applicable domain from this list:
        - general
        - document_processing
        - finance
        - healthcare
        - customer_service
        - legal
        - education
        - marketing
        - software_development
        - data_analysis
        
        TASK REQUIREMENTS:
        {task_requirements}
        
        Return only the domain name, nothing else.
        """
        
        try:
            domain_response = await self.llm.generate(domain_prompt)
            domain = domain_response.strip().lower()
            
            # Validate that it's one of the expected domains
            valid_domains = [
                "general", "document_processing", "finance", "healthcare", 
                "customer_service", "legal", "education", "marketing",
                "software_development", "data_analysis"
            ]
            
            if domain in valid_domains:
                return domain
            else:
                return "general"
        except Exception as e:
            logger.warning(f"Error extracting domain: {str(e)}")
            return "general"
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract and parse JSON from text that might contain other content."""
        # Try to parse the entire text as JSON first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON blocks
            json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
            match = re.search(json_pattern, text)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Try to find anything that looks like a JSON object
            bracket_pattern = r'(\{[\s\S]*\})'
            match = re.search(bracket_pattern, text)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # If all else fails, return the raw text as a structured error
            return {
                "error": "Failed to parse JSON",
                "raw_text": text
            }


class DesignWorkflowTool(Tool[DesignWorkflowInput, None, StringToolOutput]):
    """Tool for designing a workflow based on the requirements analysis."""
    
    name = "DesignWorkflowTool"
    description = "Design a workflow of agents and tools to fulfill the task requirements, leveraging capability-based component selection"
    input_schema = DesignWorkflowInput
    
    def __init__(
        self, 
        llm_service: LLMService, 
        workflow_generator: WorkflowGenerator,
        smart_library: SmartLibrary
    ):
        super().__init__()
        self.llm = llm_service
        self.workflow_generator = workflow_generator
        self.library = smart_library
    
    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "architect", "design"],
            creator=self,
        )
    
    async def _run(
        self, 
        input: DesignWorkflowInput, 
        options: Optional[Dict[str, Any]] = None, 
        context: Optional[RunContext] = None
    ) -> StringToolOutput:
        """
        Design a workflow based on requirements analysis.
        
        Args:
            input: Analysis from AnalyzeRequirementsTool
            
        Returns:
            Workflow design including sequence, component relations, and data flow
        """
        # Extract task objective and requirements
        task_objective = input.requirements_analysis.get("task_objective", "")
        domains = input.requirements_analysis.get("domains", [])
        required_agents = input.requirements_analysis.get("required_agents", [])
        required_tools = input.requirements_analysis.get("required_tools", [])
        capabilities = input.requirements_analysis.get("required_capabilities", [])
        capability_coverage = input.requirements_analysis.get("capability_coverage", {})
        existing_components = input.requirements_analysis.get("existing_components", [])
        
        # Determine primary domain
        domain = domains[0] if isinstance(domains, list) and domains else domains
        if isinstance(domain, str) and not domain:
            domain = "general"
        
        # Find appropriate components for each capability
        capability_mapping = {}
        if capabilities and hasattr(self.library, "find_components_for_workflow"):
            workflow_components = await self.library.find_components_for_workflow(
                workflow_description=task_objective,
                required_capabilities=capabilities,
                domain=domain,
                use_llm=True
            )
            
            # Format for inclusion in the prompt
            if workflow_components:
                capability_mapping_text = []
                for cap_id, components in workflow_components.items():
                    if components:
                        component_names = [c["name"] for c in components]
                        capability_mapping_text.append(f"- {cap_id}: {', '.join(component_names)}")
                
                if capability_mapping_text:
                    capability_mapping = "\nCAPABILITY MAPPING:\n" + "\n".join(capability_mapping_text)
                else:
                    capability_mapping = "\nNo suitable existing components found for required capabilities."
        
        # Prompt the LLM to design the workflow
        design_prompt = f"""
        Design a complete workflow for the following task:

        TASK OBJECTIVE:
        {task_objective}

        DOMAIN:
        {domain}

        REQUIRED CAPABILITIES:
        {json.dumps(capabilities, indent=2) if capabilities else "No specific capabilities identified."}

        REQUIRED AGENTS:
        {json.dumps(required_agents, indent=2) if required_agents else "No specific agents identified."}

        REQUIRED TOOLS:
        {json.dumps(required_tools, indent=2) if required_tools else "No specific tools identified."}

        EXISTING COMPONENTS IN LIBRARY:
        {json.dumps(existing_components, indent=2) if existing_components else "No existing components found."}

        {capability_mapping if capability_mapping else ""}

        Please create a workflow design that:
        1. Specifies the sequence of operations
        2. Shows how agents interact with each other and with tools
        3. Defines the data flow between components
        4. Leverages existing components where possible (reuse)
        5. Identifies components that need to be evolved from existing ones
        6. Specifies components that need to be created from scratch
        7. Maps each component to the capabilities it should provide

        For each component in the workflow, specify:
        - Whether to reuse, evolve, or create from scratch
        - Which capabilities it will provide
        - What inputs it requires
        - What outputs it produces
        - How it should be implemented (high-level)

        Format your response as a structured JSON object.
        """
        
        # Generate workflow design
        design_response = await self.llm.generate(design_prompt)
        
        try:
            # Parse the JSON response
            workflow_design = self._extract_json(design_response)
            
            # Add required capabilities if not included
            if capabilities and "required_capabilities" not in workflow_design:
                workflow_design["required_capabilities"] = capabilities
            
            # Add domain information if not included
            if "domain" not in workflow_design:
                workflow_design["domain"] = domain
            
            # Generate a pseudo-YAML representation for visualization
            yaml_representation = await self._generate_yaml_representation(workflow_design)
            workflow_design["yaml_preview"] = yaml_representation
            
            return StringToolOutput(json.dumps(workflow_design, indent=2))
        
        except Exception as e:
            # If parsing fails, return a structured error
            import traceback
            error_response = {
                "status": "error",
                "message": f"Failed to design workflow: {str(e)}",
                "traceback": traceback.format_exc(),
                "raw_response": design_response
            }
            return StringToolOutput(json.dumps(error_response, indent=2))
    
    async def _generate_yaml_representation(self, workflow_design: Dict[str, Any]) -> str:
        """Generate a YAML representation of the workflow design for visualization."""
        sequence = workflow_design.get("sequence", [])
        components = workflow_design.get("components", [])
        data_flow = workflow_design.get("data_flow", [])
        domain = workflow_design.get("domain", "general")
        
        # Get the exact list of available components in the library
        library_components = []
        for record in self.library.records:
            library_components.append({
                "name": record["name"],
                "type": record["record_type"],
                "domain": record.get("domain", "general"),
                "id": record["id"]
            })
        
        yaml_prompt = f"""
        Convert this workflow design to a YAML workflow representation.

        SEQUENCE:
        {json.dumps(sequence, indent=2) if sequence else "No sequence defined."}

        COMPONENTS:
        {json.dumps(components, indent=2) if components else "No components defined."}

        DATA FLOW:
        {json.dumps(data_flow, indent=2) if data_flow else "No data flow defined."}

        DOMAIN:
        {domain}

        AVAILABLE LIBRARY COMPONENTS (ONLY USE THESE EXACT COMPONENTS):
        {json.dumps(library_components, indent=2)}

        The YAML should follow this structure:
        ```yaml
        scenario_name: [task name]
        domain: {domain}
        description: [description]

        steps:
        - type: "CREATE" or "EXECUTE" or "DEFINE"
            item_type: "AGENT" or "TOOL"
            name: [component name]
            [additional parameters as needed]
        ```

        CRITICAL INSTRUCTIONS:
        1. ONLY use component names from the AVAILABLE LIBRARY COMPONENTS list above
        2. Ensure item_type exactly matches the component's "type" field 
        3. All components must be CREATEd before they are EXECUTEd
        4. For user inputs, use the pipe character (|) for multi-line strings
        5. Do NOT invent or rename components - use ONLY the exact names from the library
        6. If a required component doesn't exist, use the closest match or leave it out

        Return only the YAML content.
        """
        
        yaml_response = await self.llm.generate(yaml_prompt)
        
        # Clean up the response to extract just the YAML
        if "```yaml" in yaml_response and "```" in yaml_response:
            yaml_content = yaml_response.split("```yaml")[1].split("```")[0].strip()
            return yaml_content
        elif "```" in yaml_response:
            yaml_content = yaml_response.split("```")[1].split("```")[0].strip()
            return yaml_content
        
        return yaml_response
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract and parse JSON from text that might contain other content."""
        # Try to parse the entire text as JSON first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON blocks
            json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
            match = re.search(json_pattern, text)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Try to find anything that looks like a JSON object
            bracket_pattern = r'(\{[\s\S]*\})'
            match = re.search(bracket_pattern, text)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # If all else fails, return the raw text as a structured error
            return {
                "error": "Failed to parse JSON",
                "raw_text": text
            }


class GenerateLibraryEntriesHool(Tool[GenerateLibraryEntriesInput, None, StringToolOutput]):
    """Tool for generating Smart Library entries for the components needed."""
    
    name = "GenerateLibraryEntriesHool"
    description = "Generate entries for the Smart Library based on the workflow design"
    input_schema = GenerateLibraryEntriesInput
    
    def __init__(self, llm_service: LLMService, smart_library: SmartLibrary):
        super().__init__()
        self.llm = llm_service
        self.library = smart_library
    
    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "architect", "generate"],
            creator=self,
        )
    
    async def _run(
        self, 
        input: GenerateLibraryEntriesInput, 
        options: Optional[Dict[str, Any]] = None, 
        context: Optional[RunContext] = None
    ) -> StringToolOutput:
        """
        Generate library entries for the components in the workflow.
        
        Args:
            input: Workflow design from DesignWorkflowTool
            
        Returns:
            Generated library entries for new and evolved components
        """
        components = input.workflow_design.get("components", [])
        reuse_components = [c for c in components if c.get("action") == "reuse"]
        evolve_components = [c for c in components if c.get("action") == "evolve"]
        create_components = [c for c in components if c.get("action") == "create"]
        
        library_entries = {
            "reuse": [],
            "evolve": [],
            "create": []
        }
        
        # Process components to reuse
        for component in reuse_components:
            component_id = component.get("existing_component_id")
            if component_id:
                record = await self.library.find_record_by_id(component_id)
                if record:
                    library_entries["reuse"].append({
                        "id": record["id"],
                        "name": record["name"],
                        "type": record["record_type"],
                        "role_in_workflow": component.get("role", "")
                    })
        
        # Process components to evolve
        for component in evolve_components:
            parent_id = component.get("parent_component_id")
            if parent_id:
                parent_record = await self.library.find_record_by_id(parent_id)
                if parent_record:
                    # Generate evolved code for this component
                    evolved_entry = await self._generate_evolved_component(
                        parent_record,
                        component
                    )
                    library_entries["evolve"].append(evolved_entry)
        
        # Process components to create
        for component in create_components:
            # Generate new code for this component
            new_entry = await self._generate_new_component(component)
            library_entries["create"].append(new_entry)
        
        result = {
            "status": "success",
            "library_entries": library_entries
        }
        
        return StringToolOutput(json.dumps(result, indent=2))
    
    async def _generate_evolved_component(
        self, 
        parent_record: Dict[str, Any],
        component_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate code for an evolved component."""
        component_name = component_spec.get("name", "")
        component_type = component_spec.get("type", "AGENT")
        component_purpose = component_spec.get("purpose", "")
        evolution_changes = component_spec.get("evolution_changes", "")
        
        # Prompt the LLM to evolve the component
        evolution_prompt = f"""
        Evolve this existing component:

        PARENT COMPONENT:
        Name: {parent_record['name']}
        Type: {parent_record['record_type']}
        Description: {parent_record['description']}
        
        ORIGINAL CODE:
        ```
        {parent_record['code_snippet']}
        ```

        NEW REQUIREMENTS:
        Component Name: {component_name}
        Component Type: {component_type}
        Purpose: {component_purpose}
        Required Changes: {evolution_changes}

        Please generate an evolved version of this component that fulfills the new requirements.
        Return the updated component as a JSON object with these fields:
        - name: The component name
        - record_type: "AGENT" or "TOOL"
        - domain: The component domain
        - description: Detailed description of the component
        - code_snippet: The complete code for the component
        - version: Incremented version (e.g., if original is 1.0.0, use 1.0.1)
        - changes: Summary of changes made
        """
        
        # Generate evolved component
        evolution_response = await self.llm.generate(evolution_prompt)
        
        try:
            # Parse the JSON response
            evolved_component = json.loads(evolution_response)
            evolved_component["parent_id"] = parent_record["id"]
            return evolved_component
        
        except json.JSONDecodeError:
            # If parsing fails, create a structured representation
            return {
                "name": component_name,
                "record_type": component_type,
                "domain": parent_record.get("domain", "general"),
                "description": component_purpose,
                "code_snippet": evolution_response,
                "version": self._increment_version(parent_record.get("version", "1.0.0")),
                "parent_id": parent_record["id"],
                "changes": evolution_changes,
                "error": "Failed to parse as JSON, using raw response"
            }
    
    async def _generate_new_component(self, component_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code for a new component from scratch."""
        component_name = component_spec.get("name", "")
        component_type = component_spec.get("type", "AGENT")
        component_purpose = component_spec.get("purpose", "")
        capabilities = component_spec.get("capabilities", [])
        
        # Determine template type based on component type
        template_type = "agent" if component_type == "AGENT" else "tool"
        
        # Prompt the LLM to create the component
        creation_prompt = f"""
        Create a new {component_type.lower()} from scratch:

        COMPONENT DETAILS:
        Name: {component_name}
        Type: {component_type}
        Purpose: {component_purpose}
        Required Capabilities: {', '.join(capabilities) if isinstance(capabilities, list) else str(capabilities)}

        Generate a complete implementation for this component using the BeeAI framework.
        
        If creating an AGENT, the code should define a class that:
        - Provides a create_agent static method that returns a ReActAgent
        - Properly configures the agent with tools and metadata
        
        If creating a TOOL, the code should define a class that:
        - Extends the Tool class from BeeAI
        - Implements the _run method
        - Defines appropriate input schema

        Return the component as a JSON object with these fields:
        - name: The component name
        - record_type: "AGENT" or "TOOL"
        - domain: The component domain
        - description: Detailed description of the component
        - code_snippet: The complete code for the component
        - version: Initial version (1.0.0)
        """
        
        # Generate new component
        creation_response = await self.llm.generate(creation_prompt)
        
        try:
            # Parse the JSON response
            new_component = json.loads(creation_response)
            return new_component
        
        except json.JSONDecodeError:
            # If parsing fails, create a structured representation
            domain = "general"
            if isinstance(capabilities, list):
                for capability in capabilities:
                    if "finance" in str(capability).lower():
                        domain = "finance"
                    elif "medical" in str(capability).lower() or "health" in str(capability).lower():
                        domain = "medical"
                    elif "document" in str(capability).lower():
                        domain = "document_processing"
            
            return {
                "name": component_name,
                "record_type": component_type,
                "domain": domain,
                "description": component_purpose,
                "code_snippet": creation_response,
                "version": "1.0.0",
                "error": "Failed to parse as JSON, using raw response"
            }
    
    def _increment_version(self, version: str) -> str:
        """Increment the patch version number."""
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


# evolving_agents/agents/architect_zero.py (partial update - CreateWorkflowTool class)

class CreateWorkflowTool(Tool[CreateWorkflowInput, None, StringToolOutput]):
    """Tool for creating a YAML workflow from the workflow design."""
    
    name = "CreateWorkflowTool"
    description = "Create a complete YAML workflow based on the workflow design and library entries, leveraging capability-based component selection"
    input_schema = CreateWorkflowInput
    
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
        input: CreateWorkflowInput, 
        options: Optional[Dict[str, Any]] = None, 
        context: Optional[RunContext] = None
    ) -> StringToolOutput:
        """
        Create a YAML workflow from the workflow design and library entries.
        
        Args:
            input: Dict containing workflow_design and library_entries
            
        Returns:
            Complete YAML workflow ready for execution
        """
        # Check if smart_library is available through the workflow_generator
        library = getattr(self.workflow_generator, "smart_library", None)
        
        # Extract required capabilities from the workflow design
        required_capabilities = []
        workflow_domain = input.workflow_design.get("domain", "general")
        
        # Extract from components section
        for component in input.workflow_design.get("components", []):
            # Extract capabilities from component definitions
            for cap in component.get("capabilities", []):
                if isinstance(cap, str):
                    required_capabilities.append(cap)
                elif isinstance(cap, dict) and "id" in cap:
                    required_capabilities.append(cap["id"])
        
        # Also extract from workflow sequence
        for step in input.workflow_design.get("sequence", []):
            for req in step.get("requirements", []):
                if isinstance(req, str) and "capability" in req.lower():
                    # Extract the capability name
                    cap_match = re.search(r"capability\s*['\"]?([^'\"]+)['\"]?", req, re.IGNORECASE)
                    if cap_match:
                        required_capabilities.append(cap_match.group(1))
        
        # Find existing components for these capabilities if library is available
        component_mapping = {}
        if library and required_capabilities:
            for cap_id in set(required_capabilities):  # Deduplicate capabilities
                component = await library.find_component_by_capability(
                    capability_id=cap_id,
                    domain=workflow_domain
                )
                if component:
                    component_mapping[cap_id] = component
        
        # Generate the workflow with capability info
        try:
            yaml_workflow = await self.workflow_generator.generate_workflow_from_design_with_capabilities(
                input.workflow_design, 
                input.library_entries,
                component_mapping,
                workflow_domain
            )
            
            # Analyze the workflow for potential issues
            workflow_analysis = None
            if library and hasattr(library, "analyze_workflow_component_compatibility") and component_mapping:
                workflow_analysis = await library.analyze_workflow_component_compatibility(
                    workflow_description=input.workflow_design.get("description", ""),
                    selected_components=component_mapping
                )
            
            result = {
                "status": "success",
                "yaml_workflow": yaml_workflow,
                "message": "YAML workflow created successfully",
                "capability_mappings": {k: v["name"] for k, v in component_mapping.items()} if component_mapping else {},
                "workflow_analysis": workflow_analysis
            }
            
            return StringToolOutput(json.dumps(result, indent=2))
        except Exception as e:
            import traceback
            return StringToolOutput(json.dumps({
                "status": "error",
                "message": f"Error creating workflow: {str(e)}",
                "traceback": traceback.format_exc()
            }, indent=2))


class ExecuteWorkflowTool(Tool[ExecuteWorkflowInput, None, StringToolOutput]):
    """Tool for executing a workflow using the system agent."""
    
    name = "ExecuteWorkflowTool"
    description = "Execute a YAML workflow using the system agent"
    input_schema = ExecuteWorkflowInput
    
    def __init__(self, system_agent: ReActAgent):
        super().__init__()
        self.system_agent = system_agent
    
    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "architect", "execute"],
            creator=self,
        )
    
    async def _run(
        self, 
        input: ExecuteWorkflowInput, 
        options: Optional[Dict[str, Any]] = None, 
        context: Optional[RunContext] = None
    ) -> StringToolOutput:
        """
        Execute a workflow using the system agent.
        
        Args:
            input: Dict containing yaml_workflow and optional execution_params
            
        Returns:
            Workflow execution results
        """
        try:
            # Check for process_workflow method (which is more likely to exist)
            if hasattr(self.system_agent.workflow_processor, "process_workflow"):
                # Execute the workflow
                result = await self.system_agent.workflow_processor.process_workflow(
                    yaml_content=input.yaml_workflow,
                    params=input.execution_params
                )
            # Alternative method names that might exist
            elif hasattr(self.system_agent.workflow_processor, "execute_workflow"):
                result = await self.system_agent.workflow_processor.execute_workflow(
                    input.yaml_workflow,
                    input.execution_params
                )
            else:
                # If neither method exists, return an error
                return StringToolOutput(json.dumps({
                    "status": "error",
                    "message": "Workflow processor does not have a suitable execution method",
                    "available_methods": str(dir(self.system_agent.workflow_processor))
                }, indent=2))
            
            response = {
                "status": "success" if result.get("status") != "error" else "error",
                "execution_results": result,
                "message": "Workflow executed successfully" if result.get("status") != "error" else f"Workflow execution failed: {result.get('message')}"
            }
            
            return StringToolOutput(json.dumps(response, indent=2))
            
        except Exception as e:
            import traceback
            return StringToolOutput(json.dumps({
                "status": "error",
                "message": f"Error executing workflow: {str(e)}",
                "traceback": traceback.format_exc()
            }, indent=2))


# Main function to create the Architect-Zero agent
async def create_architect_zero(
    llm_service: LLMService,
    smart_library: SmartLibrary,
    agent_bus: SimpleAgentBus,
    system_agent_factory: Optional[callable] = None,
    system_agent: Optional[ReActAgent] = None
) -> ReActAgent:
    """
    Create and configure the Architect-Zero agent.
    
    Args:
        llm_service: LLM service for text generation
        smart_library: Smart library for component management
        agent_bus: Agent bus for component communication
        system_agent_factory: Optional factory function for creating the system agent
        system_agent: Optional existing system agent instance
        
    Returns:
        Configured Architect-Zero agent
    """
    return await ArchitectZeroAgentInitializer.create_agent(
        llm_service=llm_service,
        smart_library=smart_library,
        agent_bus=agent_bus,
        system_agent_factory=system_agent_factory,
        system_agent=system_agent
    )