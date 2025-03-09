# Evolving Agents Framework

A production-grade framework for creating, managing, and evolving AI agents with intelligent agent-to-agent communication. The framework enables you to build collaborative agent ecosystems that can semantically understand requirements, evolve based on past experiences, and communicate effectively to solve complex tasks.

![Evolving Agents](evolving-agents-logo.png)

## Key Features

- **Intelligent Agent Evolution**: Reuse, adapt, or create agents based on semantic similarity to existing components
- **Agent-to-Agent Communication**: Enable specialized agents to delegate tasks and collaborate on complex problems
- **Smart Library with Semantic Search**: Find the most relevant tools and agents using OpenAI embeddings
- **Self-improving System**: Agents get better over time through continuous evolution and learning
- **Human-readable YAML Workflows**: Define complex agent collaborations with simple, version-controlled YAML
- **Multi-Framework Support**: Seamlessly integrate agents from different frameworks (BeeAI, OpenAI, etc.)
- **Governance through Firmware**: Enforce domain-specific rules across all agents
- **ACP Integration**: Support for BeeAI's Agent Communication Protocol in pre-alpha

## Quick Start

```bash
# Clone the repository
git clone https://github.com/matiasmolinas/evolving-agents.git
cd evolving-agents

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Setup the agent library
python examples/setup_simplified_agent_library.py

# Run the comprehensive example
python examples/simplified_agent_communication.py
```

## Example: Agent Collaboration and Evolution

The framework lets you create agent ecosystems where specialized agents communicate and evolve:

```python
import asyncio
from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.core.llm_service import LLMService
from evolving_agents.core.system_agent import SystemAgent

async def main():
    # Initialize the framework components
    library = SmartLibrary("agent_library.json")
    llm = LLMService(provider="openai", model="gpt-4o")
    system = SystemAgent(library, llm)
    
    # 1. Ask the System Agent for an invoice analysis agent
    # The System Agent will decide whether to reuse, evolve, or create one
    result = await system.decide_and_act(
        request="I need an agent that can analyze invoices and extract the total amount",
        domain="document_processing",
        record_type="AGENT"
    )
    
    print(f"Decision: {result['action']}")  # 'reuse', 'evolve', or 'create'
    print(f"Agent: {result['record']['name']}")
    
    # 2. Execute the agent with an invoice document
    invoice_text = """
    INVOICE #12345
    Date: 2023-05-15
    Vendor: TechSupplies Inc.
    Total Due: $1,822.80
    """
    
    execution = await system.execute_item(
        result['record']['name'], 
        invoice_text
    )
    
    print("\nInvoice Analysis Result:")
    print(execution["result"])

if __name__ == "__main__":
    asyncio.run(main())
```

## Understanding the Comprehensive Example

The framework includes a comprehensive example (`examples/simplified_agent_communication.py`) that demonstrates four key capabilities. This example shows in detail how the Evolving Agents Framework creates, manages, and evolves AI agents to handle real-world tasks.

### Step 1: Setting Up the Agent Library

Before running the main example, we need to set up an initial library of agents and tools:

```bash
python examples/setup_simplified_agent_library.py
```

This script creates a foundation of BeeAI-compatible agents and tools:

- **DocumentAnalyzer Tool**: A real BeeAI tool that uses an LLM to analyze documents and identify their type
- **AgentCommunicator Tool**: A real BeeAI tool that facilitates communication between agents
- **SpecialistAgent**: A BeeAI agent specialized in detailed document analysis
- **CoordinatorAgent**: A BeeAI agent that orchestrates the document processing workflow

### Step 2: Demonstrating the System Agent's Decision Logic

The first part of the example demonstrates how the System Agent intelligently decides whether to reuse, evolve, or create a new agent based on semantic similarity:

```python
# The System Agent dynamically decides what to do based on your request
invoice_agent_result = await system_agent.decide_and_act(
    request="I need an agent that can analyze invoices and extract the total amount",
    domain="document_processing",
    record_type="AGENT"
)

print(f"System Agent Decision: {invoice_agent_result['action']}")  # 'create', 'evolve', or 'reuse'
```

The System Agent implements a sophisticated decision mechanism:
- If similarity ≥ 0.8: Reuse an existing agent/tool
- If 0.4 ≤ similarity < 0.8: Evolve an existing agent/tool 
- If similarity < 0.4: Create a new agent/tool

In the example, when we ask for an invoice analysis agent, it creates a new one. When we ask for a medical record analyzer, it evolves the existing SpecialistAgent (since it has a similarity score of around 0.46).

### Step 3: Agent-to-Agent Communication with Workflows

The second part demonstrates how agents communicate with each other through workflows defined in YAML:

```yaml
# A workflow where agents communicate with each other
workflow_yaml = """
scenario_name: "Document Processing with Agent Communication"
domain: "document_processing"
description: "Process documents by delegating specialized tasks to expert agents"

steps:
  # Create the tools from the library
  - type: "CREATE"
    item_type: "TOOL"
    name: "DocumentAnalyzer"

  - type: "CREATE"
    item_type: "TOOL"
    name: "AgentCommunicator"

  # Create the agents from the library
  - type: "CREATE"
    item_type: "AGENT"
    name: "SpecialistAgent"
    config:
      memory_type: "token"

  # Create the coordinator agent with the tools
  - type: "CREATE"
    item_type: "AGENT"
    name: "CoordinatorAgent"
    config:
      memory_type: "token"

  # Execute with an invoice document
  - type: "EXECUTE"
    item_type: "AGENT"
    name: "CoordinatorAgent"
    user_input: "Process this document: {invoice}"
"""
```

In this workflow, the CoordinatorAgent delegates specialized tasks to the SpecialistAgent through the AgentCommunicator tool. The example shows a complete workflow execution for both an invoice and a medical record.

### Step 4: Agent Evolution

The third part demonstrates how agents can be evolved to create enhanced versions:

```yaml
# Evolution workflow
evolution_workflow = """
scenario_name: "Enhanced Invoice Processing"
domain: "document_processing"
description: "Evolve the specialist agent to provide better invoice analysis"

steps:
  # Define an evolved version of the specialist agent
  - type: "DEFINE"
    item_type: "AGENT"
    name: "EnhancedInvoiceSpecialist"
    from_existing_snippet: "SpecialistAgent"
    evolve_changes:
      docstring_update: "Improved with enhanced invoice analysis capabilities including line item detection"
    description: "Enhanced specialist that provides more detailed invoice analysis with line item extraction"

  # Create and execute the evolved agent
  - type: "CREATE"
    item_type: "AGENT"
    name: "EnhancedInvoiceSpecialist"
    config:
      memory_type: "token"

  # Test the evolved agent with an invoice
  - type: "EXECUTE"
    item_type: "AGENT"
    name: "EnhancedInvoiceSpecialist"
    user_input: "{invoice}"
"""
```

This evolution process takes an existing agent (SpecialistAgent) and creates an enhanced version (EnhancedInvoiceSpecialist) with improved capabilities specific to invoice analysis.

### Step 5: Semantic Search with OpenAI Embeddings

The final part demonstrates how to find semantically similar components in the library:

```python
# Search for document processing agents
search_results = await library.semantic_search(
    query="agent that can process and understand documents",
    record_type="AGENT",
    threshold=0.3
)
```

This shows how the framework uses OpenAI embeddings to find the most relevant agents for a given task, allowing you to discover and reuse existing components based on their semantic meaning rather than just exact keyword matches.

## Agent Communication Protocol (ACP) Support

The Evolving Agents Framework includes preliminary support for BeeAI's Agent Communication Protocol (ACP), a standardized approach for agent-to-agent communication.

### What is ACP?

The Agent Communication Protocol (ACP) is a protocol designed to standardize how agents communicate, enabling automation, agent-to-agent collaboration, UI integration, and developer tooling. Currently in pre-alpha, ACP extends the Model Context Protocol (MCP), leveraging the simplicity and versatility of JSON-RPC for efficient interactions between agents, platforms, and external services.

For more information, see the [official BeeAI ACP documentation](https://docs.beeai.dev/acp/pre-alpha/introduction#relationship-with-mcp).

### ACP Integration in Evolving Agents

Our framework provides a preliminary implementation of ACP that allows agents to communicate using standardized message formats:

```python
# Example: Using ACP for agent communication
from evolving_agents.acp.client import ACPClient
from evolving_agents.providers.acp_provider import ACPProvider

# Create an ACP client
acp_client = ACPClient(transport="memory")

# Register the ACP provider with your system
provider_registry = ProviderRegistry()
provider_registry.register_provider(ACPProvider(llm_service, acp_client))

# Process an ACP-enabled workflow
acp_workflow = """
scenario_name: "ACP Document Analysis"
domain: "document_processing"
description: "Process documents using ACP-enabled agents"

steps:
  # Register agents with ACP
  - type: "ACP_REGISTER"
    name: "AnalysisAgent"

  - type: "ACP_REGISTER"
    name: "SummaryAgent"

  # Execute with ACP communication
  - type: "ACP_COMMUNICATE"
    sender: "AnalysisAgent"
    recipient: "SummaryAgent"
    message: "Please analyze this document: {document_text}"
"""

results = await workflow_processor.process_acp_workflow(acp_workflow)
```

### ACP Features Currently Supported

- **Message Types**: Support for text and structured JSON messages
- **Transport Layers**: In-memory transport (with stubs for HTTP/SSE, WebSocket, and Stdio)
- **Agent Registration**: Register and discover agents through the ACP registry
- **Workflow Integration**: Define ACP-specific workflow steps in YAML
- **Message History**: Track message exchanges for debugging and analysis

### Next Steps for ACP Integration

1. **Standard Compliance**: Align our implementation with evolving ACP standards as they mature
2. **Transport Implementation**: Complete HTTP/SSE and WebSocket transport implementations
3. **UI Integration**: Add support for ACP-powered user interfaces
4. **Advanced Message Types**: Expand support for more sophisticated message schemas
5. **Official Integration**: Prepare for seamless transition to official BeeAI ACP implementation

As BeeAI's ACP moves from pre-alpha to more stable versions, we'll update our implementation to match the official standards while maintaining backward compatibility with existing workflows.

## Implementation Details

The example uses real BeeAI agents and tools, not just simulations. The key components are:

1. **BeeAI ReActAgent Implementation**: Fully functional agents that use the Reasoning + Acting (ReAct) pattern to solve tasks.

2. **LLM-based Tools**: Tools that leverage language models to analyze documents and facilitate agent communication.

3. **Semantic Library**: A smart storage system that tracks agent versions, performance metrics, and supports semantic search.

4. **YAML Workflow Definition**: A declarative way to describe complex agent interactions.

5. **Provider Architecture**: A pluggable system that supports multiple agent frameworks (currently BeeAI).

## Known Issues

1. **Markdown Code Blocks**: The LLM sometimes includes markdown formatting in its responses (like ```python), which can cause syntax errors when trying to execute the code.

2. **String Literal Handling**: Some generated code may have syntax issues with string literals, especially when escaping special characters.

3. **Parameter Consistency**: When evolving agents, parameter types and names may not always be consistently maintained.

4. **Medical Record Processing**: In the current example workflow, there might be issues with the medical record processing not correctly analyzing medical record inputs.

## Roadmap for Future Improvements

1. **Enhanced Code Parsing**: Add a more sophisticated code parsing system to better handle markdown and other formatting in the generated code.

2. **Code Validation**: Add a validation step when loading agents and tools to ensure their code is well-formed and executable.

3. **Test Mode for Agents**: Implement a "test mode" for agents where they can be checked for basic functionality before being added to the library.

4. **Error Analysis System**: Add a system to record and analyze specific errors that agents encounter to better guide their evolution.

5. **Improved Cross-Domain Collaboration**: Enhance the ability of agents from different domains to collaborate effectively.

6. **Agent Memory Persistence**: Implement more sophisticated memory models that allow agents to retain knowledge across sessions.

7. **Self-improvement Metrics**: Add quantitative measurements of agent improvement over time and across evolutions.

8. **Visual Debugging Tools**: Create tools to visualize agent execution paths and communications for easier debugging.

9. **Full ACP Implementation**: Complete the integration with BeeAI's Agent Communication Protocol as it moves beyond pre-alpha.

## Core Components

### Smart Library

The central repository for agents, tools, and firmware:

```python
# Store a component in the library
await library.create_record(
    name="InvoiceAnalyzer",
    record_type="TOOL",
    domain="finance",
    description="Analyzes and extracts data from invoice documents",
    code_snippet=tool_code,
    tags=["invoice", "finance", "extraction"]
)

# Semantic search to find components
results = await library.semantic_search(
    query="tool that extracts data from invoices",
    record_type="TOOL",
    domain="finance"
)
```

### System Agent

The orchestrator that decides whether to reuse, evolve, or create components:

```python
# Process a request using the system agent
result = await system_agent.decide_and_act(
    request="I need a tool to analyze medical records",
    domain="healthcare",
    record_type="TOOL"
)

# Execute the resulting tool or agent
if result["action"] in ["reuse", "evolve", "create"]:
    execution = await system_agent.execute_item(
        result["record"]["name"],
        "Patient has high blood pressure and diabetes"
    )
```

### Workflow Processor

Process YAML-defined agent workflows:

```python
# Initialize workflow processor
processor = WorkflowProcessor(system_agent)

# Process a workflow 
results = await processor.process_workflow(workflow_yaml)
```

### Provider Architecture

Support for multiple agent frameworks:

```python
# Register providers
provider_registry = ProviderRegistry()
provider_registry.register_provider(BeeAIProvider(llm_service))
provider_registry.register_provider(ACPProvider(llm_service))  # ACP support

# Initialize system agent with providers
system = SystemAgent(library, llm, provider_registry=provider_registry)
```

## Use Cases

- **Document Processing**: Create specialized agents for different document types that collaborate to extract and analyze information
- **Healthcare**: Medical agents communicating with pharmacy and insurance agents to coordinate patient care
- **Financial Analysis**: Portfolio management agents collaborating with market analysis agents
- **Customer Service**: Routing agents delegating to specialized support agents
- **Multi-step Reasoning**: Break complex problems into components handled by specialized agents

## Advanced Features

- **Firmware Injection**: Enforce governance rules and constraints across all agents
- **Version Control**: Track the evolution of agents over time
- **Cross-domain Collaboration**: Enable agents from different domains to work together
- **Observability**: Monitor agent communications and decision processes

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Apache v2.0](LICENSE)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=matiasmolinas/evolving-agents&type=Timeline)](https://star-history.com/#matiasmolinas/evolving-agents&Timeline)

## Acknowledgements

- [Matias Molinas](https://github.com/matiasmolinas) and [Ismael Faro](https://github.com/ismaelfaro) for the original concept and architecture
- BeeAI framework for integrated agent capabilities
Add to Conversation
