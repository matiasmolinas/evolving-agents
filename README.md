# Evolving Agents Toolkit

A toolkit for building autonomous AI agents that can evolve, understand requirements, and build new agents - all while operating within governance guardrails.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/matiasmolinas/evolving-agents.git
cd evolving-agents

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install OpenAI Agents SDK (optional)
pip install -r requirements-openai-agents.txt

# Run the Architect-Zero example
python examples/invoice_processing/architect_zero_comprehensive_demo.py
```

## Core Features

- **Autonomous Evolution**: Agents that learn from experience and improve themselves
- **Semantic Capability Matching**: SmartAgentBus for discovering and routing requests by capability
- **Component Reuse**: Smart Library to discover, reuse, and adapt existing components
- **Framework Agnostic**: Works with BeeAI, OpenAI Agents SDK, and custom frameworks
- **Governance Firmware**: Built-in guardrails to ensure agents stay within safe boundaries
- **Self-Building Capabilities**: Agents can design and implement entire agent systems

## Key Concepts

### SmartAgentBus - The Agent Nervous System

The SmartAgentBus provides intelligent routing and execution of capabilities with:

```python
# Semantic capability discovery
result = await agent_bus.request_service(
    capability_query="I need sentiment analysis for customer feedback",
    input_data={"text": customer_review},
    min_confidence=0.7
)

# Direct capability execution
invoice_result = await agent_bus.request_service(
    capability_query="invoice_processing",
    input_data={"document": invoice_pdf},
    provider_id="invoice_specialist_v3"
)
```

Key features:
- **Semantic Matching**: Find capabilities using natural language
- **Circuit Breakers**: Automatic failure handling for unreliable providers
- **Execution Monitoring**: Detailed logging of all service requests
- **Provider Management**: Register and manage capability providers

### Smart Library

The Smart Library provides semantic discovery, storage, and evolution of components using real vector embeddings with ChromaDB:

```python
# Semantic component discovery
similar_agents = await smart_library.semantic_search(
    query="tool that can analyze invoices",
    record_type="TOOL",
    threshold=0.5
)

# Automatic provider registration
await agent_bus.initialize_from_library()
```

### Dependency Management System

The framework includes a robust dependency container to manage component dependencies:

```python
# Create a dependency container
container = DependencyContainer()

# Register core components
container.register('llm_service', llm_service)
container.register('smart_library', smart_library)

# Initialize all components with proper dependency wiring
await container.initialize()

# Get any component
system_agent = container.get('system_agent')
```

### Workflow Generation

Create and execute multi-agent workflows from natural language requirements:

```python
# Generate a workflow from requirements
workflow_yaml = await workflow_generator.generate_workflow(
    requirements="Build a pipeline to extract and verify invoice data",
    domain="finance"
)

# Execute the workflow
result = await workflow_processor.process_workflow(workflow_yaml)
```

### Agent Evolution

Evolve existing agents to adapt to new requirements or domains:

```python
# Evolve an agent to a new domain
evolved_agent = await evolve_component_tool.run(
    parent_id=original_agent_id,
    changes="Adapt to a new domain with different requirements",
    target_domain="healthcare",
    evolution_strategy="domain_adaptation"
)
```

## Example Applications

### 1. Architect-Zero Comprehensive Demo
The flagship example in `examples/invoice_processing/architect_zero_comprehensive_demo.py` demonstrates:
- Automated requirements analysis
- Semantic capability discovery through SmartAgentBus
- Circuit breaker patterns for reliable execution
- End-to-end workflow generation and execution

### 2. Semantic Capability Routing
New example in `examples/capability_routing/semantic_routing_demo.py` shows:
- Natural language capability discovery
- Automatic provider selection
- Fallback handling and circuit breakers
- Execution monitoring

### 3. OpenAI Agent Evolution
The `examples/agent_evolution/openai_agent_evolution_demo.py` demonstrates:
- Creating OpenAI agents with the Agents SDK
- Evolving agents through different strategies
- Domain adaptation
- A/B testing to compare agent versions

### 4. Conversational Forms
The `examples/forms/run_conversational_form.py` demonstrates:
- Natural language form definition
- Dynamic conversation flow
- Validation and conditional logic

## Architecture

The framework uses a three-phase initialization pattern to manage dependencies:

1. **Registration Phase**: Components are registered with the dependency container
2. **Wiring Phase**: Components receive their dependencies
3. **Initialization Phase**: Components complete their setup with proper dependencies

This approach eliminates circular reference issues and makes the system more modular and testable.

## Why Another Agent Toolkit?

Most agent frameworks focus on creating individual agents, not agent ecosystems that can build themselves. Key differences:

1. **Semantic Capability Network**: SmartAgentBus enables agents to discover and use capabilities semantically
2. **Self-Healing Architecture**: Circuit breakers and automatic failover
3. **Multi-Framework Support**: Seamlessly integrate agents from different frameworks
4. **Execution Transparency**: Comprehensive logging and monitoring

## Firmware and Guardrails

Our governance system ensures safe operation of autonomous agents:

- **Capability Validation**: All registered capabilities are validated
- **Circuit Breakers**: Prevent cascading failures
- **Execution Logging**: Complete audit trail of all operations
- **Semantic Constraints**: Prevent capability drift through embedding-based validation

## Installation

```bash
pip install evolving-agents-framework
```

## Development Features

- **LLM Caching**: Built-in caching for completions and embeddings
- **Vector Search**: Integrated ChromaDB for semantic capability discovery
- **Hot Reloading**: Components can be modified without restarting
- **Detailed Logging**: Execution logs for debugging and auditing
- **Dependency Management**: Comprehensive dependency injection system to manage circular references

## License

[Apache v2.0](LICENSE)

## Acknowledgements

- [Matias Molinas](https://github.com/matiasmolinas) and [Ismael Faro](https://github.com/ismaelfaro) for the original concept
- [BeeAI Framework](https://github.com/i-am-bee/beeai-framework/tree/main/python) for agent capabilities
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) for additional integrations
- [ChromaDB](https://www.trychroma.com/) for semantic capability matching