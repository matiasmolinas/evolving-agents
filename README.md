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
- **Component Reuse**: Smart Library to discover, reuse, and adapt existing components
- **Framework Agnostic**: Works with BeeAI, OpenAI Agents SDK, and custom frameworks
- **Governance Firmware**: Built-in guardrails to ensure agents stay within safe boundaries
- **Self-Building Capabilities**: Agents can design and implement entire agent systems

## Key Concepts

### Agent-Centric Architecture

Unlike most agent frameworks that prioritize human-to-AI interactions, Evolving Agents focuses on agent-to-agent workflows:

```python
# Example: Agents finding other agents through capabilities
analysis_agent = await agent_bus.request_service(
    capability="document_analysis",
    content={"text": document_text}
)
```

### Smart Library

The Smart Library provides semantic discovery, storage, and evolution of components using real vector embeddings with ChromaDB:

```python
# Example: Searching for components semantically
similar_agents = await smart_library.semantic_search(
    query="tool that can analyze invoices",
    record_type="TOOL",
    threshold=0.5
)
```

### Workflow Generation

Create and execute multi-agent workflows from natural language requirements:

```python
# Example: Generate a workflow from requirements
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
# Example: Evolve an agent to a new domain
evolved_agent = await evolve_component_tool.run(
    parent_id=original_agent_id,
    changes="Adapt to a new domain with different requirements",
    target_domain="healthcare",
    evolution_strategy="domain_adaptation"
)
```

## Example Applications

The repository includes several examples demonstrating key capabilities:

### 1. Architect-Zero Comprehensive Demo
The flagship example in `examples/invoice_processing/architect_zero_comprehensive_demo.py` shows our most sophisticated meta-agent building a complete invoice processing system from scratch. This example demonstrates:
- Automated requirements analysis
- Component discovery and reuse
- Evolution of existing components
- Generation of complete workflows
- End-to-end execution

### 2. OpenAI Agent Evolution
The `examples/agent_evolution/openai_agent_evolution_demo.py` demonstrates:
- Creating OpenAI agents with the Agents SDK
- Evolving agents through different strategies (standard, aggressive)
- Domain adaptation
- A/B testing to compare agent versions

### 3. Conversational Forms
The `examples/forms/run_conversational_form.py` demonstrates:
- Natural language form definition
- Dynamic conversation flow
- Validation and conditional logic
- Response summary generation

### 4. Smart Autocomplete
The `examples/autocomplete/run_autocomplete_system.py` shows:
- Context-aware text completion
- Learning from multiple input sources
- Adaptive suggestions based on user context

## Why Another Agent Toolkit?

Most agent frameworks focus on creating individual agents, not agent ecosystems that can build themselves. Key differences:

1. **Agent Autonomy**: Our agents can decide when to reuse, evolve, or create components
2. **Multi-Framework Support**: Seamlessly integrate agents from different frameworks
3. **Firmware Governance**: Built-in guardrails ensure safety even as agents evolve
4. **Self-Improvement**: Agents track their own performance and improve over time

## Firmware and Guardrails

In an ecosystem where agents can evolve and create new components, governance is essential. Our firmware:

- Embeds safe behavior rules directly into agent creation and evolution
- Provides domain-specific rules (medical, financial, etc.)
- Ensures ethical constraints are maintained across generations
- Prevents capability drift through continuous validation

Without proper guardrails, autonomous agent ecosystems could develop unintended behaviors or safety issues. The Evolving Agents Toolkit tackles this challenge head-on by making governance a core part of the agent lifecycle, not an afterthought.

## Installation

```bash
pip install evolving-agents-framework
```

## Development Features

- **LLM Caching**: Built-in caching for completions and embeddings speeds up development and testing
- **Vector Search**: Integrated ChromaDB for efficient semantic search of components
- **Hot Reloading**: Components can be modified and reloaded without restarting the system

## License

[Apache v2.0](LICENSE)

## Acknowledgements

- [Matias Molinas](https://github.com/matiasmolinas) and [Ismael Faro](https://github.com/ismaelfaro) for the original concept and architecture
- [BeeAI Framework](https://github.com/i-am-bee/beeai-framework/tree/main/python) for integrated agent capabilities
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) for additional agent integration