# Evolving Agents Toolkit

A toolkit for agent autonomy, evolution, and governance. Create agents that can understand requirements, evolve through experience, communicate effectively, and build new agents and tools - all while operating within governance guardrails.

## Why This Toolkit?

Current agent systems are designed primarily for humans to build and control AI agents. The Evolving Agents Toolkit takes a fundamentally different approach: agents building agents.

Our toolkit provides:

- **Autonomous Evolution**: Agents learn from experience and improve themselves without human intervention
- **Agent Self-Discovery**: Agents discover and collaborate with other specialized agents to solve complex problems
- **Governance Firmware**: Enforceable guardrails that ensure agents evolve and operate within safe boundaries
- **Self-Building Systems**: The ability for agents to create new tools and agents when existing ones are insufficient
- **Agent-Centric Architecture**: Communication and capabilities built for agents themselves, not just their human creators

We build on existing frameworks like BeeAI and OpenAI Agents SDK to create a layer that enables agent autonomy, evolution, and self-governance - moving us closer to truly autonomous AI systems that improve themselves while staying within safe boundaries.

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Optional: Install OpenAI Agents SDK
pip install -r requirements-openai-agents.txt

# Create the initial libraries
python evolving_agents/generate_smart_library.py

# Run the examples
python evolving_agents/run_examples.py
```

## Core Architecture

The toolkit's architecture is centered around these key components:

### 1. Smart Library with Components

The Smart Library serves as a repository for all agents, tools, and capabilities. Components are stored with:

- Semantic embeddings for intelligent search
- Performance metrics for experience-based selection
- Capability contracts defining inputs and outputs

```python
# Search for components in the library
search_results = await smart_library.semantic_search(
    query="agent that can process invoices and analyze costs",
    domain="finance",
    limit=3,
    threshold=0.6
)
```

### 2. Agent Bus for Capability-Based Communication

The Agent Bus enables agents to discover and communicate with each other through capabilities:

```python
# Register a component with capabilities
provider_id = await agent_bus.register_provider(
    name="InvoiceAnalyzer",
    capabilities=[{
        "id": "invoice_analysis",
        "name": "Invoice Analysis",
        "description": "Analyzes invoice contents to extract key information"
    }]
)

# Request service by capability
result = await agent_bus.request_service(
    capability="invoice_analysis",
    content={"invoice_text": "INVOICE #12345..."}
)
```

### 3. Governance Firmware

The Firmware system implements guardrails that apply across all components:

```python
# Get domain-specific firmware rules
finance_firmware = firmware.get_firmware_prompt("finance")

# Apply firmware to component creation
agent_record = await smart_library.create_record(
    name="Financial Analyzer",
    record_type="AGENT",
    domain="finance",
    description="Analyzes financial documents and provides insights",
    code_snippet=code_with_firmware
)
```

### 4. Evolving Agents with Experience

Agents can evolve based on their experiences:

```python
# Evolve an agent with specific changes
evolved_agent = await evolution_manager.evolve_agent(
    agent_id="invoice_processor_v1",
    changes="Improve extraction of line items and verify calculations",
    evolution_type="standard"
)

# Compare agent versions with A/B testing
test_results = await evolution_manager.compare_agents(
    agent_a_id="invoice_processor_v1",
    agent_b_id=evolved_agent["evolved_agent_id"],
    test_inputs=test_data,
    domain="finance"
)
```

## Example Use Cases

### Financial Invoice Processing

The `financial_example.py` demonstrates how to build an invoice processing system:

```python
# Run the financial example
python evolving_agents/financial_example.py
```

This example uses the Architect-Zero agent to:
1. Create document analyzers for invoice detection
2. Extract structured data from invoices (date, vendor, line items, totals)
3. Verify calculations for correctness
4. Generate concise summaries with key insights

### Medical Assessment System

The `medical_example.py` showcases a healthcare assessment system:

```python
# Run the medical example
python evolving_agents/medical_example.py
```

This example creates:
1. A physiological data extractor for patient records
2. BMI and cardiovascular risk calculators
3. Medical analysis agents for interpretation and recommendations
4. Components that adhere to healthcare governance rules

## Key Components

### Architect-Zero

The Architect-Zero agent is a meta-agent that designs multi-agent solutions:

```python
# Initialize Architect-Zero
architect_agent = await create_architect_zero(
    llm_service=llm_service,
    smart_library=smart_library,
    agent_bus=agent_bus,
    system_agent=system_agent
)

# Give it a high-level task
result = await architect_agent.run(
    "Create a comprehensive medical diagnostic system that analyzes patient records..."
)
```

### Smart Library Tools

Tools for component management:

```python
# Search for components
search_results = await search_tool.run(
    query="component that extracts financial data",
    record_type="TOOL",
    domain="finance"
)

# Create a new component
new_component = await create_tool.run(
    name="InvoiceLineItemExtractor",
    record_type="TOOL",
    domain="finance",
    description="Extracts line items from invoices",
    requirements="Create a tool that can extract individual line items..."
)

# Evolve an existing component
evolved_component = await evolve_tool.run(
    parent_id="existing_component_id",
    changes="Add support for multi-currency line items",
    evolution_strategy="standard"
)
```

### Agent Bus Tools

Tools for agent communication:

```python
# Register a provider with the bus
provider_id = await register_tool.run(
    name="DataExtractor",
    capabilities=[
        "text_extraction",
        "table_extraction"
    ]
)

# Request a service
result = await request_tool.run(
    capability="text_extraction",
    content="Extract data from this document: ..."
)

# Discover available capabilities
capabilities = await discover_tool.run(
    query="extraction capabilities"
)
```

## Technical Architecture

The toolkit integrates with different agent frameworks through a provider system:

```python
# Create a BeeAI agent
agent_record = await create_tool.run(
    name="DataAnalyzer",
    record_type="AGENT",
    domain="data_analysis",
    requirements="...",
    framework="beeai"
)

# Create an OpenAI agent
openai_agent = await create_openai_agent_tool.run(
    name="FinancialAssistant",
    domain="finance",
    description="...",
    model="gpt-4o"
)
```

## Framework Integration

The toolkit provides adapters for different frameworks:

- **BeeAI Provider**: Creates and manages BeeAI ReActAgents
- **OpenAI Provider**: Creates and manages OpenAI Agents SDK agents
- **Tool Adapters**: Convert between different tool formats

## API Reference

### SmartLibrary

```python
# Create a component
record = await smart_library.create_record(
    name="ComponentName",
    record_type="AGENT",  # or "TOOL" 
    domain="domain_name",
    description="Description",
    code_snippet="# Code goes here",
    tags=["tag1", "tag2"],
    metadata={"key": "value"}
)

# Search for components
results = await smart_library.semantic_search(
    query="search query",
    record_type="AGENT",  # optional filter
    domain="domain_name",  # optional filter
    limit=5
)

# Find components by capability
component = await smart_library.find_component_by_capability(
    capability_id="capability_name",
    domain="domain_name"
)

# Evolve a component
evolved = await smart_library.evolve_record(
    parent_id="original_record_id",
    new_code_snippet="# New code",
    description="Updated description"
)
```

### AgentBus

```python
# Register a provider
provider_id = await agent_bus.register_provider(
    name="ProviderName",
    capabilities=[{
        "id": "capability_id",
        "name": "Capability Name",
        "description": "Description"
    }]
)

# Request a service
result = await agent_bus.request_service(
    capability="capability_id",
    content={"key": "value"}
)

# Find providers for a capability
providers = await agent_bus.find_providers_for_capability(
    capability="capability_id",
    min_confidence=0.7
)
```

### Workflow Processor

```python
# Process a workflow from YAML
result = await workflow_processor.process_workflow(yaml_content)

# Generate a workflow from design
yaml_workflow = await workflow_generator.generate_workflow_from_design(
    workflow_design=design_dict,
    library_entries=entries_dict
)
```

## License

[Apache v2.0](LICENSE)

## Acknowledgements

- [Matias Molinas](https://github.com/matiasmolinas) and [Ismael Faro](https://github.com/ismaelfaro) for the original concept and architecture
- BeeAI framework for integrated agent capabilities
- OpenAI for the Agents SDK