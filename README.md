# Evolving Agents Toolkit

**Build intelligent AI agent ecosystems that design, generate, and execute complex workflows.**

[![License: Apache v2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<!-- Add other badges like build status, PyPI version etc. here -->

This toolkit provides a robust, production-grade framework for building autonomous AI agents and multi-agent systems. It uniquely focuses on enabling agents to understand requirements, design solutions, discover capabilities, evolve components, and orchestrate complex tasks, all while operating within defined governance boundaries.

## Core Features

*   **Agent-Centric Orchestration:** A central `SystemAgent` manages the ecosystem using specialized tools for component management, communication, and workflow execution.
*   **Autonomous Solution Design:** The `ArchitectZero` agent analyzes high-level requirements and designs detailed multi-component solutions.
*   **Automated Workflow Generation & Execution:** The `SystemAgent` translates designs into executable YAML workflows and processes them step-by-step using its tools.
*   **Semantic Capability Discovery:** The `SmartAgentBus` allows agents to find and utilize capabilities based on natural language descriptions, enabling dynamic service discovery and routing.
*   **Intelligent Component Management:** The `SmartLibrary` provides persistent storage, semantic search (via vector embeddings), versioning, and evolution capabilities for agents and tools.
*   **Adaptive Evolution:** Components can be evolved based on requirements, feedback, or performance data using various strategies (standard, conservative, aggressive, domain adaptation).
*   **Multi-Framework Support:** Seamlessly integrate agents built with different frameworks (e.g., BeeAI, OpenAI Agents SDK) through a flexible provider architecture.
*   **Governance & Safety:** Built-in `Firmware` injects rules, and guardrails (like the `OpenAIGuardrailsAdapter`) ensure safe and compliant operation.
*   **Self-Building Potential:** The architecture allows agents (like `ArchitectZero` and `SystemAgent`) to collaboratively design and implement *new* agent systems based on user needs.

## Why This Toolkit?

While many frameworks focus on building individual agents, the Evolving Agents Toolkit focuses on creating **intelligent, self-improving agent ecosystems**. Key differentiators include:

1.  **Design/Generate/Execute Loop:** Clear separation of concerns where one agent (`ArchitectZero`) designs the "what" and "how" (the blueprint), and another (`SystemAgent`) handles the "doing" (generating the executable plan and executing it).
2.  **Semantic Capability Network:** `SmartAgentBus` creates a dynamic network where agents discover and interact based on function, not fixed names.
3.  **Deep Component Lifecycle Management:** Beyond creation, the `SmartLibrary` and evolution tools support searching, reusing, versioning, and adapting components intelligently.
4.  **Agent-Driven Orchestration:** The `SystemAgent` isn't just a script runner; it's a `ReActAgent` using its own tools to manage the entire process, including workflow execution.
5.  **True Multi-Framework Integration:** Provides abstractions (`Providers`, `AgentFactory`) to treat agents from different SDKs as first-class citizens.

## Key Concepts

### 1. Agent-Centric Orchestration (`SystemAgent`)

The `SystemAgent` acts as the central nervous system. It's a `ReActAgent` equipped with specialized tools to manage the entire ecosystem. It doesn't hardcode logic but decides which tool to use based on the prompt.

```python
# Example: Prompting the SystemAgent to perform multiple tasks
prompt = """
1. Find tools in the library related to 'invoice data extraction' with similarity > 0.7.
2. Using the best tool found, or creating one if needed, process the attached invoice data: {invoice_content}.
3. Register the 'InvoiceProcessor_V2' agent with the Agent Bus, providing 'invoice_processing' and 'data_verification' capabilities.
4. Generate a YAML workflow based on the design stored in 'design.json' for the 'finance' domain.
5. Process the generated YAML workflow using the parameters: {'input_file': 'invoice.pdf'}.
"""
# The SystemAgent's ReAct loop will invoke tools like:
# - SearchComponentTool
# - CreateComponentTool / EvolveComponentTool
# - RequestAgentTool (to execute the found/created tool)
# - RegisterAgentTool
# - GenerateWorkflowTool
# - ProcessWorkflowTool
# - ...and potentially others based on the execution plan from ProcessWorkflowTool
final_result = await system_agent.run(prompt)
```

### 2. Solution Design (`ArchitectZero`)

`ArchitectZero` is a specialized agent that takes high-level requirements and outputs a detailed *solution design* (typically JSON), outlining the components needed (new, evolved, reused) and the logical workflow sequence.

```python
# Example: Prompting ArchitectZero to design a system
design_prompt = """
Design an automated customer support system for e-commerce.
It should handle FAQs via a dedicated agent, escalate complex issues
to a human support agent (represented by a tool), and use a sentiment
analyzer tool to prioritize urgent requests. Analyze existing library
components for potential reuse or evolution. Output the complete design as JSON.
"""
architect_response = await architect_agent.run(design_prompt)
solution_design_json = extract_json_from_response(architect_response.result.text)
# >> solution_design_json now contains the blueprint
```

### 3. Smart Library (Component Management & Discovery)

Stores agents, tools, and firmware definitions. Enables semantic search and lifecycle management.

```python
# Semantic component discovery
similar_tools = await smart_library.semantic_search(
    query="Tool that can validate financial calculations in documents",
    record_type="TOOL",
    domain="finance",
    threshold=0.6 # Find moderately similar tools
)
# >> Returns list: [(tool_record_dict, similarity_score), ...]

# Get a specific record
record = await smart_library.find_record_by_id("agent_id_123")

# Evolve an existing component
evolved_record = await smart_library.evolve_record(
    parent_id="tool_id_abc",
    new_code_snippet="# New improved Python code...",
    description="Enhanced version with better error handling",
    new_version="1.1.0" # Optional: otherwise increments automatically
)
```

### 4. Smart Agent Bus (Capability Routing)

Enables dynamic, capability-based communication between registered components.

```python
# Register a component as a capability provider (often done via SystemAgent's tool)
await agent_bus.register_agent(
    name="SentimentAnalyzerTool_v2",
    agent_type="TOOL",
    description="Analyzes text sentiment with high accuracy",
    capabilities=[{
        "id": "sentiment_analysis",
        "name": "Sentiment Analysis",
        "description": "Detects positive, negative, neutral sentiment and score.",
        "confidence": 0.9
    }]
)

# Request a service based on capability (often done via SystemAgent's tool)
result = await agent_bus.request_capability(
    capability="sentiment_analysis",
    content={"text": "This service is amazing!"},
    min_confidence=0.8 # Find providers with high confidence in this capability
)
# >> result might contain: {'agent_id': '...', 'agent_name': 'SentimentAnalyzerTool_v2', 'content': {'sentiment': 'positive', 'score': 0.95}, ...}
```

### 5. Workflow Lifecycle (Design -> Generate -> Process -> Execute)

Complex tasks are handled via a structured workflow lifecycle managed primarily by the `SystemAgent`.

1.  **Design:** `ArchitectZero` creates a solution design (JSON) based on requirements.
2.  **Generate:** `SystemAgent` receives the design and uses its `GenerateWorkflowTool` (powered by an LLM) to create an executable YAML workflow string.
3.  **Process:** `SystemAgent` receives the YAML and uses its `ProcessWorkflowTool` to parse it, validate structure, substitute parameters, and produce a structured *execution plan* (list of steps).
4.  **Execute:** `SystemAgent`'s ReAct loop iterates through the plan, using its *other* tools (`CreateComponentTool`, `EvolveComponentTool`, `RequestAgentTool`, framework-specific execution tools, etc.) to perform the action defined in each step (`DEFINE`, `CREATE`, `EXECUTE`).

```python
# Conceptual Example (within the SystemAgent's operation)

# --- Step: Generate Workflow ---
# design_json = ... (obtained from ArchitectZero)
generation_prompt = f"Use GenerateWorkflowTool for this design: {design_json}"
generation_result = await system_agent.run(generation_prompt)
# workflow_yaml = extract_yaml_from_response(generation_result...)

# --- Step: Execute Workflow ---
# workflow_yaml = ... (obtained from generation)
# params = {"input_file": "invoice.pdf"}
execution_prompt = f"Execute this YAML workflow with params {params}: {workflow_yaml}"
execution_result = await system_agent.run(execution_prompt)
# >> execution_result contains the final output after SystemAgent
#    used ProcessWorkflowTool to get the plan, then executed plan steps.
```

### 6. Component Evolution

Existing components can be adapted or improved using the `EvolveComponentTool` (typically invoked by `SystemAgent`).

```python
# Conceptual Example (within the SystemAgent's operation)
evolve_prompt = f"""
Use EvolveComponentTool to enhance agent 'id_123'.
Changes needed: Add support for processing PDF files directly.
Strategy: standard
"""
evolve_result = await system_agent.run(evolve_prompt)
# >> evolve_result indicates success and provides ID of the new evolved agent version.
```

### 7. Multi-Framework Support

Integrate agents/tools from different SDKs via `Providers` managed by the `AgentFactory`.

```python
# Example: Creating agents from different frameworks via AgentFactory
# (AgentFactory is usually used internally by tools like CreateComponentTool)

bee_record = await smart_library.find_record_by_name("BeeAgentName")
openai_record = await smart_library.find_record_by_name("OpenAIAgentName")

if bee_record:
    bee_agent_instance = await agent_factory.create_agent(bee_record)
    # >> Uses BeeAIProvider internally

if openai_record:
    openai_agent_instance = await agent_factory.create_agent(openai_record)
    # >> Uses OpenAIAgentsProvider internally
```

### 8. Governance & Firmware

Safety and operational rules are embedded via `Firmware`.

*   `Firmware` provides base rules + domain-specific constraints.
*   Prompts used by `CreateComponentTool` / `EvolveComponentTool` include firmware content.
*   `OpenAIGuardrailsAdapter` converts firmware rules into runtime checks for OpenAI agents.

## Installation

```bash
# Recommended: Create a virtual environment
python -m venv venv
source venv/bin/activate # On Windows use `venv\Scripts\activate`

# Install from PyPI (when available)
# pip install evolving-agents-framework

# Or install from source
git clone https://github.com/matiasmolinas/evolving-agents.git
cd evolving-agents
pip install -r requirements.txt
pip install -e . # Install in editable mode
```

## Quick Start

1.  **Set up Environment:**
    *   Copy `.env.example` to `.env`.
    *   Add your `OPENAI_API_KEY` to the `.env` file.
    *   Configure other settings like `LLM_MODEL` if needed.

2.  **Run the Comprehensive Demo:**
    This demo showcases ArchitectZero designing a solution and SystemAgent generating and executing the workflow for invoice processing.

    ```bash
    python examples/invoice_processing/architect_zero_comprehensive_demo_refactored.py
    ```

3.  **Explore Output:** Check the generated files:
    *   `architect_design_output.json`: The solution blueprint from ArchitectZero.
    *   `generated_invoice_workflow.yaml`: The executable workflow generated by SystemAgent.
    *   `workflow_execution_output.json`: The final result and log from SystemAgent executing the workflow.
    *   `smart_library_demo.json`: The state of the component library after the run.
    *   `smart_agent_bus_demo.json`: The agent registry state.

## Example Applications

Explore the `examples/` directory:

*   **`invoice_processing/architect_zero_comprehensive_demo.py`**: The flagship demo showing the full Design -> Generate -> Execute loop.
*   **`agent_evolution/`**: Demonstrates creating and evolving agents/tools using both BeeAI and OpenAI frameworks, including A/B testing.
*   **`forms/`**: Shows how the system can design and process conversational forms based on natural language descriptions.
*   **`autocomplete/`**: Illustrates designing a context-aware autocomplete system.
*   *(Add more examples as they are created)*

## Architecture Overview

The toolkit employs an **agent-centric architecture**. The `SystemAgent` (a ReAct agent) orchestrates operations by leveraging specialized tools that interact with core components like the `SmartLibrary` (for component persistence and semantic search via ChromaDB) and the `SmartAgentBus` (for capability-based routing). Solution design is handled by `ArchitectZero`. Multi-framework support is achieved through `Providers` and `Adapters`. Dependencies are managed via a `DependencyContainer`.

For a detailed breakdown, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Development Features

*   **LLM Caching:** Reduces API costs during development by caching completions and embeddings (`.llm_cache_demo/`).
*   **Vector Search:** Integrated ChromaDB for powerful semantic discovery of components.
*   **Modular Design:** Core components are decoupled, facilitating extension and testing.
*   **Dependency Injection:** Simplifies component wiring and initialization.
*   **Clear Logging:** Provides insights into agent thinking and component interactions.

## License

This project is licensed under the [Apache License Version 2.0](LICENSE).

## Acknowledgements

*   [BeeAI Framework](https://github.com/i-am-bee/beeai-framework): Used for the core `ReActAgent` implementation and tool structures.
*   [OpenAI Agents SDK](https://github.com/openai/openai-python/tree/main/src/openai/resources/beta): Integrated via providers for multi-framework support.
*   [ChromaDB](https://www.trychroma.com/): Powers semantic search capabilities in the `SmartLibrary` and `SmartAgentBus`.
*   Original Concept Contributors: [Matias Molinas](https://github.com/matiasmolinas) and [Ismael Faro](https://github.com/ismaelfaro)
