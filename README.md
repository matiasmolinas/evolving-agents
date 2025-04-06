# Evolving Agents Toolkit

**Build intelligent AI agent ecosystems that orchestrate complex tasks based on high-level goals.**

[![License: Apache v2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<!-- Add other badges like build status, PyPI version etc. here -->

This toolkit provides a robust, production-grade framework for building autonomous AI agents and multi-agent systems. It uniquely focuses on enabling agents to understand requirements, design solutions (potentially using specialized design agents like `ArchitectZero`), discover capabilities, evolve components, and **orchestrate complex task execution based on high-level goals**, all while operating within defined governance boundaries.

## Core Features

*   **Goal-Oriented Orchestration:** A central `SystemAgent` acts as the primary entry point, taking high-level goals and autonomously determining the steps needed, orchestrating component creation, communication, and execution.
*   **Intelligent Solution Design (Optional):** Agents like `ArchitectZero` can analyze requirements and design detailed multi-component solutions, providing blueprints for the `SystemAgent`.
*   **Internal Workflow Management:** For complex tasks, the `SystemAgent` can *internally* generate, process, and execute multi-step workflow plans using specialized tools (`GenerateWorkflowTool`, `ProcessWorkflowTool`), abstracting this complexity from the caller.
*   **Semantic Capability Discovery:** The `SmartAgentBus` allows agents to find and utilize capabilities based on natural language descriptions, enabling dynamic service discovery and routing via a logical "Data Bus".
*   **Ecosystem Management:** The `SmartAgentBus` also provides a logical "System Bus" for managing agent registration, health, and discovery.
*   **Intelligent Component Management:** The `SmartLibrary` provides persistent storage, semantic search (via vector embeddings), versioning, and evolution capabilities for agents and tools.
*   **Adaptive Evolution:** Components can be evolved based on requirements, feedback, or performance data using various strategies (standard, conservative, aggressive, domain adaptation), orchestrated by the `SystemAgent`.
*   **Multi-Framework Support:** Seamlessly integrate agents built with different frameworks (e.g., BeeAI, OpenAI Agents SDK) through a flexible provider architecture.
*   **Governance & Safety:** Built-in `Firmware` injects rules, and guardrails (like the `OpenAIGuardrailsAdapter`) ensure safe and compliant operation.
*   **Self-Building Potential:** The architecture allows agents (like `ArchitectZero` and `SystemAgent`) to collaboratively design and implement *new* agent systems based on user needs.

## Why This Toolkit?

While many frameworks focus on building individual agents, the Evolving Agents Toolkit focuses on creating **intelligent, self-improving agent ecosystems capable of handling complex tasks autonomously.** Key differentiators include:

1.  **High-Level Goal Execution:** Interact with the system via goals given to the `SystemAgent`, which then handles the "how" (planning, component management, execution).
2.  **Internal Orchestration:** Complex workflows (design -> generate -> process -> execute) are managed *internally* by the `SystemAgent`, abstracting the mechanics.
3.  **Semantic Capability Network:** `SmartAgentBus` creates a dynamic network where agents discover and interact based on function, not fixed names, via capability requests.
4.  **Deep Component Lifecycle Management:** Beyond creation, the `SmartLibrary` and evolution tools support searching, reusing, versioning, and adapting components intelligently.
5.  **Agent-Driven Ecosystem:** The `SystemAgent` isn't just a script runner; it's a `ReActAgent` using its own tools to manage the entire process, including complex workflow execution when needed.
6.  **True Multi-Framework Integration:** Provides abstractions (`Providers`, `AgentFactory`) to treat agents from different SDKs as first-class citizens.

## Key Concepts

### 1. Agent-Centric Orchestration (`SystemAgent`)

The `SystemAgent` acts as the central nervous system and primary entry point. It's a `ReActAgent` equipped with specialized tools to manage the entire ecosystem. It receives high-level goals and autonomously plans and executes the necessary steps.

**Example: Prompting the `SystemAgent` with a high-level goal**
```python
# Define the high-level task for the System Agent
invoice_content = "..." # Load or define the invoice text here

high_level_prompt = f"""
**Goal:** Accurately process the provided invoice document and return structured, verified data.

**Functional Requirements:**
- Extract key fields: Invoice #, Date, Vendor, Bill To, Line Items (Description, Quantity, Unit Price, Item Total), Subtotal, Tax Amount, Shipping (if present), Total Due, Payment Terms, Due Date.
- Verify calculations: The sum of line item totals should match the Subtotal. The sum of Subtotal, Tax Amount, and Shipping (if present) must match the Total Due. Report any discrepancies.

**Non-Functional Requirements:**
- High accuracy is critical.
- Output must be a single, valid JSON object containing the extracted data and a 'verification' section (status: 'ok'/'failed', discrepancies: list).

**Input Data:**

{invoice_content}


**Action:** Achieve this goal using the best approach available. Create, evolve, or reuse components as needed. Return ONLY the final JSON result.
"""

# Execute the task via the SystemAgent
final_result_obj = await system_agent.run(high_level_prompt)

# Process the final result (assuming extract_json_from_response is defined elsewhere)
# final_json_result = extract_json_from_response(final_result_obj.result.text)
# print(final_json_result)
```

***SystemAgent Internal Process (Conceptual):***

When the `SystemAgent` receives the `high_level_prompt`, its internal ReAct loop orchestrates the following:

1.  **Receives** the high-level goal and input data.
2.  **Analyzes** the goal using its reasoning capabilities.
3.  **Uses Tools** (`SearchComponentTool`, `DiscoverAgentTool`) to find existing capabilities suitable for the task.
4.  **Decides If Workflow Needed:** If the task is complex or no single component suffices, it determines a multi-step plan is necessary.
    *   *(Optional)* It might internally request a detailed design blueprint from `ArchitectZero` using `RequestAgentTool`.
    *   It uses `GenerateWorkflowTool` internally to create an executable YAML workflow based on the design or its analysis.
    *   It uses `ProcessWorkflowTool` internally to parse the YAML into a step-by-step execution plan.
5.  **Executes the Plan:** It iterates through the plan, using appropriate tools for each step:
    *   `CreateComponentTool` or `EvolveComponentTool` for `DEFINE` steps.
    *   Internal `AgentFactory` calls during component creation.
    *   `RequestAgentTool` for `EXECUTE` steps, invoking other agents/tools via the `SmartAgentBus`.
6.  **Returns Result:** It returns the final result specified by the plan's `RETURN` step (or the result of a direct action if no complex workflow was needed).

*(This internal complexity is hidden from the user interacting with the `SystemAgent`)*.

### 2. Solution Design (`ArchitectZero`)

`ArchitectZero` is a specialized agent, typically invoked *by the SystemAgent via the Agent Bus* when a complex task requires a detailed plan before execution.

```python
# Conceptual: SystemAgent requesting design from ArchitectZero via AgentBus
# This happens INTERNALLY within the SystemAgent's ReAct loop if needed.
design_request_prompt = f"""
Use RequestAgentTool to ask ArchitectZero to design an automated customer support system.
Requirements: Handle FAQs, escalate complex issues, use sentiment analysis.
Input for ArchitectZero: {{ "requirements": "Design customer support system..." }}
"""
# system_agent_internal_response = await system_agent.run(design_request_prompt)
# solution_design_json = extract_json_from_response(...)
# >> SystemAgent now has the design to proceed internally.
```
*Note: End users typically interact with the `SystemAgent` directly with their goal, not `ArchitectZero`.*

### 3. Smart Library (Component Management & Discovery)

Stores agents, tools, and firmware definitions. Enables semantic search and lifecycle management.

```python
# Semantic component discovery (often used internally by SystemAgent)
similar_tools = await smart_library.semantic_search(
    query="Tool that can validate financial calculations in documents",
    record_type="TOOL",
    domain="finance",
    threshold=0.6
)

# Evolve an existing component (often invoked by SystemAgent's EvolveComponentTool)
evolved_record = await smart_library.evolve_record(
    parent_id="tool_id_abc",
    new_code_snippet="# New improved Python code...",
    description="Enhanced version with better error handling"
)
```

### 4. Smart Agent Bus (Capability Routing & Ecosystem Management)

Enables dynamic, capability-based communication ("Data Bus") and provides system management functions ("System Bus").

```python
# Register a component (System Bus operation, often via SystemAgent's tool)
await agent_bus.register_agent(
    name="SentimentAnalyzerTool_v2",
    agent_type="TOOL",
    description="Analyzes text sentiment with high accuracy",
    capabilities=[{ "id": "sentiment_analysis", "name": "Sentiment Analysis", ... }]
)

# Request a service based on capability (Data Bus operation, often via SystemAgent's RequestAgentTool)
result_payload = await agent_bus.request_capability(
    capability="sentiment_analysis",
    content={"text": "This service is amazing!"},
    min_confidence=0.8
)
# >> result_payload might contain: {'agent_id': '...', 'agent_name': 'SentimentAnalyzerTool_v2', 'content': {'sentiment': 'positive', 'score': 0.95}, ...}
```

### 5. Workflow Lifecycle (Managed Internally by `SystemAgent`)

Complex tasks are handled via a structured workflow lifecycle orchestrated *internally* by the `SystemAgent`.

1.  **Goal Intake:** `SystemAgent` receives a high-level goal from the user/caller.
2.  **Analysis & Planning (Internal):** `SystemAgent` analyzes the goal and checks if existing components suffice.
3.  **Design Query (Optional/Internal):** If needed, `SystemAgent` requests a *solution design* (JSON) from `ArchitectZero` via the Agent Bus.
4.  **Workflow Generation (Internal):** If a multi-step plan is required, `SystemAgent` uses its `GenerateWorkflowTool` to translate the design (or its internal analysis) into an executable YAML workflow string.
5.  **Plan Processing (Internal):** `SystemAgent` uses its `ProcessWorkflowTool` to parse the YAML, validate, substitute parameters, and produce a structured *execution plan*.
6.  **Plan Execution (Internal):** `SystemAgent`'s ReAct loop iterates through the plan, using its *other* tools (`CreateComponentTool`, `EvolveComponentTool`, `RequestAgentTool`, etc.) to perform the action defined in each step (`DEFINE`, `CREATE`, `EXECUTE`).
7.  **Result Return:** `SystemAgent` returns the final result to the caller.

*The external caller interacts only at Step 1 and receives the result at Step 7, unaware of the internal workflow mechanics.*

### 6. Component Evolution

Existing components can be adapted or improved using the `EvolveComponentTool` (typically invoked internally by `SystemAgent`).

```python
# Conceptual Example (within the SystemAgent's internal operation)
evolve_prompt = f"""
Use EvolveComponentTool to enhance agent 'id_123'.
Changes needed: Add support for processing PDF files directly.
Strategy: standard
"""
# evolve_result = await system_agent.run(evolve_prompt)
# >> evolve_result indicates success and provides ID of the new evolved agent version.
```

### 7. Multi-Framework Support

Integrate agents/tools from different SDKs via `Providers` managed by the `AgentFactory` (used internally by tools like `CreateComponentTool`).

```python
# Example: Creating agents from different frameworks via AgentFactory
# (AgentFactory is usually used internally by tools like CreateComponentTool)

# bee_record = await smart_library.find_record_by_name("BeeAgentName")
# openai_record = await smart_library.find_record_by_name("OpenAIAgentName")

# if bee_record:
#     bee_agent_instance = await agent_factory.create_agent(bee_record)
#     # >> Uses BeeAIProvider internally

# if openai_record:
#     openai_agent_instance = await agent_factory.create_agent(openai_record)
#     # >> Uses OpenAIAgentsProvider internally
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
    This demo initializes the framework and gives the `SystemAgent` a high-level goal to process an invoice, requiring design, component creation/evolution, and execution orchestrated internally.

    ```bash
    python examples/invoice_processing/architect_zero_comprehensive_demo.py
    ```

3.  **Explore Output:** Check the generated files:
    *   `final_processing_output.json`: Contains the final structured result from the SystemAgent executing the task, along with the agent's full output log for debugging.
    *   `smart_library_demo.json`: The state of the component library after the run (shows created/evolved components).
    *   `smart_agent_bus_demo.json`: The agent registry state.
    *   `agent_bus_logs_demo.json`: Logs of agent interactions via the bus.
    *   *(Optional Debug)* `architect_design_output.json`: The demo still saves the design blueprint generated internally by ArchitectZero for inspection.

## Example Applications

Explore the `examples/` directory:

*   **`invoice_processing/architect_zero_comprehensive_demo.py`**: The flagship demo showing the `SystemAgent` handling a complex invoice processing task based on a high-level goal, orchestrating design, generation, and execution internally.
*   **`agent_evolution/`**: Demonstrates creating and evolving agents/tools using both BeeAI and OpenAI frameworks.
*   **`forms/`**: Shows how the system can design and process conversational forms.
*   **`autocomplete/`**: Illustrates designing a context-aware autocomplete system.
*   *(Add more examples as they are created)*

## Architecture Overview

The toolkit employs an **agent-centric architecture**. The `SystemAgent` (a ReAct agent) is the main orchestrator, taking high-level goals. It leverages specialized tools to interact with core components like the `SmartLibrary` (for component persistence and semantic search via ChromaDB) and the `SmartAgentBus` (for capability-based routing and system management). For complex tasks, it internally manages the full workflow lifecycle, potentially requesting designs from agents like `ArchitectZero` and using internal tools to generate, process, and execute plans. Multi-framework support is achieved through `Providers` and `Adapters`. Dependencies are managed via a `DependencyContainer`.

For a detailed breakdown, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Development Features

*   **LLM Caching:** Reduces API costs during development by caching completions and embeddings (`.llm_cache_demo/`).
*   **Vector Search:** Integrated ChromaDB for powerful semantic discovery of components.
*   **Modular Design:** Core components are decoupled, facilitating extension and testing.
*   **Dependency Injection:** Simplifies component wiring and initialization.
*   **Clear Logging:** Provides insights into agent thinking and component interactions via bus logs and standard logging.

## License

This project is licensed under the [Apache License Version 2.0](LICENSE).

## Acknowledgements

*   [BeeAI Framework](https://github.com/i-am-bee/beeai-framework): Used for the core `ReActAgent` implementation and tool structures.
*   [OpenAI Agents SDK](https://platform.openai.com/docs/assistants/overview): Integrated via providers for multi-framework support.
*   [ChromaDB](https://www.trychroma.com/): Powers semantic search capabilities in the `SmartLibrary` and `SmartAgentBus`.
*   Original Concept Contributors: [Matias Molinas](https://github.com/matiasmolinas) and [Ismael Faro](https://github.com/ismaelfaro)