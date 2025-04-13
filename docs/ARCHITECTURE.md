# Evolving Agents Toolkit - Architecture

This document details the architectural design of the Evolving Agents Toolkit, focusing on its core components, interactions, and design philosophy.

## 1. Introduction & Philosophy

The Evolving Agents Toolkit aims to provide a robust framework for building *ecosystems* of autonomous AI agents, rather than just individual agents. The core philosophy is **agent-centric**: the system itself is managed and orchestrated by specialized agents (like the `SystemAgent`), which leverage tools to interact with underlying services and manage other components.

Key goals of the architecture include:

*   **Autonomy & Evolution:** Enable agents and components to be created, evaluated, and improved over time, potentially automatically.
*   **Modularity & Reusability:** Promote the reuse of components (agents, tools) through discovery and adaptation via the `SmartLibrary`.
*   **Interoperability:** Support agents and tools built with different underlying frameworks (e.g., BeeAI, OpenAI Agents SDK) through a provider pattern.
*   **Decoupled Communication:** Facilitate communication based on *capabilities* rather than direct references using the `SmartAgentBus`.
*   **Governance & Safety:** Embed safety and ethical considerations through `Firmware` and guardrails.
*   **Task-Relevant Context:** Provide agents with context that is not just topically similar but specifically relevant to the task they are performing (leveraging the implemented Dual Embedding Strategy in `SmartLibrary` and `SmartContext`).
*   **Orchestration:** Provide mechanisms for achieving complex goals. The `SystemAgent` handles complex, multi-step tasks by potentially designing (via `ArchitectZero`), generating, processing, and executing workflows *internally*. External callers interact via high-level goals.

## 2. Core Components

The toolkit is composed of several key interacting components:

### 2.1. SystemAgent

The central orchestrator and primary entry point of the ecosystem.

*   **Implementation:** A `beeai_framework.agents.react.ReActAgent`.
*   **Role:** Receives high-level goals or tasks and determines the best execution strategy. It manages component lifecycles (search, create, evolve via tools), facilitates communication (via Agent Bus tools), and handles complex task execution. It uses `SmartContext` to understand the task at hand and leverages the `SmartLibrary`'s task-aware search to find relevant components. For multi-step tasks, it *internally* orchestrates the use of its specialized workflow tools (`GenerateWorkflowTool`, `ProcessWorkflowTool`) to create and execute a plan.
*   **Key Tools:**
    *   **SmartLibrary Tools:** `SearchComponentTool`, `CreateComponentTool`, `EvolveComponentTool`, `TaskContextTool`, `ContextualSearchTool` for managing components and task-aware context.
    *   **AgentBus Tools:** `RegisterAgentTool`, `RequestAgentTool`, `DiscoverAgentTool` for managing agent registration and communication/execution.
    *   **Workflow Tools (Internal Use):**
        *   `GenerateWorkflowTool`: Translates solution designs (often obtained internally, e.g., from ArchitectZero via the AgentBus) into executable YAML workflow strings. *Not typically called directly by external users.*
        *   `ProcessWorkflowTool`: Parses workflow YAML, substitutes parameters, validates structure, and produces a step-by-step execution plan for the SystemAgent's ReAct loop. *Not typically called directly by external users.*
    *   **(Optional) Framework-Specific Tools:** Tools for interacting directly with specific frameworks (e.g., `CreateOpenAIAgentTool`, `EvolveOpenAIAgentTool`).

```mermaid
graph TD
    UserGoal["High-Level Goal / Task"] --> SA["SystemAgent (ReActAgent)"]

    subgraph System Agent Internal Orchestration
        direction LR
        SA --> |Uses| SLT["SmartLibrary Tools"]
        SA --> |Uses| SBT["AgentBus Tools"]
        SA --> |Uses Internally| WT["Workflow Tools"]
        SA --> |Uses Optional| FST["Framework-Specific Tools"]

        SLT -->|Interacts| SL["Smart Library"]
        SBT -->|Interacts| SB["Smart Agent Bus"]
        WT ---|Generates/Parses| WorkflowPlan["Internal Workflow Plan"]
        FST -->|Interacts| ExternalSDKs["External Agent SDKs"]

        SA --> |Executes Steps| SLT
        SA --> |Executes Steps| SBT
        SA --> |Executes Steps| FST
    end

     subgraph "External Interaction (Optional/Internal)"
         SA --> |Requests via Bus| ArchZ["ArchitectZero (Optional Design)"]
         SB --> ArchZ
         ArchZ --> SB
         SB --> SA
     end

    SA --> FinalResult["Final Task Result"]

    style SA fill:#ccf,stroke:#333,stroke-width:2px
```

### 2.2. ArchitectZero Agent

A specialized agent responsible for *designing* solutions, typically invoked by the `SystemAgent` via the `SmartAgentBus` when needed.

*   **Implementation:** Typically a `ReActAgent` (as shown in `agents/architect_zero.py`).
*   **Role:** Analyzes high-level requirements, queries the `SmartLibrary`, designs multi-component solutions, and specifies components.
*   **Output:** Produces a structured *solution design* (e.g., JSON). This design serves as a blueprint *for the SystemAgent* to understand how to orchestrate component creation and execution, potentially involving internal workflow generation.
*   **Key Tools (Internal):** `AnalyzeRequirementsTool`, `DesignSolutionTool`, `ComponentSpecificationTool`.

### 2.3. Smart Library

The persistent storage and discovery mechanism for all reusable components.

*   **Stores:** Agents, Tools, Firmware definitions as structured records (typically JSON). Each record includes original content/code (`T_orig`) and potentially LLM-generated applicability text (`T_raz`).
*   **Discovery (Dual Embedding Implemented):**
    *   Stores two embeddings per record: `E_orig` (for `T_orig` content) and `E_raz` (for `T_raz` applicability text).
    *   Supports **Task-Aware Semantic Search**: Finds components based on a **Task-Context Query** by primarily searching against `E_raz` (applicability) and optionally refining results using `E_orig` (content) similarity. This provides context relevant to the *specific task* being performed. (See `tools.smart_library.search_component_tool.py` and `smart_library.py`).
    *   Also supports standard semantic search (primarily using `E_orig`).
*   **Versioning & Evolution:** Tracks component versions and parentage, facilitating evolution (`evolve_record`).
*   **Interface:** Provides methods like `create_record`, `find_record_by_id`, `semantic_search` (standard), `task_aware_search` (dual embedding implemented via semantic_search with task_context parameter), `evolve_record`.
*   **Backends:** Pluggable storage (JSON file default) and vector database (ChromaDB default).
*   **Indexing Pipeline:** Includes a step to generate `T_raz` using an LLM (`generate_applicability_text` in `LLMService`) and compute/store both `E_orig` and `E_raz` (`_sync_vector_db`).

```mermaid
graph TD
    SLI["SmartLibrary Interface"] -->|Manages| Records["Component Records (JSON)\n(T_orig, T_raz)"]
    SLI -->|Uses| VDB["Vector Database (ChromaDB)"]:::infra
    VDB -->|Stores| DualEmbeddings["Dual Embeddings (E_orig, E_raz)"]
    SLI -->|Uses| LLMEmb["LLM Embedding Service"]
    SLI -->|Uses| LLMReason["LLM Reasoning Service (for T_raz)"]

    subgraph Storage & Search
        direction LR
        Records -- Stored in --> StorageBackend["JSON File / Database"]
        DualEmbeddings -- Stored in --> VectorDBBackend["ChromaDB / Other Vector Store"]
    end

    LLMEmb -- Generates --> DualEmbeddings
    LLMReason -- Generates --> Records[T_raz]

    classDef infra fill:#eee,stroke:#666,stroke-width:1px,color:#333;
```

### 2.4. Smart Agent Bus (Service Bus)

Manages inter-agent communication and capability discovery/execution. Implements a logical **Dual Bus** structure:

*   **System Bus:** Handles system-level operations like agent registration (`register_agent`), discovery (`discover_agents`), health monitoring (`_check_circuit_breaker`), and status retrieval (`get_agent_status`). These operations focus on managing the agent ecosystem. Logging for these events is marked with `bus_type='system'`. Direct execution via `execute_with_agent` is also considered a system-level/debug operation.
*   **Data Bus:** Handles agent-to-agent communication and task execution via capability requests (`request_capability`). This is the primary channel for agents (including the `SystemAgent`) to interact and delegate tasks based on function. Logging for these events is marked with `bus_type='data'`.

*   **Role:** Acts as a central "nervous system" allowing agents to request services based on capability descriptions rather than specific agent names.
*   **Discovery:** Uses semantic matching (via embeddings generated by `LLMService`) to find registered providers (agents/tools) that match a requested capability query.
*   **Routing:** Directs requests to the best-matching, healthy provider.
*   **Resilience:** Implements circuit breakers to temporarily disable failing providers.
*   **Monitoring:** Logs agent executions and interactions (distinguishing System vs. Data Bus).
*   **Interface:** `register_agent`, `discover_agents`, `request_capability`, `get_agent_status`, `health_check`.

```mermaid
graph TD
    SABI["AgentBus Interface"] -->|Manages| Reg["Agent Registry (JSON/DB)"]
    SABI -->|Uses| CapDB["Capability Index (ChromaDB)"]
    CapDB -->|Stores| CapEmbeds["Capability Embeddings"]
    SABI -->|Uses| LLMEmb["LLM Embedding Service"]
    SABI -->|Monitors| CB["Circuit Breakers"]
    SABI -->|Logs to| ExecLogs["Execution Logs (System & Data)"]

    subgraph "Data Bus Communication (request_capability)"
        direction LR
        Requester --> SABI
        SABI -->|Finds Provider| Reg
        SABI -->|Routes Request| Provider["Agent/Tool Instance"]
        Provider -->|Returns Result| SABI
        SABI -->|Returns Result| Requester
    end
```

### 2.5. Smart Context

A data structure used to pass relevant information between agents and tools within a workflow or task execution.

*   **Role:** Carries task-specific data, user input, intermediate results, and importantly, the **current task context description**.
*   **Task Relevance:** The current task context stored within `SmartContext` is used by components like `SystemAgent` and `SearchComponentTool` to perform task-aware retrieval from the `SmartLibrary`.
*   **Dual Embedding Interaction:** While `SmartContext` holds the task description, the dual embeddings themselves reside within the `SmartLibrary`. The context enables the *querying* mechanism to leverage those embeddings effectively.

### 2.6. Providers & Agent Factory

Abstract interaction with different underlying agent frameworks.

*   **`FrameworkProvider` (Abstract Base Class):** Defines the interface (`create_agent`, `execute_agent`, `supports_framework`) that concrete providers (e.g., `BeeAIProvider`, `OpenAIAgentsProvider`) must implement.
*   **`ProviderRegistry`:** Holds instances of available providers.
*   **`AgentFactory`:** Used internally (e.g., by `CreateComponentTool` or the `SystemAgent`) via the `ProviderRegistry` to find the correct provider for a given framework and delegate agent creation/execution to that provider. Allows the rest of the system to work with agents generically.

```mermaid
graph TD
    AF["Agent Factory"] -->|Used by SystemAgent/Tools| PR["Provider Registry"]
    PR -->|Contains| BP["BeeAI Provider"]
    PR -->|Contains| OP["OpenAI Provider"]
    PR -->|Contains| FP["Future Providers..."]

    AF -->|Delegates Create/Execute| BP
    AF -->|Delegates Create/Execute| OP

    style BP fill:#f9d,stroke:#333,stroke-width:2px
    style OP fill:#ccf,stroke:#333,stroke-width:2px
```

### 2.7. Dependency Container

Manages the instantiation and wiring of core components.

*   **Role:** Handles dependency injection to avoid circular imports and manage the initialization lifecycle. Ensures components like `SmartLibrary`, `AgentBus`, `LLMService`, `SystemAgent` receive their required dependencies during setup.

### 2.8. LLM Service

*   **Role:** Provides interface to underlying LLM models for text generation, standard embeddings (`E_orig`), applicability embeddings (`E_raz`), and reasoning capabilities needed for generating applicability text (`T_raz`). Includes caching capabilities. See `core/llm_service.py`.

### 2.9. Firmware

Provides governance rules and operational constraints.

*   **Role:** Injects safety guidelines, ethical constraints, and domain-specific rules into agents and tools during creation or evolution.
*   **Mechanism:** Typically provides prompts or configuration data used by `CreateComponentTool`, `EvolveComponentTool`, and framework providers (e.g., via `OpenAIGuardrailsAdapter`).

### 2.10. Adapters

Bridge different interfaces or formats.

*   **`OpenAIGuardrailsAdapter`:** Converts `Firmware` rules into OpenAI Agents SDK guardrail functions.
*   **`OpenAIToolAdapter`:** Converts Evolving Agents/BeeAI tools into a format compatible with the OpenAI Agents SDK `function_tool`.
*   **`OpenAITracingAdapter`:** (Optional) Integrates OpenAI Agents SDK tracing with the toolkit's monitoring.

## 3. Key Architectural Patterns & Flows

### 3.1. Agent Communication (via Agent Bus)

Agents interact based on *what* needs to be done (capability), not *who* does it, primarily using the Data Bus (`request_capability`).

```mermaid
sequenceDiagram
    participant AA as Agent A (Requester)
    participant SB as Smart Agent Bus
    participant CR as Capability Registry (in Bus)
    participant AB as Agent B (Provider, Not selected)
    participant AC as Agent C (Provider, Selected)

    AA->>SB: request_capability("Analyze Sentiment", data)
    SB->>CR: Find Providers for "Analyze Sentiment" (Semantic Search)
    CR->>SB: Return [Agent C (0.9), Agent B (0.7)]
    SB->>AC: Route Request(data) [Selects best healthy provider]
    AC->>AC: Process Request
    AC->>SB: Return Result(sentiment_score)
    SB->>AA: Return Result(sentiment_score)
```

### 3.2. Task-Aware Context Retrieval (Dual Embedding Strategy - Implemented)

When an agent (e.g., `SystemAgent`) needs context for a specific task, it leverages the dual embedding capability of the `SmartLibrary` via tools like `ContextualSearchTool` or by passing `task_context` to `SearchComponentTool`.

1.  **Query Formulation:** The agent formulates a query (`query`) and provides a description of its current task (`task_context`).
2.  **Query Embedding:** The `SmartLibrary` (or `LLMService`) embeds the `query` (for content relevance) and the `task_context` (for applicability relevance).
3.  **Applicability Search:** The `SmartLibrary` searches its vector index using the `task_context` embedding against the stored Applicability Embeddings (`E_raz`). This retrieves chunks whose *inferred applicability* matches the task.
4.  **Refinement & Scoring:** The library combines the task relevance score (from `E_raz` match) and the content relevance score (similarity between `query` embedding and `E_orig`) using a weighting mechanism (e.g., `task_weight`). This produces a `final_score`. Usage metrics may also boost the score.
5.  **Content Retrieval:** The `SmartLibrary` fetches the original text content (`T_orig`) for the final ranked list of chunk identifiers based on `final_score`.
6.  **Context Provision:** The agent receives the highly relevant `T_orig` chunks tailored to its specific task.

```mermaid
sequenceDiagram
    participant Agent as Requesting Agent (e.g., SystemAgent)
    participant SL as Smart Library
    participant VDB as Vector DB (Dual Embeddings)
    participant LLM_Emb as Embedding Service

    Agent->>Agent: Formulate Query & Task Context
    Agent->>LLM_Emb: Embed Query (E_orig Model) & Task Context (E_raz Model)
    LLM_Emb-->>Agent: Query Embeddings (E_orig, E_raz)

    Agent->>SL: search(Query, Task Context, ...)

    SL->>VDB: Search E_raz Index with Task Context Embedding
    VDB-->>SL: Top-K Candidate Chunk IDs (Task Relevant)

    SL->>VDB: Retrieve E_orig for Candidates
    SL->>SL: Calculate Content Similarity (Query E_orig vs Candidate E_orig)
    SL->>SL: Combine Task Score & Content Score (Weighted) -> Final Score
    SL->>SL: Apply Usage Boost
    SL->>SL: Sort/Filter by Final Score

    SL->>SL: Fetch T_orig for Final Candidates
    SL-->>Agent: Return Relevant T_orig Chunks & Scores
```

### 3.3. Workflow Generation & Execution (Orchestrated by SystemAgent)

Complex tasks requiring multiple steps or new components are handled internally by the `SystemAgent`. The external caller simply provides the high-level goal.

1.  **Goal Intake:** An external caller (User or another Agent) provides a high-level goal and necessary input data to the `SystemAgent`.
2.  **Analysis & Planning (Internal):** The `SystemAgent`'s ReAct loop analyzes the goal. It uses its tools (`SearchComponentTool`, `DiscoverAgentTool`) potentially with task context to check if existing, suitable components can achieve the goal directly.
3.  **Design & Workflow Decision (Internal):**
    *   *If* the task is complex, requires multiple steps, or necessitates new/evolved components:
        *   The `SystemAgent` *may* internally request a *solution design* from `ArchitectZero` (using `RequestAgentTool` on the Agent Bus).
        *   Based on the design (or internal analysis), it uses its `GenerateWorkflowTool` to create an executable YAML workflow string.
        *   It then uses its `ProcessWorkflowTool` to parse the YAML and create a structured *execution plan* (list of steps).
    *   *Else* (if a direct execution path exists): The `SystemAgent` proceeds to Step 4 using a simple plan (e.g., one `EXECUTE` step).
4.  **Plan Execution (Internal):** The `SystemAgent`'s ReAct loop iterates through the execution plan (if generated). For each step, it uses the appropriate tool (`CreateComponentTool` for `DEFINE`, `AgentFactory` via tool for `CREATE`, `RequestAgentTool` for `EXECUTE`, etc.) to perform the action. Data is passed between steps using context variables managed by the agent.
5.  **Result Return:** The `SystemAgent` returns the final result (as specified by the plan's `RETURN` step or the direct execution) to the original caller.

**Key Point:** Steps 3 and 4 (Design, Generate, Process, Execute Plan) are internal mechanisms of the `SystemAgent` and are abstracted away from the external caller.

```mermaid
sequenceDiagram
    participant Caller as User / Agent
    participant SysA as SystemAgent
    participant InternalTools as SysA Internal Tools (All)
    participant AgentBus as Smart Agent Bus
    participant ArchZ as ArchitectZero (Via Bus)
    participant Library as Smart Library
    participant TargetComp as Target Component (Via Bus)

    Caller->>SysA: Run(High-Level Goal, Input Data)

    SysA->>SysA: Analyze Goal (ReAct Loop)
    SysA->>InternalTools: Use Search/Discover Tools (Check Library/Bus, potentially with Task Context)
    InternalTools-->>Library: Query
    InternalTools-->>AgentBus: Query
    Library-->>InternalTools: Component Info
    AgentBus-->>InternalTools: Capability Info
    InternalTools-->>SysA: Analysis Result

    alt Task is Complex or Needs Design
        SysA->>SysA: Decide Design/Workflow Needed
        SysA->>InternalTools: Use RequestAgentTool (Request Design from ArchZ)
        InternalTools->>AgentBus: request_capability('solution_design', ...)
        AgentBus->>ArchZ: Route Request
        ArchZ->>ArchZ: Generate Design
        ArchZ->>AgentBus: Return Design JSON
        AgentBus->>InternalTools: Design Result
        InternalTools-->>SysA: Solution Design Received

        SysA->>SysA: Use GenerateWorkflowTool (Internal)
        SysA->>SysA: Use ProcessWorkflowTool (Internal)
        SysA->>SysA: Obtain Execution Plan (List of Steps)
    else Task is Simple / Direct Execution
        SysA->>SysA: Create Simple Plan (e.g., one EXECUTE step)
    end

    loop For Each Step in Plan
        SysA->>SysA: Analyze Step (e.g., type=CREATE, name=CompX)
        SysA->>InternalTools: Use Appropriate Tool (CreateTool, RequestTool, etc.)
        InternalTools-->>Library: (Create/Evolve Record)
        InternalTools-->>AgentBus: (Execute Capability on TargetComp)
        AgentBus-->>TargetComp: Route Request
        TargetComp-->>AgentBus: Result
        AgentBus-->>InternalTools: Result
        InternalTools-->>SysA: Step Result
        SysA->>SysA: Update Context/Variables
    end

    SysA->>Caller: Return Final Workflow/Task Result
```

### 3.4. Component Evolution

Components can be improved or adapted, typically orchestrated by the `SystemAgent` using the `EvolveComponentTool`.

1.  User or an agent (like `SystemAgent`) identifies a need to evolve a component (e.g., `ComponentA_v1`).
2.  The `EvolveComponentTool` is invoked (usually by `SystemAgent`) with the `parent_id`, description of `changes`, and potentially `new_requirements` and an `evolution_strategy`.
3.  The tool uses the `LLMService` to generate a new `code_snippet` based on the original code, changes, strategy, and firmware.
4.  The tool calls `SmartLibrary.evolve_record` to create a new record (`ComponentA_v2`) linked to the parent, saving the new code and incrementing the version. This also triggers indexing of the new version, including T_raz generation and dual embedding.

```mermaid
graph TD
    Start --> Need{Need for Evolution Identified}
    Need -->|Invoke via SystemAgent| ECT["EvolveComponentTool.run(parent_id, changes, strategy)"]
    ECT -->|Gets Original| SL["Smart Library"]
    ECT -->|Uses| LLM["LLM Service"]
    ECT -->|Applies| FW["Firmware Rules"]
    LLM -->|Generates| NewCode["New Code Snippet"]
    ECT -->|Calls| SLevolve["SmartLibrary.evolve_record(...)"]
    SLevolve -->|Creates| NewRecord["New Component Record (v2)"]
    NewRecord -->|Links To| ParentRecord["Parent Record (v1)"]
    SLevolve -->|Returns| EvolvedInfo["Evolved Record Info"]
    EvolvedInfo --> End

    subgraph Evolution Strategies
        strategy["(e.g., standard, conservative, aggressive, domain_adaptation)"]
    end
    ECT -->|Selects| strategy
```

### 3.5. Dependency Injection & Initialization

Managed by the `DependencyContainer`.

1.  **Registration Phase:** Core components (`LLMService`, `SmartLibrary`, `AgentBus`, `SystemAgent`, `ArchitectZero`, etc.) are instantiated (often by factories using the container for *their* dependencies) and registered with the container.
2.  **Wiring Phase:** Dependencies are resolved. For example, when `SystemAgent` is created, its factory gets `LLMService`, `SmartLibrary`, `AgentBus`, etc. from the container. Circular dependencies are handled (e.g., `AgentBus` might get the `SystemAgent` instance after it's created).
3.  **Initialization Phase:** Components perform setup that requires their dependencies to be present (e.g., `AgentBus.initialize_from_library` is called after the `SmartLibrary` and `SystemAgent` are available).

## 4. Multi-Framework Integration

The Provider pattern (`providers/`, `AgentFactory`) is key to supporting different agent frameworks.

*   `AgentFactory` uses `ProviderRegistry` to select the correct `FrameworkProvider`.
*   The `FrameworkProvider` handles framework-specific details of agent creation (e.g., initializing `beeai_framework.ReActAgent` vs. `agents.Agent`) and execution.
*   `Adapters` help bridge specific components like tools (`OpenAIToolAdapter`) and governance (`OpenAIGuardrailsAdapter`) between the toolkit's concepts and the specific SDK requirements.

## 5. Governance and Safety

Integrated via the `Firmware` component and `SmartAgentBus` health checks.

*   `Firmware` provides baseline ethical and safety rules.
*   Allows defining domain-specific constraints (medical, finance).
*   Rules are injected into prompts during component creation/evolution.
*   Guardrails (especially for OpenAI Agents via the adapter) can enforce rules at runtime.
*   `AgentBus` circuit breakers prevent cascading failures by temporarily disabling unhealthy agents.

This architecture promotes a flexible, extensible, and governable system for building complex AI agent solutions capable of adaptation, task-aware context retrieval, and self-improvement, orchestrated primarily through the `SystemAgent` interacting with high-level goals.