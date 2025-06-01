# Evolving Agents Toolkit - Architecture

This document details the architectural design of the Evolving Agents Toolkit (EAT), focusing on its core components, interactions, its unified **MongoDB backend**, and the role of the **Smart Memory ecosystem** in enabling advanced autonomous learning and evolution.

## 1. Introduction & Philosophy

The Evolving Agents Toolkit aims to provide a robust framework for building *ecosystems* of autonomous AI agents. The core philosophy is **agent-centric**: the system itself is managed and orchestrated by specialized agents (like the `SystemAgent`), which leverage tools to interact with underlying services and manage other components. All primary data, including component metadata, embeddings, agent registrations, operational logs, LLM caches, intent plans, and **detailed agent experiences (Smart Memory)**, is now persisted in **MongoDB**.

Key goals of the architecture include:

*   **Autonomy & Evolution:** Enable agents, components, and even system strategies to be created, evaluated, and improved over time, significantly informed by past experiences.
*   **Modularity & Reusability:** Promote component reuse through discovery and adaptation via the `SmartLibrary`.
*   **Interoperability:** Support agents and tools built with different underlying frameworks.
*   **Decoupled Communication:** Facilitate capability-based communication via the `SmartAgentBus`.
*   **Governance & Safety:** Embed safety through `Firmware` and an optional human-in-the-loop review process.
*   **Deep Contextual Understanding:** Provide agents with rich, task-relevant context dynamically constructed from `SmartLibrary`, current task data, and **historical experiences from Smart Memory**.
*   **Orchestration:** Enable complex goal achievement through `SystemAgent`-driven workflow generation and execution.
*   **Cumulative Learning & Self-Improvement:** The **Smart Memory ecosystem** is central to enabling agents, particularly `SystemAgent`, to learn from past workflows, decisions, and outcomes. This learning directly informs better problem-solving, more effective component evolution, and potentially the evolution of the system's own operational strategies.
*   **Unified & Scalable Backend:** Utilize MongoDB for all persistent data.

---
```mermaid
graph TD
    User["User / External System"] -- High-Level Goal --> SA[("SystemAgent\n(Orchestrator, Learner)")];;;agent

    subgraph "Core Infrastructure & Services (MongoDB Backend)"
        direction LR
        SL["Smart Library\n(Components, Versions)"];;;service
        SM["Smart Memory\n(Agent Experiences - `eat_agent_experiences`)"];;;service
        SB["Smart Agent Bus\n(Discovery, Routing - `eat_agent_registry`, `eat_agent_bus_logs`)"];;;service
        LLMS["LLM Service\n(Reasoning, Embeddings, Generation - Cache: `eat_llm_cache`)"];;;service
        FW["Firmware\n(Governance Rules)"];;;service
        MongoDB[("MongoDB Atlas / Server\n(Primary Data Store, Vector Search)")]:::infra
    end
    
    subgraph "Key Agents & Factories"
        direction LR
        MMA[("MemoryManagerAgent\n(Manages Smart Memory via Bus)")];;;agent
        AgentF["Agent Factory"];;;infra
        ToolF["Tool Factory"];;;infra
        ArchZ["ArchitectZero\n(Optional: Solution Design)"];;;agent
        EvoS["EvolutionStrategistAgent\n(Optional: Proactive Evolution Suggestions)"];;;agent
    end

    %% SystemAgent Core Interactions
    SA -- Uses --> ToolsSystem["SystemAgent Tools\n(Search, Create, Evolve, Request, Workflow, IntentReview, ContextBuilder, ExperienceRecorder...)"];;;tool
    SA -- Uses --> LLMS
    SA -- Relies on --> AgentF
    SA -- Relies on --> ToolF

    %% Smart Memory Interactions
    ToolsSystem -- Records/Retrieves Experiences via Bus --> MMA
    MMA -- Manages --> SM
    SM -- Stores/Retrieves --> MongoDB
    MMA -- Uses Internal Tools --> LLMS % For summarization, embedding (if not done by MongoExperienceStoreTool)

    %% SmartLibrary Interactions
    ToolsSystem -- Manages Components --> SL
    SL -- Stores/Retrieves --> MongoDB
    SL -- Uses --> LLMS % For T_raz, embeddings

    %% AgentBus Interactions
    ToolsSystem -- Interacts via --> SB
    SB -- Stores Registry/Logs --> MongoDB
    MMA -- Registers/Uses --> SB
    ArchZ -- Registers/Uses --> SB
    EvoS -- Registers/Uses --> SB
    Ecosystem["Managed Agents / Tools"] -- Registers/Uses --> SB

    %% Evolution Strategist Interactions (Optional)
    SA -.->|Optional: Requests Evolution Insights| EvoS
    EvoS -- Analyzes --> SL
    EvoS -- Analyzes --> SM
    EvoS -- Analyzes --> ComponentTracker["ComponentExperienceTracker\n(Metrics in AgentBus logs or dedicated store)"] % ComponentExperienceTracker data source

    %% Other Dependencies
    AgentF -- Uses --> Providers["Providers\n(BeeAI, OpenAI, etc)"]
    AgentF -- Uses --> SL
    ToolF -- Uses --> SL
    ToolF -- Uses --> LLMS
    ToolsSystem -- Influenced By --> FW

    %% Optional Design Interaction
    SA -.->|Optional: Requests Design via Bus| ArchZ

    %% Final Result Flow
    SA -- Final Result --> User

    classDef agent fill:#9cf,stroke:#333,stroke-width:2px;
    classDef service fill:#f9f,stroke:#333,stroke-width:2px;
    classDef tool fill:#ccf,stroke:#333,stroke-width:2px;
    classDef infra fill:#eee,stroke:#666,stroke-width:1px,color:#333;
    style SA fill:#69c,stroke:#000,stroke-width:3px,color:#fff;
    style SM fill:#c9f,stroke:#333,stroke-width:2px; % Highlight Smart Memory
    style MMA fill:#9cf,stroke:#333,stroke-width:2px;
```
*Diagram Key: `agent` = Core EAT Agent, `service` = Core EAT Service, `tool` = SystemAgent's Internal Tools, `infra` = Supporting Infrastructure. Smart Memory is highlighted as a key service.*

---

## 2. Core Components

The toolkit is composed of several key interacting components, all leveraging MongoDB for persistence where applicable.

### 2.1. SystemAgent

The central orchestrator, a `beeai_framework.agents.react.ReActAgent`.
*   **Role:**
    *   Manages the lifecycle of components (agents, tools) by searching the `SmartLibrary`, creating new ones (using `CreateComponentTool`), or evolving existing ones (using `EvolveComponentTool`).
    *   Facilitates communication and task delegation via the `SmartAgentBus` (using `RequestAgentTool`, `DiscoverAgentTool`).
    *   Handles complex, multi-step task execution, often by generating and processing internal workflows (using `GenerateWorkflowTool`, `ProcessWorkflowTool`).
    *   **Crucially, actively utilizes the Smart Memory ecosystem for enhanced decision-making:**
        *   Uses `ContextBuilderTool` before significant planning or component selection to gather relevant past experiences and message summaries from the `MemoryManagerAgent`. This provides deep, historical context.
        *   Uses `ExperienceRecorderTool` after completing tasks or workflows to record significant outcomes, decisions, and learnings into Smart Memory, fostering continuous improvement for itself and the system.
    *   Manages optional human-in-the-loop review processes via intent review tools.
*   **Key Tools (Expanded):**
    *   **SmartLibrary Tools:** `SearchComponentTool`, `CreateComponentTool`, `EvolveComponentTool`.
    *   **SmartContext & Memory Tools:** `TaskContextTool` (for T_raz generation for search queries), `ContextBuilderTool` (constructs rich context using Smart Memory and SmartLibrary), `ExperienceRecorderTool` (records outcomes to Smart Memory).
    *   **AgentBus Tools:** `RegisterAgentTool`, `RequestAgentTool`, `DiscoverAgentTool`.
    *   **Workflow Tools:** `GenerateWorkflowTool`, `ProcessWorkflowTool`.
    *   **Intent Review Tools (Optional):** `WorkflowDesignReviewTool`, `ComponentSelectionReviewTool`, `ApprovePlanTool`.

### 2.2. ArchitectZero Agent

A specialized `ReActAgent` for designing solutions, typically invoked by `SystemAgent` via the `SmartAgentBus`.
*   **Role:** Analyzes complex requirements, queries `SmartLibrary` and potentially `SmartMemory` (if equipped with tools or via SystemAgent proxy) for existing patterns or components, and designs multi-component solutions.
*   **Output:** A structured solution design (e.g., JSON) that `SystemAgent` can use as a blueprint for workflow generation or component creation.

### 2.3 Smart Memory Ecosystem

This is a critical addition for enabling advanced learning and autonomous evolution.
*   **`MemoryManagerAgent`**:
    *   **Role**: A `ReActAgent` acting as the central orchestrator for the storage and retrieval of long-term memories (experiences). Registered on the `SmartAgentBus` (e.g., `memory_manager_agent_default_id`). It exposes a general `process_task` capability, allowing other agents (primarily `SystemAgent` via its tools) to send natural language requests for memory operations (e.g., "store this experience: {...}", "find experiences related to: 'invoice processing errors'").
    *   **Internal Tools**:
        *   `MongoExperienceStoreTool`: Handles CRUD operations for experiences in the `eat_agent_experiences` MongoDB collection, including generating embeddings for searchable text fields within an experience.
        *   `SemanticExperienceSearchTool`: Performs semantic (vector) searches over stored experiences in `eat_agent_experiences` based on natural language queries.
        *   `MessageSummarizationTool`: Uses an LLM to summarize message histories relevant to a specific goal, often requested by `ContextBuilderTool`.
*   **`ContextBuilderTool` (Tool for `SystemAgent`)**:
    *   **Purpose**: Dynamically constructs an optimized `SmartContext` instance for `SystemAgent` to use for a specific sub-task.
    *   **Functionality**: Invoked by `SystemAgent`. It uses `RequestAgentTool` to query the `MemoryManagerAgent` (to retrieve relevant past experiences and message summaries) and also directly queries `SmartLibrary` for relevant components. It assembles this information into a focused `SmartContext`.
*   **`ExperienceRecorderTool` (Tool for `SystemAgent`)**:
    *   **Purpose**: Facilitates the recording of completed tasks, sub-tasks, or entire workflows as structured experiences.
    *   **Functionality**: Invoked by `SystemAgent`. It structures task details (goal, components used, inputs, decisions, outputs, outcome, reasoning snippets) and uses `RequestAgentTool` to send this structured data to the `MemoryManagerAgent` for persistence in Smart Memory.
*   **MongoDB Collection: `eat_agent_experiences`**:
    *   **Purpose**: Stores structured records of agent experiences.
    *   **Key Fields**: Includes `experience_id`, `primary_goal_description` (embedded), `sub_task_description` (embedded), `input_context_summary` (embedded), `components_used`, `key_decisions_made`, `final_outcome`, `output_summary` (embedded), `feedback_signals`, `timestamp`, and generated `embeddings` for its searchable text fields. (A detailed schema is in `eat_agent_experiences_schema.md`).
    *   **Vector Search Index:** Requires an Atlas Vector Search index (e.g., `vector_index_experiences_default`) on its embedding fields.

### 2.4. Smart Library (MongoDB Backend)

(Content largely the same, just re-numbering from original doc)
Persistent storage and discovery for reusable components (agents, tools, firmware).
*   **MongoDB Collection:** `eat_components`.
*   **Stores:** Component documents with code, description, metadata, `content_embedding` (E_orig), and `applicability_embedding` (E_raz).
*   **Discovery:** Uses MongoDB Atlas Vector Search for dual embedding strategy.
*   **Interface & Indexing:** Methods like `create_record`, `semantic_search`, `evolve_record` interact with MongoDB. Embeddings (E_orig, E_raz) and applicability text (T_raz) are generated and stored.

*(Mermaid diagram for SmartLibrary can remain as is)*

### 2.5. Smart Agent Bus (MongoDB Backend)

(Content largely the same, just re-numbering)
Manages inter-agent communication and capability discovery.
*   **Agent Registry (MongoDB Collection):** `eat_agent_registry`.
*   **Execution Logs (MongoDB Collection):** `eat_agent_bus_logs`.
*   **Role, Discovery, Resilience, Interface:** As described previously, interacts with MongoDB. `MemoryManagerAgent` is a key agent registered here.

*(Mermaid diagram for SmartAgentBus can remain as is)*

### 2.6. Smart Context

Data structure for passing task-relevant information.
*   **Role:** Carries current task data, user input, intermediate results, and task context description.
*   **Enhancement via `ContextBuilderTool`**: It is **dynamically enriched** by `ContextBuilderTool` which populates it with summaries of relevant past experiences and message histories (retrieved via `MemoryManagerAgent`), and relevant components from `SmartLibrary`. This provides `SystemAgent` with deeper, historically-informed context.
*   **Interaction with `SmartLibrary`:** Its `current_task` description (often enhanced by `ContextBuilderTool`) is used by `SmartLibrary.semantic_search` for task-aware retrieval.

### 2.7. EvolutionStrategistAgent (Optional)

*   **Role:** A specialized agent (potentially a `ReActAgent`) that proactively analyzes system performance and learning to suggest or even initiate component evolutions.
*   **Data Sources:**
    *   `ComponentExperienceTracker` (for quantitative performance metrics like success/failure rates, execution times - this data can be sourced from `SmartAgentBus` logs or a dedicated metrics store).
    *   `SmartMemory` (via `MemoryManagerAgent`): To analyze patterns in successful/failed experiences, understand *why* components are performing a certain way in specific contexts.
    *   `SmartLibrary`: To understand the current state of available components.
    *   A/B Test Results (if applicable, potentially stored in a dedicated collection like `eat_ab_test_results`).
*   **Output:** Evolution proposals (e.g., "Evolve component X to better handle Y, based on N failed experiences showing Z"), which can be fed to `SystemAgent` or reviewed by a human.

### 2.8. Providers & Agent Factory
### 2.9. Dependency Container
### 2.10. LLM Service (MongoDB Cache)
### 2.11. Firmware
### 2.12. Adapters
### 2.13. Intent Review System (MongoDB Backend)
(These sections 2.8 - 2.13 remain largely the same as your previous version, just re-numbered)

## 3. Key Architectural Patterns & Flows

### 3.1. Agent Communication (via Agent Bus)
(Remains the same)

### 3.2. Task-Aware Context Retrieval (Dual Embedding on MongoDB)
(Remains the same, emphasizing that `task_context` for search is often now enriched by Smart Memory via `ContextBuilderTool`)

*(Sequence diagram for Task-Aware Context Retrieval can remain as is)*

### 3.3. Agent Learning and Context Enrichment Flow (with Smart Memory)

This is a **central flow** enabled by the Smart Memory ecosystem:

1.  **Task Initiation:** `SystemAgent` receives a high-level goal.
2.  **Context Building (Proactive):** `SystemAgent` uses `ContextBuilderTool`.
    *   `ContextBuilderTool` sends a request to `MemoryManagerAgent` (via `SmartAgentBus`) with the current sub-task description.
    *   `MemoryManagerAgent` uses `SemanticExperienceSearchTool` to find relevant past experiences from `eat_agent_experiences`.
    *   `MemoryManagerAgent` uses `MessageSummarizationTool` to summarize relevant recent message history (if provided to `ContextBuilderTool`).
    *   `ContextBuilderTool` also queries `SmartLibrary` (MongoDB `eat_components`) for potentially relevant existing components.
    *   `ContextBuilderTool` returns a structured dataset containing these findings to `SystemAgent`.
3.  **Informed Planning & Action:** `SystemAgent` incorporates this rich context (past successes/failures, relevant components, message summaries) into its planning:
    *   More accurate component selection (via `SearchComponentTool` now using better task context, or by directly choosing from `ContextBuilderTool`'s suggestions).
    *   Better workflow design (via `GenerateWorkflowTool`).
    *   More targeted parameters for tool/agent execution.
4.  **Task Execution:** `SystemAgent` orchestrates the execution of the planned steps.
5.  **Experience Recording:** After a significant sub-task or the overall task is completed (or fails), `SystemAgent` uses `ExperienceRecorderTool`.
    *   `ExperienceRecorderTool` structures the key details of the just-completed experience (goal, input summary, components used, decisions, output summary, outcome).
    *   It sends this structured experience to `MemoryManagerAgent` (via `SmartAgentBus`).
    *   `MemoryManagerAgent` uses `MongoExperienceStoreTool` to save the experience (including generating its embeddings) into the `eat_agent_experiences` collection in MongoDB.
6.  **Learning Loop Closure:** The newly recorded experience is now available for future `ContextBuilderTool` queries, allowing the system to learn and improve over time.

*(Consider adding a sequence diagram for this Smart Memory Learning Loop if desired)*

### 3.4. Informed Component Evolution

Smart Memory plays a vital role in making component evolution more targeted and effective:

1.  **Identification of Need:**
    *   `SystemAgent` might identify a need for evolution based on repeated failures for a specific task type (retrieved from Smart Memory).
    *   Or, `EvolutionStrategistAgent` might analyze `eat_agent_experiences` and `ComponentExperienceTracker` data to proactively suggest evolutions.
2.  **Contextual Evolution Prompting:** When `SystemAgent` decides to evolve a component using `EvolveComponentTool`:
    *   It first uses `ContextBuilderTool` to gather context related to the component-to-be-evolved and the problem it's failing to solve (or the new capability needed).
    *   This context (e.g., "Component X failed 3 times on task Y when input was Z, resulting in error W. Component A handles similar inputs P successfully.") is used to formulate a much more specific and informed "changes" description for `EvolveComponentTool`.
3.  **LLM-Driven Evolution:** `EvolveComponentTool` uses the LLM to generate new code based on the parent component's code and this rich, contextual "changes" description.
4.  **New Version in Library:** The evolved component is saved as a new version in `SmartLibrary`.
5.  **Future Use:** This new, contextually-evolved component is now available for future tasks and will be discoverable via `SmartLibrary` searches, further improving system performance.

### 3.5. Workflow Generation & Execution
(Remains largely the same, but workflow generation can now be informed by context from Smart Memory if `SystemAgent` uses `ContextBuilderTool` before calling `GenerateWorkflowTool`)

### 3.6. Dependency Injection & Initialization
(Remains the same)

### 3.7. Intent Review / Human-in-the-Loop Flow (MongoDB Backend)
(Remains the same)

*(Sequence diagram for Intent Review can remain as is)*

## 4. Multi-Framework Integration
(Remains the same)

## 5. Governance and Safety
(Remains the same, with Smart Memory adding to audit trails)

This enhanced architecture, with Smart Memory at its core, provides a more powerful foundation for EAT to achieve its goal of building adaptive, learning, and autonomously evolving AI agent systems. The ability to learn from rich, contextualized past experiences allows the system to make more intelligent decisions about current tasks and future evolutions.