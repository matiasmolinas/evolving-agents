# Tutorial: Testing EAT Examples with Local Dockerized MongoDB

This tutorial will guide you through setting up the Evolving Agents Toolkit (EAT) framework and running its example scripts using a local MongoDB instance managed by Docker.

## 1. Prerequisites

Before you begin, ensure you have the following installed:

*   **Git:** For cloning the repository.
*   **Python 3.11+:** EAT is designed for Python 3.11 or newer.
*   **pip:** Python's package installer.
*   **Docker Desktop (or Docker Engine + Docker Compose CLI):** For running MongoDB in a container.
    *   [Install Docker](https://docs.docker.com/get-docker/)
    *   [Install Docker Compose](https://docs.docker.com/compose/install/)
*   **OpenAI API Key:** Most examples, especially those involving LLM interactions, will require an OpenAI API key. Other LLM providers might be configurable, but OpenAI is the default for many examples.

## 2. Setup Instructions

### Step 2.1: Clone the Repository

Open your terminal and clone the `Adaptive-Agents-Framework` repository:

```bash
git clone https://github.com/matiasmolinas/evolving-agents.git
cd Adaptive-Agents-Framework
```

*(Note: Replace the URL if your repository is hosted elsewhere or has a different name)*

### Step 2.2: Set Up Python Virtual Environment & Install Dependencies

It's highly recommended to use a virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install required Python packages
pip install -r requirements.txt

# Install the EAT package in editable mode
pip install -e .
```

### Step 2.3: Configure Environment Variables (`.env` file)

The EAT framework uses a `.env` file for configuration.

1.  Copy the example environment file:
    ```bash
    cp .env.example .env
    ```

2.  Edit the `.env` file with your preferred text editor (e.g., `nano .env`, `code .env`):
    *   **Crucial:** Add your `OPENAI_API_KEY`:
        ```env
        OPENAI_API_KEY="your-openai-api-key-here"
        ```
    *   **MongoDB Settings (for Docker Compose):**
        The `docker-compose.yml` file is configured to set `MONGODB_URI` and `MONGODB_DATABASE_NAME` specifically for the application container to connect to the MongoDB container within the Docker network. The values for these in your `.env` file will be overridden by `docker-compose.yml` for the `app` service.
        However, it's good practice to have them in your `.env` if you plan to run scripts *outside* of Docker that connect to a local MongoDB instance (e.g., if you run MongoDB directly without Docker for some reason, or connect to a different Atlas instance).
        For this Docker-based tutorial, the `OPENAI_API_KEY` is the most critical setting in `.env`. The `docker-compose.yml` will handle MongoDB connection strings for the app container.
    *   **Other Settings:**
        You can review and adjust other settings like `LLM_MODEL`, `LLM_EMBEDDING_MODEL`, `LLM_USE_CACHE` as needed. The defaults from `.env.example` are generally fine for starting.
        For example, the `.env.example` has:
        ```env
        LLM_PROVIDER="openai"
        LLM_MODEL="gpt-4.1-mini"
        LLM_EMBEDDING_MODEL="text-embedding-3-small"
        MONGODB_URI="your_mongodb_srv_connection_string_with_username_and_password" # This will be overridden by docker-compose
        MONGODB_DATABASE_NAME="evolving_agents_db" # This will be overridden by docker-compose
        ```

### Step 2.4: Build and Start Docker Services (MongoDB & App)

This step will use Docker Compose to build the application image and start both the EAT application container and a MongoDB container.

1.  **Build the Docker images:**
    Navigate to the project root directory (where `docker-compose.yml` and `Dockerfile` are) and run:
    ```bash
    docker-compose build
    ```

2.  **Start the services:**
    Run the following command to start the containers in detached mode (in the background):
    ```bash
    docker-compose up -d
    ```
    *   This will:
        *   Start a MongoDB container named `eat_mongo` (MongoDB version 7.0 as per `docker-compose.yml`).
        *   Start the EAT application container named `eat_app`.
        *   Create a Docker volume named `mongo-data` to persist MongoDB data across container restarts.
        *   Create a Docker network named `eat_network` for the containers to communicate.
        *   The `app` service in `docker-compose.yml` is configured with `MONGODB_URI=mongodb://mongo:27017/evolving_agents_db` and `MONGODB_DATABASE_NAME=evolving_agents_db`, so it will connect to the `eat_mongo` service on the internal Docker network.

3.  **Check container status:**
    To ensure both containers are running, use:
    ```bash
    docker-compose ps
    ```
    You should see both `eat_app` and `eat_mongo` with a state of `Up` (or `Up (healthy)` for `eat_mongo` after its health check passes).

### Step 2.5: Create MongoDB Vector Search Indexes

For the EAT framework's `SmartLibrary`, `SmartAgentBus`, and `SmartMemory` (via `MongoExperienceStoreTool` and `SemanticExperienceSearchTool`) to function correctly with semantic search, you **must** create Vector Search Indexes in your MongoDB instance.

1.  **Access `mongosh` in the MongoDB Container:**
    Open a new terminal window and execute:
    ```bash
    docker-compose exec mongo mongosh
    ```
    This connects you to the MongoDB shell running inside the `eat_mongo` container.

2.  **Switch to the EAT Database:**
    Inside the `mongosh` prompt, switch to the database used by the application (as defined in `docker-compose.yml`'s `MONGODB_URI` for the `app` service, typically `evolving_agents_db`):
    ```javascript
    use evolving_agents_db;
    ```

3.  **Create Vector Search Indexes:**
    You need to create **four** vector search indexes.
    **IMPORTANT:** Replace `YOUR_EMBEDDING_DIMENSION` in the commands below with the actual dimension of your embedding model. For OpenAI's `text-embedding-3-small` (the default in `.env.example`), this is **1536**.

    *   **Index 1: `eat_components` - Content Embedding**
        (Internal EAT name: `idx_components_content_embedding`)
        ```javascript
        db.eat_components.createSearchIndex({
          "name": "idx_components_content_embedding",
          "definition": { "mappings": { "dynamic": false, "fields": {
            "content_embedding": { "type": "vector", "dimensions": YOUR_EMBEDDING_DIMENSION, "similarity": "cosine" },
            "record_type": { "type": "string", "analyzer": "keyword" }, "domain": { "type": "string", "analyzer": "keyword" },
            "status": { "type": "string", "analyzer": "keyword" }, "tags": { "type": "string", "analyzer": "keyword", "multi": true }
          }}}
        });
        ```

    *   **Index 2: `eat_components` - Applicability Embedding**
        (Internal EAT name: `applicability_embedding`)
        ```javascript
        db.eat_components.createSearchIndex({
          "name": "applicability_embedding",
          "definition": { "mappings": { "dynamic": false, "fields": {
            "applicability_embedding": { "type": "vector", "dimensions": YOUR_EMBEDDING_DIMENSION, "similarity": "cosine" },
            "record_type": { "type": "string", "analyzer": "keyword" }, "domain": { "type": "string", "analyzer": "keyword" },
            "status": { "type": "string", "analyzer": "keyword" }, "tags": { "type": "string", "analyzer": "keyword", "multi": true }
          }}}
        });
        ```

    *   **Index 3: `eat_agent_registry` - Agent Description Embedding**
        (Internal EAT name: `vector_index_agent_description`)
        ```javascript
        db.eat_agent_registry.createSearchIndex({
          "name": "vector_index_agent_description",
          "definition": { "mappings": { "dynamic": false, "fields": {
            "description_embedding": { "type": "vector", "dimensions": YOUR_EMBEDDING_DIMENSION, "similarity": "cosine" },
            "name": { "type": "string", "analyzer": "keyword" }, "status": { "type": "string", "analyzer": "keyword" },
            "type": { "type": "string", "analyzer": "keyword" }
          }}}
        });
        ```

    *   **Index 4: `eat_agent_experiences` - Smart Memory Embeddings**
        (Internal EAT name: `vector_index_experiences_default`)
        ```javascript
        db.eat_agent_experiences.createSearchIndex({
          "name": "vector_index_experiences_default",
          "definition": { "mappings": { "dynamic": false, "fields": {
            "embeddings.primary_goal_description_embedding": { "type": "vector", "dimensions": YOUR_EMBEDDING_DIMENSION, "similarity": "cosine" },
            "embeddings.sub_task_description_embedding": { "type": "vector", "dimensions": YOUR_EMBEDDING_DIMENSION, "similarity": "cosine" },
            "embeddings.input_context_summary_embedding": { "type": "vector", "dimensions": YOUR_EMBEDDING_DIMENSION, "similarity": "cosine" },
            "embeddings.output_summary_embedding": { "type": "vector", "dimensions": YOUR_EMBEDDING_DIMENSION, "similarity": "cosine" }
            // Example filterable fields (add if needed, see docs/MONGO-SETUP.md for more):
            // ,"initiating_agent_id": { "type": "string", "analyzer": "keyword" }
          }}}
        });
        ```
    Wait for each command to return a success message (e.g., `{ ok: 1.0 }`). Index creation might take a moment. You can check index status within `mongosh` using `db.collectionName.getSearchIndexes()`.

4.  **Exit `mongosh`:**
    Type `exit` or press `Ctrl+D`.

Your local MongoDB instance is now set up and ready for EAT.

## 3. Running the Examples

With the environment set up, you can now run the example scripts. The Python scripts will execute inside the `eat_app` Docker container, which has access to the `eat_mongo` container.

### Step 3.1: Run the Comprehensive Demo

The `README.md` highlights `examples/invoice_processing/architect_zero_comprehensive_demo.py` as a comprehensive demo. Let's run this first.

Open your terminal in the project root and execute:

```bash
docker-compose exec app python examples/invoice_processing/architect_zero_comprehensive_demo.py
```

This command tells Docker Compose to execute the Python script inside the `app` service container.

*   **Observe the output:** You'll see logs from the EAT framework, including agent interactions, tool usage, and potentially LLM calls.
*   If `INTENT_REVIEW_ENABLED=true` and `INTENT_REVIEW_LEVELS` includes `intents` in your `.env` file, the script might pause for human review. Intent plans will be stored in MongoDB.

### Step 3.2: Run Other Examples

All examples in the `examples/` directory have been updated to use the MongoDB backend. You can run them similarly. For example, to run the Smart Memory test script:

```bash
docker-compose exec app python scripts/test_smart_memory.py
```

Or another example:

```bash
docker-compose exec app python examples/smart_agent_bus/dual_bus_demo.py
```

Replace the script path accordingly for other examples you wish to test.

## 4. Verifying Results

After running an example, you can verify its effects:

### Step 4.1: Check Output Files

*   Some demos, like `architect_zero_comprehensive_demo.py`, might produce output files in the project root (or a specified output directory). For example:
    *   `final_processing_output.json`: Contains the final structured JSON result from the `SystemAgent`.
    *   `intent_plan_demo.json` (or similar, path configured in `.env` via `INTENT_REVIEW_OUTPUT_PATH`): An optional file copy of generated intent plans if review is enabled.

### Step 4.2: Inspect MongoDB Collections

This is the primary way to verify data persistence.

1.  **Connect to `mongosh`:**
    ```bash
    docker-compose exec mongo mongosh
    ```

2.  **Switch to the database:**
    ```javascript
    use evolving_agents_db;
    ```

3.  **Inspect collections:**
    *   **`eat_components` (SmartLibrary):** Stores agents, tools, firmware.
        ```javascript
        db.eat_components.find().limit(5).pretty(); // View first 5 components
        db.eat_components.countDocuments(); // Get total count
        ```
    *   **`eat_agent_registry` (SmartAgentBus Registry):** Stores agent registrations.
        ```javascript
        db.eat_agent_registry.find().limit(5).pretty();
        db.eat_agent_registry.countDocuments();
        ```
    *   **`eat_agent_bus_logs` (SmartAgentBus Logs):** Stores logs of agent interactions.
        ```javascript
        db.eat_agent_bus_logs.find().sort({timestamp: -1}).limit(5).pretty(); // View 5 most recent logs
        db.eat_agent_bus_logs.countDocuments();
        ```
    *   **`eat_llm_cache` (LLM Cache):** Stores cached LLM responses and embeddings (if `LLM_USE_CACHE=true`).
        ```javascript
        db.eat_llm_cache.find().limit(5).pretty();
        db.eat_llm_cache.countDocuments();
        ```
    *   **`eat_intent_plans` (Intent Review):** Stores `IntentPlan` objects (if review for 'intents' level is enabled).
        ```javascript
        db.eat_intent_plans.find().limit(5).pretty();
        db.eat_intent_plans.countDocuments();
        ```
    *   **`eat_agent_experiences` (Smart Memory):** Stores agent experiences if `ExperienceRecorderTool` or `MongoExperienceStoreTool` are used.
        ```javascript
        db.eat_agent_experiences.find().limit(5).pretty();
        db.eat_agent_experiences.countDocuments();
        ```

    You can also use **MongoDB Compass** or another GUI tool to connect to `mongodb://localhost:27017` (since port 27017 is exposed by the `eat_mongo` service) and explore the `evolving_agents_db` database and its collections visually.

### Step 4.3: View Docker Logs

If an example doesn't behave as expected, check the logs:

*   **Application logs:**
    ```bash
    docker-compose logs app
    ```
*   **MongoDB logs:**
    ```bash
    docker-compose logs mongo
    ```
    Add `-f` to follow logs in real-time (e.g., `docker-compose logs -f app`).

## 5. Troubleshooting Common Issues

*   **`OPENAI_API_KEY` not set:** Ensure it's correctly set in your `.env` file in the project root.
*   **Docker not running:** Make sure Docker Desktop or Docker daemon is running.
*   **Port `27017` conflict:** If another MongoDB instance is using port 27017 on your host, the `eat_mongo` service might fail. You can change the host port mapping in `docker-compose.yml` (e.g., `"27018:27017"`) and adjust your connection string if connecting from the host (the `app` service will still use `mongodb://mongo:27017` internally).
*   **`app` container connection issues to `mongo`:**
    *   Wait for `eat_mongo` to become healthy. Check `docker-compose ps`. The healthcheck in `docker-compose.yml` allows time for startup.
    *   Ensure the `eat_network` Docker network was created correctly.
*   **Vector Search Index Errors:**
    *   Double-check that all four vector search indexes were created correctly in `mongosh` (Step 2.5).
    *   Verify `YOUR_EMBEDDING_DIMENSION` was replaced with the correct value (e.g., 1536).
    *   Look for error messages like "index not found" or "Invalid $vectorSearch" in the `app` logs or MongoDB logs.
*   **Python script errors:** If a script fails, the `docker-compose exec ...` command will show the Python traceback. Debug as you would a normal Python script, keeping in mind it's running inside the container.

## 6. Stopping the Environment

When you're finished testing:

1.  **Stop and remove containers, network:**
    ```bash
    docker-compose down
    ```

2.  **(Optional) Remove the MongoDB data volume:**
    If you want a completely fresh MongoDB instance next time (this will **delete all data** stored by the `eat_mongo` service):
    ```bash
    docker-compose down -v
    ```

## 7. Conclusion

This tutorial provided steps to set up your EAT development environment with a local Dockerized MongoDB, create the necessary database indexes, and run the example scripts. You should now be able to test and explore the Evolving Agents Toolkit's capabilities with its new MongoDB backend. Remember to consult the `README.md` and `docs/` directory for more in-depth information on architecture and specific features.