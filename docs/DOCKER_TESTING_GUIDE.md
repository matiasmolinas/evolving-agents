# Dockerized Testing Guide for Evolving Agents Toolkit (EAT)

This guide explains how to set up and use a Dockerized environment for testing the Evolving Agents Toolkit, including the Smart Memory features. This setup uses Docker Compose to run the EAT application and a MongoDB instance.

## Prerequisites

*   **Docker Desktop (or Docker Engine + Docker Compose CLI):** Ensure Docker and Docker Compose are installed and running on your system. ([Install Docker](https://docs.docker.com/get-docker/), [Install Docker Compose](https://docs.docker.com/compose/install/))
*   **Git:** For cloning the EAT repository.
*   **OpenAI API Key:** Required for `LLMService`. Other LLM providers might require different configurations.
*   **EAT Project Code:** You should have a local copy of the `evolving-agents-toolkit` repository.

## Setup Instructions

### 1. Clone the Repository (if you haven't already)

```bash
# Replace with actual repo URL if different
git clone https://github.com/your-username/evolving-agents-toolkit.git
cd evolving-agents-toolkit
```

### 2. Configure Environment Variables

The EAT application uses a `.env` file for configuration, primarily for API keys.

*   If an `.env.example` file exists in the project root, copy it to `.env`:
    ```bash
    cp .env.example .env
    ```
*   If not, create a new file named `.env` in the project root.
*   Edit the `.env` file and add your `OPENAI_API_KEY`:
    ```env
    OPENAI_API_KEY="your_openai_api_key_here"

    # Other EAT configurations can also be set here (e.g., LLM_MODEL, LLM_PROVIDER)
    # MONGODB_URI and MONGODB_DATABASE_NAME in this file will be
    # overridden by docker-compose.yml for the app container,
    # but are needed if you run scripts outside Docker that connect to a custom MongoDB.
    # Example values (they will be overridden by docker-compose for the app service):
    # MONGODB_URI="mongodb://localhost:27017/evolving_agents_db"
    # MONGODB_DATABASE_NAME="evolving_agents_db"
    ```
    **Important:** The `docker-compose.yml` file is configured to provide the `MONGODB_URI` and `MONGODB_DATABASE_NAME` directly to the EAT application container, pointing it to the MongoDB container within the Docker network. The values in `.env` are primarily for other settings like API keys when running within Docker Compose.

## Building and Running the Docker Environment

### 1. Build the Docker Images

Navigate to the project root directory (where `docker-compose.yml` and `Dockerfile` are located) and run:

```bash
docker-compose build
```
This command builds the EAT application image based on the `Dockerfile`.

### 2. Start the Services

To start the EAT application and the MongoDB service, run:

```bash
docker-compose up -d
```
*   The `-d` flag runs the containers in detached mode (in the background). You can omit it to see live logs from both containers in your terminal.
*   This will:
    *   Start a MongoDB container named `eat_mongo`.
    *   Start an EAT application container named `eat_app`.
    *   Create a Docker volume named `mongo-data` to persist MongoDB data.
    *   Create a Docker network named `eat_network` for the containers.

To check the status of your containers:
```bash
docker-compose ps
```

## Initial MongoDB Setup (Creating Vector Search Indexes)

The first time you start the MongoDB container, or if you clear its data volume (`docker-compose down -v`), you'll need to create the necessary vector search indexes for EAT to function fully.

1.  **Access `mongosh` in the MongoDB Container:**
    Once the `mongo` service is running (check with `docker-compose ps`), open a new terminal and run:
    ```bash
    docker-compose exec mongo mongosh
    ```
    This will connect you to the `mongosh` shell inside the `eat_mongo` container.

2.  **Switch to the EAT Database:**
    In the `mongosh` prompt, switch to the database used by the application (as defined in `docker-compose.yml`, e.g., `evolving_agents_db`):
    ```javascript
    use evolving_agents_db;
    ```

3.  **Create Vector Search Indexes:**
    Copy and paste the following `mongosh` commands to create the 4 required indexes. **Replace `YOUR_EMBEDDING_DIMENSION`** with the dimension of your embedding model (e.g., 1536 for OpenAI's `text-embedding-3-small`, 3072 for `text-embedding-3-large`).

    *   **Index 1: `eat_components` - Content Embedding**
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
        ```javascript
        db.eat_agent_experiences.createSearchIndex({
          "name": "vector_index_experiences_default",
          "definition": { "mappings": { "dynamic": false, "fields": {
            "embeddings.primary_goal_description_embedding": { "type": "vector", "dimensions": YOUR_EMBEDDING_DIMENSION, "similarity": "cosine" },
            "embeddings.sub_task_description_embedding": { "type": "vector", "dimensions": YOUR_EMBEDDING_DIMENSION, "similarity": "cosine" },
            "embeddings.input_context_summary_embedding": { "type": "vector", "dimensions": YOUR_EMBEDDING_DIMENSION, "similarity": "cosine" },
            "embeddings.output_summary_embedding": { "type": "vector", "dimensions": YOUR_EMBEDDING_DIMENSION, "similarity": "cosine" }
            // Add filterable fields as needed, e.g.:
            // "initiating_agent_id": { "type": "string", "analyzer": "keyword" },
          }}}
        });
        ```
    Wait for each command to return success (e.g., `{ ok: 1.0 }`). Index creation might take a moment. You can check index status with `db.collectionName.getSearchIndexes()`.

4.  **Exit `mongosh`:**
    Type `exit` or press `Ctrl+D`.

Your MongoDB instance within Docker is now set up with the necessary vector indexes.

## Running Tests

### Option A: Focused Smart Memory Test Script

A dedicated script `scripts/test_smart_memory.py` is provided to test core Smart Memory functionalities directly.

To run this test:
```bash
docker-compose exec app python scripts/test_smart_memory.py
```
Observe the logs for success/failure messages from the script.

### Option B: Comprehensive Demo Script

You can also run the more comprehensive demo script (`examples/invoice_processing/architect_zero_comprehensive_demo.py`), which initializes a fuller EAT environment. The Smart Memory tools are available to `SystemAgent` in this demo. This can be useful for observing more complex interactions.

To run this demo:
```bash
docker-compose exec app python examples/invoice_processing/architect_zero_comprehensive_demo.py
```
This script is more complex and its output will be broader. Look for logs indicating `MemoryManagerAgent` activity or `SystemAgent` using `ContextBuilderTool` or `ExperienceRecorderTool`.

## Viewing Logs

*   **Live Logs (if not detached):** If you ran `docker-compose up` without `-d`, logs from all services will appear in your terminal.
*   **Service-Specific Logs:**
    ```bash
    docker-compose logs app
    docker-compose logs mongo
    ```
    You can use the `-f` flag to follow logs in real-time: `docker-compose logs -f app`

## Stopping the Environment

To stop and remove the containers, network:
```bash
docker-compose down
```
To remove the `mongo-data` volume as well (for a completely fresh start next time, **this will delete your MongoDB data**):
```bash
docker-compose down -v
```

## Troubleshooting

*   **OpenAI API Key:** Ensure `OPENAI_API_KEY` is correctly set in your `.env` file in the project root and is accessible to the `app` container.
*   **Docker Not Running:** Make sure Docker Desktop or Docker daemon is running.
*   **Port Conflicts:** If port `27017` is already in use on your host for another MongoDB instance, the `mongo` service in `docker-compose` might fail to start or be inaccessible from the host. You can change the host port mapping in `docker-compose.yml` (e.g., `"27018:27017"`) and adjust your connection string if connecting from the host (the `app` service will still use `mongodb://mongo:27017` internally).
*   **Build Issues:** If `docker-compose build` fails, check the output for errors. Common issues include missing system dependencies for a Python package (you might need to uncomment and adjust the `RUN apt-get update ...` line in the `Dockerfile` if a package needs C compilers or other build tools not present in `python:3.11-slim`).
*   **MongoDB Healthcheck/Startup:** If the `app` service has issues connecting to `mongo`, ensure the `mongo` container started correctly and its healthcheck (if enabled and robust) is passing. It can take MongoDB a little while to initialize fully after the container starts.
