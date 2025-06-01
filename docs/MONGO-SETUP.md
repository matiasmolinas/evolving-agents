# MongoDB Setup Guide for EAT: Atlas CLI Local Deployment (Dev/Test)

This guide provides instructions for setting up MongoDB as the unified backend for the Evolving Agents Toolkit (EAT), specifically using the **MongoDB Atlas CLI Local Deployment method.** This approach is **highly recommended for local development and testing** as it provides an Atlas-like environment on your machine, including full support for Atlas Vector Search features, without the 3-search-index limit of the M0 cloud free tier.

**Contents:**

1.  [Why Atlas CLI Local Deployment for Dev/Test?](#1-why-atlas-cli-local-deployment-for-devtest)
2.  [Prerequisites](#2-prerequisites)
3.  [Setup Instructions](#3-setup-instructions)
    *   [Step 3.1: Install the Atlas CLI](#step-31-install-the-atlas-cli)
    *   [Step 3.2: Set Up Your Local Atlas Deployment](#step-32-set-up-your-local-atlas-deployment)
    *   [Step 3.3: Connect to Your Local Atlas Deployment (`mongosh`)](#step-33-connect-to-your-local-atlas-deployment-mongosh)
    *   [Step 3.4: Create Vector Search Indexes via `mongosh` (CRITICAL)](#step-34-create-vector-search-indexes-via-mongosh-critical)
4.  [Configure EAT Project (`.env` file)](#4-configure-eat-project-env-file)
5.  [Running the EAT Application](#5-running-the-eat-application)
6.  [Managing Your Local Atlas Deployment](#6-managing-your-local-atlas-deployment)
7.  [Troubleshooting](#7-troubleshooting)
8.  [Note on Production](#8-note-on-production)

---

## 1. Why Atlas CLI Local Deployment for Dev/Test?

*   **Full Atlas Vector Search Features Locally:** Test all EAT semantic search capabilities without needing a cloud instance for basic development.
*   **No Cloud Index Limits During Development:** Create all necessary vector search indexes (currently 4 for full EAT functionality) without being constrained by the M0 free cloud tier's 3-index limit.
*   **Consistent Environment:** Develop in an environment that closely mirrors MongoDB Atlas cloud deployments, easing future transitions if needed.
*   **Local Resources & Control:** Runs on your machine using Docker (managed by the Atlas CLI), with no cloud costs during development.
*   **Simplified Vector Search Setup:** Easier than configuring advanced vector search on a traditional self-hosted MongoDB.

**Important:** This local deployment is for **development and testing only** and is not suitable for production environments due to factors like single-node replica sets and default lack of authentication.

---

## 2. Prerequisites

*   **Docker:** Docker Desktop or Docker Engine must be installed and running. The Atlas CLI uses Docker to run its local MongoDB instance.
*   **Python 3.11+:** For the EAT framework.
*   **EAT Project:** You should have the EAT project code cloned.
*   **OpenAI API Key (or other LLM provider configured):** Required by EAT for generating embeddings.

---

## 3. Setup Instructions

### Step 3.1: Install the Atlas CLI

Follow the official MongoDB instructions to install the Atlas CLI on your operating system:
*   [Install the Atlas CLI](https://www.mongodb.com/docs/atlas/cli/stable/install-atlas-cli/)

Verify the installation:
```bash
atlas --version
```

### Step 3.2: Set Up Your Local Atlas Deployment

1.  **Login to Atlas (Recommended, but optional for basic local setup):**
    Logging in can integrate with your Atlas account for other CLI features. If you skip this, `atlas local` commands should still work.
    ```bash
    atlas auth login
    ```
    Follow the prompts to log in via your web browser.

2.  **Create and Start the Local Deployment:**
    Open your terminal. This command will download necessary Docker images (if not already present) and set up a local MongoDB instance. This might take a few minutes the first time.
    ```bash
    atlas deployments setup eat-local-dev --type local --port 27017 --mdbVersion 7.0
    ```
    *   `eat-local-dev`: This is a suggested name for your local deployment. You can choose another.
    *   `--type local`: Specifies a local deployment.
    *   `--port 27017`: Sets the port MongoDB will listen on. `27017` is the standard MongoDB port. Change if `27017` is already in use on your host.
    *   `--mdbVersion 7.0`: Specifies MongoDB version 7.0. Check `atlas deployments availableversions --type local` for other supported versions.

    Upon successful setup, the Atlas CLI will indicate the deployment is running and provide a connection string, typically `mongodb://localhost:27017/`.
    By default, these local deployments run **without authentication enabled.**

### Step 3.3: Connect to Your Local Atlas Deployment (`mongosh`)

You'll need to connect to your local Atlas deployment to create the necessary search indexes.

```bash
atlas deployments connect eat-local-dev --connectWith mongosh
```
(Replace `eat-local-dev` if you used a different name in Step 3.2). This will open a `mongosh` session connected to your local Atlas instance.

### Step 3.4: Create Vector Search Indexes via `mongosh` (CRITICAL)

EAT's `SmartLibrary`, `SmartAgentBus`, and `SmartMemory` rely on specific Vector Search Indexes.

1.  **Switch to the EAT Database in `mongosh`:**
    The database name you choose here must match what you configure in your EAT project's `.env` file.
    ```javascript
    use evolving_agents_db; // Or your preferred MONGODB_DATABASE_NAME
    ```

2.  **Create the Vector Search Indexes:**
    **IMPORTANT:** Replace `YOUR_EMBEDDING_DIMENSION` in the commands below with the actual dimension of your embedding model (e.g., **1536** for OpenAI's `text-embedding-3-small`).

    *   **Index 1: `eat_components` - Content Embedding**
        *   Atlas Search Index Name (and EAT internal reference): `idx_components_content_embedding`
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
        *   Atlas Search Index Name (and EAT internal reference): `applicability_embedding`
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
        *   Atlas Search Index Name (and EAT internal reference): `vector_index_agent_description`
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
        *   Atlas Search Index Name (and EAT internal reference): `vector_index_experiences_default`
        ```javascript
        db.eat_agent_experiences.createSearchIndex({
          "name": "vector_index_experiences_default",
          "definition": { "mappings": { "dynamic": false, "fields": {
            "embeddings.primary_goal_description_embedding": { "type": "vector", "dimensions": YOUR_EMBEDDING_DIMENSION, "similarity": "cosine" },
            "embeddings.sub_task_description_embedding": { "type": "vector", "dimensions": YOUR_EMBEDDING_DIMENSION, "similarity": "cosine" },
            "embeddings.input_context_summary_embedding": { "type": "vector", "dimensions": YOUR_EMBEDDING_DIMENSION, "similarity": "cosine" },
            "embeddings.output_summary_embedding": { "type": "vector", "dimensions": YOUR_EMBEDDING_DIMENSION, "similarity": "cosine" }
            // Add other filterable fields as needed based on your eat_agent_experiences_schema.md
          }}}
        });
        ```
    Wait for each `createSearchIndex` command to return a success message (e.g., `{ ok: 1.0 }`). Index building may take a few moments. You can check status with `db.collectionName.getSearchIndexes()`.

3.  **Exit `mongosh`:** Type `exit` or press `Ctrl+D`.

Your local Atlas deployment is now ready with the necessary vector indexes for EAT.

---

## 4. Configure EAT Project (`.env` file)

Your EAT application needs to connect to this local Atlas deployment.

1.  Navigate to the root directory of your EAT project.
2.  If it doesn't exist, copy `.env.example` to `.env`: `cp .env.example .env`
3.  Open `.env` and **ensure these lines are correctly set**:
    ```env
    # ... other settings like OPENAI_API_KEY ...

    MONGODB_URI="mongodb://localhost:27017/" # Or the port you used in Step 3.2, e.g., mongodb://localhost:27018/
    MONGODB_DATABASE_NAME="evolving_agents_db" # Must match the DB name used in mongosh for index creation

    # ... other LLM settings ...
    ```
    *   The local Atlas deployment typically does **not** require username/password in the URI.

---

## 5. Running the EAT Application

With your local Atlas deployment running (started via `atlas deployments start eat-local-dev`) and your `.env` file configured:

*   **If running EAT scripts directly on host (Recommended for this setup):**
    1.  Ensure your Python virtual environment for EAT is activated.
    2.  Run example scripts from the EAT project root:
        ```bash
        python examples/invoice_processing/architect_zero_comprehensive_demo.py
        python scripts/test_smart_memory.py
        ```
        The EAT application will use the `MONGODB_URI` from your `.env` file to connect to your local Atlas deployment.

*   **If running EAT application inside a Docker container (e.g., using the project's `docker-compose.yml` for the `app` service only):**
    1.  The `app` service in `docker-compose.yml` must be configured to pick up the `MONGODB_URI` from the `.env` file (which points to `localhost:27017`).
    2.  Ensure any `environment` settings for `MONGODB_URI` within the `app` service in `docker-compose.yml` that point to an internal Docker alias (like `mongodb://mongo:27017`) are removed or commented out.
    3.  The `depends_on: mongo` for the `app` service and the `mongo` service block itself in `docker-compose.yml` would typically be removed or commented out, as the database is managed externally by the Atlas CLI.
    4.  Then run: `docker-compose up -d app` (or however you run just the app service). Docker Desktop usually allows containers to connect to `localhost` on the host machine.

---

## 6. Managing Your Local Atlas Deployment

Use the Atlas CLI to manage your local MongoDB deployment:

*   **Start:** `atlas deployments start eat-local-dev`
*   **Stop:** `atlas deployments stop eat-local-dev`
*   **Pause (preserves data, stops Docker container):** `atlas deployments pause eat-local-dev`
*   **Resume:** `atlas deployments resume eat-local-dev`
*   **Delete (removes deployment and its data volume):** `atlas deployments delete eat-local-dev --force`
*   **List Deployments:** `atlas deployments list`
*   **View Logs:** `atlas deployments logs eat-local-dev -f`

---

## 7. Troubleshooting

*   **Atlas CLI `deployments setup` fails:** Ensure Docker is running and you have network connectivity.
*   **EAT cannot connect to `mongodb://localhost:27017/`:**
    *   Verify your local Atlas deployment is running (`atlas deployments list`).
    *   Check the port number in your `MONGODB_URI` matches the port used during `atlas deployments setup`.
*   **Vector Search "Index Not Found" Errors in EAT:**
    *   Ensure all four indexes were created in the correct database (`MONGODB_DATABASE_NAME`) using the exact names and field paths specified in Step 3.4.
    *   Verify `YOUR_EMBEDDING_DIMENSION` was correct.
    *   Check index status via `db.collectionName.getSearchIndexes()` in `mongosh` connected to your local Atlas deployment.
*   **`.env` not loaded:** Ensure `python-dotenv` is installed in your virtual environment.

---

## 8. Note on Production

The MongoDB Atlas CLI Local Deployment is **for development and testing only.** For staging or production, you should use a managed **MongoDB Atlas cloud cluster** for reliability, scalability, backups, and security. The process of creating Vector Search Indexes in a cloud Atlas cluster is similar (using the Atlas UI JSON editor).

---
```

---

**Updated File: `EAT_EXAMPLES_TESTING_TUTORIAL.md`** (This replaces the previous `DOCKER_TESTING_GUIDE.md` content)

```markdown
# Tutorial: Testing EAT Examples with Atlas CLI Local MongoDB

This tutorial guides you through setting up the Evolving Agents Toolkit (EAT) and running its example scripts using a **MongoDB Atlas CLI Local Deployment** for your database. This method is recommended for local development as it provides full Atlas Vector Search features without using the project's `docker-compose.yml` for the database service.

## 1. Prerequisites

*   **Git:** For cloning the repository.
*   **Python 3.11+:** EAT is designed for Python 3.11 or newer.
*   **pip:** Python's package installer.
*   **Docker Desktop (or Docker Engine):** The Atlas CLI uses Docker to run its local MongoDB instance. Docker Compose is needed if you choose to run the EAT *application* itself in a container.
    *   [Install Docker](https://docs.docker.com/get-docker/)
*   **MongoDB Atlas CLI:** For creating and managing your local Atlas MongoDB deployment.
    *   [Install the Atlas CLI](https://www.mongodb.com/docs/atlas/cli/stable/install-atlas-cli/)
*   **OpenAI API Key:** Most EAT examples require an OpenAI API key.

## 2. Setup Instructions

### Step 2.1: Clone the EAT Repository

```bash
git clone https://github.com/matiasmolinas/evolving-agents.git
cd Adaptive-Agents-Framework
```
*(Note: Adjust URL if your repository is hosted elsewhere)*

### Step 2.2: Set Up Python Virtual Environment & Install EAT Dependencies

```bash
python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### Step 2.3: Set Up and Start MongoDB Atlas CLI Local Deployment

This step creates and starts your local MongoDB instance with Atlas Search capabilities. This MongoDB instance runs in a Docker container managed by the Atlas CLI.

1.  **Follow `docs/MONGO-SETUP.md` (Sections 3.1, 3.2):**
    *   Install the Atlas CLI if you haven't already.
    *   Use the Atlas CLI to set up and start your local deployment. A typical command is:
        ```bash
        atlas deployments setup eat-local-dev --type local --port 27017 --mdbVersion 7.0
        ```
        Ensure this local deployment is **running** before proceeding (check with `atlas deployments list`).

2.  **Create Vector Search Indexes in your Local Atlas Deployment:**
    *   This is a **critical step** for EAT's semantic search features.
    *   Connect to your local Atlas deployment using `mongosh` (as shown in `docs/MONGO-SETUP.md`, Step 3.3: `atlas deployments connect eat-local-dev --connectWith mongosh`).
    *   Inside `mongosh`, execute the `db.collection.createSearchIndex({...})` commands provided in **Step 3.4** of `docs/MONGO-SETUP.md`. You need to create all four indexes detailed there for `eat_components` (x2), `eat_agent_registry`, and `eat_agent_experiences`.
    *   **Remember to replace `YOUR_EMBEDDING_DIMENSION`** (e.g., with **1536** for `text-embedding-3-small`).
    *   Exit `mongosh` after indexes are successfully created.

### Step 2.4: Configure EAT Environment Variables (`.env` file)

Your EAT application needs to know how to connect to the local Atlas deployment.

1.  In the EAT project root, copy `.env.example` to `.env` if it doesn't exist:
    ```bash
    cp .env.example .env
    ```
2.  Edit the `.env` file:
    *   Add your `OPENAI_API_KEY`:
        ```env
        OPENAI_API_KEY="your-openai-api-key-here"
        ```
    *   Configure MongoDB connection details:
        ```env
        MONGODB_URI="mongodb://localhost:27017/" # Adjust port if you used a different one for `atlas deployments setup`
        MONGODB_DATABASE_NAME="evolving_agents_db" # Must match the database name used when creating indexes
        ```
    *   Review other EAT settings (e.g., `LLM_MODEL`) as needed.

## 3. Running EAT Examples

You have two main ways to run the EAT application examples:

### Option 3.A: Run Python Scripts Directly on Host (Recommended for simplicity with this setup)

With your Atlas CLI Local MongoDB running and the Python virtual environment activated:

1.  Navigate to the EAT project root (`Adaptive-Agents-Framework`).
2.  Execute example scripts directly:
    ```bash
    # Comprehensive Demo
    python examples/invoice_processing/architect_zero_comprehensive_demo.py

    # Smart Memory Test Script
    python scripts/test_smart_memory.py

    # Smart Agent Bus Demo
    python examples/smart_agent_bus/dual_bus_demo.py
    ```
    The scripts will use the `MONGODB_URI` from your `.env` file to connect to the local Atlas deployment running on `localhost`.

### Option 3.B: Run EAT Application in Docker (using `docker-compose.yml` for the `app` service only)

If you prefer to run the EAT application itself within a Docker container while still using the externally managed Atlas CLI Local MongoDB:

1.  **Modify `docker-compose.yml` (Important):**
    *   Open `docker-compose.yml` in the EAT project root.
    *   **Remove or comment out the entire `mongo` service block.**
    *   In the `app` service definition:
        *   **Remove or comment out `depends_on: mongo`**.
        *   **Remove or comment out any `environment` variables that set `MONGODB_URI` or `MONGODB_DATABASE_NAME`** (e.g., lines like `- MONGODB_URI=mongodb://mongo:27017/...`). This ensures the `app` container uses the values from your `.env` file, which point to `localhost:27017`.
    *   Your `app` service definition should effectively use the `.env` file for its MongoDB connection. Example `app` service after modification:
        ```yaml
        services:
          app:
            build:
              context: .
            container_name: eat_app
            volumes:
              - .:/app
            env_file:
              - ./.env # This will provide MONGODB_URI="mongodb://localhost:27017/"
            # Removed: depends_on: mongo
            # Removed: environment block that hardcoded MONGODB_URI to the 'mongo' service
            networks:
              - eat_network # Or remove if only 'app' service remains and it connects to host's localhost
            tty: true
            stdin_open: true
        
        # mongo service block would be removed or commented out
        # volumes: mongo-data would be removed or commented out if mongo service is removed

        # networks: eat_network might still be needed if other EAT services are added later,
        # or can be simplified if app is the only service.
        # For connecting to host's localhost from Docker, often no explicit network config beyond default is needed.
        ```

2.  **Build the EAT application image (if not already built or if Dockerfile changed):**
    ```bash
    docker-compose build app
    ```

3.  **Start *only* the EAT application service:**
    ```bash
    docker-compose up -d app
    ```
    The `eat_app` container will start and connect to your Atlas CLI Local MongoDB running on `localhost:27017` (Docker Desktop usually bridges `localhost` to the host).

4.  **Run examples using `docker-compose exec`:**
    ```bash
    docker-compose exec app python examples/invoice_processing/architect_zero_comprehensive_demo.py
    ```

## 4. Verifying Results

### Step 4.1: Check Local Output Files

Some demos create files in the project directory (e.g., `final_processing_output.json`).

### Step 4.2: Inspect MongoDB Collections (in your Local Atlas Deployment)

1.  Connect to `mongosh` of your local Atlas deployment:
    ```bash
    atlas deployments connect eat-local-dev --connectWith mongosh
    ```
2.  Switch to database: `use evolving_agents_db;`
3.  Inspect collections (e.g., `db.eat_components.find().limit(1).pretty();`, `db.eat_agent_experiences.countDocuments();`).

## 5. Troubleshooting

*   **EAT App connection issues to `localhost:27017` MongoDB:**
    *   Ensure your Atlas CLI Local Deployment (`eat-local-dev`) is running (`atlas deployments list`).
    *   Verify `MONGODB_URI` in `.env` is `mongodb://localhost:27017/` (or correct port).
    *   If running EAT app in Docker (Option 3.B), ensure `docker-compose.yml` for the `app` service is *not* overriding `MONGODB_URI` to point to an internal Docker alias.
*   **Vector Search Index errors:** Confirm indexes were created correctly in your local Atlas deployment (Step 2.3.2) with the right names, paths, and dimensions.
*   Refer to `docs/MONGO-SETUP.md` for more detailed Atlas CLI troubleshooting.

## 6. Stopping the Environment

1.  **If running EAT app in Docker (Option 3.B):**
    ```bash
    docker-compose down
    ```
2.  **Stop your MongoDB Atlas CLI Local Deployment:**
    ```bash
    atlas deployments stop eat-local-dev # Or your deployment name
    ```
    To completely remove it and its data: `atlas deployments delete eat-local-dev --force`

## 7. Conclusion

This tutorial outlined using the MongoDB Atlas CLI Local Deployment for your database needs while developing with EAT. This provides a powerful, Atlas-feature-rich local MongoDB environment, allowing you to run EAT scripts directly or via a simplified Docker Compose setup for the application.
```

**Key Changes and Rationale for these Updates:**

*   **Central Focus on Atlas CLI Local Deployment:** Both documents now clearly state this as the recommended method for local development and testing, explicitly to avoid user-managed Docker for the database while still getting Atlas Search features.
*   **`MONGO-SETUP.md` Streamlined:** It's now almost entirely dedicated to the Atlas CLI Local method, making it the primary instruction set for DB setup.
*   **Main Tutorial (`EAT_EXAMPLES_TESTING_TUTORIAL.md`):**
    *   It directs users to `MONGO-SETUP.md` for the Atlas CLI Local DB setup.
    *   It then provides two options for running the EAT application:
        *   **Option 3.A (Direct Python):** This is the simplest "no Docker for the app" path.
        *   **Option 3.B (Docker for App Only):** For users who still want to containerize the EAT application, it explains how to modify `docker-compose.yml` to remove the `mongo` service and ensure the `app` service connects to the `localhost` where the Atlas CLI's MongoDB is running.
*   **Consistency:** The vector index creation commands and troubleshooting tips are now more consistently referenced from `MONGO-SETUP.MD`.

This approach provides a clear, focused path for developers wanting the full features locally with minimal direct Docker management for the database component.