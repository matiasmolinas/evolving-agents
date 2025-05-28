# MongoDB Setup Guide for Evolving Agents Toolkit (EAT)

This guide provides instructions for setting up MongoDB as the unified backend for the Evolving Agents Toolkit (EAT). Using MongoDB, especially with features like Atlas Vector Search (available in cloud and local Atlas CLI deployments), simplifies the data stack and enhances performance and scalability.

**Contents:**

1.  [Prerequisites](#prerequisites)
2.  [Choosing Your MongoDB Option](#choosing-your-mongodb-option)
3.  [Option 1: MongoDB Atlas CLI Local Deployment (Recommended for Development)](#option-1-mongodb-atlas-cli-local-deployment-recommended-for-development)
    *   [Step 1: Install the Atlas CLI](#step-1-install-the-atlas-cli)
    *   [Step 2: Set Up a Local Atlas Deployment](#step-2-set-up-a-local-atlas-deployment)
    *   [Step 3: Connect to the Local Deployment](#step-3-connect-to-the-local-deployment)
    *   [Step 4: Create Vector Search Indexes via mongosh](#step-4-create-vector-search-indexes-via-mongosh)
4.  [Option 2: MongoDB Atlas (Cloud)](#option-2-mongodb-atlas-cloud)
    *   [Step 1: Create an Atlas Account and Cluster](#step-1-create-an-atlas-account-and-cluster)
    *   [Step 2: Configure Network Access](#step-2-configure-network-access)
    *   [Step 3: Create a Database User](#step-3-create-a-database-user)
    *   [Step 4: Get Your Connection String](#step-4-get-your-connection-string)
    *   [Step 5: Create Vector Search Indexes via Atlas UI](#step-5-create-vector-search-indexes-via-atlas-ui)
5.  [Option 3: Traditional Self-Hosted MongoDB](#option-3-traditional-self-hosted-mongodb)
6.  [Configure EAT Project](#configure-eat-project)
7.  [Create Vector Search Indexes (CRITICAL) - Structure Overview](#create-vector-search-indexes-critical---structure-overview)
    *   [Index 1: SmartLibrary Content Embedding (`eat_components` collection)](#index-1-smartlibrary-content-embedding-eat_components-collection)
    *   [Index 2: SmartLibrary Applicability Embedding (`eat_components` collection)](#index-2-smartlibrary-applicability-embedding-eat_components-collection)
    *   [Index 3: SmartAgentBus Agent Description Embedding (`eat_agent_registry` collection)](#index-3-smartagentbus-agent-description-embedding-eat_agent_registry-collection)
    *   [Index 4: Smart Memory Experience Embeddings (`eat_agent_experiences` collection)](#index-4-smart-memory-experience-embeddings-eat_agent_experiences-collection)
8.  [Cloud M0 Free Tier Index Limitations](#cloud-m0-free-tier-index-limitations)
9.  [Running the Application](#running-the-application)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

*   Python 3.11+
*   `pymongo` and `motor` (Python drivers for MongoDB, included in `requirements.txt`)
*   An OpenAI API key (or other LLM provider configured) for generating embeddings.
*   Access to a MongoDB instance.
*   Docker (for Option 1: MongoDB Atlas CLI Local Deployment).

---
## Choosing Your MongoDB Option

You have a few options for your MongoDB setup, each with its own advantages:

1.  **MongoDB Atlas CLI Local Deployment (Recommended for Development):**
    *   **Pros:** Allows you to run an Atlas-like environment locally, potentially bypassing the cloud M0 free tier's 3-vector-index limit for development purposes. Provides a consistent experience if you later deploy to Atlas cloud. Enables all EAT search features locally without cloud costs.
    *   **Cons:** Intended for development and testing only, not production. Relies on your local machine's resources. Docker is typically required.
    *   Setup for this option is detailed in the next section.

2.  **MongoDB Atlas (Cloud - M0 Free Tier or Paid Tiers):**
    *   **Pros:** Fully managed service, easy to get started, robust Vector Search, scalable. M0 Free Tier is available for basic use.
    *   **Cons (for M0 Free Tier):** Limited to 3 vector search indexes, which requires choosing which EAT search features to prioritize (see "Cloud M0 Free Tier Index Limitations" section below). Paid tiers remove this limit and are suitable for production.
    *   Setup for this option is detailed under "Option 2: MongoDB Atlas (Cloud)".

3.  **Traditional Self-Hosted MongoDB (Community/Enterprise Server):**
    *   **Pros:** Full control over your MongoDB environment, no artificial index limits (beyond your hardware's capability).
    *   **Cons:** Requires manual installation, configuration, security, backups, and potentially more complex setup for robust vector search if not using a recent MongoDB version (6.0.11+ or 7.0.2+ recommended for native vector search).
    *   Setup for this option is detailed under "Option 3: Traditional Self-Hosted MongoDB".

This guide will provide details for all options, with a strong recommendation for **MongoDB Atlas CLI Local Deployment** for most development scenarios requiring full EAT features without cloud costs. If you plan to deploy to production on Atlas, developing with the CLI local deployment can also be beneficial.
---

## Option 1: MongoDB Atlas CLI Local Deployment (Recommended for Development)

This option allows you to run a local MongoDB environment managed by the Atlas CLI. It's ideal for development as it can provide an Atlas-like experience and allows you to work around the 3-vector-index limit of the M0 cloud free tier for local testing, enabling all EAT search features. Docker is typically required to be running on your system.

**Important:** This local deployment is for development and testing only and is not suitable for production environments due to factors like single-node replica sets and lack of default authentication.

### Step 1: Install the Atlas CLI

Follow the official MongoDB instructions to install the Atlas CLI on your operating system:
*   [Install the Atlas CLI](https://www.mongodb.com/docs/atlas/cli/stable/install-atlas-cli/)

Verify the installation:
```bash
atlas --version
```

### Step 2: Set Up a Local Atlas Deployment

1.  **Login to Atlas (Optional but Recommended):**
    While not strictly required for `deployments setup --type local` for basic functionality, logging in can integrate with your Atlas account for other CLI features.
    ```bash
    atlas auth login
    ```
    Follow the prompts to log in via your web browser.

2.  **Create the Local Deployment:**
    Open your terminal and run:
    ```bash
    atlas deployments setup eat-local-dev --type local --port 27017
    ```
    *   `eat-local-dev` is a suggested name for your local deployment. You can choose another.
    *   `--type local` specifies a local deployment.
    *   `--port 27017` (optional) sets the port. If omitted, it might choose a default or an available port. Using the standard MongoDB port can simplify connection strings.
    *   The CLI will download necessary Docker images (if not already present) and set up a local MongoDB instance. This might take a few minutes the first time.
    *   Note the connection string provided upon successful setup. By default, authentication is typically not enabled for these local deployments.

### Step 3: Connect to the Local Deployment

1.  **Connection String:**
    The typical connection string for a local instance started by Atlas CLI on the default port is:
    `mongodb://localhost:27017/`
    If you named your deployment (e.g., `eat-local-dev`) and it's running, you can also connect using `mongosh` via the Atlas CLI:
    ```bash
    atlas deployments connect eat-local-dev --connectWith mongosh
    ```
    This will directly open a `mongosh` session to the local deployment.

2.  **Configure EAT Project:**
    Update your `.env` file as described in the "Configure EAT Project" section (Section 6), using the appropriate connection string (e.g., `mongodb://localhost:27017/`) and your desired `MONGODB_DATABASE_NAME`.

### Step 4: Create Vector Search Indexes via mongosh

Once connected to your local Atlas deployment using `mongosh` (and after selecting your database with `use YOUR_DATABASE_NAME`), you can create the necessary vector search indexes.
*   Use the `db.collection.createSearchIndex()` method.
*   The logical structures for all 4 recommended vector indexes are detailed in Section 7 ("Create Vector Search Indexes (CRITICAL) - Structure Overview"). Use the mapping definitions provided there to construct your `mongosh` commands. For example, for "Index 1: SmartLibrary Content Embedding", your `mongosh` command would look like:
    ```javascript
    // Ensure you are using the correct database: use YOUR_DATABASE_NAME
    db.eat_components.createSearchIndex({
      name: "idx_components_content_embedding", // This is the EAT internal reference name
      definition: {
        mappings: {
          dynamic: false,
          fields: {
            content_embedding: {
              type: "vector",
              dimensions: YOUR_EMBEDDING_DIMENSION, // Replace with your model's dimension
              similarity: "cosine"
            },
            record_type: { type: "string", analyzer: "keyword" },
            domain: { type: "string", analyzer: "keyword" },
            status: { type: "string", analyzer: "keyword" },
            tags: { type: "string", analyzer: "keyword", "multi": "true" }
          }
        }
      }
    });
    // Repeat for the other 3 indexes, adapting collection name and mapping definitions.
    ```
*   **With this local Atlas deployment, you should be able to create all 4 recommended vector indexes without the 3-index limit of the M0 cloud tier.**

---

## Option 2: MongoDB Atlas (Cloud)

This section guides you through setting up EAT with MongoDB Atlas, a fully managed cloud database service. This is recommended for production or when you prefer a managed cloud solution.

### Step 1: Create an Atlas Account and Cluster

1.  **Sign Up/Log In:** Go to [MongoDB Atlas](https://cloud.mongodb.com/) and sign up for a new account or log in if you already have one.
2.  **Create a Project:** Organize your clusters within a project. If you're new, Atlas might guide you through creating your first project and cluster.
3.  **Build a Database (Create a Cluster):**
    *   Click "Build a Database" or "Create" to start a new cluster.
    *   **Cluster Tier for Vector Search:**
        *   **Free Tier (M0):** The free tier **supports Atlas Vector Search** with up to 512MB of storage and a limit of **3 search indexes per cluster**. This is suitable for getting started and small projects with EAT but requires careful index selection (see Section 8: "Cloud M0 Free Tier Index Limitations").
        *   **Shared Tiers (M2/M5):** These also support Vector Search and offer more resources and higher index limits if your project grows.
        *   **Dedicated Tiers (M10+):** For larger datasets, production workloads, or more demanding performance, consider M10 or higher shared clusters, or any dedicated cluster.
    *   **MongoDB Version for Vector Search:** Ensure your cluster is running a compatible MongoDB version.
        *   For Approximate Nearest Neighbor (ANN) search: MongoDB version **6.0.11+, 7.0.2+, or later**.
        *   For Exact Nearest Neighbor (ENN) search (preview): MongoDB version **7.0.10+, 7.3.2+, or later**.
        *   When creating your cluster, Atlas usually defaults to a recent, compatible version. You can verify this during setup.
    *   Select your preferred cloud provider and region.
    *   Configure additional settings like cluster name (e.g., `EAT-Cluster`), backup, etc., as needed.
    *   Click "Create Cluster." Deployment will take a few minutes.

### Step 2: Configure Network Access

You need to allow your application's IP address to connect to the Atlas cluster.

1.  In your Atlas project, navigate to **Network Access** under the "Security" section in the left sidebar.
2.  Click **"Add IP Address"**.
3.  **Options:**
    *   **"Allow Access From Anywhere" (0.0.0.0/0):** Easiest for development but less secure.
    *   **"Add Current IP Address":** If you're running the EAT project from your current machine.
    *   **Specific IP/CIDR:** If your application will run from a server with a static IP.
4.  Add a description and confirm.

### Step 3: Create a Database User

Your EAT application will need credentials to connect to the database.

1.  In your Atlas project, navigate to **Database Access** under the "Security" section.
2.  Click **"Add New Database User"**.
3.  Choose an **Authentication Method** (e.g., "Password").
4.  Enter a **Username** (e.g., `eat_user`) and **Password**. Store these securely.
5.  **Database User Privileges:**
    *   Assign roles like **"Read and write to any database"** (for development simplicity) or restrict to your specific `MONGODB_DATABASE_NAME`.
6.  Click **"Add User"**.

### Step 4: Get Your Connection String

1.  Navigate to **Database** under the "Deployment" section.
2.  For your cluster, click **"Connect"**.
3.  Choose **"Drivers"**.
4.  Select Driver: **Python** and your `pymongo` version.
5.  Copy the **Connection String (SRV address)**. It will look like:
    `mongodb+srv://<username>:<password>@yourcluster.mongodb.net/?retryWrites=true&w=majority`
6.  Replace `<username>` and `<password>` with your database user's credentials.
    *   ⚠️ **Important:** Ensure any special characters in your password are URL-encoded if necessary.

### Step 5: Create Vector Search Indexes via Atlas UI

For cloud Atlas deployments, you'll use the Atlas UI to create search indexes.
1.  Navigate to your cluster in Atlas, then click the "Search" tab.
2.  For each of the 4 indexes detailed in Section 7 ("Create Vector Search Indexes (CRITICAL) - Structure Overview"):
    *   Click "Create Search Index."
    *   Choose "Atlas Vector Search" and then the "JSON Editor."
    *   Set the **Database** (your `MONGODB_DATABASE_NAME`) and **Collection** (e.g., `eat_components`).
    *   For the **Index Name (in Atlas UI)**, use the EAT internal reference name specified in Section 7 (e.g., `idx_components_content_embedding`).
    *   For the **JSON Configuration**, use the "Core Mapping Definition" provided for that index in Section 7, placing it inside a structure like:
        ```json
        {
          "name": "YOUR_ATLAS_UI_INDEX_NAME_MATCHING_EAT_REF", // e.g., "idx_components_content_embedding"
          "fields": [
            // Paste the content of the "Core Mapping Definition" (the fields array) here for older Atlas UI.
            // For newer Atlas UI that expects the full structure:
            // {
            //   "type": "vector",
            //   "path": "embedding_field_name",
            //   "numDimensions": YOUR_EMBEDDING_DIMENSION,
            //   "similarity": "cosine"
            // },
            // ... other filter fields ...
          ]
        }
        // Newer Atlas UI (from approx. late 2023/early 2024 for Vector Search) expects:
        // {
        //   "name": "YOUR_ATLAS_UI_INDEX_NAME_MATCHING_EAT_REF",
        //   "mappings": {
        //     "dynamic": false, // Or true
        //     "fields": {
        //       "embedding_field_name": {
        //         "type": "vector",
        //         "dimensions": YOUR_EMBEDDING_DIMENSION,
        //         "similarity": "cosine"
        //       },
        //       // ... other filterable fields as objects ...
        //     }
        //   }
        // }
        // Use the "Core Mapping Definition" from Section 7 to fill the "fields" object within "mappings".
        ```
        **Adapt the structure based on what the Atlas UI JSON editor for Vector Search expects.** The "Core Mapping Definitions" in Section 7 provide the `fields` object content.
    *   Replace `YOUR_EMBEDDING_DIMENSION` with your model's dimension.
    *   Click "Next," then "Create Search Index."
3.  If using the M0 Free Tier, be mindful of the 3-index limit (see Section 8).

---

## Option 3: Traditional Self-Hosted MongoDB

If you choose to self-host:

1.  Install MongoDB Community Server or Enterprise Server on your machine/server. Follow the official MongoDB installation guides for your operating system.
2.  Ensure the MongoDB service (`mongod`) is running.
3.  Your connection string will typically be simpler, e.g., `mongodb://localhost:27017/`.
4.  **Vector Search:** For self-hosted vector search, you'll need to investigate MongoDB's current offerings for on-premise vector search. This is significantly more complex to set up and manage than Atlas Vector Search and might involve specific MongoDB versions, configurations, or additional search engine integrations. Ensure your self-hosted version meets the MongoDB version requirements for vector search (e.g., 6.0.11+ or 7.0.2+ for ANN). If your self-hosted MongoDB version supports vector search, you can create all the vector indexes detailed in Section 7 ('Create Vector Search Indexes (CRITICAL) - Structure Overview') without the 3-index limit imposed by the Atlas free tier. The JSON "Core Mapping Definitions" provided for Atlas indexes in Section 7 can be adapted for use with the `db.collection.createSearchIndex()` command in `mongosh` (similar to the example in Option 1, Step 4). **This guide focuses on Atlas Vector Search for detailed UI steps, but the index structures are relevant for self-hosting.**

---

## Configure EAT Project

Once you have your MongoDB connection string (from any of the options):

1.  Navigate to the root directory of your Evolving Agents Toolkit project.
2.  If it doesn't exist, copy `.env.example` to a new file named `.env`:
    ```bash
    cp .env.example .env
    ```
3.  Open the `.env` file in a text editor.
4.  Add or update the following lines, replacing the placeholder with your actual connection string and your desired database name:
    ```env
    MONGODB_URI="mongodb://your_connection_string_details_here/"
    MONGODB_DATABASE_NAME="evolving_agents_db" # Or your preferred database name
    ```
    *   Ensure there are no extra spaces or quotes around the URI or database name in the `.env` file.
    *   The `MONGODB_DATABASE_NAME` will be used by the application to create/connect to the specified database.

The application will now use these environment variables to connect to your MongoDB instance.

---

## Create Vector Search Indexes (CRITICAL) - Structure Overview

This section describes the **logical structure** of the 4 ideal vector search indexes required for full EAT functionality. The specific creation commands depend on your chosen MongoDB setup:
*   **Option 1 (Atlas CLI Local):** Use `db.collection.createSearchIndex()` in `mongosh` with the `definition.mappings` structure shown here (see example in Option 1, Step 4).
*   **Option 2 (Atlas Cloud):** Use the Atlas UI JSON Editor. The "Core Mapping Definition" below corresponds to the `fields` object within the `mappings` object in the Atlas UI's JSON editor for Vector Search indexes.
*   **Option 3 (Traditional Self-Hosted):** Use `db.collection.createSearchIndex()` in `mongosh`, similar to Option 1.

**IMPORTANT:** The `YOUR_EMBEDDING_DIMENSION` in your index definitions **must match** the embedding dimension of your `LLM_EMBEDDING_MODEL`.
*   For `text-embedding-3-small` (OpenAI default), `YOUR_EMBEDDING_DIMENSION` is **1536**.
*   For `text-embedding-3-large` (OpenAI), `YOUR_EMBEDDING_DIMENSION` is **3072**.
*   For `nomic-embed-text` (common with Ollama), `YOUR_EMBEDDING_DIMENSION` might be **768**.
*   **Verify your model's output dimension!**

### Index 1: SmartLibrary Content Embedding (`eat_components` collection)
*   **Target Collection:** `eat_components`
*   **Index Name (for Atlas Search & EAT):** `idx_components_content_embedding`
*   **Purpose:** Enables semantic search of library components by their main content.
*   **Core Mapping Definition (for `mappings.fields` in `mongosh` or Atlas UI):**
    ```json
    {
      "content_embedding": {
        "type": "vector",
        "dimensions": YOUR_EMBEDDING_DIMENSION,
        "similarity": "cosine"
      },
      "record_type": { "type": "string", "analyzer": "keyword" },
      "domain": { "type": "string", "analyzer": "keyword" },
      "status": { "type": "string", "analyzer": "keyword" },
      "tags": { "type": "string", "analyzer": "keyword", "multi": "true" }
    }
    ```

### Index 2: SmartLibrary Applicability Embedding (`eat_components` collection)
*   **Target Collection:** `eat_components`
*   **Index Name (for Atlas Search & EAT):** `applicability_embedding`
*   **Purpose:** Enables task-aware semantic search of library components based on their applicability text (E_raz).
*   **Core Mapping Definition (for `mappings.fields` in `mongosh` or Atlas UI):**
    ```json
    {
      "applicability_embedding": {
        "type": "vector",
        "dimensions": YOUR_EMBEDDING_DIMENSION,
        "similarity": "cosine"
      },
      "record_type": { "type": "string", "analyzer": "keyword" },
      "domain": { "type": "string", "analyzer": "keyword" },
      "status": { "type": "string", "analyzer": "keyword" },
      "tags": { "type": "string", "analyzer": "keyword", "multi": "true" }
    }
    ```

### Index 3: SmartAgentBus Agent Description Embedding (`eat_agent_registry` collection)
*   **Target Collection:** `eat_agent_registry`
*   **Index Name (for Atlas Search & EAT):** `vector_index_agent_description`
*   **Purpose:** Enables semantic search of registered agents by their overall description.
*   **Core Mapping Definition (for `mappings.fields` in `mongosh` or Atlas UI):**
    ```json
    {
      "description_embedding": {
        "type": "vector",
        "dimensions": YOUR_EMBEDDING_DIMENSION,
        "similarity": "cosine"
      },
      "name": { "type": "string", "analyzer": "keyword" },
      "status": { "type": "string", "analyzer": "keyword" },
      "type": { "type": "string", "analyzer": "keyword" }
    }
    ```

### Index 4: Smart Memory Experience Embeddings (`eat_agent_experiences` collection)
*   **Target Collection:** `eat_agent_experiences`
*   **Index Name (for Atlas Search & EAT):** `vector_index_experiences_default`
*   **Purpose:** Enables semantic search of past agent experiences.
*   **Core Mapping Definition (for `mappings.fields` in `mongosh` or Atlas UI):**
    This example primarily indexes `embeddings.primary_goal_description_embedding`. You can add other fields from the `embeddings` object if needed and supported by your indexing strategy.
    ```json
    {
      "embeddings.primary_goal_description_embedding": {
        "type": "vector",
        "dimensions": YOUR_EMBEDDING_DIMENSION,
        "similarity": "cosine"
      }
      // Example for other embedded fields from the 'embeddings' object:
      // ,"embeddings.sub_task_description_embedding": {
      //   "type": "vector",
      //   "dimensions": YOUR_EMBEDDING_DIMENSION,
      //   "similarity": "cosine"
      // }
      // Example for filterable fields from the experience document:
      // ,"initiating_agent_id": { "type": "string", "analyzer": "keyword" },
      // "final_outcome": { "type": "string", "analyzer": "keyword" },
      // "tags": { "type": "string", "analyzer": "keyword", "multi": "true" }
    }
    ```
    *Remember to replace `YOUR_EMBEDDING_DIMENSION`.*

---

## Cloud M0 Free Tier Index Limitations

With the Atlas M0 (Free Tier) limit of 3 search indexes, you need to choose carefully. The EAT framework can potentially leverage several vector indexes:

1.  **`eat_components` (SmartLibrary - Content):** Indexed as `idx_components_content_embedding`. (Essential for core component search by content).
2.  **`eat_components` (SmartLibrary - Applicability):** Indexed as `applicability_embedding`. (Essential for task-aware component search).
3.  **`eat_agent_registry` (Agent Bus - Descriptions):** Indexed as `vector_index_agent_description`. (Optional, for semantic search of agents by their overall description).
4.  **`eat_agent_experiences` (Smart Memory):** Indexed as `vector_index_experiences_default`. (Essential for the new Smart Memory semantic search).

**This means you have 4 potential vector indexes, but only 3 slots on the free tier.**

**Recommendations for Free Tier Users:**
*   **Option A (Prioritize Smart Memory & Full SmartLibrary):**
    1.  `idx_components_content_embedding`
    2.  `applicability_embedding`
    3.  `vector_index_experiences_default`
    (This means omitting the semantic search for agent descriptions in `eat_agent_registry`).
*   **Option B (Prioritize Core EAT without Smart Memory search or with limited SmartLibrary):**
    1.  `idx_components_content_embedding`
    2.  `applicability_embedding`
    3.  `vector_index_agent_description`
    (This means omitting the `eat_agent_experiences` search, or sacrificing one of the `eat_components` indexes if you deem agent description search more critical than one aspect of component search).

Consider your primary use case. If the Smart Memory functionality is central to your current work, Option A is recommended. If you need to reduce further, you might investigate if your version of Atlas allows a single index definition on `eat_components` to effectively serve both content and applicability searches, potentially freeing up a slot. Always refer to the latest MongoDB Atlas documentation on search index capabilities and limits.

---

## Running the Application

Once MongoDB is configured and your `.env` file is updated:

1.  Install dependencies: `pip install -r requirements.txt`
2.  Run any of the example scripts, for instance:
    ```bash
    python examples/invoice_processing/architect_zero_comprehensive_demo.py
    ```

The application should connect to MongoDB. Collections (`eat_components`, `eat_agent_registry`, `eat_agent_bus_logs`, `eat_llm_cache`, `eat_intent_plans`, `eat_agent_experiences`) will be created automatically when data is first written to them, if they don't already exist.

---

## Troubleshooting

*   **Connection Errors:**
    *   Verify your `MONGODB_URI` in `.env` is correct (username, password, cluster address).
    *   Ensure your IP address is whitelisted in Atlas Network Access (for cloud).
    *   Verify your local Atlas deployment or self-hosted MongoDB is running and accessible on the configured port/address.
    *   Check for firewall issues if self-hosting.
*   **Authentication Errors:**
    *   Double-check the username and password for your database user (primarily for cloud Atlas). Local Atlas CLI deployments often don't have auth enabled by default.
    *   Ensure the user has the necessary read/write permissions for the specified database.
*   **Vector Search "Index Not Found" Errors or Poor Results:**
    *   **Verify Atlas Search Index Names:** Ensure the names you gave your Search Index Definitions in the Atlas UI or `mongosh` *exactly match* the names expected by the code (e.g., `idx_components_content_embedding`, `applicability_embedding`, etc.).
    *   **Verify `path` and `numDimensions`:** In each index definition, ensure the `path` correctly points to the embedding field in your documents (e.g., `content_embedding`, `applicability_embedding`, `embeddings.primary_goal_description_embedding`) and that `numDimensions` matches your LLM's embedding output.
    *   **Index Status:** Check the status of your search indexes. In Atlas UI ("Search" tab), they should be "Active." For local/self-hosted, you can use `db.collection.getSearchIndexes()` in `mongosh`. Building can take a few minutes after creation.
    *   **MongoDB Version:** Verify your Atlas cluster or local/self-hosted instance meets the MongoDB version requirements for Vector Search (e.g., 6.0.11+, 7.0.2+). Atlas CLI local deployments should use a compatible version.
*   **`pymongo.errors.ConfigurationError: ... requires a an S PKCS #8 private key ...`:**
    *   This can happen with some Python versions or environments (especially on macOS) if TLS/SSL certificates are not correctly configured.
    *   Try installing `certifi`: `pip install certifi`
    *   You might need to pass `tlsCAFile=certifi.where()` to your `MongoClient` call, or ensure your Python installation has up-to-date CA certificates. (The EAT `MongoDBClient` would need to be modified to support this if it becomes a common issue). For EAT, ensure your system's Python environment can make secure TLS connections. This is less common with `localhost` connections.
*   **TypeErrors for Embeddings (e.g., `BSONTypeError: EOO needed something of type 'array' but got 'ndarray'`):**
    *   Ensure that embeddings (lists of floats) are being stored as Python `list` objects, not NumPy arrays, directly into MongoDB. The EAT framework generally handles this conversion.
*   **Ensure `.env` is Loaded:** If you see errors about missing API keys or MongoDB URI, double-check that your `.env` file is in the project root and correctly formatted.
*   **Docker Issues (for Option 1):** Ensure Docker is running and has sufficient resources if you are using the Atlas CLI local deployment. Check `atlas deployments logs eat-local-dev` for issues.

---

You are now ready to use the Evolving Agents Toolkit with a unified MongoDB backend!