# MongoDB Setup Guide for Evolving Agents Toolkit (EAT)

This guide provides instructions for setting up MongoDB as the unified backend for the Evolving Agents Toolkit (EAT), replacing the previous JSON file and ChromaDB setup. Using MongoDB, especially with Atlas Vector Search, simplifies the data stack and enhances performance and scalability.

**Contents:**

1.  [Prerequisites](#prerequisites)
2.  [Choosing Your MongoDB Option](#choosing-your-mongodb-option)
3.  [Option 1: MongoDB Atlas (Recommended)](#option-1-mongodb-atlas-recommended)
    *   [Step 1: Create an Atlas Account and Cluster](#step-1-create-an-atlas-account-and-cluster)
    *   [Step 2: Configure Network Access](#step-2-configure-network-access)
    *   [Step 3: Create a Database User](#step-3-create-a-database-user)
    *   [Step 4: Get Your Connection String](#step-4-get-your-connection-string)
4.  [Option 2: Self-Hosted MongoDB](#option-2-self-hosted-mongodb)
5.  [Configure EAT Project](#configure-eat-project)
6.  [Create Vector Search Indexes (CRITICAL)](#create-vector-search-indexes-critical)
    *   [SmartLibrary Component Embeddings (`eat_components` collection)](#smartlibrary-component-embeddings-eat_components-collection)
    *   [SmartAgentBus Agent Description Embeddings (`eat_agent_registry` collection)](#smartagentbus-agent-description-embeddings-eat_agent_registry-collection)
    *   [Free Tier Considerations (3 Index Limit)](#free-tier-considerations-3-index-limit)
7.  [Running the Application](#running-the-application)
8.  [Troubleshooting](#troubleshooting)

---

## Prerequisites

*   Python 3.11+
*   `pymongo` and `motor` (Python drivers for MongoDB, included in `requirements.txt`)
*   An OpenAI API key (or other LLM provider configured) for generating embeddings.
*   Access to a MongoDB instance (MongoDB Atlas Free Tier is sufficient for getting started).

---

## Choosing Your MongoDB Option

You have two main options for your MongoDB setup:

1.  **MongoDB Atlas (Recommended):** A fully managed cloud database service by MongoDB. This is the easiest way to get started, especially for leveraging **Atlas Vector Search**.
2.  **Self-Hosted MongoDB:** Running MongoDB on your own server or local machine. Setting up robust vector search capabilities in a self-hosted environment can be more complex than with Atlas.

This guide will primarily focus on **MongoDB Atlas** due to its integrated Vector Search functionality.

---

## Option 1: MongoDB Atlas (Recommended)

### Step 1: Create an Atlas Account and Cluster

1.  **Sign Up/Log In:** Go to [MongoDB Atlas](https://cloud.mongodb.com/) and sign up for a new account or log in if you already have one.
2.  **Create a Project:** Organize your clusters within a project. If you're new, Atlas might guide you through creating your first project and cluster.
3.  **Build a Database (Create a Cluster):**
    *   Click "Build a Database" or "Create" to start a new cluster.
    *   **Cluster Tier for Vector Search:**
        *   **Free Tier (M0):** The free tier **supports Atlas Vector Search** with up to 512MB of storage and a limit of **3 search indexes per cluster**. This is suitable for getting started and small projects with EAT.
        *   **Shared Tiers (M2/M5):** These also support Vector Search and offer more resources and higher index limits if your project grows.
        *   **Dedicated Tiers (M10+):** For larger datasets, production workloads, or more demanding performance, consider M10 or higher shared clusters, or any dedicated cluster.
    *   **MongoDB Version for Vector Search:** Ensure your cluster is running a compatible MongoDB version.
        *   For Approximate Nearest Neighbor (ANN) search: MongoDB version **6.0.11+, 7.0.2+, or later**.
        *   For Exact Nearest Neighbor (ENN) search (preview): MongoDB version **7.0.10+, 7.3.2+, or later**.
        *   When creating your cluster, Atlas usually defaults to a recent, compatible version. You can verify this during setup.
    *   Select your preferred cloud provider and region (choose one geographically close to you or your application servers).
    *   Configure additional settings like cluster name (e.g., `EAT-Cluster`), backup (not typically needed for M0 dev), etc., as needed.
    *   Click "Create Cluster." Deployment will take a few minutes.

### Step 2: Configure Network Access

You need to allow your application's IP address to connect to the Atlas cluster.

1.  In your Atlas project, navigate to **Network Access** under the "Security" section in the left sidebar.
2.  Click **"Add IP Address"**.
3.  **Options:**
    *   **"Allow Access From Anywhere" (0.0.0.0/0):** Easiest for development but less secure. Use with caution and consider restricting it later.
    *   **"Add Current IP Address":** If you're running the EAT project from your current machine. This is a good default for local development.
    *   **Specific IP/CIDR:** If your application will run from a server with a static IP or a specific range.
4.  Add a description (e.g., "My Dev Machine" or "EAT Development Access") and confirm. It might take a minute for the network rule to become active.

### Step 3: Create a Database User

Your EAT application will need credentials to connect to the database.

1.  In your Atlas project, navigate to **Database Access** under the "Security" section.
2.  Click **"Add New Database User"**.
3.  Choose an **Authentication Method**. "Password" is common for application users.
4.  Enter a **Username** (e.g., `eat_user`) and **Password**. Use a strong, unique password and securely store these credentials; you'll need them for the connection string.
5.  **Database User Privileges:**
    *   For simplicity during development, you can assign built-in roles like **"Read and write to any database"**.
    *   For better security, especially in later stages or production, select **"Only read and write to specific databases"** and specify the database name you intend to use (e.g., `evolving_agents_db` as configured in your `.env` file). You might also need to grant the `atlasAdmin` role if your application needs to manage search indexes (though EAT currently requires manual index creation). For EAT's core functionality, read/write to its specific database is usually sufficient once indexes are set up.
6.  Click **"Add User"**.

### Step 4: Get Your Connection String

This string allows your application to connect to the Atlas cluster.

1.  Navigate to **Database** under the "Deployment" section in the left sidebar.
2.  For your cluster, click the **"Connect"** button.
3.  A modal will pop up. Choose **"Drivers"** (it might be labeled "Connect your application" or similar).
4.  Under "1. Select your driver and version", choose:
    *   Driver: **Python**
    *   Version: Select a version compatible with your installed `pymongo` (e.g., 3.11 or later).
5.  You will see a **Connection String (SRV address)**. It will look something like:
    `mongodb+srv://<username>:<password>@yourcluster.mongodb.net/?retryWrites=true&w=majority`
6.  **Copy this string.** You will need to replace `<username>` and `<password>` with the credentials of the database user you created in Step 3.
    *   ⚠️ **Important:** When replacing `<password>`, ensure any special characters in your password are URL-encoded if necessary, though PyMongo usually handles this well. If you encounter issues, try URL-encoding special characters in the password.

---

## Option 2: Self-Hosted MongoDB

If you choose to self-host:

1.  Install MongoDB Community Server or Enterprise Server on your machine/server. Follow the official MongoDB installation guides for your operating system.
2.  Ensure the MongoDB service (`mongod`) is running.
3.  Your connection string will typically be simpler, e.g., `mongodb://localhost:27017/`.
4.  **Vector Search:** For self-hosted vector search, you'll need to investigate MongoDB's current offerings for on-premise vector search. This is significantly more complex to set up and manage than Atlas Vector Search and might involve specific MongoDB versions, configurations, or additional search engine integrations (like Lucene-based search with vector capabilities if supported). Ensure your self-hosted version meets the MongoDB version requirements for vector search (e.g., 6.0.11+ or 7.0.2+ for ANN). **This guide focuses on Atlas Vector Search.**

---

## Configure EAT Project

Once you have your MongoDB connection string:

1.  Navigate to the root directory of your Evolving Agents Toolkit project.
2.  If it doesn't exist, copy `.env.example` to a new file named `.env`:
    ```bash
    cp .env.example .env
    ```
3.  Open the `.env` file in a text editor.
4.  Add or update the following lines, replacing the placeholder with your actual connection string (with your username and password inserted) and your desired database name:
    ```env
    MONGODB_URI="mongodb+srv://your_username:your_password@yourcluster.mongodb.net/?retryWrites=true&w=majority"
    MONGODB_DATABASE_NAME="evolving_agents_db" # Or your preferred database name
    ```
    *   Ensure there are no extra spaces or quotes around the URI or database name in the `.env` file.
    *   The `MONGODB_DATABASE_NAME` will be used by the application to create/connect to the specified database.

The application will now use these environment variables to connect to your MongoDB instance.

---

## Create Vector Search Indexes (CRITICAL)

For the `SmartLibrary` and `SmartAgentBus` semantic search to function correctly, you **must** create Vector Search Indexes in MongoDB Atlas. The application code *does not* create these specific types of indexes.

**IMPORTANT:** The `numDimensions` in your index definition **must match** the embedding dimension of your `LLM_EMBEDDING_MODEL` (defined in `.env` or `evolving_agents/config.py`).
*   For `text-embedding-3-small` (OpenAI default), `numDimensions` is **1536**.
*   For `text-embedding-3-large` (OpenAI), `numDimensions` is **3072**.
*   For `nomic-embed-text` (common with Ollama), `numDimensions` might be **768** or as specified by the model.
*   **Verify your model's output dimension!**

The Atlas Free Tier (M0) allows up to **3 search indexes**. The following setup describes the recommended 3 indexes for optimal EAT functionality.

### SmartLibrary Component Embeddings (`eat_components` collection)

You need **two** vector search indexes on the `eat_components` collection. This collection will be created automatically by the application if it doesn't exist when data is first written.

1.  **Navigate to Atlas Search:**
    *   In your MongoDB Atlas dashboard, go to your cluster.
    *   Click on the "Search" tab (or it might be under "Data Services" -> "Atlas Search").
    *   If you haven't used search before, you might need to enable it by clicking "Create Search Index" and then choosing the "Atlas Vector Search" option.
2.  **Create Index (using JSON Editor):**
    *   Click "Create Search Index."
    *   Choose "Atlas Vector Search" as the configuration type/builder.
    *   Select the "JSON Editor" option for configuration.
    *   Set the **Database and Collection**:
        *   Database: Your `MONGODB_DATABASE_NAME` (e.g., `evolving_agents_db`).
        *   Collection: `eat_components`.
    *   Give the index a **Name** in the Atlas UI (this is the name for the Search Index Definition itself).

3.  **Index 1: For `content_embedding` (E_orig)**
    *   **Atlas Search Index Name (in Atlas UI):** `idx_components_content_embedding`
        *(This name is referenced in `SmartLibrary.py`)*
    *   **JSON Configuration:**
        ```json
        {
          "name": "idx_components_content_embedding",
          "fields": [
            {
              "type": "vector",
              "path": "content_embedding",
              "numDimensions": YOUR_EMBEDDING_DIMENSION,
              "similarity": "cosine"
            },
            { "type": "filter", "path": "record_type" },
            { "type": "filter", "path": "domain" },
            { "type": "filter", "path": "status" },
            { "type": "filter", "path": "tags" }
          ]
        }
        ```
    *   Replace `YOUR_EMBEDDING_DIMENSION` with the correct number (e.g., 1536).
    *   Click "Next," then "Create Search Index."

4.  **Index 2: For `applicability_embedding` (E_raz)**
    *   **Atlas Search Index Name (in Atlas UI):** `applicability_embedding`
        *(This name is referenced in `SmartLibrary.py` based on your confirmed setup)*
    *   **JSON Configuration:**
        ```json
        {
          "name": "applicability_embedding",
          "fields": [
            {
              "type": "vector",
              "path": "applicability_embedding",
              "numDimensions": YOUR_EMBEDDING_DIMENSION,
              "similarity": "cosine"
            },
            { "type": "filter", "path": "record_type" },
            { "type": "filter", "path": "domain" },
            { "type": "filter", "path": "status" },
            { "type": "filter", "path": "tags" }
          ]
        }
        ```
    *   Replace `YOUR_EMBEDDING_DIMENSION` with the correct number (e.g., 1536).
    *   Click "Next," then "Create Search Index."

Index building can take a few minutes. You can monitor its status in the Atlas Search tab. It should show "Active" when ready.

### SmartAgentBus Agent Description Embeddings (`eat_agent_registry` collection)

You need **one** vector search index on the `eat_agent_registry` collection for `SmartAgentBus` to semantically search agents by their overall description. This collection will be created automatically by the application.

1.  **Navigate to Atlas Search** as above.
2.  **Create Index (using JSON Editor):**
    *   Database: Your `MONGODB_DATABASE_NAME`.
    *   Collection: `eat_agent_registry`.
    *   **Atlas Search Index Name (in Atlas UI):** `vector_index_agent_description`
        *(This name is referenced in `SmartAgentBus.py`)*
    *   **JSON Configuration:**
        ```json
        {
          "name": "vector_index_agent_description",
          "fields": [
            {
              "type": "vector",
              "path": "description_embedding",
              "numDimensions": YOUR_EMBEDDING_DIMENSION,
              "similarity": "cosine"
            },
            { "type": "filter", "path": "name" },
            { "type": "filter", "path": "status" },
            { "type": "filter", "path": "type" }
            // Optional: { "type": "filter", "path": "capabilities.id" }
          ]
        }
        ```
    *   Replace `YOUR_EMBEDDING_DIMENSION` with the correct number (e.g., 1536).
    *   Click "Next," then "Create Search Index."

### Free Tier Considerations (3 Index Limit)

With the Atlas M0 (Free Tier) limit of 3 search indexes, the setup described above (2 for `eat_components` and 1 for `eat_agent_registry`) utilizes all available slots and provides the core semantic search capabilities for EAT.

If you were previously using an index for `capabilities.capability_description_embedding` on `eat_agent_registry`, you would need to decide which feature is more important or upgrade your Atlas tier to support more indexes. The current `SmartAgentBus.discover_agents(task_description="...")` is designed to use the index on the agent's overall `description_embedding`.

---

## Running the Application

Once MongoDB is configured and your `.env` file is updated:

1.  Install dependencies: `pip install -r requirements.txt`
2.  Run any of the example scripts, for instance:
    ```bash
    python examples/invoice_processing/architect_zero_comprehensive_demo.py
    ```

The application should connect to MongoDB. Collections (`eat_components`, `eat_agent_registry`, `eat_agent_bus_logs`, `eat_llm_cache`, `eat_intent_plans`) will be created automatically when data is first written to them, if they don't already exist.

---

## Troubleshooting

*   **Connection Errors:**
    *   Verify your `MONGODB_URI` in `.env` is correct (username, password, cluster address).
    *   Ensure your IP address is whitelisted in Atlas Network Access.
    *   Check for firewall issues if self-hosting.
*   **Authentication Errors:**
    *   Double-check the username and password for your database user.
    *   Ensure the user has the necessary read/write permissions for the specified database.
*   **Vector Search "Index Not Found" Errors or Poor Results:**
    *   **Verify Atlas Search Index Names:** Ensure the names you gave your Search Index Definitions in the Atlas UI *exactly match* the names expected by the code:
        *   For `eat_components` collection: One index definition should be named `idx_components_content_embedding` and the other `applicability_embedding`.
        *   For `eat_agent_registry` collection: The index definition should be named `vector_index_agent_description`.
    *   **Verify `path` and `numDimensions`:** In each index definition, ensure the `path` correctly points to the embedding field in your documents (e.g., `content_embedding`, `applicability_embedding`, `description_embedding`) and that `numDimensions` matches your LLM's embedding output.
    *   **Index Status:** Check the status of your search indexes in the Atlas UI ("Search" tab for your cluster). They should be "Active." Building can take a few minutes after creation.
    *   **MongoDB Version:** Verify your Atlas cluster meets the MongoDB version requirements for Vector Search (e.g., 6.0.11+, 7.0.2+).
*   **`pymongo.errors.ConfigurationError: ... requires a an S PKCS #8 private key ...`:**
    *   This can happen with some Python versions or environments (especially on macOS) if TLS/SSL certificates are not correctly configured.
    *   Try installing `certifi`: `pip install certifi`
    *   You might need to pass `tlsCAFile=certifi.where()` to your `MongoClient` call, or ensure your Python installation has up-to-date CA certificates. (The EAT `MongoDBClient` would need to be modified to support this if it becomes a common issue). For EAT, ensure your system's Python environment can make secure TLS connections.
*   **TypeErrors for Embeddings (e.g., `BSONTypeError: EOO needed something of type 'array' but got 'ndarray'`):**
    *   Ensure that embeddings (lists of floats) are being stored as Python `list` objects, not NumPy arrays, directly into MongoDB. The `SmartLibrary.save_record()` method includes a conversion step for this.
*   **Ensure `.env` is Loaded:** If you see errors about missing API keys or MongoDB URI, double-check that your `.env` file is in the project root and correctly formatted (no extra spaces, no comments on the same line as values).

---

You are now ready to use the Evolving Agents Toolkit with a unified MongoDB backend!