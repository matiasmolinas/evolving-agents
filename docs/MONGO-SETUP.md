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
    *   [SmartLibrary Component Embeddings](#smartlibrary-component-embeddings)
    *   [SmartAgentBus Capability Embeddings (Optional)](#smartagentbus-capability-embeddings-optional)
7.  [Running the Application](#running-the-application)
8.  [Troubleshooting](#troubleshooting)

---

## Prerequisites

*   Python 3.11+
*   `pip` and `venv` (recommended for virtual environments)
*   The Evolving Agents Toolkit project cloned to your local machine.
*   `pymongo` library: Install it in your project's virtual environment:
    ```bash
    pip install pymongo
    ```

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
2.  **Create a Project:** Organize your clusters within a project.
3.  **Build a Database (Create a Cluster):**
    *   **Cluster Tier for Vector Search:**
        *   **Free Tier (M0):** The free tier **supports Atlas Vector Search** with up to 512MB of storage. This is suitable for getting started and small projects.
        *   **Shared Tiers (M2/M5):** These also support Vector Search and offer more resources.
        *   **Dedicated Tiers (M10+):** For larger datasets, production workloads, or more demanding performance, consider M10 or higher shared clusters, or any dedicated cluster.
    *   **MongoDB Version for Vector Search:** Ensure your cluster is running a compatible MongoDB version.
        *   For Approximate Nearest Neighbor (ANN) search: MongoDB version **6.0.11+, 7.0.2+, or later**.
        *   For Exact Nearest Neighbor (ENN) search (preview): MongoDB version **7.0.10+, 7.3.2+, or later**.
        *   When creating your cluster, Atlas usually defaults to a recent, compatible version.
    *   Select your preferred cloud provider and region.
    *   Configure additional settings like cluster name, backup, etc., as needed.
    *   Click "Create Cluster." Deployment will take a few minutes.

### Step 2: Configure Network Access

You need to allow your application's IP address to connect to the Atlas cluster.

1.  In your Atlas project, navigate to **Network Access** under the "Security" section in the left sidebar.
2.  Click **"Add IP Address"**.
3.  **Options:**
    *   **"Allow Access From Anywhere" (0.0.0.0/0):** Easiest for development but less secure. Use with caution.
    *   **"Add Current IP Address":** If you're running the EAT project from your current machine.
    *   **Specific IP/CIDR:** If your application will run from a server with a static IP.
4.  Add a description (e.g., "My Dev Machine") and confirm.

### Step 3: Create a Database User

Your EAT application will need credentials to connect to the database.

1.  In your Atlas project, navigate to **Database Access** under the "Security" section.
2.  Click **"Add New Database User"**.
3.  Choose an **Authentication Method**. "Password" is common.
4.  Enter a **Username** and **Password**. Securely store these credentials; you'll need them for the connection string.
5.  **Database User Privileges:**
    *   You can assign built-in roles like "Read and write to any database" for simplicity during development. For production, use more granular permissions.
    *   Alternatively, select "Only read and write to specific databases" and specify the database name you intend to use (e.g., `evolving_agents_db`).
6.  Click **"Add User"**.

### Step 4: Get Your Connection String

This string allows your application to connect to the Atlas cluster.

1.  Navigate to **Database** under the "Deployment" section.
2.  For your cluster, click the **"Connect"** button.
3.  Choose **"Drivers"** (or "Connect your application").
4.  Select **Python** as your driver and the **version** of PyMongo you have installed (e.g., 3.11 or later).
5.  You will see a **Connection String (SRV address)**. It will look something like:
    `mongodb+srv://<username>:<password>@yourcluster.mongodb.net/?retryWrites=true&w=majority`
6.  **Copy this string.** You will need to replace `<username>` and `<password>` with the credentials of the database user you created in Step 3.
    *   ⚠️ **Important:** When replacing `<password>`, ensure any special characters in your password are URL-encoded if necessary, though PyMongo usually handles this well.

---

## Option 2: Self-Hosted MongoDB

If you choose to self-host:

1.  Install MongoDB Community Server or Enterprise Server on your machine/server. Follow the official MongoDB installation guides for your operating system.
2.  Ensure the MongoDB service (`mongod`) is running.
3.  Your connection string will typically be simpler, e.g., `mongodb://localhost:27017/`.
4.  **Vector Search:** For self-hosted vector search, you'll need to investigate MongoDB's current offerings for on-premise vector search, which might involve specific versions, configurations, or additional search engine integrations (like Lucene-based search with vector capabilities if supported). This is significantly more complex to set up and manage than Atlas Vector Search. Ensure your self-hosted version meets the MongoDB version requirements for vector search (e.g., 6.0.11+ or 7.0.2+).

---

## Configure EAT Project

Once you have your MongoDB connection string:

1.  Navigate to the root directory of your Evolving Agents Toolkit project.
2.  If it doesn't exist, copy `.env.example` to a new file named `.env`:
    ```bash
    cp .env.example .env
    ```
3.  Open the `.env` file in a text editor.
4.  Add or update the following lines:
    ```env
    MONGODB_URI="your_mongodb_srv_connection_string_with_username_and_password"
    MONGODB_DATABASE_NAME="evolving_agents_db" # Or your preferred database name
    ```
    Replace `"your_mongodb_srv_connection_string_with_username_and_password"` with the actual connection string you obtained, making sure to insert your database username and password.

The application will now use these environment variables to connect to your MongoDB instance.

---

## Create Vector Search Indexes (CRITICAL)

For the `SmartLibrary`'s semantic search to function correctly, you **must** create Vector Search Indexes in MongoDB Atlas on the `eat_components` collection. The application code *does not* create these specific types of indexes.

**IMPORTANT:** The `numDimensions` in your index definition **must match** the embedding dimension of your `LLM_EMBEDDING_MODEL` (defined in `.env` or `evolving_agents/config.py`).
*   For `text-embedding-3-small` (OpenAI default), `numDimensions` is **1536**.
*   For `text-embedding-3-large` (OpenAI), `numDimensions` is **3072**.
*   For `nomic-embed-text` (common with Ollama), `numDimensions` might be **768** or as specified by the model.
*   **Verify your model's output dimension!**

### SmartLibrary Component Embeddings

You need two vector search indexes on the `eat_components` collection (this collection will be created automatically by the application if it doesn't exist when data is first written).

1.  **Navigate to Atlas:**
    *   In your MongoDB Atlas dashboard, go to your cluster.
    *   Click on the "Search" tab (or it might be under "Data Services" -> "Atlas Search").
    *   If you haven't used search before, you might need to enable it.
2.  **Create Index:**
    *   Click "Create Search Index."
    *   Choose "Atlas Vector Search" as the type.
    *   Select "JSON Editor" for configuration.
    *   Set the **Database and Collection**:
        *   Database: Your `MONGODB_DATABASE_NAME` (e.g., `evolving_agents_db`).
        *   Collection: `eat_components`.
    *   Give the index a **Name**.

3.  **Index 1: For `content_embedding` (E_orig)**
    *   **Index Name Example:** `idx_components_content_embedding`
    *   **JSON Configuration:**
        ```json
        {
          "fields": [
            {
              "type": "vector",
              "path": "content_embedding",
              "numDimensions": YOUR_EMBEDDING_DIMENSION, // E.g., 1536 for text-embedding-3-small
              "similarity": "cosine"
            }
            // You can add other fields to filter on here if needed, e.g.,
            // { "type": "filter", "path": "record_type" },
            // { "type": "filter", "path": "domain" }
          ]
        }
        ```
    *   Replace `YOUR_EMBEDDING_DIMENSION` with the correct number.
    *   Click "Next," then "Create Search Index."

4.  **Index 2: For `applicability_embedding` (E_raz)**
    *   Repeat the "Create Index" process.
    *   **Index Name Example:** `idx_components_applicability_embedding`
    *   **JSON Configuration:**
        ```json
        {
          "fields": [
            {
              "type": "vector",
              "path": "applicability_embedding",
              "numDimensions": YOUR_EMBEDDING_DIMENSION, // E.g., 1536 for text-embedding-3-small
              "similarity": "cosine"
            }
            // Add filterable fields as needed:
            // { "type": "filter", "path": "record_type" },
            // { "type": "filter", "path": "domain" }
          ]
        }
        ```
    *   Replace `YOUR_EMBEDDING_DIMENSION` with the correct number.
    *   Click "Next," then "Create Search Index."

Index building can take a few minutes. You can monitor its status in the Atlas Search tab.

### SmartAgentBus Capability Embeddings (Optional)

If you plan to use semantic discovery for agent capabilities directly via MongoDB vector search (rather than just metadata matching), you'll need an index on the `eat_agent_registry` collection.

1.  **Navigate to Atlas Search** as above.
2.  **Create Index:**
    *   Database: Your `MONGODB_DATABASE_NAME`.
    *   Collection: `eat_agent_registry`.
    *   **Index Name Example:** `idx_agent_registry_capability_embedding`
    *   **JSON Configuration (if `capability_description_embedding` is a field within each capability object in an array called `capabilities`):**
        ```json
        {
          "fields": [
            {
              "type": "vector",
              "path": "capabilities.capability_description_embedding", // Adjust path if structure is different
              "numDimensions": YOUR_EMBEDDING_DIMENSION,
              "similarity": "cosine"
            }
            // Filterable fields for agents:
            // { "type": "filter", "path": "name" },
            // { "type": "filter", "path": "status" }
          ]
        }
        ```
    *   This setup assumes `capability_description_embedding` is a field within objects in the `capabilities` array. The actual implementation of storing these embeddings in `SmartAgentBus` will dictate the correct `path`. If capability embeddings are stored differently, adjust the path.

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
*   **Vector Search Not Working / Slow:**
    *   Verify your Atlas cluster meets the MongoDB version requirements for Vector Search (e.g., 6.0.11+, 7.0.2+). Free tier (M0) now supports this with limitations. For larger deployments, M10+ or dedicated clusters are recommended.
    *   Ensure the Vector Search Indexes are created correctly in Atlas for the `eat_components` collection, with the **correct `numDimensions`** matching your embedding model, and on the correct fields (`content_embedding`, `applicability_embedding`).
    *   Check the status of your search indexes in the Atlas UI; they should be "Active."
*   **`pymongo.errors.ConfigurationError: ... requires a an S PKCS #8 private key ...`:**
    *   This can happen with some Python versions or environments on macOS if TLS/SSL certificates are not correctly configured.
    *   Try installing `certifi`: `pip install certifi`
    *   You might need to pass `tlsCAFile=certifi.where()` to your `MongoClient` call, or ensure your Python installation has up-to-date CA certificates. (The EAT `MongoDBClient` would need to be modified to support this if it becomes a common issue).
*   **TypeErrors for Embeddings (e.g., `BSONTypeError: EOO needed something of type 'array' but got 'ndarray'`):**
    *   Ensure that embeddings (lists of floats) are being stored as Python `list` objects, not NumPy arrays, directly into MongoDB. The `SmartLibrary.save_record()` method provided in the migration guide includes a conversion step.

---

You are now ready to use the Evolving Agents Toolkit with a unified MongoDB backend!