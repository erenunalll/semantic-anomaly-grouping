# Technical Specification: Semantic Anomaly Grouping & Deduplication Engine

* **Project Duration**: 6 Weeks
* **Target Audience**: Software Engineer / Data Scientist / Integration Engineer
* **Tech Stack**: Python (USearch preferred, Faiss CPU acceptable, requests/urllib), Docker, Ollama / HuggingFace TEI / vLLM

---

## 📝 Project Summary & Objectives

This project specifies the design and implementation of a **Semantic Anomaly Grouping and Deduplication microservice**.

### Problem Statement:
The parent application currently identifies duplicate error logs (anomalies) by analyzing pairs of error messages using a generative Large Language Model (LLM). This pairwise approach grows quadratically ($O(N^2)$ checks), introduces significant execution latency, and incurs high API costs. Furthermore, string-matching heuristics fail to detect anomalies that express identical semantic meanings but use different wordings (e.g. `"Failed to resolve hostname"` vs `"DNS lookup failure"`).

### Project Goals:
The objective is to implement a Python-based engine that represents error messages as dense mathematical vectors (embeddings) and groups them using a threshold-based similarity graph. 
* Instead of running costly generative LLM calls for every pair, the service will run a local embedding model and use an approximate nearest-neighbor (ANN) vector index to retrieve only the most relevant candidate neighbors for each anomaly.
* A graph connected-component (or Union-Find) algorithm will group similar anomalies together based on a configured similarity threshold (e.g., $\ge 0.82$).
* The service will automatically select a representative title (a "canonical title") for each duplicate group using metadata rules (e.g., selecting the most frequent or descriptive log title).
* The engine will be packaged as a containerized **FastAPI** web service to be queried by the main Spring Boot backend.

---

## 🌐 Model Server Interaction

To ensure the script remains lightweight and optimized, the microservice must not load PyTorch or HuggingFace libraries directly. Instead, the embedding model runs in a dedicated local model server (e.g., Ollama, HuggingFace TEI, or vLLM) on `localhost`. 

The Python script must connect to this local model server using the standard OpenAI-compatible embeddings endpoint:
* **Default Endpoint**: `http://localhost:11434/v1/embeddings` (Ollama), `http://localhost:8080/v1/embeddings` (HuggingFace TEI), or `http://localhost:8000/v1/embeddings` (vLLM).
* **Payload Format**: OpenAI-compatible Embedding Request:
  ```json
  {
    "model": "all-minilm",
    "input": ["Text 1", "Text 2"]
  }
  ```
* **Response Format**: OpenAI-compatible Embedding Response:
  ```json
  {
    "data": [
      { "embedding": [0.12, -0.43, ...] },
      { "embedding": [-0.05, 0.81, ...] }
    ]
  }
  ```

The implementation must support batching (sending all texts in a single request to the model server) to minimize round-trip latency.

---

## 📥 JSON Input Contract Specification (via `stdin`)

The Python CLI tool and the FastAPI endpoints must accept a single, UTF-8 encoded JSON object.

### Input JSON Structure:
- **`model_server_url`** *(string, optional)*: The HTTP URL of the local model server (default: `http://localhost:11434/v1/embeddings`).
- **`model_name`** *(string, optional)*: The model name to pass to the model server (default: `all-minilm`).
- **`threshold`** *(float, required)*: The minimum cosine similarity score $[0.0, 1.0]$ required to establish a match.
- **`top_k`** *(integer, required)*: The maximum number of final reported matches involving any individual item.
- **`items`** *(array of objects, required)*: The text documents to analyze. Each object contains:
  - **`id`** *(string, required)*: A unique identifier for the document.
  - **`text`** *(string, required)*: The text string to embed and analyze.
  - **`total_count`** *(integer, optional)*: The frequency of this log's occurrence (used for canonical title selection).

### Example Input Payload:
```json
{
  "model_server_url": "http://localhost:11434/v1/embeddings",
  "model_name": "all-minilm",
  "threshold": 0.80,
  "top_k": 2,
  "items": [
    {
      "id": "doc-a",
      "text": "Database connection timed out during handshake",
      "total_count": 142
    },
    {
      "id": "doc-b",
      "text": "Failed to connect to the database: connection timeout",
      "total_count": 8
    },
    {
      "id": "doc-c",
      "text": "NullPointerException: user profile object is null",
      "total_count": 35
    }
  ]
}
```

---

## 📤 JSON Output Contract Specification (via `stdout` and `POST /similarity`)

The CLI tool and the `POST /similarity` endpoint must output a single JSON object.

### Output JSON Structure:
- **`candidate_pairs`** *(array of objects)*: List of matches exceeding the threshold. Each object contains:
  - **`left_id`** *(string)*: The ID of one matching document.
  - **`right_id`** *(string)*: The ID of the other matching document.
  - **`similarity`** *(float)*: The calculated cosine similarity score between the two items, rounded to 4 decimal places.

### Formatting Rules:
1. **No Output Pollution**: Print **only** the raw JSON string to `stdout`. Do not output informational logs, progress bars, debug traces, or warning banners to `stdout`. Any logs, debug statements, or execution updates must be sent exclusively to standard error (`stderr`).
2. **Bidirectional Uniqueness**: Each matching pair must be reported exactly once. Do not return duplicate pairs in reverse order (e.g., if you return `{"left_id": "doc-a", "right_id": "doc-b"}`, do not also return `{"left_id": "doc-b", "right_id": "doc-a"}`).
3. **ID Sorting Convention**: Ensure that `left_id` is lexicographically smaller than `right_id` (i.e. `left_id < right_id`).

### Example Output Payload:
```json
{
  "candidate_pairs": [
    {
      "left_id": "doc-a",
      "right_id": "doc-b",
      "similarity": 0.9142
    }
  ]
}
```

---

## 📤 JSON Cluster Output Contract Specification (`POST /cluster`)

The `POST /cluster` endpoint must accept the same input JSON structure as the CLI and `POST /similarity` endpoint. It must return a single JSON object containing the candidate edges and the duplicate groups derived from those edges.

### Cluster Output JSON Structure:
- **`candidate_pairs`** *(array of objects)*: The same threshold-qualified edge list returned by `/similarity`.
- **`groups`** *(array of objects)*: Connected duplicate groups derived from `candidate_pairs`. Each object contains:
  - **`group_id`** *(string)*: A deterministic group identifier. Use the lexicographically smallest member ID unless the backend later provides a persistent group ID.
  - **`canonical_id`** *(string)*: The member ID selected as the representative record.
  - **`canonical_title`** *(string)*: The representative title/text selected from the group.
  - **`member_ids`** *(array of strings)*: All member IDs in the group, sorted lexicographically.
  - **`total_count`** *(integer)*: Sum of available `total_count` values for all group members. Missing `total_count` values should be treated as `0`.

Groups with fewer than 2 members must be omitted. If there are no threshold-qualified duplicate pairs, `/cluster` must return `"candidate_pairs": []` and `"groups": []`.

### Canonical Selection Rule:
1. Prefer the member with the highest `total_count`.
2. If counts tie or are missing, prefer the member with the shortest sanitized text.
3. If still tied, prefer the lexicographically smallest `id`.

### Example Cluster Output Payload:
```json
{
  "candidate_pairs": [
    {
      "left_id": "doc-a",
      "right_id": "doc-b",
      "similarity": 0.9142
    }
  ],
  "groups": [
    {
      "group_id": "doc-a",
      "canonical_id": "doc-a",
      "canonical_title": "Database connection timed out during handshake",
      "member_ids": ["doc-a", "doc-b"],
      "total_count": 150
    }
  ]
}
```

---

## ⚙️ Execution Constraints & Performance Requirements

1. **Subprocess Startup Performance**: Because the script offloads the heavy embedding calculations to `localhost`, process startup and argument/input parsing should complete in under **200 milliseconds** on a warm environment. End-to-end runtime additionally includes model-server latency, vector normalization, index build time, and index query time.
2. **Batching HTTP requests**: Ensure all texts from the input `items` are packed into a single array payload when calling the local model server, rather than making individual HTTP calls for each text.
3. **Vector Indexing Strategy**: After embeddings are returned, normalize vectors and insert them into a lightweight ANN index such as **USearch** (`usearch.index.Index`) or **Faiss CPU** using an ANN-capable index family such as HNSW or IVF. Candidate generation must query each vector against the localized index space for `candidate_k + 1` neighbors, where the extra result accounts for the self-match. `candidate_k` must be greater than or equal to `top_k`; a practical default is `min(N - 1, max(32, top_k * 4))`, with tuning based on ANN recall benchmarks.
4. **Sub-Quadratic Candidate Search Target**: Production matching should target approximately $O(N \log N)$ or $O(N)$ candidate retrieval behavior, depending on the selected ANN index and parameters. The implementation must not generate an `N x N` pairwise similarity matrix for production matching. Similarity edges must be produced from index search results, which keeps candidate exploration bounded by `candidate_k` and index-search complexity rather than by all possible document pairs.
5. **Localized Query Scope**: For request-scoped CLI/API calls, build an in-memory index for the current input batch. For future persistent deployments, maintain a tenant/project-scoped index and query only the relevant partition before inserting or merging new anomalies. This preserves the existing JSON contract while avoiding global brute-force comparisons.
6. **Deterministic `top_k` Pair Selection**: For each item, sort threshold-qualified neighbors by similarity descending, then by neighbor ID ascending. Process source items by lexicographic ID order. For each source item, accept neighbors from its sorted list until the source reaches `top_k`, skipping any neighbor that already appears in `top_k` accepted pairs. Convert accepted matches into unordered candidate pairs by sorting each pair into `left_id` and `right_id`, remove duplicate directions, and keep the highest observed similarity for each unordered pair. This makes `top_k` a final undirected degree cap while ensuring each item is considered against its closest available neighbors in a deterministic order.
7. **Verification Mode vs Production Mode**: The 5 must-pass test cases use tiny batches and require deterministic behavior. Implementations may use an exact/high-recall index configuration for these tests, including USearch configured for exhaustive recall or a flat Faiss verification index. Production configuration must use ANN-capable indexes and record recall/latency tradeoffs during benchmarking.
8. **Error Handling & Exit Codes**: 
   - If the input JSON is malformed or missing required keys, print a descriptive message to `stderr` and exit with code `1`.
   - If the local model server is unreachable or returns a non-200 HTTP status, log the details to `stderr` and exit with code `2`.
   - If `items` has fewer than 2 elements, output an empty `candidate_pairs` list and exit successfully with code `0`. For `/cluster`, the equivalent successful response is `{"candidate_pairs": [], "groups": []}`.
   - FastAPI validation errors should return HTTP `400` for malformed or semantically invalid request bodies and HTTP `502` when the configured local model server is unreachable or returns a non-200 response.

---

## 📅 Weekly Tasks & Deliverables

### 📦 Week 1: Research, Local Model Server Setup, & CLI Prototype
The objective of the first week is to set up a local AI sandbox and implement a baseline script.

#### Requirements:
1. **Set Up the Environment**:
   - Install a local model server (Ollama, HuggingFace TEI, or vLLM) on the development machine.
   - Download the default embedding model (`all-minilm`).
2. **Implement the CLI Script**:
   - Write a Python script that reads batch queries from standard input (`stdin`), sends text batches to the model server API on `localhost`, builds an in-memory vector index, and retrieves threshold-qualified nearest-neighbor candidates without constructing a full pairwise similarity matrix.
   - Ensure the script outputs the matches to standard output (`stdout`) in JSON.
3. **Pass Verification Test Cases**:
   - Verify the script against the **5 Must-Pass Test Cases** defined below.

---

### 🧹 Week 2: Input Sanitization & Data Cleaning Pipeline
Log files contain variable noise that can interfere with embedding quality. This week focuses on filtering out this noise.

#### Requirements:
1. **Analyze Anomaly Message Patterns**:
   - Identify common dynamic noise (hex memory addresses, UUIDs, numeric IDs, timestamps, thread names) in error logs.
2. **Build a Pre-Processing Pipeline**:
   - Implement a text sanitization function in Python that cleans log titles prior to embedding.
3. **Evaluate Cleaning Impact**:
   - Test and compare the similarity scores of raw logs versus cleaned logs to verify that sanitization improves semantic accuracy.

---

### 📊 Week 3: Dataset Construction & Model Benchmarking
This week focuses on empirical analysis of different model choices.

#### Requirements:
1. **Construct an Evaluation Dataset**:
   - Create a labeled set of 100+ representative mock error messages containing true duplicates (worded differently) and distinct errors.
2. **Benchmark Embedding Models**:
   - Test at least 3 models (e.g., `all-MiniLM-L6-v2`, `bge-small-en-v1.5`, `nomic-embed-text`).
   - Measure embedding generation latency (ms), vector dimensions, index build/query latency, ANN recall, CPU/RAM footprint, and semantic matching accuracy.

---

### 🎯 Week 4: Hyperparameter Tuning & Threshold Evaluation
This week is dedicated to analyzing verification metrics and selecting production settings.

#### Requirements:
1. **Tune Similarity Threshold**:
   - Plot Precision, Recall, and F1-Score curves against similarity thresholds ranging from `0.50` to `0.95`.
   - Identify the threshold that prevents false merges (high precision) while catching true duplicates (high recall).
   - Report semantic precision/recall separately from ANN candidate-retrieval recall so missed index candidates are not confused with poor embedding or threshold quality.

#### Required Deliverable:
* 📝 **Report 1: Model Benchmarking & Threshold Evaluation Report** (due end of Week 4). *See details below for format and contents.*

---

### 🕸️ Week 5: Similarity Graph Grouping & Canonical Title Selection
Pairwise checks do not scale in production. The implementation must group anomalies from ANN-retrieved candidate edges using graph connected components.

#### Requirements:
1. **Implement Graph Grouping (Union-Find)**:
   - Create a simple graph connected-components or Union-Find algorithm in Python.
   - Nodes represent logs, and edges are established from ANN-retrieved candidate neighbors whose cosine similarity meets or exceeds the similarity threshold.
   - Extract clusters of related logs as connected sub-graphs.
   - Apply a chain-link mitigation policy before finalizing a group: either require every member to connect to the selected canonical item above the threshold, or split weakly connected components when the minimum canonical-to-member similarity falls below the threshold.
2. **Design a Metadata-Based Canonical Selector**:
   - Implement a selection rule to choose a representative title (the "canonical title") for each group.
   - Factor in metadata: prefer the title with the highest frequency (`total_count`) to ensure familiarity, using string length as a tie-breaker.

#### Required Deliverable:
* 📝 **Report 2: Similarity Graph Grouping & Metadata Selection Analysis** (due end of Week 5). *See details below for format and contents.*

---

### 🚀 Week 6: Production Packaging, Docker, & Integration Support
The final week is dedicated to packaging the engine as a production-grade microservice and supporting backend integration.

#### Requirements:
1. **Develop the API Wrapper**:
   - Wrap the python similarity and graph grouping code into a REST API using **FastAPI** and **Uvicorn**.
   - Create endpoints: `POST /similarity` and `POST /cluster`.
2. **Containerize the Application**:
   - Write a `Dockerfile` for the FastAPI service.
   - Define a `docker-compose.yml` service wrapper to demonstrate how the service connects to the model server and database.
3. **Support Integration Testing**:
   - Support the engineering team in connecting the core Spring Boot application to the REST API.

#### Required Deliverable:
* 📝 **Report 3: API Integration & Service Handover Documentation** (due end of Week 6). *See details below for format and contents.*

---

## 🧪 Must-Pass Test Cases

The script will be verified using the following scenarios. It must produce the exact matching behaviors and outputs shown.

### Test Case 1: Semantic Synonym Match
This verifies that the model detects semantic duplicates with completely different words but identical meanings.

#### Input:
```json
{
  "threshold": 0.78,
  "top_k": 2,
  "items": [
    {
      "id": "1",
      "text": "Network connection dropped because of server failure"
    },
    {
      "id": "2",
      "text": "Server error caused link disconnect"
    },
    {
      "id": "3",
      "text": "OutOfMemoryError: Java heap space exhausted"
    }
  ]
}
```
#### Expected Behavior:
- Items `1` and `2` must match because they share a close semantic meaning, despite using different lexical tokens.
- Item `3` is completely unrelated and must not match either item.
- Outputs a single candidate pair representing `1` and `2`.

---

### Test Case 2: Zero Matches Under High Threshold
This verifies that the script respects the similarity threshold.

#### Input:
```json
{
  "threshold": 0.98,
  "top_k": 5,
  "items": [
    {
      "id": "x",
      "text": "Database read timeout"
    },
    {
      "id": "y",
      "text": "Database connection lost"
    }
  ]
}
```
#### Expected Behavior:
- Even though the items are related, their cosine similarity will be below the strict `0.98` threshold.
- The output `candidate_pairs` array must be empty.

---

### Test Case 3: Empty and Boundary Inputs
This verifies robust behavior on minimal inputs.

#### Input A (Empty items):
```json
{
  "threshold": 0.50,
  "top_k": 3,
  "items": []
}
```
#### Input B (Single item):
```json
{
  "threshold": 0.50,
  "top_k": 3,
  "items": [
    { "id": "single", "text": "Generic API error" }
  ]
}
```
#### Expected Behavior for both:
- Exit successfully with code `0`.
- Output: `{"candidate_pairs": []}`.

---

### Test Case 4: Top-K Constraints
This verifies that candidate returns are limited per item using the `top_k` parameter.

#### Input:
```json
{
  "threshold": 0.60,
  "top_k": 1,
  "items": [
    {
      "id": "item-a",
      "text": "Database transaction timeout error"
    },
    {
      "id": "item-b",
      "text": "Database read timed out"
    },
    {
      "id": "item-c",
      "text": "Database write timed out"
    }
  ]
}
```
#### Expected Behavior:
- Items `item-b` and `item-c` will both have a similarity score above `0.60` when compared to `item-a`.
- Because `top_k` is set to `1`, `item-a` must only be paired with its **single closest neighbor** (the item with the highest similarity score), rather than matching with both.
- The final `candidate_pairs` list must not contain both `item-a`/`item-b` and `item-a`/`item-c`.

---

### Test Case 5: Bidirectional Deduplication
This verifies that matching pairs are formatted correctly and not duplicated.

#### Input:
```json
{
  "threshold": 0.70,
  "top_k": 5,
  "items": [
    { "id": "20", "text": "SSL Handshake failed on server" },
    { "id": "10", "text": "Server failed to establish SSL handshake" }
  ]
}
```
#### Expected Behavior:
- Output must contain exactly one match object.
- The pair must be sorted by ID to maintain consistency: `left_id` must be `"10"` and `right_id` must be `"20"`. An output of `left_id: "20"` and `right_id: "10"` is incorrect.

---

## 📝 Required Reports & Document Structures

Three progress and architectural reports are required throughout the project. These documents should be saved in markdown format.

### Report 1: Model Benchmarking & Threshold Evaluation Report
* **Purpose**: Present data justifying the choice of embedding model and the recommended similarity threshold.
* **Required Sections**:
  1. **Executive Summary**: Core recommendation (model name, threshold).
  2. **Model Comparison Table**:
     - Columns: Model Name, Size (MB), Embedding Latency per 100 items (ms), Vector Dimension, Index Build Time, Query Latency, ANN Recall, RAM usage.
  3. **Threshold Performance Analysis**:
     - Graph/Table showing Precision, Recall, and F1-score for threshold increments of `0.05`.
     - Analysis of edge-case failures (e.g. false merges vs missed merges).
  4. **Pre-Processing Impact**: Analysis showing how text sanitization rules affected similarity scores.

### Report 2: Similarity Graph Grouping & Metadata Selection Analysis
* **Purpose**: Evaluate the threshold-based graph connected components approach and metadata naming logic.
* **Required Sections**:
  1. **Graph Grouping Algorithm**: Explanation of the connected components or Union-Find structure used to define similarity boundaries.
  2. **Chain-Link Mitigation Analysis**: Definition and evaluation of the policy used to handle indirect connections (e.g. A matches B, B matches C, but A and C are different).
  3. **Canonical Title Rule Definition**: Specification of the metadata priorities used to select the primary title (e.g., occurrence frequency vs. title length).

### Report 3: API Integration & Service Handover Documentation
* **Purpose**: Hand over a production-ready microservice to the core backend team.
* **Required Sections**:
  1. **API Reference**: 
     - HTTP request/response JSON schemas for `/similarity` and `/cluster` endpoints.
     - HTTP status codes and error-handling definitions.
  2. **Deployment Instructions**:
     - How to build the Docker image.
     - Environment variables list (e.g., model server URLs, port configurations).
  3. **Verification Steps**:
     - Step-by-step curl commands or scripts to run a health check and verify the end-to-end flow.

---

## 💡 Implementation FAQs & Design Guidelines

These guidelines are provided to support the developer and prevent common architectural pitfalls:

### 1. Library Dependencies
* **For CLI Scripting & HTTP Requests**: Use Python’s built-in `urllib.request` (standard library, zero dependencies) or the standard `requests` library.
* **For Vector Search / Similarity Retrieval**: Use **USearch** as the preferred lightweight ANN library for request-scoped vector indexing and top-K neighbor lookup. Store embeddings in a cosine-compatible index configuration, normalize vectors before insertion when using inner-product metrics, and query the index for `candidate_k + 1` neighbors so the current item can be removed from its own result set.
* **Acceptable Fallback**: `faiss-cpu` may be used when USearch is unavailable or when the deployment already standardizes on Faiss. Use ANN-capable Faiss indexes such as HNSW or IVF for production-scale matching. Flat exhaustive indexes such as `IndexFlatIP` are acceptable only for tiny unit tests or deterministic verification because they still perform brute-force search internally.
* **Avoid Brute-Force Matrix Math**: Do not use `numpy.dot(embeddings, embeddings.T)` or equivalent `N x N` matrix expansion for production candidate generation. NumPy may still be used for small helper operations such as vector normalization, dtype conversion, and validation, but it must not be the primary matching engine.
* **For Top-K Result Capping**: Apply `top_k` as a final undirected degree cap while processing source items in lexicographic ID order. For each source item, walk its neighbors by similarity descending and accept the closest available neighbors until either the source reaches `top_k` or no eligible neighbors remain. This makes Test Case 4 deterministic while preserving the no-duplicate-pairs output contract.
* **For Graph Traversal**: No heavy graph libraries (like NetworkX) are required. Simple connected components can be written in a few lines using Depth First Search (DFS) or a standard Union-Find structure in Python.

### 2. Local Model Server Initialization (Ollama Example)
1. Install Ollama on the local machine.
2. Initialize the embedding model:
   ```bash
   ollama pull all-minilm
   ```
3. Verify connection via the terminal:
   ```bash
   curl http://localhost:11434/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{"model": "all-minilm", "input": ["Hello World"]}'
   ```

### 3. Evaluation Dataset Generation (Week 3)
* Extract 15-20 representative logs from staging or database entries.
* Create 3 to 4 variations for each entry by altering syntax, word order, inserting generic error tokens, or appending noise (timestamps/numbers).
* Label matches and non-matches explicitly. This dataset will serve as the ground-truth benchmark to verify Precision and Recall rates.

---

## 📐 Definitions & Mathematical Examples of Deliverables

To clarify the required deliverables, this section defines the data structures and mathematical models to be implemented:

### 1. Labeled Evaluation Dataset (Example)
The **Labeled Evaluation Dataset** is a structured dataset (commonly saved in JSON or CSV format) used to validate the model's accuracy. It contains clusters of log messages representing ground-truth relationships.

* **Example JSON Structure**:
```json
{
  "ground_truth_groups": [
    {
      "group_id": "network_timeout",
      "records": [
        "Connection timed out when connecting to DB host db-prod-01",
        "Failed to connect to the database: connection timeout",
        "Network connection lost during handshake"
      ]
    },
    {
      "group_id": "null_pointer",
      "records": [
        "NullPointerException: user profile is null",
        "NPE: Null value dereferenced for profile object"
      ]
    }
  ]
}
```

### 2. Mathematical Model (Example)

#### Cosine Similarity Model
Used as the scoring metric returned by the vector index for two vector representations $A$ and $B$. Implementations should rely on the selected index library for neighbor retrieval rather than evaluating this formula across all vector pairs:
$$\text{Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum A_i B_i}{\sqrt{\sum A_i^2} \sqrt{\sum B_i^2}}$$

### 3. Graph Grouping Example
If similarity between two logs is $\ge \text{threshold}$, they are linked. For example:
- `doc-a` is linked to `doc-b` (similarity 0.89)
- `doc-b` is linked to `doc-c` (similarity 0.85)
- If the selected chain-link mitigation policy confirms that all members remain similar enough to the canonical item, the algorithm extracts `[doc-a, doc-b, doc-c]` as a single group. Otherwise, it splits the weakly connected component into safer sub-groups.

---

## 📋 Required Final Deliverables Checklist

To complete this project, the following deliverables must be successfully implemented, verified, and handed over:

### Code & Execution Deliverables
- [✅] **Python CLI Similarity Tool**: A standalone script (`semantic_match.py` or equivalent) that reads JSON inputs via `stdin`, calls the localhost model server, retrieves nearest-neighbor candidates through a vector index, filters by cosine similarity threshold, and outputs JSON via `stdout`.
- [✅] **Input Sanitization Pipeline**: Pre-processing rules integrated into the CLI tool and API that successfully filter timestamps, UUIDs, numeric IDs, and hex addresses.
- [✅] **FastAPI Microservice App**: A lightweight REST API wrapper exposing endpoints for `/similarity` and `/cluster`.
- [✅] **Docker assets**: A fully working `Dockerfile` containerizing the FastAPI app.
- [✅] **Docker Compose Integration**: A `docker-compose.yml` file demonstrating how the FastAPI service runs in an isolated network and connects to the local model server.

### Datasets & Graph Models
- [✅] **Graph Connected Components / Grouping**: Verified Union-Find or DFS module that groups similar vectors.
- [✅] **Metadata-Based Canonical Selector**: Verified algorithm that selects the canonical title of a group based on occurrence frequency and string length.
- [✅] **Labeled Evaluation Dataset**: A dataset of 100+ categorized test logs used for model benchmarking and threshold tuning.

### Documentation & Reports
- [✅] **Verification Suite Result**: Verified output confirming that the final code successfully passes all **5 Must-Pass Test Cases**.
- [✅] **Report 1**: Model Benchmarking & Threshold Evaluation Report.
- [✅] **Report 2**: Similarity Graph Grouping & Metadata Selection Analysis.
- [✅] **Report 3**: API Integration & Service Handover Documentation.
