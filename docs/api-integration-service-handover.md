# API Integration & Service Handover Documentation
## 1. API Reference
### `POST /similarity` Endpoint
**Description**

Finds semantically similar anomaly pairs using embedding vectors and ANN search. 

**HTTP Request Schema**
```json
{
    "model_server_url": "string (optional, default: http://localhost:11434/v1/embeddings)",
    "model_name": "string (optional, default: bge-m3)",
    "threshold": "float (required, 0.0 - 1.0)",
    "top_k": "integer (required, >0)",
    "items": [
        {
            "id": "string (required)",
            "text": "string (required)",
            "total_count": "integer (optional, default: 0)"
        },
        ...
    ]
}
```
**Response Schema**
```json
{
    "candidate_pairs": [
        {
            "left_id": "string",
            "right_id": "string",
            "similarity": "float"
        },
        ...
    ]
}
```
### `POST /cluster` Endpoint
**Description**

Groups similar anomalies into connected components.

**HTTP Request Schema**
```json
{
    "model_server_url": "string (optional, default: http://localhost:11434/v1/embeddings)",
    "model_name": "string (optional, default: bge-m3)",
    "threshold": "float (required, 0.0 - 1.0)",
    "top_k": "integer (required, >0)",
    "items": [
        {
            "id": "string (required)",
            "text": "string (required)",
            "total_count": "integer (optional, default: 0)"
        },
        ...
    ]
}
```
**Response Schema**
```json
{
    "candidate_pairs": [
        {
            "left_id": "string",
            "right_id": "string",
            "similarity": "float"
        },
        ...
    ],
    "groups":[
        {
            "group_id": "string",
            "canonical_id": "string",
            "canonical_title": "string",
            "member_ids": ["string"],
            "total_count": "integer"
        },
        ...
    ]
}
```
**HTTP Status Codes**
| HTTP Status | Condition | Response |
|-------------|-----------|----------|
| 200 OK | Request processed succesfully | Returns `candidate_pairs` (`POST /similarity`) or `candidate_pairs` + `groups` (`POST /cluster`) |
| 400 Bad Request | Malformed input or validation fail | Returns a JSON object describing the error |
| 502 Bad Gateway | Model server unreachable or returned a non-200 HTTP response | Returns an error message describing the model server failure. |

**HTTP 400 Example**
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": [
        "body",
        "top_k"
      ],
      "msg": "Field required",
      "input": {
        "model_server_url": "http://localhost:11434/v1/embeddings",
        "model_name": "bge-m3",
        "threshold": 1,
        "top_": 1,
        "items": [
          {
            "id": "string",
            "text": "string",
            "total_count": 0
          }
        ]
      }
    }
  ]
}
```

**HTTP 502 Example**
```json
{
  "detail": "Model server unreachable: HTTPConnectionPool(host='localhost', port=11434): Max retries exceeded with url: /v1/embeddings (Caused by NewConnectionError(\"HTTPConnection(host='localhost', port=11434): Failed to establish a new connection: [WinError 10061] No connection could be made because the target machine actively refused it\"))"
}
```

## 2. Deployment Instructions
### Building the Docker Image
```bash
docker build -t semantic-anomaly-grouping-api .
```

### Running Docker Compose
```bash
docker-compose up -d --build
```

### Pulling the embedding model (Required before first use)
```bash
docker exec -it semantic-anomaly-grouping-ollama-1 ollama pull bge-m3
```

### Environment Variables
| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `bge-m3` | Name of the embedding model to use by default when a request does not specify one. |
| `MODEL_SERVER_URL` | `http://localhost:11434/v1/embeddings` | URL of the embedding model server. Overridden to `http://ollama:11434/v1/embeddings` in `docker-compose.yml` so the API container can reach the Ollama container by service name. |
> Both variables can also be overridden per-request via the `model_server_url` and `model_name` fields in the JSON request body, which take precedence over these environment defaults.


> Before specifying `model_name`in the JSON request body, the model that is going to be used must be downloaded first via:
>```bash
>docker exec -it semantic-anomaly-grouping-ollama-1 ollama pull <model_name>
>```
### Port Configuration
The `api` service listens on port `8000` inside its container. Mapped to the host via `docker-compose.yml`:
```yml
api:
  ports:
    - "8000:8000"
```

The `ollama` service listens on port `11434` inside its container, also mapped to the host:
```yml
ollama:
  ports:
    - "11434:11434"
```

### Other Configuration Details

**Consistent storage (volume):** The `ollama` service has a named volume, so pulled models don't need to be re-downloaded after every `docker-compose down`.
```yml
ollama:
  volumes:
    - ollama_data:/root/.ollama
volumes:
  ollama_data:
```

**Startup order (healthcheck):** The `ollama` service defines a `healthcheck`. The `api` service waits on `condition: service_healthy` before starting, ensuring it does not attempt to reach the model server before it is fully ready.
```yml
api:
  depends_on:
    ollama:
      condition: service_healthy
ollama:
  healthcheck:
    test: ["CMD", "ollama", "list"]
    interval: 5s
    timeout: 10s
    retries: 5
```

## 3. Verification Steps
### 1. Step: Confirm both containers are running
```bash
docker ps
```
- Expected to see both `semantic-anomaly-grouping-api-1` and `semantic-anomaly-grouping-ollama-1` containers with **STATUS** "Up".

### 2. Step: Confirm the model is downloaded
```bash
docker exec -it semantic-anomaly-grouping-ollama-1 ollama list
```
- `bge-m3` (or any embedding model downloaded) should appear in the list.

### 3. Step: End to end health check on `POST /similarity`
**On Windows (PowerShell):**
```powershell
curl.exe -X POST http://localhost:8000/similarity `
  -H "Content-Type: application/json" `
  -d '{\"threshold\": 0.7, \"top_k\": 2, \"items\": [{\"id\": \"1\", \"text\": \"Database connection timed out\"}, {\"id\": \"2\", \"text\": \"DB connection failed due to timeout\"}]}'
```

**On macOS/Linux (Bash/Zsh):**
```bash
curl -X POST http://localhost:8000/similarity \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.7, "top_k": 2, "items": [{"id": "1", "text": "Database connection timed out"}, {"id": "2", "text": "DB connection failed due to timeout"}]}'
```

- Expected HTTP 200 with a non-empty `candidate_pairs` array showing items 1 and 2 matched.

### 4. Step: Health Check on `POST /cluster`
**On Windows (PowerShell):**
```powershell
curl.exe -X POST http://localhost:8000/cluster `
  -H "Content-Type: application/json" `
  -d '{\"threshold\": 0.7, \"top_k\": 2, \"items\": [{\"id\": \"1\", \"text\": \"Database connection timed out\"}, {\"id\": \"2\", \"text\": \"DB connection failed due to timeout\"}]}'
```

**On macOS/Linux (Bash/Zsh):**
```bash
curl -X POST http://localhost:8000/cluster \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.7, "top_k": 2, "items": [{"id": "1", "text": "Database connection timed out"}, {"id": "2", "text": "DB connection failed due to timeout"}]}'
```

-----