# LLM DSA Reasoning Engine

A FastAPI backend that solves Data Structures and Algorithms (DSA) problems using an LLM (Groq) and renders step-by-step visualizations.

## What It Does

- Accepts a natural-language DSA prompt (e.g. "Explain Bubble Sort for [5,3,1]")
- Detects the problem type: **array**, **graph**, **tree**, or **dynamic programming**
- Routes to the appropriate solver:
  - **Deterministic engine** for sorting (bubble, selection, insertion) and BFS graph traversal — guaranteed correct, no LLM call
  - **LLM engine** (Groq) for everything else — with retries, repair prompts, and strict JSON schema validation
- Returns a structured response with:
  - `explanation` — human-readable algorithm description
  - `steps` — ordered state snapshots for each operation
  - `visualization` — renderer-ready data (arrays, graphs, trees, DP tables)
- Serves a minimal web UI at `/` for interactive testing

## Project Structure

```
app/
├── main.py                  # FastAPI entrypoint, routes, static files
├── models/
│   └── schemas.py           # Pydantic models: ArrayResponse, GraphResponse, TreeResponse, DPResponse
├── routes/
│   ├── solve.py             # POST /solve — main solving endpoint
│   └── debug.py             # GET /debug/*, /evaluate/sample — observability
├── services/
│   ├── llm_service.py       # Groq client, retries, JSON extraction, validation
│   └── deterministic_engine.py  # Bubble/Selection/Insertion sort simulators
├── engines/
│   └── graph_engine.py      # Deterministic BFS parser + executor
├── core/
│   ├── router.py            # Problem type detection + engine routing
│   ├── normalizer.py        # Response normalization for frontend
│   ├── debug_store.py       # In-memory ring buffer of last 10 requests
│   └── config.py            # Settings from environment (GROQ_API_KEY, etc.)
├── prompts/
│   └── array_prompt.txt     # LLM prompt template for array problems
└── static/
    └── index.html           # Interactive visualization UI
```

## Workflow: Request to Response

This section describes the full data flow when a user submits a problem.

### 1. Request Entry — `POST /solve`

The client sends a JSON payload:

```json
{"problem": "Explain Bubble Sort for [5,3,1,4]"}
```

`app/routes/solve.py` receives this via the `SolveRequest` Pydantic model, resets per-request LLM metadata, and calls the router.

### 2. Problem Type Detection — `core/router.py`

`detect_problem_type()` scores the input against keyword banks:

| Type | Keywords |
|------|----------|
| array | sort, bubble, selection, insertion, merge, quick, two pointer, sliding window, subarray, binary search |
| graph | bfs, dfs, dijkstra, shortest path, graph, adjacency, vertex, edge, topological, kruskal, prim |
| tree | tree, binary tree, bst, inorder, preorder, postorder, level order, avl, heap, trie |
| dp | dynamic programming, dp, memoization, tabulation, knapsack, lcs, edit distance, coin change, fibonacci |

The type with the highest keyword match score wins. If no keywords match, it defaults to `array`.

### 3. Engine Routing

`route_problem_with_meta()` decides which engine handles the problem:

**Deterministic path (no LLM call):**
- Array sorting problems (bubble, selection, insertion) → `deterministic_engine.py`
- Graph BFS traversal → `engines/graph_engine.py`

**LLM path:**
- Everything else → `services/llm_service.py` with type-specific prompts and Pydantic validators

### 4. Deterministic Engine — `services/deterministic_engine.py`

For sorting problems:
1. `extract_array()` pulls integers from the prompt (bracket notation or raw numbers)
2. `detect_algorithm()` identifies bubble, selection, or insertion from keywords
3. The matching simulator runs the algorithm step-by-step, recording:
   - Each comparison or swap
   - Highlighted indices
   - Sorted boundary progress
4. Returns a fully-formed `ArrayResponse`-compatible dict

For graph BFS:
1. `engines/graph_engine.py` parses nodes and edges from natural language
2. `run_bfs()` executes the algorithm, recording queue state, visited nodes, and active node at each step
3. Returns a `GraphResponse`-compatible dict

### 5. LLM Service — `services/llm_service.py`

For non-deterministic problems, the LLM pipeline runs:

```
Build prompt from template (app/prompts/*.txt)
    ↓
Call Groq API with response_format: json_object
    ↓
Extract JSON from response (handles markdown fences)
    ↓
Validate against Pydantic model (ArrayResponse / GraphResponse / TreeResponse / DPResponse)
    ↓
Run semantic validation (e.g., final array is sorted, no duplicate steps)
    ↓
If invalid → repair prompt with error details → retry (up to max_retries)
    ↓
If all retries fail → fallback to deterministic engine (arrays only)
```

Metadata about each call (engine used, raw LLM output) is stored in context variables for observability.

### 6. Response Normalization — `core/normalizer.py`

Before returning to the client, `normalize_response()` sanitizes the payload:

- Ensures `problem_type` is one of the supported types
- Sorts steps by step number
- Normalizes state structures per type:
  - **array**: validates indices, clamps sorted_boundary
  - **graph**: ensures active node exists in nodes list
  - **tree**: ensures current node exists in nodes list
  - **dp**: clamps current_cell to table bounds
- Builds a consistent `visualization` section from step data
- Guarantees at least one step exists (adds a default init step if empty)

### 7. Debug Capture — `core/debug_store.py`

Every solve attempt is logged to an in-memory ring buffer (max 10 entries):

- Problem text
- Detected type and engine used
- Raw LLM output (if applicable)
- Parsed/normalized output
- Valid flag and error message (if failed)

Accessible via `/debug/last`, `/debug/history`, `/debug/errors`.

### 8. Response Delivery

The final `SolveResponse` is returned:

```json
{
  "problem_type": "array",
  "explanation": "Bubble Sort repeatedly compares adjacent elements...",
  "steps": [
    {
      "step": 1,
      "description": "Compare indices 0 and 1.",
      "state": {
        "array": [5, 3, 1, 4],
        "highlight": [0, 1],
        "sorted_boundary": -1
      }
    }
  ],
  "visualization": {
    "type": "array",
    "data": {"initial_array": [5, 3, 1, 4]}
  }
}
```

On failure, a 502 or 500 error is returned with an `ErrorResponse` containing the error code and details.

### 9. Frontend Visualization — `static/index.html`

The UI renders the response based on `problem_type`:

- **array**: Animated bar chart with highlighted indices and sorted boundary
- **graph**: SVG node-link diagram with visited/active state coloring
- **tree**: SVG tree layout with current node highlighting
- **dp**: Table grid with active cell highlighting

Playback controls allow stepping through manually or auto-playing at configurable speed.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your Groq API key:
   ```bash
   export GROQ_API_KEY="your-key-here"
   ```

3. Run the server:
   ```bash
   uvicorn app.main:app --reload
   ```

4. Open `http://localhost:8000` for the interactive UI, or POST to `/solve`:
   ```bash
   curl -X POST http://localhost:8000/solve \
     -H "Content-Type: application/json" \
     -d '{"problem": "Explain Bubble Sort for [5,3,1,4]"}'
   ```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Required for LLM-powered problem types |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Model ID for Groq completions |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | Groq API base URL |
| `LLM_TIMEOUT_SECONDS` | `30` | Request timeout |
| `LLM_MAX_RETRIES` | `2` | Retry attempts on validation failure |
| `LLM_RETRY_DELAY_SECONDS` | `1` | Delay between retries |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Interactive visualization UI |
| GET | `/health` | Health check |
| POST | `/solve` | Solve a DSA problem |
| GET | `/debug/last` | Last request debug record |
| GET | `/debug/history` | Last 10 request records |
| GET | `/debug/errors` | Failed request records only |
| GET | `/evaluate/sample` | Run sample problems across all types |

## Supported Problem Types

| Type | Deterministic | LLM | Examples |
|------|--------------|-----|----------|
| Array | Bubble, Selection, Insertion sort | Other array problems | "Sort [3,1,2] with bubble sort" |
| Graph | BFS traversal | DFS, Dijkstra, etc. | "Run BFS on graph A-B, A-C" |
| Tree | — | Inorder, preorder, BST | "Inorder traversal of binary tree [4,2,6,1,3,5,7]" |
| DP | — | Knapsack, LCS, etc. | "0/1 knapsack weights [1,3,4], values [15,20,30]" |

## How Solving Works

1. **Detect** — Keyword scoring classifies the problem into array/graph/tree/dp
2. **Route** — Array sorting and graph BFS use the deterministic engine; everything else uses the LLM
3. **Generate** — LLM calls use strict JSON-schema prompts with `response_format: json_object`
4. **Validate** — Pydantic models enforce field presence, step sequencing, state transitions
5. **Repair** — On validation failure, the LLM is re-prompted with the error and invalid output
6. **Normalize** — Responses are sanitized into a consistent frontend-ready shape
7. **Visualize** — The UI renders bars (arrays), SVG graphs, trees, or DP tables from the step states
