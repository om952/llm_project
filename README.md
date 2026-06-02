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
