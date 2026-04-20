"""Problem type detection and routing layer for the multi-DSA engine."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

logger = logging.getLogger(__name__)

_DETERMINISTIC_BFS_KEYWORDS: tuple[str, ...] = (
    "bfs",
    "traversal",
    "graph traversal",
)

# ---------------------------------------------------------------------------
# Keyword banks for each problem type
# ---------------------------------------------------------------------------

_GRAPH_KEYWORDS: tuple[str, ...] = (
    "bfs",
    "breadth first",
    "breadth-first",
    "dfs",
    "depth first",
    "depth-first",
    "dijkstra",
    "shortest path",
    "graph",
    "adjacency",
    "vertex",
    "vertices",
    "edge",
    "edges",
    "connected component",
    "topological",
    "kruskal",
    "prim",
    "bellman",
    "floyd",
    "cycle detection",
)

_TREE_KEYWORDS: tuple[str, ...] = (
    "tree",
    "binary tree",
    "bst",
    "binary search tree",
    "inorder",
    "preorder",
    "postorder",
    "level order",
    "level-order",
    "avl",
    "red black",
    "red-black",
    "heap",
    "trie",
    "traversal",
    "subtree",
    "leaf",
    "root",
    "left child",
    "right child",
)

_DP_KEYWORDS: tuple[str, ...] = (
    "dynamic programming",
    "dp",
    "memoization",
    "tabulation",
    "knapsack",
    "longest common subsequence",
    "lcs",
    "edit distance",
    "coin change",
    "fibonacci",
    "climbing stairs",
    "rod cutting",
    "matrix chain",
    "optimal substructure",
    "overlapping subproblem",
    "subset sum",
)

_ARRAY_KEYWORDS: tuple[str, ...] = (
    "array",
    "sort",
    "sorted",
    "bubble",
    "selection",
    "insertion",
    "merge sort",
    "quick sort",
    "two pointer",
    "sliding window",
    "subarray",
    "reverse",
    "rotate",
    "search",
    "binary search",
    "linear search",
)


def _score(problem_lower: str, keywords: tuple[str, ...]) -> int:
    """Count how many keywords from a bank appear in the problem text."""
    return sum(1 for kw in keywords if kw in problem_lower)


def detect_problem_type(problem: str) -> str:
    """Classify a problem description into one of the supported types.

    Returns one of: ``"array"``, ``"graph"``, ``"tree"``, ``"dp"``, ``"unknown"``.
    """
    lowered = problem.lower()

    scores: dict[str, int] = {
        "graph": _score(lowered, _GRAPH_KEYWORDS),
        "tree": _score(lowered, _TREE_KEYWORDS),
        "dp": _score(lowered, _DP_KEYWORDS),
        "array": _score(lowered, _ARRAY_KEYWORDS),
    }

    best_type = max(scores, key=lambda k: scores[k])
    best_score = scores[best_type]

    if best_score == 0:
        logger.info("Problem type detection: no keywords matched → unknown")
        return "unknown"

    logger.info(
        "Problem type detection: scores=%s → selected '%s'",
        scores,
        best_type,
    )
    return best_type


def _should_use_deterministic_bfs(problem: str, problem_type: str) -> bool:
    """Determine whether graph prompts should use deterministic BFS routing."""
    if problem_type != "graph":
        return False
    lowered = problem.lower()
    return any(keyword in lowered for keyword in _DETERMINISTIC_BFS_KEYWORDS)


def _run_deterministic_graph_bfs(problem: str) -> dict[str, Any]:
    """Run deterministic BFS parsing+execution and validate final shape."""
    from app.engines.graph_engine import GraphParseError, parse_graph_problem, run_bfs
    from app.models.schemas import GraphResponse

    graph, start = parse_graph_problem(problem)
    payload = run_bfs(graph, start)
    validated = GraphResponse.model_validate(payload)
    return validated.model_dump()


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

async def route_problem(problem: str) -> dict[str, Any]:
    """Route a problem to the correct solving engine and return the result.

    * ``array`` → deterministic engine (for sorting) or array LLM
    * ``graph`` → graph LLM
    * ``tree``  → tree LLM
    * ``dp``    → dp LLM
    * ``unknown`` → defaults to array LLM
    """
    # Import here to avoid circular imports
    from app.services.llm_service import (
        call_llm_array,
        call_llm_dp,
        call_llm_graph,
        call_llm_tree,
    )

    problem_type = detect_problem_type(problem)
    logger.info("Routing problem to engine: type='%s'", problem_type)

    if problem_type == "graph":
        if _should_use_deterministic_bfs(problem, problem_type):
            try:
                logger.info("Using deterministic BFS engine for graph problem")
                result = _run_deterministic_graph_bfs(problem)
            except (ValidationError, ValueError) as exc:
                logger.warning(
                    "Deterministic BFS parse/validation failed, falling back to graph LLM: %s",
                    str(exc),
                )
                result = await call_llm_graph(problem)
        else:
            result = await call_llm_graph(problem)
    elif problem_type == "tree":
        result = await call_llm_tree(problem)
    elif problem_type == "dp":
        result = await call_llm_dp(problem)
    else:
        # "array" or "unknown" → use existing array pipeline
        result = await call_llm_array(problem)

    logger.info("Engine returned successfully for type='%s'", problem_type)
    return result


async def route_problem_with_meta(problem: str) -> dict[str, Any]:
    """Route a problem and return payload plus routing/engine metadata."""
    from app.services.llm_service import (
        call_llm_array,
        call_llm_dp,
        call_llm_graph,
        call_llm_tree,
        get_last_call_metadata,
    )

    problem_type = detect_problem_type(problem)
    logger.info("Routing with metadata: type='%s'", problem_type)

    engine_used = "llm"
    raw_llm_output = ""

    if problem_type == "graph":
        if _should_use_deterministic_bfs(problem, problem_type):
            try:
                logger.info("Using deterministic BFS engine for graph problem (meta route)")
                parsed_output = _run_deterministic_graph_bfs(problem)
                engine_used = "deterministic"
                raw_llm_output = ""
                return {
                    "detected_type": problem_type,
                    "engine_used": engine_used,
                    "raw_llm_output": raw_llm_output,
                    "parsed_output": parsed_output,
                }
            except (ValidationError, ValueError) as exc:
                logger.warning(
                    "Deterministic BFS parse/validation failed in meta route, using graph LLM: %s",
                    str(exc),
                )

        parsed_output = await call_llm_graph(problem)
        service_meta = get_last_call_metadata()
        engine_used = service_meta.get("engine_used", "llm")
        raw_llm_output = service_meta.get("raw_llm_output", "")
    elif problem_type == "tree":
        parsed_output = await call_llm_tree(problem)
        service_meta = get_last_call_metadata()
        engine_used = service_meta.get("engine_used", "llm")
        raw_llm_output = service_meta.get("raw_llm_output", "")
    elif problem_type == "dp":
        parsed_output = await call_llm_dp(problem)
        service_meta = get_last_call_metadata()
        engine_used = service_meta.get("engine_used", "llm")
        raw_llm_output = service_meta.get("raw_llm_output", "")
    else:
        parsed_output = await call_llm_array(problem)
        service_meta = get_last_call_metadata()
        engine_used = service_meta.get("engine_used", "llm")
        raw_llm_output = service_meta.get("raw_llm_output", "")

    return {
        "detected_type": problem_type,
        "engine_used": engine_used,
        "raw_llm_output": raw_llm_output,
        "parsed_output": parsed_output,
    }
