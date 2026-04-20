"""Debug and evaluation routes for observability and quick testing."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.core.debug_store import (
    get_debug_errors,
    get_debug_history,
    get_last_debug_entry,
)
from app.core.normalizer import normalize_response
from app.core.router import detect_problem_type, route_problem_with_meta
from app.services.llm_service import InvalidLLMResponseError, LLMServiceError

router = APIRouter(tags=["debug", "evaluation"])


@router.get("/debug/last")
async def get_last_debug() -> dict[str, Any]:
    """Return the last debug record."""
    entry = get_last_debug_entry()
    return {"entry": entry}


@router.get("/debug/history")
async def get_debug_history_route() -> dict[str, list[dict[str, Any]]]:
    """Return all debug records (up to last 10)."""
    return {"entries": get_debug_history()}


@router.get("/debug/errors")
async def get_debug_errors_route() -> dict[str, list[dict[str, Any]]]:
    """Return only failed debug records."""
    return {"entries": get_debug_errors()}


@router.get("/evaluate/sample")
async def evaluate_sample() -> dict[str, Any]:
    """Run a fixed sample set across all problem types and return results."""
    samples: dict[str, str] = {
        "array": "Explain Bubble Sort for [5,3,1,4]",
        "graph": "Run BFS on graph with nodes A,B,C,D and edges A-B, A-C, B-D.",
        "tree": "Explain inorder traversal for binary tree with nodes [4,2,6,1,3,5,7].",
        "dp": "Solve 0/1 knapsack with weights [1,3,4], values [15,20,30], capacity 4 using DP.",
    }

    results: dict[str, Any] = {}

    for key, problem in samples.items():
        detected_type = detect_problem_type(problem)
        try:
            routed = await route_problem_with_meta(problem)
            normalized = normalize_response(routed["parsed_output"])
            results[key] = {
                "problem": problem,
                "detected_type": detected_type,
                "engine_used": routed["engine_used"],
                "valid": True,
                "error": None,
                "result": normalized,
            }
        except (InvalidLLMResponseError, LLMServiceError) as exc:
            results[key] = {
                "problem": problem,
                "detected_type": detected_type,
                "engine_used": "llm",
                "valid": False,
                "error": str(exc),
                "result": None,
            }

    return {"samples": results}
