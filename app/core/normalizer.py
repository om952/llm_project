"""Response normalization utilities for frontend-ready payloads."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_SUPPORTED_TYPES = {"array", "graph", "tree", "dp", "linked_list", "hashmap"}


def _safe_int(value: Any, default: int = 0) -> int:
    """Convert to int safely, using default when conversion fails."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_array_state(state: dict[str, Any]) -> dict[str, Any]:
    """Normalize array step state to a strict frontend-friendly structure."""
    array = state.get("array")
    if not isinstance(array, list):
        array = [0]
    parsed_array = [_safe_int(item, 0) for item in array]
    if not parsed_array:
        parsed_array = [0]

    highlight = state.get("highlight")
    if not isinstance(highlight, list):
        highlight = []
    parsed_highlight = [_safe_int(index, -1) for index in highlight]
    parsed_highlight = [
        index for index in parsed_highlight if 0 <= index < len(parsed_array)
    ]

    sorted_boundary = _safe_int(state.get("sorted_boundary", -1), -1)
    sorted_boundary = max(-1, min(sorted_boundary, len(parsed_array)))

    return {
        "array": parsed_array,
        "highlight": parsed_highlight,
        "sorted_boundary": sorted_boundary,
    }


def _normalize_graph_state(state: dict[str, Any]) -> dict[str, Any]:
    """Normalize graph step state with required keys and defaults."""
    nodes = state.get("nodes") if isinstance(state.get("nodes"), list) else []
    parsed_nodes = [str(node) for node in nodes]
    if not parsed_nodes:
        parsed_nodes = ["A"]

    edges_raw = state.get("edges") if isinstance(state.get("edges"), list) else []
    edges: list[list[str]] = []
    for edge in edges_raw:
        if isinstance(edge, list) and len(edge) >= 2:
            edges.append([str(edge[0]), str(edge[1])])

    visited_raw = state.get("visited") if isinstance(state.get("visited"), list) else []
    visited = [str(node) for node in visited_raw if str(node) in parsed_nodes]

    queue_raw = state.get("queue") if isinstance(state.get("queue"), list) else []
    queue = [str(node) for node in queue_raw if str(node) in parsed_nodes]

    active = str(state.get("active", parsed_nodes[0]))
    if active not in parsed_nodes:
        active = parsed_nodes[0]

    return {
        "nodes": parsed_nodes,
        "edges": edges,
        "visited": visited,
        "active": active,
        "queue": queue,
    }


def _normalize_tree_state(state: dict[str, Any]) -> dict[str, Any]:
    """Normalize tree step state with required keys and defaults."""
    nodes = state.get("nodes") if isinstance(state.get("nodes"), list) else []
    parsed_nodes = [_safe_int(node, 0) for node in nodes]
    if not parsed_nodes:
        parsed_nodes = [0]

    edges_raw = state.get("edges") if isinstance(state.get("edges"), list) else []
    edges: list[list[int]] = []
    for edge in edges_raw:
        if isinstance(edge, list) and len(edge) >= 2:
            edges.append([_safe_int(edge[0], 0), _safe_int(edge[1], 0)])

    current = _safe_int(state.get("current", parsed_nodes[0]), parsed_nodes[0])
    if current not in parsed_nodes:
        current = parsed_nodes[0]

    return {
        "nodes": parsed_nodes,
        "edges": edges,
        "current": current,
    }


def _normalize_dp_state(state: dict[str, Any]) -> dict[str, Any]:
    """Normalize DP step state with required keys and defaults."""
    table_raw = state.get("table") if isinstance(state.get("table"), list) else []
    table: list[list[int]] = []
    for row in table_raw:
        if isinstance(row, list):
            table.append([_safe_int(cell, 0) for cell in row])

    if not table or not table[0]:
        table = [[0]]

    current_cell_raw = state.get("current_cell")
    if isinstance(current_cell_raw, list) and len(current_cell_raw) >= 2:
        i = _safe_int(current_cell_raw[0], 0)
        j = _safe_int(current_cell_raw[1], 0)
    else:
        i, j = 0, 0

    max_i = len(table) - 1
    max_j = len(table[0]) - 1
    i = max(0, min(i, max_i))
    j = max(0, min(j, max_j))

    return {
        "table": table,
        "current_cell": [i, j],
    }


def _normalize_linked_list_state(state: dict[str, Any]) -> dict[str, Any]:
    """Normalize linked list step state with required keys and defaults."""
    nodes = state.get("nodes") if isinstance(state.get("nodes"), list) else []
    parsed_nodes: list[dict[str, Any]] = []
    for node in nodes:
        if isinstance(node, dict):
            parsed_nodes.append({
                "value": _safe_int(node.get("value"), 0),
                "index": _safe_int(node.get("index"), 0),
                "next": _safe_int(node.get("next"), -1) if node.get("next") is not None else None,
            })
    if not parsed_nodes:
        parsed_nodes = [{"value": 0, "index": 0, "next": None}]

    highlight = state.get("highlight") if isinstance(state.get("highlight"), list) else []
    parsed_highlight = [_safe_int(index, -1) for index in highlight]

    pointers = state.get("pointers") if isinstance(state.get("pointers"), dict) else {}
    parsed_pointers: dict[str, Any] = {}
    for key, value in pointers.items():
        if value is None:
            parsed_pointers[key] = None
        else:
            parsed_pointers[key] = _safe_int(value, -1)

    return {
        "nodes": parsed_nodes,
        "highlight": parsed_highlight,
        "pointers": parsed_pointers,
    }


def _normalize_hashmap_state(state: dict[str, Any]) -> dict[str, Any]:
    """Normalize hashmap step state with required keys and defaults."""
    buckets_raw = state.get("buckets") if isinstance(state.get("buckets"), list) else []
    buckets: list[list[dict[str, Any]]] = []
    for bucket in buckets_raw:
        if isinstance(bucket, list):
            parsed_bucket: list[dict[str, Any]] = []
            for item in bucket:
                if isinstance(item, dict):
                    parsed_bucket.append({
                        "key": str(item.get("key", "")),
                        "value": _safe_int(item.get("value"), 0),
                    })
            buckets.append(parsed_bucket)
    if not buckets:
        buckets = [[]]

    highlight_bucket = state.get("highlight_bucket")
    if highlight_bucket is not None:
        highlight_bucket = _safe_int(highlight_bucket, -1)

    highlight_key = state.get("highlight_key")
    if highlight_key is not None:
        highlight_key = str(highlight_key)

    operation = str(state.get("operation", "init"))

    return {
        "buckets": buckets,
        "highlight_bucket": highlight_bucket,
        "highlight_key": highlight_key,
        "operation": operation,
    }


def _normalize_state(problem_type: str, state: dict[str, Any]) -> dict[str, Any]:
    """Normalize state structure based on problem type."""
    if problem_type == "graph":
        return _normalize_graph_state(state)
    if problem_type == "tree":
        return _normalize_tree_state(state)
    if problem_type == "dp":
        return _normalize_dp_state(state)
    if problem_type == "linked_list":
        return _normalize_linked_list_state(state)
    if problem_type == "hashmap":
        return _normalize_hashmap_state(state)
    return _normalize_array_state(state)


def _default_state(problem_type: str) -> dict[str, Any]:
    """Return a default state for the given problem type."""
    return _normalize_state(problem_type, {})


def _normalize_visualization(
    problem_type: str,
    visualization: dict[str, Any],
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    """Normalize visualization section for consistent frontend usage."""
    if not isinstance(visualization, dict):
        visualization = {}
    data = visualization.get("data") if isinstance(visualization.get("data"), dict) else {}

    if problem_type == "graph":
        source = steps[-1]["state"] if steps else _default_state("graph")
        return {
            "type": "graph",
            "data": {
                "nodes": data.get("nodes") if isinstance(data.get("nodes"), list) else source["nodes"],
                "edges": data.get("edges") if isinstance(data.get("edges"), list) else source["edges"],
            },
        }

    if problem_type == "tree":
        source = steps[-1]["state"] if steps else _default_state("tree")
        return {
            "type": "tree",
            "data": {
                "nodes": data.get("nodes") if isinstance(data.get("nodes"), list) else source["nodes"],
                "edges": data.get("edges") if isinstance(data.get("edges"), list) else source["edges"],
            },
        }

    if problem_type == "dp":
        source = steps[-1]["state"] if steps else _default_state("dp")
        table = data.get("table") if isinstance(data.get("table"), list) else source["table"]
        if not table or not isinstance(table[0], list):
            table = [[0]]
        dimensions = data.get("dimensions")
        if not (isinstance(dimensions, list) and len(dimensions) >= 2):
            dimensions = [len(table), len(table[0])]
        return {
            "type": "dp",
            "data": {
                "table": table,
                "dimensions": [_safe_int(dimensions[0], len(table)), _safe_int(dimensions[1], len(table[0]))],
            },
        }

    if problem_type == "linked_list":
        source = steps[-1]["state"] if steps else _default_state("linked_list")
        viz_data = data if data else {}
        return {
            "type": "linked_list",
            "data": {
                "initial_values": viz_data.get("initial_values") if isinstance(viz_data.get("initial_values"), list) else [node.get("value", 0) for node in source.get("nodes", [])],
                "nodes": viz_data.get("nodes") if isinstance(viz_data.get("nodes"), list) else source.get("nodes", []),
            },
        }

    if problem_type == "hashmap":
        source = steps[-1]["state"] if steps else _default_state("hashmap")
        viz_data = data if data else {}
        return {
            "type": "hashmap",
            "data": {
                "buckets": viz_data.get("buckets") if isinstance(viz_data.get("buckets"), list) else source.get("buckets", [[]]),
                "capacity": viz_data.get("capacity") if isinstance(viz_data.get("capacity"), int) else 8,
            },
        }

    initial_array = data.get("initial_array") if isinstance(data.get("initial_array"), list) else None
    if initial_array is None:
        if steps:
            initial_array = steps[0]["state"].get("array", [0])
        else:
            initial_array = [0]

    return {
        "type": "array",
        "data": {
            "initial_array": [_safe_int(item, 0) for item in initial_array] or [0],
        },
    }


def normalize_response(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize response payload for deterministic frontend consumption."""
    payload = deepcopy(data) if isinstance(data, dict) else {}

    problem_type = str(payload.get("problem_type", "array")).lower()
    if problem_type not in _SUPPORTED_TYPES:
        problem_type = "array"

    explanation = str(payload.get("explanation", "")).strip()
    if not explanation:
        explanation = "No explanation provided."

    steps_input = payload.get("steps") if isinstance(payload.get("steps"), list) else []
    raw_steps: list[dict[str, Any]] = [step for step in steps_input if isinstance(step, dict)]
    raw_steps.sort(key=lambda step: _safe_int(step.get("step"), 10**9))

    normalized_steps: list[dict[str, Any]] = []
    for index, step in enumerate(raw_steps, start=1):
        description = str(step.get("description", "")).strip() or f"Step {index}"
        state_input = step.get("state") if isinstance(step.get("state"), dict) else {}
        normalized_steps.append(
            {
                "step": index,
                "description": description,
                "state": _normalize_state(problem_type, state_input),
            }
        )

    if not normalized_steps:
        normalized_steps = [
            {
                "step": 1,
                "description": "Initialize state.",
                "state": _default_state(problem_type),
            }
        ]

    visualization = _normalize_visualization(
        problem_type,
        payload.get("visualization") if isinstance(payload.get("visualization"), dict) else {},
        normalized_steps,
    )

    return {
        "problem_type": problem_type,
        "explanation": explanation,
        "steps": normalized_steps,
        "visualization": visualization,
    }
