"""Deterministic graph BFS engine and graph parsing utilities."""

from __future__ import annotations

from collections import deque
import re
from typing import Any


class GraphParseError(ValueError):
    """Raised when a graph problem cannot be parsed into nodes/edges/start."""


def _normalize_node(token: str) -> str:
    """Normalize a node token from natural language input."""
    return token.strip().strip(",.;:()[]{}")


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    """Return unique values while preserving insertion order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _extract_edges(problem: str) -> list[list[str]]:
    """Extract graph edges from natural language using common edge patterns."""
    edge_matches: list[tuple[str, str]] = []
    patterns = (
        r"([A-Za-z0-9_]+)\s*-\s*([A-Za-z0-9_]+)",
        r"([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)",
        r"([A-Za-z0-9_]+)\s+to\s+([A-Za-z0-9_]+)",
    )

    for pattern in patterns:
        for left, right in re.findall(pattern, problem, flags=re.IGNORECASE):
            a = _normalize_node(left)
            b = _normalize_node(right)
            if a and b:
                edge_matches.append((a, b))

    unique_edges: list[list[str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for left, right in edge_matches:
        pair_key = (left, right)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        unique_edges.append([left, right])

    return unique_edges


def _extract_nodes(problem: str, edges: list[list[str]]) -> list[str]:
    """Extract node identifiers from text and edges."""
    extracted_nodes: list[str] = []

    node_section = re.search(
        r"nodes?\s*(?:are|=|:)?\s*([A-Za-z0-9_,\s]+)",
        problem,
        flags=re.IGNORECASE,
    )
    if node_section:
        raw_nodes = node_section.group(1)
        raw_nodes = re.split(
            r"\b(?:and|with|where|starting|start|edges?)\b",
            raw_nodes,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        extracted_nodes.extend(re.findall(r"[A-Za-z0-9_]+", raw_nodes))

    for left, right in edges:
        extracted_nodes.append(left)
        extracted_nodes.append(right)

    normalized = [_normalize_node(node) for node in extracted_nodes]
    normalized = [node for node in normalized if node]
    return _dedupe_preserve_order(normalized)


def _extract_start(problem: str, nodes: list[str]) -> str:
    """Extract BFS start node from text or infer a default."""
    start_patterns = (
        r"starting\s+from\s+([A-Za-z0-9_]+)",
        r"start\s+from\s+([A-Za-z0-9_]+)",
        r"start(?:ing)?\s+at\s+([A-Za-z0-9_]+)",
        r"from\s+([A-Za-z0-9_]+)",
    )

    candidate: str | None = None
    for pattern in start_patterns:
        match = re.search(pattern, problem, flags=re.IGNORECASE)
        if match:
            candidate = _normalize_node(match.group(1))
            break

    if candidate:
        for node in nodes:
            if node.lower() == candidate.lower():
                return node

    if not nodes:
        raise GraphParseError("No graph nodes available to infer a BFS start node.")

    return nodes[0]


def parse_graph_problem(problem: str) -> tuple[dict[str, Any], str]:
    """Parse a natural-language graph BFS prompt into graph + start node."""
    edges = _extract_edges(problem)
    nodes = _extract_nodes(problem, edges)

    if not nodes:
        raise GraphParseError("Unable to parse graph nodes from problem statement.")

    if not edges:
        raise GraphParseError("Unable to parse graph edges from problem statement.")

    start = _extract_start(problem, nodes)
    graph = {
        "nodes": nodes,
        "edges": edges,
    }
    return graph, start


def _append_step(
    steps: list[dict[str, Any]],
    description: str,
    nodes: list[str],
    edges: list[list[str]],
    visited: list[str],
    active: str,
    queue: list[str],
) -> None:
    """Append one BFS step while preventing duplicate consecutive states."""
    state = {
        "nodes": list(nodes),
        "edges": [list(edge) for edge in edges],
        "visited": list(visited),
        "active": active,
        "queue": list(queue),
    }

    if steps and steps[-1]["state"] == state:
        return

    steps.append(
        {
            "step": len(steps) + 1,
            "description": description,
            "state": state,
        }
    )


def run_bfs(graph: dict[str, Any], start: str) -> dict[str, Any]:
    """Run deterministic BFS and return strict graph visualization response."""
    nodes_input = graph.get("nodes", [])
    edges_input = graph.get("edges", [])

    nodes = [str(node) for node in nodes_input if str(node)]
    nodes = _dedupe_preserve_order(nodes)
    if not nodes:
        raise GraphParseError("Graph must include at least one node.")

    edges: list[list[str]] = []
    for edge in edges_input:
        if not isinstance(edge, list) or len(edge) < 2:
            continue
        left = str(edge[0]).strip()
        right = str(edge[1]).strip()
        if left and right:
            edges.append([left, right])
            if left not in nodes:
                nodes.append(left)
            if right not in nodes:
                nodes.append(right)

    if not edges:
        raise GraphParseError("Graph must include at least one edge.")

    start_node = str(start).strip()
    if not start_node:
        raise GraphParseError("A valid BFS start node is required.")

    if start_node not in nodes:
        matched = next((node for node in nodes if node.lower() == start_node.lower()), None)
        if matched is None:
            raise GraphParseError(f"Start node '{start_node}' is not present in graph nodes.")
        start_node = matched

    adjacency: dict[str, list[str]] = {node: [] for node in nodes}
    for left, right in edges:
        if right not in adjacency[left]:
            adjacency[left].append(right)
        if left not in adjacency[right]:
            adjacency[right].append(left)

    queue: deque[str] = deque([start_node])
    seen: set[str] = {start_node}
    visited: list[str] = [start_node]
    steps: list[dict[str, Any]] = []

    _append_step(
        steps,
        f"Start from node {start_node}.",
        nodes,
        edges,
        visited,
        start_node,
        list(queue),
    )

    while queue:
        current = queue.popleft()

        for neighbor in adjacency.get(current, []):
            if neighbor not in seen:
                seen.add(neighbor)
                visited.append(neighbor)
                queue.append(neighbor)

        _append_step(
            steps,
            f"Visit node {current} and enqueue unvisited neighbors.",
            nodes,
            edges,
            visited,
            current,
            list(queue),
        )

    return {
        "problem_type": "graph",
        "explanation": "Breadth First Search traversal",
        "steps": steps,
        "visualization": {
            "type": "graph",
            "data": {
                "nodes": nodes,
                "edges": edges,
            },
        },
    }
