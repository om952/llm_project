"""Deterministic hashmap algorithm visualization engine."""

from __future__ import annotations

import re
from typing import Any


def _parse_key_value_pairs(problem: str) -> list[tuple[str, int]]:
    """Extract key-value pairs from problem text."""
    pairs: list[tuple[str, int]] = []
    # Match patterns like "key: value", "key=value", "(key, value)"
    patterns = [
        r'\(([^,)]+),\s*(-?\d+)\)',
        r'"([^"]+)":\s*(-?\d+)',
        r"'([^']+)':\s*(-?\d+)",
        r'([a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]\s*(-?\d+)',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, problem):
            key = match.group(1).strip()
            value = int(match.group(2))
            pairs.append((key, value))
    return pairs


def _extract_hashmap_values(problem: str) -> dict[str, int]:
    """Extract initial hashmap key-value pairs from problem text."""
    pairs = _parse_key_value_pairs(problem)
    if pairs:
        return {k: v for k, v in pairs}
    # Fallback: try to extract numbers and assign generic keys
    numbers = [int(token) for token in re.findall(r"-?\d+", problem)]
    if numbers:
        return {f"key_{i}": v for i, v in enumerate(numbers[:5])}
    return {"key_0": 0}


def _extract_single_key(problem: str) -> str:
    """Extract a single key for search/delete operations."""
    # Try to find quoted key first (most specific)
    match = re.search(r'["\']([a-zA-Z_][a-zA-Z0-9_]*)["\']', problem)
    if match:
        return match.group(1)
    # Try to find key after search/find/delete/remove
    match = re.search(r'(?:search|find|delete|remove|get)\s+(?:for\s+)?["\']?([a-zA-Z_][a-zA-Z0-9_]*)["\']?', problem, re.IGNORECASE)
    if match:
        return match.group(1)
    return "key_0"


def _extract_single_value(problem: str) -> int:
    """Extract a single value for insertion."""
    match = re.search(r'(?:value|insert)\s+(?:of\s+)?(-?\d+)', problem, re.IGNORECASE)
    if match:
        return int(match.group(1))
    numbers = [int(token) for token in re.findall(r"-?\d+", problem)]
    return numbers[-1] if numbers else 0


def detect_hashmap_algorithm(problem: str) -> str:
    """Detect hashmap algorithm from user text."""
    lowered = problem.lower()
    if any(kw in lowered for kw in ("search", "find", "get", "lookup")):
        return "search"
    if any(kw in lowered for kw in ("delete", "remove", "erase")):
        return "deletion"
    if any(kw in lowered for kw in ("insert", "add", "put", "set")):
        return "insertion"
    if any(kw in lowered for kw in ("collision", "chaining", "linear probe", "quadratic probe")):
        return "collision"
    return "insertion"


def _simple_hash(key: str, capacity: int = 8) -> int:
    """Simple hash function for demonstration."""
    return sum(ord(c) for c in key) % capacity


def _build_buckets(data: dict[str, int], capacity: int = 8) -> list[list[dict[str, Any]]]:
    """Build hashmap bucket representation with chaining."""
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(capacity)]
    for key, value in data.items():
        idx = _simple_hash(key, capacity)
        buckets[idx].append({"key": key, "value": value})
    return buckets


def _append_step(
    steps: list[dict[str, Any]],
    description: str,
    buckets: list[list[dict[str, Any]]],
    highlight_bucket: int | None,
    highlight_key: str | None,
    operation: str,
) -> None:
    """Append one step with full state, skipping exact duplicate states."""
    new_state = {
        "buckets": [[dict(item) for item in bucket] for bucket in buckets],
        "highlight_bucket": highlight_bucket,
        "highlight_key": highlight_key,
        "operation": operation,
    }

    if steps and steps[-1]["state"] == new_state:
        return

    steps.append(
        {
            "step": len(steps) + 1,
            "description": description,
            "state": new_state,
        }
    )


def _simulate_insertion(data: dict[str, int], new_key: str, new_value: int, capacity: int = 8) -> tuple[str, list[dict[str, Any]]]:
    """Create deterministic hashmap insertion simulation steps."""
    buckets = _build_buckets(data, capacity)
    steps: list[dict[str, Any]] = []

    _append_step(
        steps,
        f"Initialize hashmap with capacity {capacity}.",
        buckets,
        None,
        None,
        "init",
    )

    hash_idx = _simple_hash(new_key, capacity)

    _append_step(
        steps,
        f"Compute hash for key '{new_key}': hash = sum(ord(c)) % {capacity} = {hash_idx}.",
        buckets,
        hash_idx,
        None,
        "hash",
    )

    # Check for existing key (update)
    found = False
    for i, item in enumerate(buckets[hash_idx]):
        _append_step(
            steps,
            f"Check bucket {hash_idx}, entry {i}: key = '{item['key']}'.",
            buckets,
            hash_idx,
            item["key"],
            "compare",
        )
        if item["key"] == new_key:
            found = True
            old_value = item["value"]
            item["value"] = new_value
            _append_step(
                steps,
                f"Key '{new_key}' found! Update value from {old_value} to {new_value}.",
                buckets,
                hash_idx,
                new_key,
                "update",
            )
            break

    if not found:
        buckets[hash_idx].append({"key": new_key, "value": new_value})
        _append_step(
            steps,
            f"Key '{new_key}' not found. Insert new entry at bucket {hash_idx}.",
            buckets,
            hash_idx,
            new_key,
            "insert",
        )

    _append_step(
        steps,
        f"Insertion complete: key '{new_key}' = {new_value} stored in bucket {hash_idx}.",
        buckets,
        hash_idx,
        new_key,
        "complete",
    )

    return (
        "Hashmap Insertion computes the hash of the key to find the target bucket, then inserts or updates the key-value pair in that bucket (using chaining for collisions).",
        steps,
    )


def _simulate_deletion(data: dict[str, int], delete_key: str, capacity: int = 8) -> tuple[str, list[dict[str, Any]]]:
    """Create deterministic hashmap deletion simulation steps."""
    buckets = _build_buckets(data, capacity)
    steps: list[dict[str, Any]] = []

    _append_step(
        steps,
        f"Initialize hashmap with capacity {capacity}.",
        buckets,
        None,
        None,
        "init",
    )

    hash_idx = _simple_hash(delete_key, capacity)

    _append_step(
        steps,
        f"Compute hash for key '{delete_key}': hash = sum(ord(c)) % {capacity} = {hash_idx}.",
        buckets,
        hash_idx,
        None,
        "hash",
    )

    found = False
    for i, item in enumerate(buckets[hash_idx]):
        _append_step(
            steps,
            f"Check bucket {hash_idx}, entry {i}: key = '{item['key']}'.",
            buckets,
            hash_idx,
            item["key"],
            "compare",
        )
        if item["key"] == delete_key:
            found = True
            removed_value = item["value"]
            buckets[hash_idx].pop(i)
            _append_step(
                steps,
                f"Found key '{delete_key}' with value {removed_value}. Remove from bucket {hash_idx}.",
                buckets,
                hash_idx,
                delete_key,
                "delete",
            )
            break

    if not found:
        _append_step(
            steps,
            f"Key '{delete_key}' not found in bucket {hash_idx}. Nothing to delete.",
            buckets,
            hash_idx,
            delete_key,
            "not_found",
        )
        return (
            f"Hashmap Deletion: key '{delete_key}' was not found in the hashmap.",
            steps,
        )

    _append_step(
        steps,
        f"Deletion complete: key '{delete_key}' removed from bucket {hash_idx}.",
        buckets,
        hash_idx,
        None,
        "complete",
    )

    return (
        "Hashmap Deletion computes the hash of the key to locate the bucket, then searches the bucket and removes the matching key-value pair.",
        steps,
    )


def _simulate_search(data: dict[str, int], search_key: str, capacity: int = 8) -> tuple[str, list[dict[str, Any]]]:
    """Create deterministic hashmap search simulation steps."""
    buckets = _build_buckets(data, capacity)
    steps: list[dict[str, Any]] = []

    _append_step(
        steps,
        f"Initialize hashmap with capacity {capacity}.",
        buckets,
        None,
        None,
        "init",
    )

    hash_idx = _simple_hash(search_key, capacity)

    _append_step(
        steps,
        f"Compute hash for key '{search_key}': hash = sum(ord(c)) % {capacity} = {hash_idx}.",
        buckets,
        hash_idx,
        None,
        "hash",
    )

    found = False
    found_value = None
    for i, item in enumerate(buckets[hash_idx]):
        _append_step(
            steps,
            f"Check bucket {hash_idx}, entry {i}: key = '{item['key']}', value = {item['value']}.",
            buckets,
            hash_idx,
            item["key"],
            "compare",
        )
        if item["key"] == search_key:
            found = True
            found_value = item["value"]
            _append_step(
                steps,
                f"Found key '{search_key}' with value {found_value} in bucket {hash_idx}.",
                buckets,
                hash_idx,
                search_key,
                "found",
            )
            break

    if not found:
        _append_step(
            steps,
            f"Key '{search_key}' not found in bucket {hash_idx}.",
            buckets,
            hash_idx,
            search_key,
            "not_found",
        )
        return (
            f"Hashmap Search: key '{search_key}' was not found in the hashmap.",
            steps,
        )

    _append_step(
        steps,
        f"Search complete: key '{search_key}' = {found_value} found in bucket {hash_idx}.",
        buckets,
        hash_idx,
        search_key,
        "complete",
    )

    return (
        "Hashmap Search computes the hash of the key to find the target bucket, then searches the bucket for the matching key and returns its value.",
        steps,
    )


def _simulate_collision(data: dict[str, int], capacity: int = 8) -> tuple[str, list[dict[str, Any]]]:
    """Create deterministic hashmap collision handling simulation steps."""
    buckets = _build_buckets(data, capacity)
    steps: list[dict[str, Any]] = []

    _append_step(
        steps,
        f"Initialize hashmap with capacity {capacity} using separate chaining.",
        buckets,
        None,
        None,
        "init",
    )

    # Find a bucket with multiple items to demonstrate collision
    collision_bucket = None
    for i, bucket in enumerate(buckets):
        if len(bucket) > 1:
            collision_bucket = i
            break

    if collision_bucket is not None:
        bucket = buckets[collision_bucket]
        _append_step(
            steps,
            f"Collision detected in bucket {collision_bucket}: {len(bucket)} keys hash to the same index.",
            buckets,
            collision_bucket,
            None,
            "collision_detected",
        )

        keys = [item["key"] for item in bucket]
        _append_step(
            steps,
            f"Keys {keys} all map to bucket {collision_bucket} via hash % {capacity}.",
            buckets,
            collision_bucket,
            None,
            "collision_explain",
        )

        _append_step(
            steps,
            f"Separate chaining stores all colliding keys in a linked list within bucket {collision_bucket}.",
            buckets,
            collision_bucket,
            None,
            "chaining",
        )
    else:
        _append_step(
            steps,
            "No collisions detected in current hashmap. All keys map to unique buckets.",
            buckets,
            None,
            None,
            "no_collision",
        )

    return (
        "Hashmap Collision Handling uses separate chaining to store multiple key-value pairs that hash to the same bucket in a linked list structure.",
        steps,
    )


def build_hashmap_solution(problem: str) -> dict[str, Any]:
    """Build a strict hashmap visualization response without relying on LLM output."""
    algorithm = detect_hashmap_algorithm(problem)
    data = _extract_hashmap_values(problem)

    if algorithm == "search":
        search_key = _extract_single_key(problem)
        explanation, steps = _simulate_search(data, search_key)
    elif algorithm == "deletion":
        delete_key = _extract_single_key(problem)
        explanation, steps = _simulate_deletion(data, delete_key)
    elif algorithm == "collision":
        explanation, steps = _simulate_collision(data)
    else:
        # Default to insertion
        new_key = _extract_single_key(problem)
        new_value = _extract_single_value(problem)
        explanation, steps = _simulate_insertion(data, new_key, new_value)

    if not steps:
        buckets = _build_buckets(data)
        steps = [
            {
                "step": 1,
                "description": "Initialize hashmap state.",
                "state": {
                    "buckets": buckets,
                    "highlight_bucket": None,
                    "highlight_key": None,
                    "operation": "init",
                },
            }
        ]

    last_state = steps[-1]["state"]

    return {
        "problem_type": "hashmap",
        "explanation": explanation,
        "steps": steps,
        "visualization": {
            "type": "hashmap",
            "data": {
                "buckets": last_state["buckets"],
                "capacity": 8,
            },
        },
    }
