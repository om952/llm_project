"""Deterministic array-based algorithm visualization fallback engine."""

from __future__ import annotations

import re
from typing import Any


def extract_array(problem: str) -> list[int]:
    """Extract an integer array from the problem text."""
    bracket_match = re.search(r"\[([^\]]+)\]", problem)
    if bracket_match:
        raw_items = [item.strip() for item in bracket_match.group(1).split(",")]
        values: list[int] = []
        for item in raw_items:
            if not item:
                continue
            if re.fullmatch(r"-?\d+", item):
                values.append(int(item))
        if values:
            return values

    fallback_values = [int(token) for token in re.findall(r"-?\d+", problem)]
    return fallback_values if fallback_values else [0]


def detect_algorithm(problem: str) -> str:
    """Detect sorting algorithm from user text, defaulting to bubble sort."""
    lowered = problem.lower()
    if "selection" in lowered:
        return "selection"
    if "insertion" in lowered:
        return "insertion"
    return "bubble"


def _append_step(
    steps: list[dict[str, Any]],
    description: str,
    array: list[int],
    highlight: list[int],
    sorted_boundary: int,
) -> None:
    """Append one step with full state, skipping exact duplicate states."""
    new_state = {
        "array": list(array),
        "highlight": list(highlight),
        "sorted_boundary": sorted_boundary,
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


def _simulate_bubble(initial_array: list[int]) -> tuple[str, list[dict[str, Any]]]:
    """Create deterministic Bubble Sort simulation steps."""
    arr = list(initial_array)
    n = len(arr)
    steps: list[dict[str, Any]] = []
    sorted_count = -1

    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            _append_step(
                steps,
                f"Compare indices {j} and {j + 1}.",
                arr,
                [j, j + 1],
                sorted_count,
            )
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
                _append_step(
                    steps,
                    f"Swap values at indices {j} and {j + 1}.",
                    arr,
                    [j, j + 1],
                    sorted_count,
                )

        sorted_count = i + 1
        _append_step(
            steps,
            "Mark one more element at the end as sorted.",
            arr,
            [],
            sorted_count,
        )

        if not swapped:
            break

    return (
        "Bubble Sort repeatedly compares adjacent elements and swaps them if they are in the wrong order, moving larger values to the end each pass.",
        steps,
    )


def _simulate_selection(initial_array: list[int]) -> tuple[str, list[dict[str, Any]]]:
    """Create deterministic Selection Sort simulation steps."""
    arr = list(initial_array)
    n = len(arr)
    steps: list[dict[str, Any]] = []
    sorted_count = -1

    for i in range(n):
        min_idx = i
        _append_step(
            steps,
            f"Set current minimum index to {i}.",
            arr,
            [i],
            sorted_count,
        )

        for j in range(i + 1, n):
            _append_step(
                steps,
                f"Compare current minimum index {min_idx} with index {j}.",
                arr,
                [min_idx, j],
                sorted_count,
            )
            if arr[j] < arr[min_idx]:
                min_idx = j
                _append_step(
                    steps,
                    f"Update minimum index to {min_idx}.",
                    arr,
                    [min_idx],
                    sorted_count,
                )

        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
            _append_step(
                steps,
                f"Swap index {i} with minimum index {min_idx}.",
                arr,
                [i, min_idx],
                sorted_count,
            )

        sorted_count = i + 1
        _append_step(
            steps,
            "Expand the sorted prefix by one element.",
            arr,
            [],
            sorted_count,
        )

    return (
        "Selection Sort repeatedly selects the minimum element from the unsorted part and places it at the next sorted position.",
        steps,
    )


def _simulate_insertion(initial_array: list[int]) -> tuple[str, list[dict[str, Any]]]:
    """Create deterministic Insertion Sort simulation steps."""
    arr = list(initial_array)
    n = len(arr)
    steps: list[dict[str, Any]] = []
    sorted_count = 1 if n > 0 else -1

    _append_step(
        steps,
        "Initialize sorted prefix with the first element.",
        arr,
        [0] if n > 0 else [],
        sorted_count,
    )

    for i in range(1, n):
        key = arr[i]
        j = i - 1

        _append_step(
            steps,
            f"Select key at index {i} for insertion.",
            arr,
            [i],
            sorted_count,
        )

        while j >= 0 and arr[j] > key:
            _append_step(
                steps,
                f"Compare key with index {j}.",
                arr,
                [j, j + 1],
                sorted_count,
            )
            arr[j + 1] = arr[j]
            _append_step(
                steps,
                f"Shift value from index {j} to index {j + 1}.",
                arr,
                [j, j + 1],
                sorted_count,
            )
            j -= 1

        arr[j + 1] = key
        _append_step(
            steps,
            f"Insert key at index {j + 1}.",
            arr,
            [j + 1],
            sorted_count,
        )

        sorted_count = i + 1
        _append_step(
            steps,
            "Expand the sorted prefix by one element.",
            arr,
            [],
            sorted_count,
        )

    return (
        "Insertion Sort builds a sorted prefix by inserting each new element into its correct position within the already sorted part.",
        steps,
    )


def build_deterministic_solution(problem: str) -> dict[str, Any]:
    """Build a strict array visualization response without relying on LLM output."""
    initial_array = extract_array(problem)
    algorithm = detect_algorithm(problem)

    if algorithm == "selection":
        explanation, steps = _simulate_selection(initial_array)
    elif algorithm == "insertion":
        explanation, steps = _simulate_insertion(initial_array)
    else:
        explanation, steps = _simulate_bubble(initial_array)

    if not steps:
        steps = [
            {
                "step": 1,
                "description": "Initialize array state.",
                "state": {
                    "array": list(initial_array),
                    "highlight": [],
                    "sorted_boundary": -1,
                },
            }
        ]

    return {
        "problem_type": "array",
        "explanation": explanation,
        "steps": steps,
        "visualization": {
            "type": "array",
            "data": {
                "initial_array": list(initial_array),
            },
        },
    }
