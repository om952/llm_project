"""Deterministic linked list algorithm visualization engine."""

from __future__ import annotations

import re
from typing import Any


def extract_linked_list_values(problem: str) -> list[int]:
    """Extract integer values for the linked list from the problem text."""
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


def detect_linked_list_algorithm(problem: str) -> str:
    """Detect linked list algorithm from user text."""
    lowered = problem.lower()
    if "reverse" in lowered:
        return "reverse"
    if "detect cycle" in lowered or "cycle detection" in lowered or "has cycle" in lowered:
        return "cycle_detection"
    if "merge" in lowered:
        return "merge"
    if "insert" in lowered:
        return "insertion"
    if "delete" in lowered or "remove" in lowered:
        return "deletion"
    return "traversal"


def _build_nodes(values: list[int]) -> list[dict[str, Any]]:
    """Build linked list node representations from values."""
    nodes: list[dict[str, Any]] = []
    for i, val in enumerate(values):
        nodes.append({
            "value": val,
            "index": i,
            "next": i + 1 if i < len(values) - 1 else None,
        })
    return nodes


def _append_step(
    steps: list[dict[str, Any]],
    description: str,
    nodes: list[dict[str, Any]],
    highlight: list[int],
    pointers: dict[str, int | None],
) -> None:
    """Append one step with full state, skipping exact duplicate states."""
    new_state = {
        "nodes": [dict(node) for node in nodes],
        "highlight": list(highlight),
        "pointers": dict(pointers),
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


def _simulate_traversal(values: list[int]) -> tuple[str, list[dict[str, Any]]]:
    """Create deterministic linked list traversal simulation steps."""
    nodes = _build_nodes(values)
    steps: list[dict[str, Any]] = []
    pointers: dict[str, int | None] = {"current": 0, "prev": None}

    _append_step(
        steps,
        "Initialize traversal: set current pointer to head (index 0).",
        nodes,
        [0],
        pointers,
    )

    current = 0
    while current is not None:
        _append_step(
            steps,
            f"Visit node at index {current} with value {nodes[current]['value']}.",
            nodes,
            [current],
            {"current": current, "prev": pointers["prev"]},
        )

        prev = current
        current = nodes[current]["next"]
        pointers = {"current": current, "prev": prev}

        if current is not None:
            _append_step(
                steps,
                f"Move current pointer to next node (index {current}).",
                nodes,
                [current],
                pointers,
            )

    _append_step(
        steps,
        "Traversal complete: reached end of linked list (null).",
        nodes,
        [],
        pointers,
    )

    return (
        "Linked List Traversal visits each node sequentially by following the next pointers from the head to the end of the list.",
        steps,
    )


def _simulate_reverse(values: list[int]) -> tuple[str, list[dict[str, Any]]]:
    """Create deterministic linked list reversal simulation steps."""
    nodes = _build_nodes(values)
    steps: list[dict[str, Any]] = []

    prev: int | None = None
    current = 0 if nodes else None

    pointers: dict[str, int | None] = {"current": current, "prev": prev, "next": None}

    _append_step(
        steps,
        "Initialize reversal: prev = null, current = head.",
        nodes,
        [current] if current is not None else [],
        pointers,
    )

    while current is not None:
        next_node = nodes[current]["next"]
        pointers = {"current": current, "prev": prev, "next": next_node}

        _append_step(
            steps,
            f"Store next pointer of node {current} (value {nodes[current]['value']}).",
            nodes,
            [current],
            pointers,
        )

        # Reverse the link
        nodes[current]["next"] = prev
        pointers = {"current": current, "prev": prev, "next": next_node}

        _append_step(
            steps,
            f"Reverse link: node {current} now points to {prev if prev is not None else 'null'}.",
            nodes,
            [current],
            pointers,
        )

        # Move pointers forward
        prev = current
        current = next_node
        pointers = {"current": current, "prev": prev, "next": None}

        if current is not None:
            _append_step(
                steps,
                f"Move prev and current forward: prev = {prev}, current = {current}.",
                nodes,
                [current],
                pointers,
            )

    _append_step(
        steps,
        f"Reversal complete: new head is at index {prev} (value {nodes[prev]['value']}).",
        nodes,
        [prev] if prev is not None else [],
        pointers,
    )

    return (
        "Linked List Reversal reverses the direction of all next pointers so that the last node becomes the new head.",
        steps,
    )


def _simulate_cycle_detection(values: list[int]) -> tuple[str, list[dict[str, Any]]]:
    """Create deterministic cycle detection simulation steps (Floyd's algorithm)."""
    # For deterministic demo, we create a list without a cycle
    nodes = _build_nodes(values)
    steps: list[dict[str, Any]] = []

    if not nodes:
        return "Cycle Detection using Floyd's Tortoise and Hare algorithm.", steps

    slow = 0
    fast = 0

    pointers: dict[str, int | None] = {"slow": slow, "fast": fast}

    _append_step(
        steps,
        "Initialize Floyd's Cycle Detection: slow = head, fast = head.",
        nodes,
        [slow, fast],
        pointers,
    )

    has_cycle = False
    while True:
        pointers = {"slow": slow, "fast": fast}
        _append_step(
            steps,
            f"Compare slow ({slow}) and fast ({fast}).",
            nodes,
            [slow, fast],
            pointers,
        )

        if slow == fast and steps[-1]["step"] > 1:
            has_cycle = True
            _append_step(
                steps,
                f"Cycle detected: slow ({slow}) == fast ({fast}).",
                nodes,
                [slow, fast],
                pointers,
            )
            break

        # Move slow one step
        slow_next = nodes[slow]["next"]
        # Move fast two steps
        fast_next = nodes[fast]["next"]
        fast_next_next = nodes[fast_next]["next"] if fast_next is not None else None

        if fast_next is None or fast_next_next is None:
            _append_step(
                steps,
                "No cycle: fast pointer reached end of list.",
                nodes,
                [slow],
                {"slow": slow, "fast": fast},
            )
            break

        slow = slow_next
        fast = fast_next_next

        pointers = {"slow": slow, "fast": fast}
        _append_step(
            steps,
            f"Move slow to {slow} (1 step), fast to {fast} (2 steps).",
            nodes,
            [slow, fast],
            pointers,
        )

    result_text = (
        "Cycle detected in the linked list using Floyd's Tortoise and Hare algorithm."
        if has_cycle
        else "No cycle detected in the linked list using Floyd's Tortoise and Hare algorithm."
    )

    return result_text, steps


def _simulate_merge(list1: list[int], list2: list[int]) -> tuple[str, list[dict[str, Any]]]:
    """Create deterministic merge of two sorted linked lists simulation."""
    nodes1 = _build_nodes(list1)
    nodes2 = _build_nodes(list2)

    # Offset indices for second list
    offset = len(nodes1)
    for node in nodes2:
        node["index"] += offset
        if node["next"] is not None:
            node["next"] += offset

    all_nodes = nodes1 + nodes2
    steps: list[dict[str, Any]] = []

    merged: list[int] = []
    i = j = 0

    pointers: dict[str, int | None] = {"list1": 0 if nodes1 else None, "list2": offset if nodes2 else None, "merged_tail": None}

    _append_step(
        steps,
        "Initialize merge: point to heads of both sorted lists.",
        all_nodes,
        [0, offset] if nodes1 and nodes2 else ([0] if nodes1 else ([offset] if nodes2 else [])),
        pointers,
    )

    while i < len(list1) and j < len(list2):
        idx1 = i
        idx2 = offset + j

        pointers = {"list1": idx1, "list2": idx2, "merged_tail": merged[-1] if merged else None}
        _append_step(
            steps,
            f"Compare value {list1[i]} (list1) with {list2[j]} (list2).",
            all_nodes,
            [idx1, idx2],
            pointers,
        )

        if list1[i] <= list2[j]:
            merged.append(idx1)
            i += 1
        else:
            merged.append(idx2)
            j += 1

        pointers = {"list1": i if i < len(list1) else None, "list2": offset + j if j < len(list2) else None, "merged_tail": merged[-1] if merged else None}
        _append_step(
            steps,
            f"Append smaller value to merged list. Merged so far: {[all_nodes[idx]['value'] for idx in merged]}.",
            all_nodes,
            [merged[-1]],
            pointers,
        )

    # Append remaining elements
    while i < len(list1):
        idx = i
        merged.append(idx)
        i += 1
        pointers = {"list1": i if i < len(list1) else None, "list2": None, "merged_tail": merged[-1]}
        _append_step(
            steps,
            f"Append remaining element {all_nodes[idx]['value']} from list1.",
            all_nodes,
            [idx],
            pointers,
        )

    while j < len(list2):
        idx = offset + j
        merged.append(idx)
        j += 1
        pointers = {"list1": None, "list2": offset + j if j < len(list2) else None, "merged_tail": merged[-1]}
        _append_step(
            steps,
            f"Append remaining element {all_nodes[idx]['value']} from list2.",
            all_nodes,
            [idx],
            pointers,
        )

    # Update next pointers for merged list
    for k in range(len(merged) - 1):
        all_nodes[merged[k]]["next"] = merged[k + 1]
    if merged:
        all_nodes[merged[-1]]["next"] = None

    pointers = {"list1": None, "list2": None, "merged_tail": merged[-1] if merged else None}
    _append_step(
        steps,
        "Merge complete: all nodes linked in sorted order.",
        all_nodes,
        merged,
        pointers,
    )

    return (
        "Merge Two Sorted Linked Lists combines two sorted lists into one sorted list by comparing nodes one by one.",
        steps,
    )


def _simulate_insertion(values: list[int], insert_value: int, position: int | None = None) -> tuple[str, list[dict[str, Any]]]:
    """Create deterministic linked list insertion simulation steps."""
    nodes = _build_nodes(values)
    steps: list[dict[str, Any]] = []

    # Add new node at the end by default
    new_index = len(nodes)
    new_node = {"value": insert_value, "index": new_index, "next": None}

    pointers: dict[str, int | None] = {"current": 0 if nodes else None, "new_node": new_index}

    _append_step(
        steps,
        f"Create new node with value {insert_value} at index {new_index}.",
        nodes + [new_node],
        [new_index],
        pointers,
    )

    if not nodes:
        nodes.append(new_node)
        _append_step(
            steps,
            "List was empty: new node becomes the head.",
            nodes,
            [new_index],
            {"current": new_index, "new_node": new_index},
        )
        return (
            "Linked List Insertion adds a new node to the list. In an empty list, the new node becomes the head.",
            steps,
        )

    # Insert at specific position or append at end
    if position == 0:
        new_node["next"] = 0
        nodes.append(new_node)
        _append_step(
            steps,
            f"Insert at head: new node points to current head (index 0).",
            nodes,
            [new_index, 0],
            {"current": new_index, "new_node": new_index},
        )
    elif position is not None and 0 < position <= len(nodes):
        # Traverse to position
        current = 0
        prev = None
        count = 0

        while current is not None and count < position:
            pointers = {"current": current, "prev": prev, "new_node": new_index}
            _append_step(
                steps,
                f"Traverse to position {position}: at index {current} (value {nodes[current]['value']}).",
                nodes + [new_node],
                [current],
                pointers,
            )
            prev = current
            current = nodes[current]["next"]
            count += 1

        # Insert after prev
        if prev is not None:
            new_node["next"] = nodes[prev]["next"]
            nodes[prev]["next"] = new_index
            nodes.append(new_node)
            _append_step(
                steps,
                f"Insert new node after index {prev}: node {prev} now points to {new_index}, new node points to {new_node['next']}.",
                nodes,
                [prev, new_index],
                {"current": new_index, "prev": prev, "new_node": new_index},
            )
    else:
        # Append at end
        current = 0
        while current is not None:
            pointers = {"current": current, "new_node": new_index}
            _append_step(
                steps,
                f"Traverse to end: at index {current} (value {nodes[current]['value']}).",
                nodes + [new_node],
                [current],
                pointers,
            )

            if nodes[current]["next"] is None:
                nodes[current]["next"] = new_index
                nodes.append(new_node)
                _append_step(
                    steps,
                    f"Append at end: node {current} now points to new node {new_index}.",
                    nodes,
                    [current, new_index],
                    {"current": new_index, "new_node": new_index},
                )
                break

            current = nodes[current]["next"]

    _append_step(
        steps,
        f"Insertion complete: value {insert_value} added to the list.",
        nodes,
        [new_index],
        {"current": new_index, "new_node": new_index},
    )

    return (
        "Linked List Insertion adds a new node at the specified position (or at the end if no position is given) while maintaining the chain of next pointers.",
        steps,
    )


def _simulate_deletion(values: list[int], delete_value: int) -> tuple[str, list[dict[str, Any]]]:
    """Create deterministic linked list deletion simulation steps."""
    nodes = _build_nodes(values)
    steps: list[dict[str, Any]] = []

    if not nodes:
        _append_step(
            steps,
            "List is empty: nothing to delete.",
            nodes,
            [],
            {"current": None, "prev": None},
        )
        return "Linked List Deletion: the list is empty, so there is nothing to delete.", steps

    current = 0
    prev: int | None = None

    pointers: dict[str, int | None] = {"current": current, "prev": prev}

    _append_step(
        steps,
        f"Start searching for value {delete_value}: current = head (index 0).",
        nodes,
        [current],
        pointers,
    )

    found = False
    while current is not None:
        if nodes[current]["value"] == delete_value:
            found = True
            _append_step(
                steps,
                f"Found value {delete_value} at index {current}.",
                nodes,
                [current],
                {"current": current, "prev": prev},
            )

            if prev is None:
                # Delete head
                new_head = nodes[current]["next"]
                nodes[current]["next"] = None
                _append_step(
                    steps,
                    f"Delete head: move head to index {new_head}.",
                    nodes,
                    [new_head] if new_head is not None else [],
                    {"current": new_head, "prev": None},
                )
            else:
                # Delete middle or tail
                nodes[prev]["next"] = nodes[current]["next"]
                nodes[current]["next"] = None
                _append_step(
                    steps,
                    f"Delete node {current}: node {prev} now points to {nodes[prev]['next']}.",
                    nodes,
                    [prev],
                    {"current": nodes[prev]["next"], "prev": prev},
                )
            break

        prev = current
        current = nodes[current]["next"]
        pointers = {"current": current, "prev": prev}

        if current is not None:
            _append_step(
                steps,
                f"Value not found at {prev}, move to next node (index {current}).",
                nodes,
                [current],
                pointers,
            )

    if not found:
        _append_step(
            steps,
            f"Value {delete_value} not found in the list.",
            nodes,
            [],
            {"current": None, "prev": prev},
        )
        return f"Linked List Deletion: value {delete_value} was not found in the list.", steps

    _append_step(
        steps,
        f"Deletion complete: value {delete_value} removed from the list.",
        nodes,
        [],
        {"current": None, "prev": prev},
    )

    return (
        "Linked List Deletion removes a node with the target value by updating the previous node's next pointer to skip the deleted node.",
        steps,
    )


def build_linked_list_solution(problem: str) -> dict[str, Any]:
    """Build a strict linked list visualization response without relying on LLM output."""
    algorithm = detect_linked_list_algorithm(problem)

    if algorithm == "merge":
        # Try to extract two lists
        lists = re.findall(r"\[([^\]]+)\]", problem)
        if len(lists) >= 2:
            list1 = [int(x.strip()) for x in lists[0].split(",") if re.fullmatch(r"-?\d+", x.strip())]
            list2 = [int(x.strip()) for x in lists[1].split(",") if re.fullmatch(r"-?\d+", x.strip())]
            list1.sort()
            list2.sort()
            explanation, steps = _simulate_merge(list1, list2)
            values = list1 + list2
        else:
            values = extract_linked_list_values(problem)
            explanation, steps = _simulate_traversal(values)
    elif algorithm == "insertion":
        values = extract_linked_list_values(problem)
        # Try to extract insert value
        insert_match = re.search(r"insert\s+(?:value\s+)?(-?\d+)", problem, re.IGNORECASE)
        insert_value = int(insert_match.group(1)) if insert_match else 0
        # Try to extract position
        pos_match = re.search(r"(?:at\s+position|position\s+|index\s+)\s*(-?\d+)", problem, re.IGNORECASE)
        position = int(pos_match.group(1)) if pos_match else None
        explanation, steps = _simulate_insertion(values, insert_value, position)
    elif algorithm == "deletion":
        values = extract_linked_list_values(problem)
        delete_match = re.search(r"(?:delete|remove)\s+(?:value\s+)?(-?\d+)", problem, re.IGNORECASE)
        delete_value = int(delete_match.group(1)) if delete_match else (values[0] if values else 0)
        explanation, steps = _simulate_deletion(values, delete_value)
    elif algorithm == "reverse":
        values = extract_linked_list_values(problem)
        explanation, steps = _simulate_reverse(values)
    elif algorithm == "cycle_detection":
        values = extract_linked_list_values(problem)
        explanation, steps = _simulate_cycle_detection(values)
    else:
        values = extract_linked_list_values(problem)
        explanation, steps = _simulate_traversal(values)

    if not steps:
        steps = [
            {
                "step": 1,
                "description": "Initialize linked list state.",
                "state": {
                    "nodes": _build_nodes(values),
                    "highlight": [],
                    "pointers": {"current": None},
                },
            }
        ]

    # Extract nodes from the last step for visualization
    last_nodes = steps[-1]["state"]["nodes"] if steps else _build_nodes(values)

    return {
        "problem_type": "linked_list",
        "explanation": explanation,
        "steps": steps,
        "visualization": {
            "type": "linked_list",
            "data": {
                "initial_values": values,
                "nodes": last_nodes,
            },
        },
    }
