"""In-memory debug store for recent solve requests."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from threading import Lock
from typing import Any, TypedDict


class DebugEntry(TypedDict):
    """Debug record for one solve execution."""

    problem: str
    detected_type: str
    engine_used: str
    raw_llm_output: str
    parsed_output: dict[str, Any]
    valid: bool
    error: str | None


_MAX_ENTRIES = 10
_store: deque[DebugEntry] = deque(maxlen=_MAX_ENTRIES)
_store_lock = Lock()


def add_debug_entry(
    problem: str,
    detected_type: str,
    engine_used: str,
    raw_llm_output: str,
    parsed_output: dict[str, Any],
    valid: bool,
    error: str | None,
) -> DebugEntry:
    """Add a debug entry to the in-memory store and return it."""
    entry: DebugEntry = {
        "problem": problem,
        "detected_type": detected_type,
        "engine_used": engine_used,
        "raw_llm_output": raw_llm_output,
        "parsed_output": deepcopy(parsed_output),
        "valid": valid,
        "error": error,
    }
    with _store_lock:
        _store.append(entry)
    return deepcopy(entry)


def get_last_debug_entry() -> DebugEntry | None:
    """Return the most recent debug entry if available."""
    with _store_lock:
        if not _store:
            return None
        return deepcopy(_store[-1])


def get_debug_history() -> list[DebugEntry]:
    """Return all stored debug entries (up to the last 10)."""
    with _store_lock:
        return [deepcopy(entry) for entry in _store]


def get_debug_errors() -> list[DebugEntry]:
    """Return only failed debug entries."""
    with _store_lock:
        return [deepcopy(entry) for entry in _store if not entry["valid"]]
