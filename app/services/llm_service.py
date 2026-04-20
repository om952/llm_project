"""Groq LLM service integration with retries, strict JSON validation, and multi-type support."""

import asyncio
from contextvars import ContextVar
import json
import logging
import re
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from app.core.config import settings
from app.models.schemas import (
    ArrayResponse,
    DPResponse,
    GraphResponse,
    LLMStructuredResponse,
    TreeResponse,
)
from app.services.deterministic_engine import (
    build_deterministic_solution,
    detect_algorithm,
    extract_array,
)

logger = logging.getLogger(__name__)

_LAST_ENGINE_USED: ContextVar[str] = ContextVar("_LAST_ENGINE_USED", default="llm")
_LAST_RAW_LLM_OUTPUT: ContextVar[str] = ContextVar("_LAST_RAW_LLM_OUTPUT", default="")


def reset_last_call_metadata() -> None:
    """Reset per-request LLM metadata for observability."""
    _LAST_ENGINE_USED.set("llm")
    _LAST_RAW_LLM_OUTPUT.set("")


def get_last_call_metadata() -> dict[str, str]:
    """Return metadata captured during the latest service call in this context."""
    return {
        "engine_used": _LAST_ENGINE_USED.get(),
        "raw_llm_output": _LAST_RAW_LLM_OUTPUT.get(),
    }

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory."""
    path = _PROMPTS_DIR / filename
    return path.read_text(encoding="utf-8")


# Lazy-loaded prompt cache
_prompt_cache: dict[str, str] = {}


def _get_prompt(filename: str) -> str:
    """Get a prompt template, loading and caching on first access."""
    if filename not in _prompt_cache:
        _prompt_cache[filename] = _load_prompt(filename)
    return _prompt_cache[filename]


# Keep the original inline prompt as a fallback constant
PROMPT_TEMPLATE = """You are an expert Data Structures and Algorithms tutor and a deterministic algorithm visualization engine.

Your task is to analyze the given problem and return a response in STRICT JSON format ONLY.

---

## OUTPUT FORMAT (STRICT - DO NOT VIOLATE)

Return ONLY valid JSON matching this structure:

{
"problem_type": "array",
"explanation": "string",
"steps": [
{
"step": 1,
"description": "string",
"state": {
"array": [int],
"highlight": [int],
"sorted_boundary": int
}
}
],
"visualization": {
"type": "array",
"data": {
"initial_array": [int]
}
}
}

---

## DEFINITIONS

* problem_type: must be "array"
* explanation: clear, concise explanation of the algorithm
* steps: step-by-step execution of the algorithm
* Each step must represent ONE logical operation
* state must ALWAYS be COMPLETE (do not omit fields)

---

## STRICT RULES

1. Output ONLY JSON. No text, no markdown, no explanations outside JSON.
2. Do NOT skip steps.
3. Step numbering must start from 1 and increment sequentially.
4. Each step must modify the previous state (no duplicates).
5. "array" must always reflect the current state after the step.
6. "highlight" contains indices being compared or swapped.
7. "sorted_boundary" indicates how much of the array is sorted (use -1 if none).
8. Visualization type MUST be exactly "array".
9. Do NOT introduce any extra fields.
10. Ensure the JSON is valid and parseable.

---

## ALGORITHM SCOPE

You are currently restricted to:

* Bubble Sort
* Selection Sort
* Insertion Sort

If the problem is outside this scope:

* Still respond, but adapt it into an array-based step simulation.

---

## EXAMPLE BEHAVIOR

For input:
"Explain Bubble Sort for [5,3,1]"

You should:

* Show comparisons
* Show swaps
* Update array after each operation
* Gradually increase sorted_boundary

---

## FAILURE CONDITIONS (AVOID THESE)

* Invalid JSON
* Missing fields
* Skipped transitions
* Partial states
* Extra commentary outside JSON

---

## INPUT PROBLEM:

{{problem}}"""


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LLMServiceError(Exception):
    """Raised when the LLM provider call fails."""


class InvalidLLMResponseError(LLMServiceError):
    """Raised when the LLM response cannot be parsed or validated."""


# ---------------------------------------------------------------------------
# Prompt building helpers
# ---------------------------------------------------------------------------

def _build_messages(problem: str, prompt_template: str | None = None) -> list[dict[str, str]]:
    """Build Groq chat completion messages using the required prompt template."""
    template = prompt_template or PROMPT_TEMPLATE
    prompt = template.replace("{{problem}}", problem.strip())
    return [{"role": "user", "content": prompt}]


def _build_repair_messages(
    problem: str,
    invalid_output: str,
    failure_reason: str,
    prompt_template: str | None = None,
) -> list[dict[str, str]]:
    """Build follow-up messages that force the model to repair invalid output."""
    template = prompt_template or PROMPT_TEMPLATE
    return [
        {"role": "user", "content": template.replace("{{problem}}", problem.strip())},
        {"role": "assistant", "content": invalid_output},
        {
            "role": "user",
            "content": (
                "Your previous output was invalid.\n"
                f"Validation issue: {failure_reason}\n"
                "Regenerate and return ONLY valid JSON matching the required schema."
            ),
        },
    ]


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def _extract_json_payload(content: str) -> str:
    """Extract JSON content, handling markdown code fences if present."""
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise InvalidLLMResponseError("No JSON object found in LLM response.")

    return cleaned[start : end + 1]


# ---------------------------------------------------------------------------
# HTTP request helper
# ---------------------------------------------------------------------------

async def _request_completion(
    problem: str,
    messages: list[dict[str, str]] | None = None,
    prompt_template: str | None = None,
) -> str:
    """Call Groq chat completions endpoint and return raw text content."""
    if not settings.groq_api_key:
        raise LLMServiceError("Missing GROQ_API_KEY in environment.")

    url = f"{settings.groq_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": settings.groq_model,
        "messages": messages if messages is not None else _build_messages(problem, prompt_template),
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }

    async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMServiceError("Malformed response from Groq API.") from exc

    if not isinstance(content, str) or not content.strip():
        raise LLMServiceError("Groq API returned empty content.")

    return content


# ---------------------------------------------------------------------------
# Parse & validate (generic)
# ---------------------------------------------------------------------------

def _parse_and_validate_content(
    raw_content: str,
    response_model: type = LLMStructuredResponse,
) -> dict[str, Any]:
    """Parse response JSON and validate against the given Pydantic schema."""
    parsed_json = json.loads(_extract_json_payload(raw_content))
    validated = response_model.model_validate(parsed_json)
    return validated.model_dump()


# ---------------------------------------------------------------------------
# Array-specific semantic checks (preserved from original)
# ---------------------------------------------------------------------------

def _is_non_decreasing(values: list[int]) -> bool:
    """Return True when the list is sorted in ascending order."""
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def _is_semantically_valid(problem: str, payload: dict[str, Any]) -> bool:
    """Check semantic correctness for sorting-focused array simulations."""
    algorithm = detect_algorithm(problem)
    initial_array = extract_array(problem)
    final_array = payload["steps"][-1]["state"]["array"]

    if len(initial_array) != len(final_array):
        return False

    if sorted(initial_array) != sorted(final_array):
        return False

    if algorithm in {"bubble", "selection", "insertion"} and not _is_non_decreasing(
        final_array
    ):
        return False

    boundary_values = [step["state"]["sorted_boundary"] for step in payload["steps"]]
    if any(next_v < current_v for current_v, next_v in zip(boundary_values, boundary_values[1:])):
        return False

    if payload["visualization"]["data"]["initial_array"] != initial_array:
        return False

    return True


def _should_use_deterministic_engine(problem: str) -> bool:
    """Use deterministic generation for sorting-scope prompts."""
    lowered = problem.lower()
    keywords = ("bubble", "selection", "insertion", "sort", "sorted")
    return any(keyword in lowered for keyword in keywords)


# ---------------------------------------------------------------------------
# Generic LLM call with retries
# ---------------------------------------------------------------------------

async def _call_llm_generic(
    problem: str,
    prompt_file: str,
    response_model: type,
    semantic_validator: Any | None = None,
) -> dict[str, Any]:
    """Generic retry-based LLM call for any problem type.

    Parameters
    ----------
    problem:
        Raw user problem text.
    prompt_file:
        Filename of the prompt template in ``app/prompts/``.
    response_model:
        Pydantic model class for validation.
    semantic_validator:
        Optional callable ``(problem, payload) -> bool`` for extra semantic checks.
    """
    prompt_template = _get_prompt(prompt_file)
    attempts = max(1, settings.llm_max_retries + 1)
    last_error: Exception | None = None
    raw_content: str = ""

    for attempt in range(1, attempts + 1):
        try:
            raw_content = await _request_completion(problem, prompt_template=prompt_template)
            _LAST_ENGINE_USED.set("llm")
            _LAST_RAW_LLM_OUTPUT.set(raw_content)
            llm_payload = _parse_and_validate_content(raw_content, response_model)

            if semantic_validator is not None:
                if not semantic_validator(problem, llm_payload):
                    raise InvalidLLMResponseError(
                        "LLM output failed semantic validation."
                    )

            logger.info(
                "LLM call succeeded on attempt %s/%s for %s",
                attempt,
                attempts,
                response_model.__name__,
            )
            return llm_payload

        except (json.JSONDecodeError, ValidationError, InvalidLLMResponseError) as exc:
            last_error = exc
            logger.warning(
                "LLM output validation failed on attempt %s/%s (%s): %s",
                attempt,
                attempts,
                response_model.__name__,
                str(exc),
            )

            try:
                repaired_content = await _request_completion(
                    problem,
                    messages=_build_repair_messages(
                        problem, raw_content, str(exc), prompt_template
                    ),
                )
                _LAST_ENGINE_USED.set("llm")
                _LAST_RAW_LLM_OUTPUT.set(repaired_content)
                repaired_payload = _parse_and_validate_content(
                    repaired_content, response_model
                )

                if semantic_validator is not None:
                    if not semantic_validator(problem, repaired_payload):
                        raise InvalidLLMResponseError(
                            "Repaired LLM output failed semantic validation."
                        )

                logger.info(
                    "LLM repair succeeded on attempt %s/%s for %s",
                    attempt,
                    attempts,
                    response_model.__name__,
                )
                return repaired_payload
            except (
                httpx.HTTPError,
                json.JSONDecodeError,
                ValidationError,
                InvalidLLMResponseError,
                LLMServiceError,
            ) as repair_exc:
                last_error = repair_exc
                logger.warning(
                    "LLM repair failed on attempt %s/%s (%s): %s",
                    attempt,
                    attempts,
                    response_model.__name__,
                    str(repair_exc),
                )
                if attempt < attempts:
                    await asyncio.sleep(settings.llm_retry_delay_seconds)

        except (httpx.HTTPError, LLMServiceError) as exc:
            last_error = exc
            logger.warning(
                "LLM call failed on attempt %s/%s (%s): %s",
                attempt,
                attempts,
                response_model.__name__,
                str(exc),
            )
            if attempt < attempts:
                await asyncio.sleep(settings.llm_retry_delay_seconds)

    raise LLMServiceError(
        f"LLM request failed after retries for {response_model.__name__}."
    ) from last_error


# ---------------------------------------------------------------------------
# Type-specific public API
# ---------------------------------------------------------------------------

async def call_llm_array(problem: str) -> dict[str, Any]:
    """Solve an array problem — deterministic engine first, else LLM.

    This preserves the original ``call_llm`` behavior exactly.
    """
    # Deterministic fast-path for sorting problems
    if _should_use_deterministic_engine(problem):
        logger.info("Using deterministic engine for array problem")
        _LAST_ENGINE_USED.set("deterministic")
        _LAST_RAW_LLM_OUTPUT.set("")
        deterministic_payload = build_deterministic_solution(problem)
        validated = LLMStructuredResponse.model_validate(deterministic_payload)
        return validated.model_dump()

    try:
        _LAST_ENGINE_USED.set("llm")
        _LAST_RAW_LLM_OUTPUT.set("")
        return await _call_llm_generic(
            problem,
            prompt_file="array_prompt.txt",
            response_model=ArrayResponse,
            semantic_validator=_is_semantically_valid,
        )
    except LLMServiceError as exc:
        # Fallback to deterministic engine (preserving original behavior)
        if isinstance(exc.__cause__, (json.JSONDecodeError, ValidationError, InvalidLLMResponseError)):
            logger.warning(
                "Falling back to deterministic engine after invalid LLM JSON: %s",
                str(exc),
            )
            _LAST_ENGINE_USED.set("deterministic")
            fallback_response = build_deterministic_solution(problem)
            validated_fallback = LLMStructuredResponse.model_validate(fallback_response)
            return validated_fallback.model_dump()
        raise


async def call_llm_graph(problem: str) -> dict[str, Any]:
    """Solve a graph problem via LLM."""
    logger.info("Calling LLM for graph problem")
    _LAST_ENGINE_USED.set("llm")
    _LAST_RAW_LLM_OUTPUT.set("")
    return await _call_llm_generic(
        problem,
        prompt_file="graph_prompt.txt",
        response_model=GraphResponse,
    )


async def call_llm_tree(problem: str) -> dict[str, Any]:
    """Solve a tree problem via LLM."""
    logger.info("Calling LLM for tree problem")
    _LAST_ENGINE_USED.set("llm")
    _LAST_RAW_LLM_OUTPUT.set("")
    return await _call_llm_generic(
        problem,
        prompt_file="tree_prompt.txt",
        response_model=TreeResponse,
    )


async def call_llm_dp(problem: str) -> dict[str, Any]:
    """Solve a dynamic programming problem via LLM."""
    logger.info("Calling LLM for DP problem")
    _LAST_ENGINE_USED.set("llm")
    _LAST_RAW_LLM_OUTPUT.set("")
    return await _call_llm_generic(
        problem,
        prompt_file="dp_prompt.txt",
        response_model=DPResponse,
    )


# ---------------------------------------------------------------------------
# Backward-compatible entrypoint
# ---------------------------------------------------------------------------

async def call_llm(problem: str) -> dict[str, Any]:
    """Original entrypoint — delegates to ``call_llm_array`` for backward compatibility."""
    return await call_llm_array(problem)
