"""API routes for solving DSA problems with LLM reasoning output."""

import logging

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.core.debug_store import add_debug_entry
from app.core.normalizer import normalize_response
from app.core.router import detect_problem_type, route_problem_with_meta
from app.models.schemas import ErrorResponse, SolveRequest, SolveResponse
from app.services.llm_service import (
    InvalidLLMResponseError,
    LLMServiceError,
    get_last_call_metadata,
    reset_last_call_metadata,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["solve"])


@router.post(
    "/solve",
    response_model=SolveResponse,
    responses={
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
)
async def solve_problem(payload: SolveRequest) -> SolveResponse | JSONResponse:
    """Solve and explain a DSA problem using the configured LLM service.

    Supports: array, graph, tree, dp.
    """
    problem = payload.problem
    reset_last_call_metadata()

    # Log detected type
    detected_type = detect_problem_type(problem)
    logger.info("Detected problem type: '%s' for problem: '%s'", detected_type, problem[:80])

    try:
        routed = await route_problem_with_meta(problem)
        normalized_result = normalize_response(routed["parsed_output"])

        add_debug_entry(
            problem=problem,
            detected_type=routed.get("detected_type", detected_type),
            engine_used=routed.get("engine_used", "llm"),
            raw_llm_output=routed.get("raw_llm_output", ""),
            parsed_output=normalized_result,
            valid=True,
            error=None,
        )
    except InvalidLLMResponseError as exc:
        logger.warning("Validation failure: %s", str(exc))
        meta = get_last_call_metadata()
        add_debug_entry(
            problem=problem,
            detected_type=detected_type,
            engine_used=meta.get("engine_used", "llm"),
            raw_llm_output=meta.get("raw_llm_output", ""),
            parsed_output={},
            valid=False,
            error=str(exc),
        )
        error = ErrorResponse(error="invalid_llm_json", details=str(exc))
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=error.model_dump(),
        )
    except LLMServiceError as exc:
        logger.error("LLM service failure: %s", str(exc))
        meta = get_last_call_metadata()
        add_debug_entry(
            problem=problem,
            detected_type=detected_type,
            engine_used=meta.get("engine_used", "llm"),
            raw_llm_output=meta.get("raw_llm_output", ""),
            parsed_output={},
            valid=False,
            error=str(exc),
        )
        error = ErrorResponse(error="llm_service_failure", details=str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error.model_dump(),
        )

    logger.info(
        "Validation success for type='%s' using engine='%s'",
        normalized_result.get("problem_type"),
        routed.get("engine_used", "llm"),
    )

    return SolveResponse(
        problem_type=normalized_result["problem_type"],
        explanation=normalized_result["explanation"],
        steps=normalized_result["steps"],
        visualization=normalized_result["visualization"],
    )
