"""Pydantic schemas for API requests, responses, and strict LLM validation."""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class StrictBaseModel(BaseModel):
    """Base model that rejects unknown fields for strict schema checks."""

    model_config = ConfigDict(extra="forbid")


class SolveRequest(StrictBaseModel):
    """Incoming solve request payload."""

    problem: str = Field(..., min_length=1)


class BaseResponse(StrictBaseModel):
    """Common response fields shared across all problem types."""

    problem_type: Literal["array", "graph", "tree", "dp", "linked_list", "hashmap"]
    explanation: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# ARRAY schemas (preserved from original)
# ---------------------------------------------------------------------------

class ArrayStepState(StrictBaseModel):
    """Strict step state for array algorithm visualization."""

    array: list[int] = Field(..., min_length=1)
    highlight: list[int] = Field(default_factory=list)
    sorted_boundary: int = Field(..., ge=-1)

    @model_validator(mode="after")
    def validate_state(self) -> "ArrayStepState":
        """Ensure indices and sorted boundary are valid for the current array."""
        max_boundary = len(self.array)
        if self.sorted_boundary > max_boundary:
            raise ValueError("sorted_boundary must be -1 or a valid sorted amount/index.")

        max_index = len(self.array) - 1
        for index in self.highlight:
            if index < 0 or index > max_index:
                raise ValueError("highlight indices must be valid array indices.")

        return self


# Keep the original name as an alias for backward compatibility
StepState = ArrayStepState


class ArrayStep(StrictBaseModel):
    """One reasoning step and algorithm state snapshot for arrays."""

    step: int = Field(..., ge=1)
    description: str = Field(..., min_length=1)
    state: ArrayStepState


# Backward-compatible alias
Step = ArrayStep


class ArrayVisualizationData(StrictBaseModel):
    """Visualization payload for the array renderer."""

    initial_array: list[int] = Field(..., min_length=1)


# Backward-compatible alias
VisualizationData = ArrayVisualizationData


class ArrayVisualization(StrictBaseModel):
    """Visualization metadata and data payload for arrays."""

    type: Literal["array"]
    data: ArrayVisualizationData


# Backward-compatible alias
Visualization = ArrayVisualization


class ArrayResponse(BaseResponse):
    """Strictly validated LLM output schema for array problems."""

    problem_type: Literal["array"]
    steps: list[ArrayStep] = Field(..., min_length=1)
    visualization: ArrayVisualization

    @model_validator(mode="after")
    def validate_steps(self) -> "ArrayResponse":
        """Enforce sequential step numbers and non-duplicate state transitions."""
        for expected_step, step_obj in enumerate(self.steps, start=1):
            if step_obj.step != expected_step:
                raise ValueError("Step numbering must start at 1 and be sequential.")

        for idx in range(1, len(self.steps)):
            previous = self.steps[idx - 1].state.model_dump()
            current = self.steps[idx].state.model_dump()
            if previous == current:
                raise ValueError("Each step must modify the previous state.")

        if self.problem_type != self.visualization.type:
            raise ValueError("problem_type must match visualization.type")

        return self


# Backward-compatible alias for existing code
LLMStructuredResponse = ArrayResponse


# ---------------------------------------------------------------------------
# GRAPH schemas
# ---------------------------------------------------------------------------

class GraphStepState(StrictBaseModel):
    """Step state for graph algorithm visualization."""

    nodes: list[str] = Field(..., min_length=1)
    edges: list[list[str]] = Field(default_factory=list)
    visited: list[str] = Field(default_factory=list)
    active: str = Field(..., min_length=1)
    queue: list[str] = Field(default_factory=list)


class GraphStep(StrictBaseModel):
    """One reasoning step and state snapshot for graph algorithms."""

    step: int = Field(..., ge=1)
    description: str = Field(..., min_length=1)
    state: GraphStepState


class GraphVisualizationData(StrictBaseModel):
    """Visualization payload for the graph renderer."""

    nodes: list[str] = Field(..., min_length=1)
    edges: list[list[str]] = Field(default_factory=list)


class GraphVisualization(StrictBaseModel):
    """Visualization metadata and data payload for graphs."""

    type: Literal["graph"]
    data: GraphVisualizationData


class GraphResponse(BaseResponse):
    """Strictly validated LLM output schema for graph problems."""

    problem_type: Literal["graph"]
    steps: list[GraphStep] = Field(..., min_length=1)
    visualization: GraphVisualization

    @model_validator(mode="after")
    def validate_steps(self) -> "GraphResponse":
        """Enforce sequential steps, monotonic visited, valid active node."""
        for expected_step, step_obj in enumerate(self.steps, start=1):
            if step_obj.step != expected_step:
                raise ValueError("Step numbering must start at 1 and be sequential.")

        for idx in range(1, len(self.steps)):
            previous = self.steps[idx - 1].state.model_dump()
            current = self.steps[idx].state.model_dump()
            if previous == current:
                raise ValueError("Each step must modify the previous state.")

        # visited must grow monotonically
        for idx in range(1, len(self.steps)):
            prev_visited = set(self.steps[idx - 1].state.visited)
            curr_visited = set(self.steps[idx].state.visited)
            if not prev_visited.issubset(curr_visited):
                raise ValueError("visited must grow monotonically.")

        # active node must be in nodes list
        all_nodes = set(self.steps[0].state.nodes)
        for step_obj in self.steps:
            if step_obj.state.active not in all_nodes:
                raise ValueError(f"active node '{step_obj.state.active}' not in nodes list.")

        if self.problem_type != self.visualization.type:
            raise ValueError("problem_type must match visualization.type")

        return self


# ---------------------------------------------------------------------------
# TREE schemas
# ---------------------------------------------------------------------------

class TreeStepState(StrictBaseModel):
    """Step state for tree algorithm visualization."""

    nodes: list[int] = Field(..., min_length=1)
    edges: list[list[int]] = Field(default_factory=list)
    current: int


class TreeStep(StrictBaseModel):
    """One reasoning step and state snapshot for tree algorithms."""

    step: int = Field(..., ge=1)
    description: str = Field(..., min_length=1)
    state: TreeStepState


class TreeVisualizationData(StrictBaseModel):
    """Visualization payload for the tree renderer."""

    nodes: list[int] = Field(..., min_length=1)
    edges: list[list[int]] = Field(default_factory=list)


class TreeVisualization(StrictBaseModel):
    """Visualization metadata and data payload for trees."""

    type: Literal["tree"]
    data: TreeVisualizationData


class TreeResponse(BaseResponse):
    """Strictly validated LLM output schema for tree problems."""

    problem_type: Literal["tree"]
    steps: list[TreeStep] = Field(..., min_length=1)
    visualization: TreeVisualization

    @model_validator(mode="after")
    def validate_steps(self) -> "TreeResponse":
        """Enforce sequential steps, valid current node."""
        for expected_step, step_obj in enumerate(self.steps, start=1):
            if step_obj.step != expected_step:
                raise ValueError("Step numbering must start at 1 and be sequential.")

        for idx in range(1, len(self.steps)):
            previous = self.steps[idx - 1].state.model_dump()
            current = self.steps[idx].state.model_dump()
            if previous == current:
                raise ValueError("Each step must modify the previous state.")

        # current node must be in nodes list
        for step_obj in self.steps:
            if step_obj.state.current not in step_obj.state.nodes:
                raise ValueError(
                    f"current node {step_obj.state.current} not in nodes list."
                )

        if self.problem_type != self.visualization.type:
            raise ValueError("problem_type must match visualization.type")

        return self


# ---------------------------------------------------------------------------
# DP schemas
# ---------------------------------------------------------------------------

class DPStepState(StrictBaseModel):
    """Step state for dynamic programming visualization."""

    table: list[list[int]] = Field(..., min_length=1)
    current_cell: list[int] = Field(..., min_length=2, max_length=2)

    @model_validator(mode="after")
    def validate_current_cell(self) -> "DPStepState":
        """Ensure current_cell indices are within table bounds."""
        rows = len(self.table)
        cols = len(self.table[0]) if rows > 0 else 0
        i, j = self.current_cell
        if i < 0 or i >= rows:
            raise ValueError(f"current_cell row {i} is out of bounds (table has {rows} rows).")
        if j < 0 or j >= cols:
            raise ValueError(f"current_cell col {j} is out of bounds (table has {cols} cols).")
        return self


class DPStep(StrictBaseModel):
    """One reasoning step and state snapshot for DP algorithms."""

    step: int = Field(..., ge=1)
    description: str = Field(..., min_length=1)
    state: DPStepState


class DPVisualizationData(StrictBaseModel):
    """Visualization payload for the DP renderer."""

    table: list[list[int]] = Field(..., min_length=1)
    dimensions: list[int] = Field(..., min_length=2, max_length=2)


class DPVisualization(StrictBaseModel):
    """Visualization metadata and data payload for DP."""

    type: Literal["dp"]
    data: DPVisualizationData


class DPResponse(BaseResponse):
    """Strictly validated LLM output schema for DP problems."""

    problem_type: Literal["dp"]
    steps: list[DPStep] = Field(..., min_length=1)
    visualization: DPVisualization

    @model_validator(mode="after")
    def validate_steps(self) -> "DPResponse":
        """Enforce sequential steps and non-duplicate transitions."""
        for expected_step, step_obj in enumerate(self.steps, start=1):
            if step_obj.step != expected_step:
                raise ValueError("Step numbering must start at 1 and be sequential.")

        for idx in range(1, len(self.steps)):
            previous = self.steps[idx - 1].state.model_dump()
            current = self.steps[idx].state.model_dump()
            if previous == current:
                raise ValueError("Each step must modify the previous state.")

        if self.problem_type != self.visualization.type:
            raise ValueError("problem_type must match visualization.type")

        return self


# ---------------------------------------------------------------------------
# LINKED LIST schemas
# ---------------------------------------------------------------------------

class LinkedListStepState(StrictBaseModel):
    """Step state for linked list algorithm visualization."""

    nodes: list[dict[str, Any]] = Field(..., min_length=1)
    highlight: list[int] = Field(default_factory=list)
    pointers: dict[str, Any] = Field(default_factory=dict)


class LinkedListStep(StrictBaseModel):
    """One reasoning step and state snapshot for linked list algorithms."""

    step: int = Field(..., ge=1)
    description: str = Field(..., min_length=1)
    state: LinkedListStepState


class LinkedListVisualizationData(StrictBaseModel):
    """Visualization payload for the linked list renderer."""

    initial_values: list[int] = Field(..., min_length=1)
    nodes: list[dict[str, Any]] = Field(..., min_length=1)


class LinkedListVisualization(StrictBaseModel):
    """Visualization metadata and data payload for linked lists."""

    type: Literal["linked_list"]
    data: LinkedListVisualizationData


class LinkedListResponse(BaseResponse):
    """Strictly validated output schema for linked list problems."""

    problem_type: Literal["linked_list"]
    steps: list[LinkedListStep] = Field(..., min_length=1)
    visualization: LinkedListVisualization

    @model_validator(mode="after")
    def validate_steps(self) -> "LinkedListResponse":
        """Enforce sequential steps and non-duplicate transitions."""
        for expected_step, step_obj in enumerate(self.steps, start=1):
            if step_obj.step != expected_step:
                raise ValueError("Step numbering must start at 1 and be sequential.")

        for idx in range(1, len(self.steps)):
            previous = self.steps[idx - 1].state.model_dump()
            current = self.steps[idx].state.model_dump()
            if previous == current:
                raise ValueError("Each step must modify the previous state.")

        if self.problem_type != self.visualization.type:
            raise ValueError("problem_type must match visualization.type")

        return self


# ---------------------------------------------------------------------------
# HASHMAP schemas
# ---------------------------------------------------------------------------

class HashmapStepState(StrictBaseModel):
    """Step state for hashmap algorithm visualization."""

    buckets: list[list[dict[str, Any]]] = Field(..., min_length=1)
    highlight_bucket: int | None = Field(default=None)
    highlight_key: str | None = Field(default=None)
    operation: str = Field(default="init")


class HashmapStep(StrictBaseModel):
    """One reasoning step and state snapshot for hashmap algorithms."""

    step: int = Field(..., ge=1)
    description: str = Field(..., min_length=1)
    state: HashmapStepState


class HashmapVisualizationData(StrictBaseModel):
    """Visualization payload for the hashmap renderer."""

    buckets: list[list[dict[str, Any]]] = Field(..., min_length=1)
    capacity: int = Field(..., ge=1)


class HashmapVisualization(StrictBaseModel):
    """Visualization metadata and data payload for hashmaps."""

    type: Literal["hashmap"]
    data: HashmapVisualizationData


class HashmapResponse(BaseResponse):
    """Strictly validated output schema for hashmap problems."""

    problem_type: Literal["hashmap"]
    steps: list[HashmapStep] = Field(..., min_length=1)
    visualization: HashmapVisualization

    @model_validator(mode="after")
    def validate_steps(self) -> "HashmapResponse":
        """Enforce sequential steps and non-duplicate transitions."""
        for expected_step, step_obj in enumerate(self.steps, start=1):
            if step_obj.step != expected_step:
                raise ValueError("Step numbering must start at 1 and be sequential.")

        for idx in range(1, len(self.steps)):
            previous = self.steps[idx - 1].state.model_dump()
            current = self.steps[idx].state.model_dump()
            if previous == current:
                raise ValueError("Each step must modify the previous state.")

        if self.problem_type != self.visualization.type:
            raise ValueError("problem_type must match visualization.type")

        return self


# ---------------------------------------------------------------------------
# Union response type
# ---------------------------------------------------------------------------

ResponseModel = Union[ArrayResponse, GraphResponse, TreeResponse, DPResponse, LinkedListResponse, HashmapResponse]


# ---------------------------------------------------------------------------
# API-level response (accepts any problem type)
# ---------------------------------------------------------------------------

class SolveResponse(BaseResponse):
    """API response returned by POST /solve.

    Accepts any supported problem type.
    """

    steps: list[ArrayStep | GraphStep | TreeStep | DPStep | LinkedListStep | HashmapStep] = Field(..., min_length=1)
    visualization: ArrayVisualization | GraphVisualization | TreeVisualization | DPVisualization | LinkedListVisualization | HashmapVisualization

    @model_validator(mode="after")
    def validate_type_consistency(self) -> "SolveResponse":
        """Ensure problem_type and visualization.type match."""
        if self.problem_type != self.visualization.type:
            raise ValueError("problem_type must match visualization.type")
        return self


class ErrorResponse(StrictBaseModel):
    """Structured error response payload."""

    error: str
    details: str | None = None
