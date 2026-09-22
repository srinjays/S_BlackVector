from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskType(str, Enum):
    VQA = "vqa"
    CAPTION = "caption"
    GROUNDING = "grounding"
    CHANGE = "change"
    FUSION = "fusion"
    EARTH_FACTS = "earth_facts"
    SPECTRAL_ANALYSIS = "spectral"
    AGRICULTURE_PREDICTION = "agriculture"



class InputScope(str, Enum):
    SINGLE = "single"
    CROSS_MODAL = "cross_modal"
    BI_TEMPORAL = "bi_temporal"


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    input_scope: InputScope
    image_ids: List[str] = Field(min_length=1, max_length=2)
    task_hint: Optional[TaskType] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    request_id: UUID = Field(default_factory=uuid4)

    @field_validator("image_ids")
    @classmethod
    def validate_image_ids(cls, value: List[str]) -> List[str]:
        if any(not item.strip() for item in value):
            raise ValueError("image_ids cannot contain blank values")
        return value


class ValidationRequest(BaseModel):
    request_id: UUID
    input_scope: InputScope
    image_ids: List[str]
    task_type: TaskType
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ValidationResponse(BaseModel):
    valid: bool
    reasons: List[str] = Field(default_factory=list)
    normalized_input: Dict[str, Any] = Field(default_factory=dict)


class ToolSpec(BaseModel):
    name: str
    task_type: TaskType
    endpoint: str
    method: Literal["POST"] = "POST"
    timeout_seconds: float = 60.0
    enabled: bool = True
    required_scope: List[InputScope] = Field(default_factory=list)
    description: str = ""


class MLRequest(BaseModel):
    request_id: UUID
    task_type: TaskType
    query: str
    input_scope: InputScope
    image_ids: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MLEvidence(BaseModel):
    type: str
    data: Dict[str, Any] = Field(default_factory=dict)


class MLResponse(BaseModel):
    task_type: TaskType
    status: Literal["success", "failure"]
    facts: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[MLEvidence] = Field(default_factory=list)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    modality_attribution: Dict[str, Any] = Field(default_factory=dict)
    disagreement: Optional[bool] = None
    error: Optional[str] = None
    model_version: Optional[str] = None


class ConfidenceRequest(BaseModel):
    request_id: UUID
    task_type: TaskType
    model_confidence: Optional[float] = None
    cross_modal_agreement: Optional[float] = None
    input_quality: Optional[float] = None
    raw_block_output: Dict[str, Any] = Field(default_factory=dict)


class ConfidenceResponse(BaseModel):
    score: float = Field(ge=0, le=1)
    components: Dict[str, float] = Field(default_factory=dict)
    disagreement: bool = False
    explanation: str = ""


class TraceEvent(BaseModel):
    step: int
    state: str
    status: Literal["started", "success", "failed", "skipped"]
    message: str = ""
    block: Optional[str] = None
    endpoint: Optional[str] = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class ExecutionTrace(BaseModel):
    request_id: UUID
    task: TaskType
    final_status: Literal["success", "failed", "rejected"]
    events: List[TraceEvent] = Field(default_factory=list)


class IntelligenceAlert(BaseModel):
    """PGIL alert surfaced alongside query responses."""
    alert_id: str
    area_id: str
    alert_type: str = ""
    title: str = ""
    description: str = ""
    severity: str = "medium"
    confidence: float = 0.0
    observation_interval: Optional[str] = None
    affected_count: int = 0
    change_event_id: Optional[str] = None


class ControllerResponse(BaseModel):
    request_id: UUID
    task: TaskType
    status: Literal["success", "failed", "rejected"]
    answer_facts: Dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[ConfidenceResponse] = None
    evidence: List[MLEvidence] = Field(default_factory=list)
    modality_attribution: Dict[str, Any] = Field(default_factory=dict)
    disagreement: bool = False
    provenance: Dict[str, Any] = Field(default_factory=dict)
    trace: ExecutionTrace
    error: Optional[str] = None
    intelligence_alerts: List[IntelligenceAlert] = Field(default_factory=list)
