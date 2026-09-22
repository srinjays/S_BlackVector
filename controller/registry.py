from __future__ import annotations

from typing import Dict

from .config import settings
from .models import InputScope, TaskType, ToolSpec


TOOL_REGISTRY: Dict[TaskType, ToolSpec] = {
    TaskType.VQA: ToolSpec(
        name="vqa",
        task_type=TaskType.VQA,
        endpoint="/vqa",
        required_scope=[InputScope.SINGLE],
        description="Remote-sensing visual question answering",
    ),
    TaskType.CAPTION: ToolSpec(
        name="caption",
        task_type=TaskType.CAPTION,
        endpoint="/caption",
        required_scope=[InputScope.SINGLE],
        description="Remote-sensing image captioning from structured perception facts",
    ),
    TaskType.GROUNDING: ToolSpec(
        name="grounding",
        task_type=TaskType.GROUNDING,
        endpoint="/grounding",
        required_scope=[InputScope.SINGLE],
        description="Spatial grounding / bounding boxes",
    ),
    TaskType.CHANGE: ToolSpec(
        name="change",
        task_type=TaskType.CHANGE,
        endpoint="/change",
        required_scope=[InputScope.BI_TEMPORAL],
        description="Bi-temporal change detection",
    ),
    TaskType.FUSION: ToolSpec(
        name="fusion",
        task_type=TaskType.FUSION,
        endpoint="/fusion",
        required_scope=[InputScope.CROSS_MODAL],
        description="Optical-SAR complementary information fusion",
    ),
    TaskType.EARTH_FACTS: ToolSpec(
        name="earth_facts",
        task_type=TaskType.EARTH_FACTS,
        endpoint="/earth_facts",
        required_scope=[InputScope.SINGLE],
        description="Verifiable Earth Observation domain facts",
    ),
    TaskType.SPECTRAL_ANALYSIS: ToolSpec(
        name="spectral",
        task_type=TaskType.SPECTRAL_ANALYSIS,
        endpoint="/spectral",
        required_scope=[InputScope.SINGLE, InputScope.BI_TEMPORAL, InputScope.CROSS_MODAL],
        description="Multi-spectral indices and remote sensing indicator engine",
    ),
    TaskType.AGRICULTURE_PREDICTION: ToolSpec(
        name="agriculture",
        task_type=TaskType.AGRICULTURE_PREDICTION,
        endpoint="/agriculture",
        required_scope=[InputScope.SINGLE, InputScope.BI_TEMPORAL, InputScope.CROSS_MODAL],
        description="Context-aware probabilistic agricultural prediction engine",
    ),
}



def get_registry() -> Dict[TaskType, ToolSpec]:
    return TOOL_REGISTRY


def register_tool(spec: ToolSpec) -> None:
    TOOL_REGISTRY[spec.task_type] = spec
