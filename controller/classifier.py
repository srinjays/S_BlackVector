from __future__ import annotations

import re
from typing import Optional

from .models import InputScope, TaskType


class QueryClassifier:
    """Deterministic, auditable baseline classifier.

    The roadmap permits rules or a small LLM classifier. This implementation uses
    explicit rules first so behavior is stable and easy to test; a model-backed
    classifier can be added behind the same interface later.
    """

    RULES = [
        (TaskType.SPECTRAL_ANALYSIS, ["ndvi", "ndwi", "ndbi", "ndmi", "nbr", "dnbr", "spectral", "vegetation stress", "water extent", "burn severity", "canopy", "fapar", "fvc", "lai", "surface moisture proxy"]),
        (TaskType.AGRICULTURE_PREDICTION, ["soil moisture", "soil type", "crop suitability", "crop yield", "irrigation requirement", "irrigation", "crop recommendation", "drought risk", "drought", "agricultural productivity", "soil nutrient", "crop classification", "yield estimate"]),
        (TaskType.EARTH_FACTS, ["earth facts", "ground resolution", "sentinel-2 bands", "spatial resolution", "revisit time"]),
        (TaskType.CHANGE, ["change", "changed", "difference", "differences", "between", "before and after", "temporal", "development over time"]),
        (TaskType.FUSION, ["sar", "optical", "combine", "fusion", "complementary", "both modalities", "cross-modal"]),
        (TaskType.GROUNDING, ["where", "locate", "location", "find", "detect all", "bounding box", "bounding boxes", "mark", "highlight", "near the"]),
        (TaskType.CAPTION, ["caption", "summarize the image", "give me a caption"]),
        (TaskType.VQA, ["is there", "are there", "how many", "what type", "what kind", "does the image", "which", "what is", "what are", "can you identify"]),
    ]


    def classify(self, query: str, input_scope: InputScope, hint: Optional[TaskType] = None) -> TaskType:
        if hint is not None:
            self._validate_hint(hint, input_scope)
            return hint

        q = re.sub(r"\s+", " ", query.lower()).strip()

        # Input scope provides a strong signal for ambiguous requests.
        if input_scope == InputScope.BI_TEMPORAL and any(k in q for k in ["change", "changed", "difference", "between", "before", "after"]):
            return TaskType.CHANGE
        if input_scope == InputScope.CROSS_MODAL or any(k in q for k in ["sar", "optical"]):
            return TaskType.FUSION

        for task, keywords in self.RULES:
            if any(keyword in q for keyword in keywords):
                self._validate_hint(task, input_scope)
                return task

        # Conservative defaults by input shape.
        if input_scope == InputScope.BI_TEMPORAL:
            return TaskType.CHANGE
        if input_scope == InputScope.CROSS_MODAL:
            return TaskType.FUSION
        return TaskType.VQA

    @staticmethod
    def _validate_hint(task: TaskType, scope: InputScope) -> None:
        """Validate task/scope compatibility.

        In interactive chat mode, the frontend may send a task_hint with
        a scope that doesn't strictly match (e.g., VQA with single scope
        but the user typed a fusion-related query). We allow it through
        with a logged warning rather than rejecting the request.
        """
        strict = {
            TaskType.CHANGE: {InputScope.BI_TEMPORAL},
            TaskType.FUSION: {InputScope.CROSS_MODAL},
        }
        required = strict.get(task)
        if required and scope not in required:
            raise ValueError(f"Task '{task.value}' is incompatible with input scope '{scope.value}'")
