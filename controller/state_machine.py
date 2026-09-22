from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from .clients import ServiceClient
from .models import (
    ConfidenceRequest,
    ControllerResponse,
    ExecutionTrace,
    IntelligenceAlert,
    MLRequest,
    MLEvidence,
    QueryRequest,
    TaskType,
    TraceEvent,
    ValidationRequest,
)
from .registry import get_registry

import logging

pgil_logger = logging.getLogger("pgil.intelligence_path")


class ControllerEngine:
    def __init__(self, client: ServiceClient, classifier: Any) -> None:
        self.client = client
        self.classifier = classifier

    async def execute(self, request: QueryRequest) -> ControllerResponse:
        events: List[TraceEvent] = []
        start = datetime.now(timezone.utc)
        request_id = request.request_id

        def event(state: str, status: str, message: str = "", block: str | None = None, endpoint: str | None = None, details: Dict[str, Any] | None = None) -> None:
            events.append(
                TraceEvent(
                    step=len(events) + 1,
                    state=state,
                    status=status,
                    message=message,
                    block=block,
                    endpoint=endpoint,
                    details=details or {},
                )
            )

        try:
            event("QUERY_RECEIVED", "success", "Query accepted", details={"query_length": len(request.query)})

            try:
                task = self.classifier.classify(request.query, request.input_scope, request.task_hint)
            except ValueError as exc:
                event("CLASSIFY", "failed", str(exc))
                trace = ExecutionTrace(request_id=request_id, task=request.task_hint or TaskType.VQA, final_status="rejected", events=events)
                return ControllerResponse(
                    request_id=request_id,
                    task=request.task_hint or TaskType.VQA,
                    status="rejected",
                    trace=trace,
                    error=str(exc),
                )

            event("CLASSIFY", "success", f"Classified as {task.value}", details={"task": task.value})

            registry = get_registry()
            spec = registry.get(task)
            if spec is None or not spec.enabled:
                raise RuntimeError(f"No enabled tool registered for task '{task.value}'")
            if spec.required_scope and request.input_scope not in spec.required_scope:
                raise ValueError(f"Task '{task.value}' requires one of {[s.value for s in spec.required_scope]}")

            event("TOOL_SELECTED", "success", f"Selected {spec.name}", block=spec.name, endpoint=spec.endpoint)

            validation_req = ValidationRequest(
                request_id=request_id,
                input_scope=request.input_scope,
                image_ids=request.image_ids,
                task_type=task,
                metadata=request.metadata,
            )
            event("VALIDATE_INPUT", "started", "Calling B2 validation")
            validation = await self.client.validate(validation_req)
            if not validation.valid:
                reasons = "; ".join(validation.reasons) or "Input validation failed"
                event("VALIDATE_INPUT", "failed", reasons, details={"reasons": validation.reasons})
                trace = ExecutionTrace(request_id=request_id, task=task, final_status="rejected", events=events)
                return ControllerResponse(
                    request_id=request_id,
                    task=task,
                    status="rejected",
                    trace=trace,
                    error=reasons,
                )
            event("VALIDATE_INPUT", "success", "Input accepted", details=validation.normalized_input)

            ml_req = MLRequest(
                request_id=request_id,
                task_type=task,
                query=request.query,
                input_scope=request.input_scope,
                image_ids=request.image_ids,
                metadata={**request.metadata, "normalized_input": validation.normalized_input},
            )
            event("ML_EXECUTE", "started", f"Calling {spec.name}", block=spec.name, endpoint=spec.endpoint)
            ml_result = await self.client.ml(spec.endpoint, ml_req)
            if ml_result.status != "success":
                event("ML_EXECUTE", "failed", ml_result.error or "ML block failed", block=spec.name, endpoint=spec.endpoint)
                raise RuntimeError(ml_result.error or "ML block failed")
            event(
                "ML_EXECUTE",
                "success",
                f"{spec.name} completed",
                block=spec.name,
                endpoint=spec.endpoint,
                details={"model_version": ml_result.model_version},
            )

            event("CONFIDENCE", "started", "Aggregating confidence and disagreement")
            confidence_req = ConfidenceRequest(
                request_id=request_id,
                task_type=task,
                model_confidence=ml_result.confidence,
                cross_modal_agreement=(
                    ml_result.facts.get("cross_modal_agreement")
                    if isinstance(ml_result.facts, dict)
                    else None
                ),
                input_quality=(
                    validation.normalized_input.get("input_quality")
                    if isinstance(validation.normalized_input, dict)
                    else None
                ),
                raw_block_output=ml_result.model_dump(mode="json"),
            )
            confidence = await self.client.confidence(confidence_req)
            event(
                "CONFIDENCE",
                "success",
                f"Final confidence {confidence.score:.3f}",
                details={"components": confidence.components, "disagreement": confidence.disagreement},
            )

            provenance = {
                "controller": "satquery-b1",
                "request_id": str(request_id),
                "task": task.value,
                "block": spec.name,
                "endpoint": spec.endpoint,
                "model_version": ml_result.model_version,
                "input_image_ids": request.image_ids,
                "input_scope": request.input_scope.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            event("TRACE_FINALIZED", "success", "Execution trace completed")

            # ── PGIL Intelligence Path (runs AFTER query answer) ─────
            intelligence_alerts: List[Dict[str, Any]] = []
            try:
                intelligence_alerts = await self._run_intelligence_path(
                    request=request,
                    validation_normalized=validation.normalized_input,
                    event_fn=event,
                )
            except Exception as pgil_exc:
                pgil_logger.warning("PGIL intelligence path failed (non-fatal): %s", pgil_exc)
                event("INTELLIGENCE_PATH", "failed", f"PGIL error (non-fatal): {pgil_exc}")

            trace = ExecutionTrace(request_id=request_id, task=task, final_status="success", events=events)
            return ControllerResponse(
                request_id=request_id,
                task=task,
                status="success",
                answer_facts=ml_result.facts,
                confidence=confidence,
                evidence=ml_result.evidence,
                modality_attribution=ml_result.modality_attribution,
                disagreement=bool(confidence.disagreement or ml_result.disagreement),
                provenance=provenance,
                trace=trace,
                intelligence_alerts=[IntelligenceAlert(**a) for a in intelligence_alerts],
            )

        except Exception as exc:
            event("EXECUTION", "failed", str(exc))
            trace_task = request.task_hint or TaskType.VQA
            try:
                trace_task = self.classifier.classify(request.query, request.input_scope, request.task_hint)
            except Exception:
                pass
            trace = ExecutionTrace(request_id=request_id, task=trace_task, final_status="failed", events=events)
            return ControllerResponse(
                request_id=request_id,
                task=trace_task,
                status="failed",
                trace=trace,
                error=str(exc),
            )

    async def _run_intelligence_path(
        self,
        request: QueryRequest,
        validation_normalized: Dict[str, Any],
        event_fn: Any,
    ) -> List[Dict[str, Any]]:
        """
        PGIL Intelligence Path — now handled automatically by the ML server
        on every image upload (via _pgil_ingest_background in server.py).

        The controller no longer runs its own PGIL path because:
        1. The frontend calls the ML service directly (bypasses controller)
        2. The controller has no GPU access for ChangeFormer / object extraction
        3. The ML server's auto-ingest runs real analysis with full model access

        This method remains as a no-op stub to avoid breaking the call site.
        """
        event_fn("INTELLIGENCE_PATH", "success",
                 "PGIL: Intelligence runs automatically on ML server via upload ingestion")
        return []
