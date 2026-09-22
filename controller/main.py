from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .classifier import QueryClassifier
from .clients import ServiceClient
from .models import ControllerResponse, QueryRequest, TaskType, ToolSpec
from .registry import get_registry, register_tool
from .state_machine import ControllerEngine

app = FastAPI(title="SatQuery AI — B1 Controller", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
engine = ControllerEngine(ServiceClient(), QueryClassifier())


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "satquery-b1-controller"}


@app.get("/tools")
async def tools() -> dict:
    return {"tools": [spec.model_dump(mode="json") for spec in get_registry().values()]}


@app.post("/tools")
async def add_tool(spec: ToolSpec) -> dict:
    register_tool(spec)
    return {"registered": spec.model_dump(mode="json")}


@app.post("/query", response_model=ControllerResponse)
async def query(request: QueryRequest) -> ControllerResponse:
    if len(request.query) > 2000:
        raise HTTPException(status_code=422, detail="query exceeds 2000 characters")
    return await engine.execute(request)


@app.get("/tasks")
async def tasks() -> dict:
    return {"tasks": [task.value for task in TaskType]}


# ── PGIL Intelligence Endpoints ─────────────────────────────

@app.get("/alerts")
async def get_alerts() -> dict:
    """Return all pending PGIL intelligence alerts."""
    try:
        from pgil.store import SpatialMemoryStore
        store = SpatialMemoryStore()
        alerts = store.get_pending_alerts()
        return {"alerts": [a.model_dump() for a in alerts]}
    except Exception as exc:
        return {"alerts": [], "error": str(exc)}


@app.get("/alerts/all")
async def get_all_alerts() -> dict:
    """Return all PGIL intelligence alerts (including dismissed).

    Enriches each alert with image paths from associated observations
    so the frontend can display before/after thumbnails.
    """
    ML_BASE_URL = "http://localhost:8200"
    try:
        from pgil.store import SpatialMemoryStore
        store = SpatialMemoryStore()
        alerts = store.get_all_alerts()

        enriched = []
        for a in alerts:
            alert_dict = a.model_dump()

            # Join alert → change_event → observations to get image paths
            try:
                change_event = store.get_change_event(a.change_event_id)
                if change_event:
                    old_obs = store.get_observation(change_event.old_observation_id)
                    new_obs = store.get_observation(change_event.new_observation_id)
                    if old_obs:
                        alert_dict["old_image_path"] = old_obs.image_path
                        alert_dict["old_image_url"] = (
                            f"{ML_BASE_URL}/file?path={old_obs.image_path}"
                        )
                    if new_obs:
                        alert_dict["new_image_path"] = new_obs.image_path
                        alert_dict["new_image_url"] = (
                            f"{ML_BASE_URL}/file?path={new_obs.image_path}"
                        )
            except Exception:
                pass  # Non-fatal: alert still useful without images

            enriched.append(alert_dict)

        return {"alerts": enriched}
    except Exception as exc:
        return {"alerts": [], "error": str(exc)}


@app.get("/alerts/{alert_id}")
async def get_alert_detail(alert_id: str) -> dict:
    """Return full detail for a specific alert."""
    try:
        from pgil.store import SpatialMemoryStore
        store = SpatialMemoryStore()
        alert = store.get_alert(alert_id)
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")

        # Get associated change event
        change_event = store.get_change_event(alert.change_event_id) if alert.change_event_id else None

        # Get area info
        area = store.get_area(alert.area_id)

        result = {
            "alert": alert.model_dump(),
            "change_event": change_event.model_dump() if change_event else None,
            "area": area.model_dump() if area else None,
        }

        # Get observations if change event exists
        if change_event:
            old_obs = store.get_observation(change_event.old_observation_id)
            new_obs = store.get_observation(change_event.new_observation_id)
            result["old_observation"] = old_obs.model_dump() if old_obs else None
            result["new_observation"] = new_obs.model_dump() if new_obs else None

        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/alerts/{alert_id}/feedback")
async def submit_alert_feedback(alert_id: str, feedback: dict) -> dict:
    """Accept user feedback on an alert (confirm/dismiss)."""
    try:
        from pgil.store import SpatialMemoryStore
        from pgil.models import AlertStatus
        store = SpatialMemoryStore()

        status_str = feedback.get("status", "dismissed")
        user_note = feedback.get("feedback", "")

        status = AlertStatus.CONFIRMED if status_str == "confirmed" else AlertStatus.DISMISSED
        store.update_alert_status(alert_id, status, user_note)
        return {"status": "updated", "alert_id": alert_id, "new_status": status.value}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/areas")
async def get_areas() -> dict:
    """List all known geographic areas with observation counts."""
    try:
        from pgil.store import SpatialMemoryStore
        store = SpatialMemoryStore()
        areas = store.get_all_areas()
        return {"areas": [a.model_dump() for a in areas]}
    except Exception as exc:
        return {"areas": [], "error": str(exc)}


@app.get("/areas/{area_id}/history")
async def get_area_history(area_id: str) -> dict:
    """Get observation timeline for a geographic area."""
    try:
        from pgil.store import SpatialMemoryStore
        store = SpatialMemoryStore()
        area = store.get_area(area_id)
        if not area:
            raise HTTPException(status_code=404, detail="Area not found")

        observations = store.get_observations_for_area(area_id)
        change_events = store.get_change_events_for_area(area_id)

        return {
            "area": area.model_dump(),
            "observations": [o.model_dump() for o in observations],
            "change_events": [ce.model_dump() for ce in change_events],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/investigation/{change_id}")
async def get_investigation(change_id: str) -> dict:
    """Full investigation payload for a change event."""
    try:
        from pgil.store import SpatialMemoryStore
        store = SpatialMemoryStore()

        change_event = store.get_change_event(change_id)
        if not change_event:
            raise HTTPException(status_code=404, detail="Change event not found")

        area = store.get_area(change_event.area_id)
        old_obs = store.get_observation(change_event.old_observation_id)
        new_obs = store.get_observation(change_event.new_observation_id)

        old_objects = store.get_objects_for_observation(change_event.old_observation_id) if old_obs else []
        new_objects = store.get_objects_for_observation(change_event.new_observation_id) if new_obs else []

        return {
            "change_event": change_event.model_dump(),
            "area": area.model_dump() if area else None,
            "old_observation": old_obs.model_dump() if old_obs else None,
            "new_observation": new_obs.model_dump() if new_obs else None,
            "old_objects": [o.model_dump() for o in old_objects],
            "new_objects": [o.model_dump() for o in new_objects],
            "change_mask_b64": change_event.change_mask_b64,
            "old_image_path": old_obs.image_path if old_obs else "",
            "new_image_path": new_obs.image_path if new_obs else "",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
