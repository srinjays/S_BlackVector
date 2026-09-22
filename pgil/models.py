"""
PGIL Data Models
================
Pydantic models for the Persistent Geospatial Intelligence Layer,
matching the PDF specification §7.4 data model.

Entities: Area, Observation, DetectedObject, ChangeEvent, Alert
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    DISMISSED = "dismissed"


class ObjectChangeType(str, Enum):
    NEW = "new"
    REMOVED = "removed"
    UNCHANGED = "unchanged"
    MODIFIED = "modified"


class ChangeCategory(str, Enum):
    CONSTRUCTION = "construction"       # No building → building
    DEMOLITION = "demolition"           # Building → no building
    EXPANSION = "expansion"             # Small footprint → larger
    ROAD_DEVELOPMENT = "road_development"
    VEGETATION_LOSS = "vegetation_loss"  # Vegetation → built-up/bare
    VEGETATION_GAIN = "vegetation_gain"
    WATER_CHANGE = "water_change"       # Dry ↔ water
    GENERAL = "general"


# ── Core Entities ────────────────────────────────────────────

class GeoJSON(BaseModel):
    """Minimal GeoJSON geometry representation."""
    type: str = "Polygon"
    coordinates: Any = Field(default_factory=list)


class Area(BaseModel):
    """A geographic region with a stable identity across observations."""
    area_id: str = Field(default_factory=lambda: f"AREA_{uuid4().hex[:8].upper()}")
    geometry: Optional[Dict[str, Any]] = None       # GeoJSON bounding polygon
    centroid: Optional[Dict[str, float]] = None      # {"lat": ..., "lon": ...}
    bounds: Optional[Dict[str, float]] = None        # {"min_x", "min_y", "max_x", "max_y"}
    location_name: Optional[str] = None
    crs: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    observation_count: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Observation(BaseModel):
    """A single satellite scene linked to a geographic Area."""
    observation_id: str = Field(default_factory=lambda: f"OBS_{uuid4().hex[:8].upper()}")
    area_id: str
    acquisition_date: Optional[str] = None
    sensor: Optional[str] = None
    platform: Optional[str] = None
    resolution_m: Optional[float] = None
    image_path: str
    width: Optional[int] = None
    height: Optional[int] = None
    band_count: Optional[int] = None
    crs: Optional[str] = None
    bounds: Optional[Dict[str, float]] = None
    detected_objects_count: int = 0
    cloud_cover_pct: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DetectedObject(BaseModel):
    """A geographic object extracted from an observation."""
    object_id: str = Field(default_factory=lambda: f"OBJ_{uuid4().hex[:8].upper()}")
    observation_id: str
    class_name: str                                    # building, water body, road, etc.
    confidence: float = 0.0
    bbox: Optional[Dict[str, float]] = None            # {"x1", "y1", "x2", "y2"} in pixels
    geo_bbox: Optional[Dict[str, float]] = None        # geographic coordinates
    geometry: Optional[Dict[str, Any]] = None          # GeoJSON polygon (geographic)
    area_sqm: Optional[float] = None
    centroid: Optional[Dict[str, float]] = None        # {"lat", "lon"}
    attributes: Dict[str, Any] = Field(default_factory=dict)


class ObjectMatch(BaseModel):
    """Result of matching an old object to a new object across observations."""
    old_object_id: Optional[str] = None
    new_object_id: Optional[str] = None
    class_name: str
    change_type: ObjectChangeType
    iou: float = 0.0
    old_bbox: Optional[Dict[str, float]] = None
    new_bbox: Optional[Dict[str, float]] = None


class ChangeEvent(BaseModel):
    """A detected change between two observations of the same area."""
    change_id: str = Field(default_factory=lambda: f"CHG_{uuid4().hex[:8].upper()}")
    area_id: str
    old_observation_id: str
    new_observation_id: str
    change_type: ChangeCategory = ChangeCategory.GENERAL
    object_matches: List[ObjectMatch] = Field(default_factory=list)
    new_objects_count: int = 0
    removed_objects_count: int = 0
    modified_objects_count: int = 0
    unchanged_objects_count: int = 0
    affected_area_sqm: Optional[float] = None
    change_percentage: Optional[float] = None          # % of area changed (from ChangeFormer)
    confidence: float = 0.0
    severity: Severity = Severity.LOW
    change_mask_b64: Optional[str] = None              # base64-encoded PNG change mask
    optical_evidence: bool = False
    sar_evidence: bool = False
    summary: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Alert(BaseModel):
    """An actionable alert generated from a significant ChangeEvent."""
    alert_id: str = Field(default_factory=lambda: f"ALERT_{uuid4().hex[:8].upper()}")
    change_event_id: str
    area_id: str
    alert_type: str = ""                               # e.g., "URBAN_DEVELOPMENT", "VEGETATION_LOSS"
    title: str = ""
    description: str = ""
    severity: Severity = Severity.MEDIUM
    confidence: float = 0.0
    status: AlertStatus = AlertStatus.PENDING
    observation_interval: Optional[str] = None         # "Jan 2026 → Jul 2026"
    affected_count: int = 0
    affected_area_sqm: Optional[float] = None
    optical_evidence: bool = False
    sar_evidence: bool = False
    user_feedback: Optional[str] = None
    reviewed_at: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ── Investigation Response ───────────────────────────────────

class InvestigationPayload(BaseModel):
    """Full evidence payload for the investigation UI."""
    change_event: ChangeEvent
    alert: Alert
    area: Area
    old_observation: Observation
    new_observation: Observation
    old_objects: List[DetectedObject] = Field(default_factory=list)
    new_objects: List[DetectedObject] = Field(default_factory=list)
    object_matches: List[ObjectMatch] = Field(default_factory=list)
    change_mask_b64: Optional[str] = None
    old_image_path: str = ""
    new_image_path: str = ""
