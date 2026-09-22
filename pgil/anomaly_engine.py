"""
PGIL Anomaly Engine
===================
Rule-based anomaly scoring and alert generation per PDF §7.9 and §7.12.
Evaluates ChangeEvents against configurable thresholds and generates
actionable Alerts when significance thresholds are crossed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .models import (
    Alert,
    ChangeCategory,
    ChangeEvent,
    ObjectChangeType,
    Severity,
)
from .store import SpatialMemoryStore

logger = logging.getLogger(__name__)


@dataclass
class AnomalyRules:
    """Configurable thresholds for anomaly detection."""
    # Building change thresholds
    new_building_alert_threshold: int = 3       # alert when ≥ N new buildings
    removed_building_alert_threshold: int = 3   # alert when ≥ N buildings removed

    # Vegetation change thresholds
    vegetation_loss_pct_threshold: float = 15.0  # alert when vegetation loss > X%

    # Water change thresholds
    water_change_pct_threshold: float = 20.0     # alert when water area change > X%

    # Pixel-level change threshold
    pixel_change_pct_threshold: float = 1.0       # alert when > X% pixels changed

    # General thresholds
    min_total_changes: int = 0                    # minimum changed objects to trigger alert
    min_confidence: float = 0.3                  # minimum confidence to generate alert

    # Severity escalation
    high_severity_new_buildings: int = 10         # HIGH when ≥ N new buildings
    critical_severity_new_buildings: int = 25     # CRITICAL when ≥ N new buildings
    high_severity_pixel_change_pct: float = 20.0
    critical_severity_pixel_change_pct: float = 40.0


# Default global rules instance
DEFAULT_RULES = AnomalyRules()


def evaluate_change_event(
    event: ChangeEvent,
    store: SpatialMemoryStore,
    rules: Optional[AnomalyRules] = None,
) -> Optional[Alert]:
    """
    Evaluate a ChangeEvent against anomaly rules and generate an Alert
    if significance thresholds are crossed.

    Args:
        event: The ChangeEvent to evaluate
        store: SpatialMemoryStore for historical context
        rules: AnomalyRules (uses defaults if None)

    Returns:
        Alert if anomaly detected, None otherwise
    """
    if rules is None:
        rules = DEFAULT_RULES

    # Skip low-confidence events
    if event.confidence < rules.min_confidence:
        logger.debug("PGIL Anomaly: Skipping event %s — confidence %.2f below threshold", event.change_id, event.confidence)
        return None

    # Count object changes by class
    new_buildings = sum(
        1 for m in event.object_matches
        if m.change_type == ObjectChangeType.NEW and "building" in m.class_name.lower()
    ) + int(event.metadata.get("building_delta", 0))
    removed_buildings = sum(
        1 for m in event.object_matches
        if m.change_type == ObjectChangeType.REMOVED and "building" in m.class_name.lower()
    )
    new_water = sum(
        1 for m in event.object_matches
        if m.change_type == ObjectChangeType.NEW and "water" in m.class_name.lower()
    )
    removed_veg = sum(
        1 for m in event.object_matches
        if m.change_type == ObjectChangeType.REMOVED and "vegetation" in m.class_name.lower()
    )

    total_changes = event.new_objects_count + event.removed_objects_count + event.modified_objects_count
    pixel_pct = event.change_percentage or 0.0

    # ── Rule evaluation ──────────────────────────────────────

    alert_type = ""
    title = ""
    description = ""
    severity = Severity.LOW
    should_alert = False

    # Rule 1: New buildings threshold
    if new_buildings >= rules.new_building_alert_threshold:
        should_alert = True
        alert_type = "URBAN_DEVELOPMENT"
        title = f"⚠ Potential Urban Development — {new_buildings} new building structures detected"
        description = (
            f"{new_buildings} new building structures have been identified since the previous observation. "
            f"This may indicate construction activity, urban expansion, or infrastructure development."
        )
        if new_buildings >= rules.critical_severity_new_buildings:
            severity = Severity.CRITICAL
        elif new_buildings >= rules.high_severity_new_buildings:
            severity = Severity.HIGH
        else:
            severity = Severity.MEDIUM

    # Rule 2: Building demolition
    elif removed_buildings >= rules.removed_building_alert_threshold:
        should_alert = True
        alert_type = "DEMOLITION"
        title = f"⚠ Potential Demolition — {removed_buildings} building structures no longer detected"
        description = (
            f"{removed_buildings} previously detected building structures are no longer present. "
            f"This may indicate demolition, damage, or significant land-use change."
        )
        severity = Severity.HIGH if removed_buildings >= 10 else Severity.MEDIUM

    # Rule 3: Vegetation loss
    elif removed_veg > 0 and pixel_pct >= rules.vegetation_loss_pct_threshold:
        should_alert = True
        alert_type = "VEGETATION_LOSS"
        title = f"🌿 Vegetation Loss Detected — {pixel_pct:.1f}% area change"
        description = (
            f"Significant vegetation reduction detected with {pixel_pct:.1f}% pixel-level change. "
            f"{removed_veg} vegetation region(s) are no longer present. "
            f"This may indicate deforestation, land clearing, or seasonal change."
        )
        severity = Severity.HIGH if pixel_pct > 30 else Severity.MEDIUM

    # Rule 4: Water body change
    elif new_water > 0 or (pixel_pct >= rules.water_change_pct_threshold and event.change_type == ChangeCategory.WATER_CHANGE):
        should_alert = True
        alert_type = "WATER_CHANGE"
        title = f"💧 Water Body Change Detected"
        description = (
            f"Significant water body changes detected. "
            f"This may indicate flooding, drought, reservoir changes, or land reclamation."
        )
        severity = Severity.HIGH

    # Rule 5: General significant pixel change
    elif pixel_pct >= rules.pixel_change_pct_threshold and total_changes >= rules.min_total_changes:
        should_alert = True
        alert_type = "GENERAL_CHANGE"
        title = f"📡 Significant Area Change — {pixel_pct:.1f}% changed"
        description = (
            f"{total_changes} object-level changes detected with {pixel_pct:.1f}% pixel-level change. "
            f"New: {event.new_objects_count}, Removed: {event.removed_objects_count}, Modified: {event.modified_objects_count}."
        )
        if pixel_pct >= rules.critical_severity_pixel_change_pct:
            severity = Severity.CRITICAL
        elif pixel_pct >= rules.high_severity_pixel_change_pct:
            severity = Severity.HIGH
        else:
            severity = Severity.MEDIUM

    # Rule 6: Any detectable radiometric/seasonal change (low severity)
    elif pixel_pct > 0:
        should_alert = True
        alert_type = "ENVIRONMENTAL_CHANGE"
        title = f"\U0001f30d Environmental Change Detected \u2014 {pixel_pct:.1f}% area change"
        description = (
            f"Radiometric or seasonal changes detected with {pixel_pct:.1f}% pixel-level difference. "
            f"This may indicate vegetation growth/loss, seasonal variation, or gradual land-use change."
        )
        severity = Severity.LOW

    if not should_alert:
        logger.debug("PGIL Anomaly: No anomaly rules triggered for event %s", event.change_id)
        return None

    # ── Historical escalation ────────────────────────────────
    # Check if similar changes persist across multiple observations
    prior_events = store.get_change_events_for_area(event.area_id)
    persistent_count = sum(
        1 for pe in prior_events
        if pe.change_id != event.change_id
        and pe.change_type == event.change_type
        and pe.new_objects_count > 0
    )
    if persistent_count >= 2:
        # Escalate severity for persistent changes
        if severity == Severity.LOW:
            severity = Severity.MEDIUM
        elif severity == Severity.MEDIUM:
            severity = Severity.HIGH
        description += f" (Change pattern persists across {persistent_count + 1} observation intervals.)"

    # ── Build observation interval ───────────────────────────
    observation_interval = event.metadata.get("observation_interval", "")

    # ── Create Alert ─────────────────────────────────────────
    alert = Alert(
        change_event_id=event.change_id,
        area_id=event.area_id,
        alert_type=alert_type,
        title=title,
        description=description,
        severity=severity,
        confidence=event.confidence,
        observation_interval=observation_interval,
        affected_count=total_changes,
        affected_area_sqm=event.affected_area_sqm,
        optical_evidence=event.optical_evidence,
        sar_evidence=event.sar_evidence,
    )

    store.store_alert(alert)
    logger.info(
        "PGIL ALERT: %s [%s] — %s (confidence: %.2f)",
        alert.alert_id, severity.value.upper(), title, event.confidence,
    )
    return alert
