"""
PGIL Change Comparator
======================
Compares two observations of the same geographic area using:
1. Pixel-level change detection (ChangeFormer — already in ModelManager)
2. Object-level spatial matching (IoU between old/new object polygons)

Produces a ChangeEvent with full evidence.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    ChangeCategory,
    ChangeEvent,
    DetectedObject,
    ObjectChangeType,
    ObjectMatch,
    Severity,
)
from .store import SpatialMemoryStore

logger = logging.getLogger(__name__)

# IoU thresholds for object matching
IOU_MATCH_THRESHOLD = 0.3       # minimum IoU to consider same object
IOU_MODIFIED_THRESHOLD = 0.7    # above this → unchanged, below → modified


def match_objects_spatially(
    old_objects: List[DetectedObject],
    new_objects: List[DetectedObject],
) -> Tuple[List[ObjectMatch], int, int, int, int]:
    """
    Spatially match old and new detected objects using bounding box IoU.

    Returns:
        (matches, new_count, removed_count, modified_count, unchanged_count)
    """
    matches: List[ObjectMatch] = []
    matched_old_ids: set = set()
    matched_new_ids: set = set()

    # For each new object, find the best-matching old object of the same class
    for new_obj in new_objects:
        best_iou = 0.0
        best_old: Optional[DetectedObject] = None

        for old_obj in old_objects:
            if old_obj.object_id in matched_old_ids:
                continue
            if old_obj.class_name != new_obj.class_name:
                continue
            if not old_obj.bbox or not new_obj.bbox:
                continue

            iou = _bbox_iou(old_obj.bbox, new_obj.bbox)
            if iou > best_iou:
                best_iou = iou
                best_old = old_obj

        if best_old and best_iou >= IOU_MATCH_THRESHOLD:
            matched_old_ids.add(best_old.object_id)
            matched_new_ids.add(new_obj.object_id)

            change_type = (
                ObjectChangeType.UNCHANGED
                if best_iou >= IOU_MODIFIED_THRESHOLD
                else ObjectChangeType.MODIFIED
            )

            matches.append(ObjectMatch(
                old_object_id=best_old.object_id,
                new_object_id=new_obj.object_id,
                class_name=new_obj.class_name,
                change_type=change_type,
                iou=best_iou,
                old_bbox=best_old.bbox,
                new_bbox=new_obj.bbox,
            ))

    # Unmatched new objects → NEW
    for new_obj in new_objects:
        if new_obj.object_id not in matched_new_ids:
            matches.append(ObjectMatch(
                new_object_id=new_obj.object_id,
                class_name=new_obj.class_name,
                change_type=ObjectChangeType.NEW,
                iou=0.0,
                new_bbox=new_obj.bbox,
            ))

    # Unmatched old objects → REMOVED
    for old_obj in old_objects:
        if old_obj.object_id not in matched_old_ids:
            matches.append(ObjectMatch(
                old_object_id=old_obj.object_id,
                class_name=old_obj.class_name,
                change_type=ObjectChangeType.REMOVED,
                iou=0.0,
                old_bbox=old_obj.bbox,
            ))

    new_count = sum(1 for m in matches if m.change_type == ObjectChangeType.NEW)
    removed_count = sum(1 for m in matches if m.change_type == ObjectChangeType.REMOVED)
    modified_count = sum(1 for m in matches if m.change_type == ObjectChangeType.MODIFIED)
    unchanged_count = sum(1 for m in matches if m.change_type == ObjectChangeType.UNCHANGED)

    return matches, new_count, removed_count, modified_count, unchanged_count


def classify_change(
    new_count: int,
    removed_count: int,
    modified_count: int,
    matches: List[ObjectMatch],
) -> ChangeCategory:
    """Determine the primary change category from object-level match results."""
    # Count by class
    new_buildings = sum(1 for m in matches if m.change_type == ObjectChangeType.NEW and "building" in m.class_name.lower())
    removed_buildings = sum(1 for m in matches if m.change_type == ObjectChangeType.REMOVED and "building" in m.class_name.lower())
    new_water = sum(1 for m in matches if m.change_type == ObjectChangeType.NEW and "water" in m.class_name.lower())
    removed_veg = sum(1 for m in matches if m.change_type == ObjectChangeType.REMOVED and "vegetation" in m.class_name.lower())
    new_roads = sum(1 for m in matches if m.change_type == ObjectChangeType.NEW and "road" in m.class_name.lower())

    if new_buildings > 0 and new_buildings >= removed_buildings:
        return ChangeCategory.CONSTRUCTION
    if removed_buildings > 0 and removed_buildings > new_buildings:
        return ChangeCategory.DEMOLITION
    if removed_veg > 0:
        return ChangeCategory.VEGETATION_LOSS
    if new_water > 0:
        return ChangeCategory.WATER_CHANGE
    if new_roads > 0:
        return ChangeCategory.ROAD_DEVELOPMENT
    if modified_count > 0:
        return ChangeCategory.EXPANSION
    return ChangeCategory.GENERAL


def _fast_pixel_diff(
    old_image_path: str,
    new_image_path: str,
) -> Tuple[Optional[str], Optional[float]]:
    """Fast pixel-difference change detection (no GPU needed).

    ChangeFormer is trained for structural changes (buildings, roads) and
    often returns 0% for radiometric/seasonal shifts between satellite
    mosaics. This simple OpenCV diff catches those changes.

    Returns:
        (change_mask_b64, change_percentage) or (None, None) on failure
    """
    try:
        import cv2
        import numpy as np
        from PIL import Image
        from services.models.serving.utils.image_loader import ensure_rgb_path

        p1 = ensure_rgb_path(old_image_path)
        p2 = ensure_rgb_path(new_image_path)

        img1 = cv2.imread(p1)
        img2 = cv2.imread(p2)
        if img1 is None or img2 is None:
            img1 = cv2.cvtColor(np.array(Image.open(p1).convert("RGB")), cv2.COLOR_RGB2BGR)
            img2 = cv2.cvtColor(np.array(Image.open(p2).convert("RGB")), cv2.COLOR_RGB2BGR)

        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        diff = cv2.absdiff(img1, img2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        changed = int(np.sum(mask > 0))
        total = int(mask.size)
        pct = round((changed / total) * 100, 2) if total > 0 else 0.0

        # Encode mask as base64
        mask_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        mask_rgba[mask > 0] = [255, 80, 0, 128]
        mask_img = Image.fromarray(mask_rgba, "RGBA")
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        logger.info("PGIL: Fast pixel-diff complete. %.1f%% changed (%d/%d px)", pct, changed, total)
        return mask_b64, pct

    except Exception as exc:
        logger.error("PGIL: Fast pixel-diff failed: %s", exc)
        return None, None


async def run_pixel_change_detection(
    old_image_path: str,
    new_image_path: str,
    manager: Any,
) -> Tuple[Optional[str], Optional[float]]:
    """Run change detection: ChangeFormer (structural) + pixel-diff (radiometric) fallback.

    Returns the result with higher change percentage.
    """
    cf_b64, cf_pct = None, None
    diff_b64, diff_pct = None, None

    # 1. Try ChangeFormer (structural change detection)
    if manager.changeformer is not None:
        try:
            from PIL import Image
            import numpy as np

            result = manager.changeformer.predict(old_image_path, new_image_path)
            change_mask = result["change_mask"]
            cf_pct = result["change_pct"]

            mask_rgba = np.zeros((*change_mask.shape, 4), dtype=np.uint8)
            mask_rgba[change_mask > 128] = [255, 0, 0, 128]
            mask_img = Image.fromarray(mask_rgba, "RGBA")
            buf = io.BytesIO()
            mask_img.save(buf, format="PNG")
            cf_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            logger.info("PGIL: ChangeFormer detected %.1f%% structural change", cf_pct)
        except Exception as exc:
            logger.error("PGIL: ChangeFormer failed: %s", exc, exc_info=True)

    # 2. Always run fast pixel-diff (catches seasonal/radiometric changes)
    diff_b64, diff_pct = _fast_pixel_diff(old_image_path, new_image_path)

    # 3. Use whichever detected more change
    cf_pct = cf_pct or 0.0
    diff_pct = diff_pct or 0.0

    if cf_pct >= diff_pct:
        logger.info("PGIL: Using ChangeFormer result (%.1f%% >= pixel-diff %.1f%%)", cf_pct, diff_pct)
        return cf_b64, round(cf_pct, 2)
    else:
        logger.info("PGIL: Using pixel-diff result (%.1f%% > ChangeFormer %.1f%%)", diff_pct, cf_pct)
        return diff_b64, round(diff_pct, 2)


async def compare_observations(
    old_obs_id: str,
    new_obs_id: str,
    area_id: str,
    store: SpatialMemoryStore,
    manager: Any,
) -> Optional[ChangeEvent]:
    """
    Full change comparison between two observations:
    1. Object-level spatial matching
    2. Pixel-level ChangeFormer analysis
    3. Aggregate into a ChangeEvent

    Args:
        old_obs_id: observation_id of the older scene
        new_obs_id: observation_id of the newer scene
        area_id: the geographic area these observations belong to
        store: SpatialMemoryStore instance
        manager: ModelManager instance

    Returns:
        ChangeEvent if meaningful change detected, else None
    """
    old_obs = store.get_observation(old_obs_id)
    new_obs = store.get_observation(new_obs_id)

    if not old_obs or not new_obs:
        logger.error("PGIL: Cannot compare — observation(s) not found")
        return None

    # 1. Object-level comparison
    old_objects = store.get_objects_for_observation(old_obs_id)
    new_objects = store.get_objects_for_observation(new_obs_id)

    matches, new_count, removed_count, modified_count, unchanged_count = \
        match_objects_spatially(old_objects, new_objects)

    # 2. Pixel-level ChangeFormer
    change_mask_b64, change_pct = await run_pixel_change_detection(
        old_obs.image_path, new_obs.image_path, manager
    )

    # 3. Always create a ChangeEvent to preserve temporal history.
    #    The anomaly engine decides whether to fire an alert.
    total_changes = new_count + removed_count + modified_count

    # 4. Classify the primary change type
    change_type = classify_change(new_count, removed_count, modified_count, matches)

    # 5. Determine confidence
    # Combine object detection confidence and pixel change evidence
    avg_obj_conf = 0.0
    if matches:
        confs = [m.iou for m in matches if m.iou > 0]
        avg_obj_conf = sum(confs) / len(confs) if confs else 0.5
    pixel_conf = min((change_pct or 0) / 50.0, 1.0)  # normalize: 50%+ change → full confidence
    combined_confidence = 0.6 * avg_obj_conf + 0.4 * pixel_conf

    # 6. Build summary
    parts = []
    if new_count > 0:
        new_classes = {}
        for m in matches:
            if m.change_type == ObjectChangeType.NEW:
                new_classes[m.class_name] = new_classes.get(m.class_name, 0) + 1
        for cls, cnt in new_classes.items():
            parts.append(f"{cnt} new {cls}(s)")
    if removed_count > 0:
        parts.append(f"{removed_count} removed object(s)")
    if modified_count > 0:
        parts.append(f"{modified_count} modified object(s)")
    if change_pct and change_pct > 0:
        parts.append(f"{change_pct:.1f}% pixel-level change")
    summary = "; ".join(parts) if parts else "Minor changes detected"

    # 7. Build observation interval string
    interval = ""
    if old_obs.acquisition_date and new_obs.acquisition_date:
        interval = f"{old_obs.acquisition_date} → {new_obs.acquisition_date}"
    elif old_obs.created_at and new_obs.created_at:
        interval = f"{old_obs.created_at[:10]} → {new_obs.created_at[:10]}"

    # 8. Create ChangeEvent
    event = ChangeEvent(
        area_id=area_id,
        old_observation_id=old_obs_id,
        new_observation_id=new_obs_id,
        change_type=change_type,
        object_matches=matches,
        new_objects_count=new_count,
        removed_objects_count=removed_count,
        modified_objects_count=modified_count,
        unchanged_objects_count=unchanged_count,
        change_percentage=change_pct,
        confidence=round(combined_confidence, 3),
        change_mask_b64=change_mask_b64,
        optical_evidence=True,
        summary=summary,
        metadata={"observation_interval": interval},
    )

    store.store_change_event(event)
    logger.info(
        "PGIL: ChangeEvent %s created for area %s: %s",
        event.change_id, area_id, summary,
    )
    return event


# ── Internal helpers ─────────────────────────────────────────

def _bbox_iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Calculate IoU between two pixel bounding boxes."""
    inter_x1 = max(a["x1"], b["x1"])
    inter_y1 = max(a["y1"], b["y1"])
    inter_x2 = min(a["x2"], b["x2"])
    inter_y2 = min(a["y2"], b["y2"])

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union_area = area_a + area_b - inter_area

    return float(inter_area / union_area) if union_area > 0 else 0.0
