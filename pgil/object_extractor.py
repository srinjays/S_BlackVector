"""
PGIL Object Extractor
=====================
Extracts grounded geographic objects from a satellite observation using
Grounding DINO + MobileSAM (already loaded in the ModelManager), and
stores them in the spatial memory store.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .models import DetectedObject
from .store import SpatialMemoryStore

logger = logging.getLogger(__name__)

# Standard target classes for background intelligence extraction
DEFAULT_TARGET_CLASSES = [
    "building",
    "water body",
    "road",
    "vegetation",
    "agricultural field",
    "vehicle",
    "ship",
    "industrial structure",
]


def pixel_bbox_to_geo(
    bbox: Dict[str, float],
    affine: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, float]]:
    """
    Convert pixel bounding box to geographic coordinates using affine transform.
    Affine dict keys: a (x-res), b (row rot), c (x-origin), d (col rot), e (y-res), f (y-origin).
    """
    if not affine:
        return None

    a = affine.get("a", 1)
    b = affine.get("b", 0)
    c = affine.get("c", 0)
    d = affine.get("d", 0)
    e = affine.get("e", -1)
    f = affine.get("f", 0)

    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

    geo_x1 = a * x1 + b * y1 + c
    geo_y1 = d * x1 + e * y1 + f
    geo_x2 = a * x2 + b * y2 + c
    geo_y2 = d * x2 + e * y2 + f

    return {
        "min_x": min(geo_x1, geo_x2),
        "min_y": min(geo_y1, geo_y2),
        "max_x": max(geo_x1, geo_x2),
        "max_y": max(geo_y1, geo_y2),
    }


def estimate_area_sqm(
    bbox: Dict[str, float],
    resolution_m: Optional[float] = None,
) -> Optional[float]:
    """Estimate object area in square meters from pixel bbox and resolution."""
    if resolution_m is None:
        return None
    w_px = abs(bbox["x2"] - bbox["x1"])
    h_px = abs(bbox["y2"] - bbox["y1"])
    return w_px * h_px * resolution_m * resolution_m


async def extract_and_store_objects(
    observation_id: str,
    image_path: str,
    store: SpatialMemoryStore,
    manager: Any,
    target_classes: Optional[List[str]] = None,
    affine: Optional[Dict[str, float]] = None,
    resolution_m: Optional[float] = None,
    confidence_threshold: float = 0.25,
) -> List[DetectedObject]:
    """
    Run Grounding DINO on the image for each target class and store the
    resulting objects in the spatial memory.

    This reuses the already-loaded Grounding DINO + MobileSAM from ModelManager,
    so it adds zero VRAM overhead.

    Args:
        observation_id: ID of the observation these objects belong to
        image_path: path to the satellite image file
        store: SpatialMemoryStore instance
        manager: ModelManager instance (has .grounding_head, .sam)
        target_classes: list of class names to detect
        affine: affine transform dict for geo-referencing
        resolution_m: spatial resolution in meters
        confidence_threshold: minimum confidence for object inclusion

    Returns:
        List of stored DetectedObject instances
    """
    if target_classes is None:
        target_classes = DEFAULT_TARGET_CLASSES

    if manager.grounding_head is None:
        logger.warning("PGIL: Grounding DINO not loaded — skipping object extraction")
        return []

    try:
        from PIL import Image
        pil_image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        logger.error("PGIL: Failed to open image %s: %s", image_path, exc)
        return []

    all_objects: List[DetectedObject] = []

    # Run Grounding DINO for a combined prompt of all target classes
    combined_prompt = " . ".join(target_classes)
    try:
        from groundingdino.util.inference import predict as gdino_predict
        import torch

        gdino_model = manager.grounding_head
        from groundingdino.util.inference import load_image
        import numpy as np

        # Prepare image tensor for Grounding DINO
        image_np = np.array(pil_image)
        # Grounding DINO expects BGR numpy → we convert
        import torchvision.transforms.functional as F
        image_tensor = F.to_tensor(pil_image)

        boxes, logits, phrases = gdino_predict(
            model=gdino_model,
            image=image_tensor,
            caption=combined_prompt,
            box_threshold=confidence_threshold,
            text_threshold=confidence_threshold,
        )

        w, h = pil_image.size
        for i, (box, conf, phrase) in enumerate(zip(boxes, logits, phrases)):
            # Grounding DINO boxes are in cx, cy, w, h format normalized to [0, 1]
            cx, cy, bw, bh = box.tolist()
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h

            pixel_bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            geo_bbox = pixel_bbox_to_geo(pixel_bbox, affine)
            area_sqm = estimate_area_sqm(pixel_bbox, resolution_m)

            centroid = None
            if geo_bbox:
                centroid = {
                    "lat": (geo_bbox["min_y"] + geo_bbox["max_y"]) / 2,
                    "lon": (geo_bbox["min_x"] + geo_bbox["max_x"]) / 2,
                }

            # Clean up the phrase to a standard class name
            class_name = phrase.strip().lower()
            # Map back to nearest target class
            for tc in target_classes:
                if tc.lower() in class_name or class_name in tc.lower():
                    class_name = tc
                    break

            obj = DetectedObject(
                observation_id=observation_id,
                class_name=class_name,
                confidence=float(conf),
                bbox=pixel_bbox,
                geo_bbox=geo_bbox,
                area_sqm=area_sqm,
                centroid=centroid,
                attributes={"phrase": phrase, "detection_index": i},
            )
            all_objects.append(obj)

        logger.info(
            "PGIL: Extracted %d objects from observation %s (%s)",
            len(all_objects),
            observation_id,
            image_path,
        )

    except Exception as exc:
        logger.error("PGIL: Object extraction failed for %s: %s", image_path, exc, exc_info=True)
        return []

    # Store objects
    if all_objects:
        store.store_objects(all_objects)
        store.update_detected_objects_count(observation_id, len(all_objects))

    return all_objects
