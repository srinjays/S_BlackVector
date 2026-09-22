"""
PGIL Area Matcher
=================
Matches incoming satellite images to existing geographic Areas or creates
new ones.  Creates Observation records in the spatial memory store.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional, Tuple

from .models import Area, Observation
from .store import SpatialMemoryStore

logger = logging.getLogger(__name__)


def match_or_create_area(
    bounds: dict,
    crs: Optional[str] = None,
    sensor: Optional[str] = None,
    store: Optional[SpatialMemoryStore] = None,
    iou_threshold: float = 0.3,
) -> Tuple[str, bool]:
    """
    Match an image footprint to an existing Area or create a new one.

    Args:
        bounds: dict with min_x, min_y, max_x, max_y
        crs: coordinate reference system string
        sensor: sensor name
        store: SpatialMemoryStore instance
        iou_threshold: minimum IoU to consider an overlap match

    Returns:
        (area_id, is_new_area) tuple
    """
    if store is None:
        store = SpatialMemoryStore()

    min_x = bounds.get("min_x", 0)
    min_y = bounds.get("min_y", 0)
    max_x = bounds.get("max_x", 0)
    max_y = bounds.get("max_y", 0)

    # Search for overlapping areas
    matching_areas = store.find_overlapping_areas(min_x, min_y, max_x, max_y, iou_threshold)

    if matching_areas:
        # Use the best match (first result, already sorted)
        area = matching_areas[0]
        logger.info(
            "PGIL: Matched image footprint to existing area %s (%d prior observations)",
            area.area_id,
            area.observation_count,
        )
        return area.area_id, False

    # No match — create a new area
    centroid_lat = (min_y + max_y) / 2
    centroid_lon = (min_x + max_x) / 2

    area = Area(
        geometry={
            "type": "Polygon",
            "coordinates": [[
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y],
                [min_x, min_y],
            ]],
        },
        centroid={"lat": centroid_lat, "lon": centroid_lon},
        bounds={"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y},
        crs=crs,
        metadata={"sensor": sensor} if sensor else {},
    )
    store.store_area(area)
    logger.info("PGIL: Created new area %s at (%.4f, %.4f)", area.area_id, centroid_lat, centroid_lon)
    return area.area_id, True


def create_observation(
    area_id: str,
    image_path: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    band_count: Optional[int] = None,
    crs: Optional[str] = None,
    bounds: Optional[dict] = None,
    sensor: Optional[str] = None,
    platform: Optional[str] = None,
    acquisition_date: Optional[str] = None,
    resolution_m: Optional[float] = None,
    cloud_cover_pct: Optional[float] = None,
    store: Optional[SpatialMemoryStore] = None,
) -> str:
    """
    Create and store a new Observation record linked to an Area.

    Returns:
        observation_id
    """
    if store is None:
        store = SpatialMemoryStore()

    obs = Observation(
        area_id=area_id,
        image_path=image_path,
        width=width,
        height=height,
        band_count=band_count,
        crs=crs,
        bounds=bounds,
        sensor=sensor,
        platform=platform,
        acquisition_date=acquisition_date,
        resolution_m=resolution_m,
        cloud_cover_pct=cloud_cover_pct,
    )
    store.store_observation(obs)
    store.increment_observation_count(area_id)
    logger.info(
        "PGIL: Stored observation %s for area %s (image: %s)",
        obs.observation_id,
        area_id,
        image_path,
    )
    return obs.observation_id
