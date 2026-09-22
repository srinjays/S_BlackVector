"""
PGIL Spatial Memory Store
=========================
SQLite-backed persistent store for Areas, Observations, DetectedObjects,
ChangeEvents, and Alerts.  Zero external dependencies beyond the Python
standard library + Pydantic (already in the project).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Optional, Union

from .models import (
    Alert,
    AlertStatus,
    Area,
    ChangeEvent,
    DetectedObject,
    Observation,
)

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "pgil_memory.db"


class SpatialMemoryStore:
    """Thread-safe SQLite store for the Persistent Geospatial Intelligence Layer."""

    def __init__(self, db_path: Optional[Union[Path, str]] = None) -> None:
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self.init_db()

    # ── Connection ───────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    # ── Schema init ──────────────────────────────────────────

    def init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS areas (
                area_id        TEXT PRIMARY KEY,
                geometry       TEXT,
                centroid       TEXT,
                bounds         TEXT,
                location_name  TEXT,
                crs            TEXT,
                metadata       TEXT DEFAULT '{}',
                observation_count INTEGER DEFAULT 0,
                created_at     TEXT,
                updated_at     TEXT
            );

            CREATE TABLE IF NOT EXISTS observations (
                observation_id TEXT PRIMARY KEY,
                area_id        TEXT NOT NULL REFERENCES areas(area_id),
                acquisition_date TEXT,
                sensor         TEXT,
                platform       TEXT,
                resolution_m   REAL,
                image_path     TEXT NOT NULL,
                width          INTEGER,
                height         INTEGER,
                band_count     INTEGER,
                crs            TEXT,
                bounds         TEXT,
                detected_objects_count INTEGER DEFAULT 0,
                cloud_cover_pct REAL,
                metadata       TEXT DEFAULT '{}',
                created_at     TEXT
            );

            CREATE TABLE IF NOT EXISTS detected_objects (
                object_id      TEXT PRIMARY KEY,
                observation_id TEXT NOT NULL REFERENCES observations(observation_id),
                class_name     TEXT NOT NULL,
                confidence     REAL DEFAULT 0.0,
                bbox           TEXT,
                geo_bbox       TEXT,
                geometry       TEXT,
                area_sqm       REAL,
                centroid       TEXT,
                attributes     TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS change_events (
                change_id          TEXT PRIMARY KEY,
                area_id            TEXT NOT NULL REFERENCES areas(area_id),
                old_observation_id TEXT NOT NULL REFERENCES observations(observation_id),
                new_observation_id TEXT NOT NULL REFERENCES observations(observation_id),
                change_type        TEXT DEFAULT 'general',
                object_matches     TEXT DEFAULT '[]',
                new_objects_count  INTEGER DEFAULT 0,
                removed_objects_count INTEGER DEFAULT 0,
                modified_objects_count INTEGER DEFAULT 0,
                unchanged_objects_count INTEGER DEFAULT 0,
                affected_area_sqm  REAL,
                change_percentage  REAL,
                confidence         REAL DEFAULT 0.0,
                severity           TEXT DEFAULT 'low',
                change_mask_b64    TEXT,
                optical_evidence   INTEGER DEFAULT 0,
                sar_evidence       INTEGER DEFAULT 0,
                summary            TEXT DEFAULT '',
                metadata           TEXT DEFAULT '{}',
                created_at         TEXT
            );

            CREATE TABLE IF NOT EXISTS alerts (
                alert_id         TEXT PRIMARY KEY,
                change_event_id  TEXT NOT NULL REFERENCES change_events(change_id),
                area_id          TEXT NOT NULL REFERENCES areas(area_id),
                alert_type       TEXT DEFAULT '',
                title            TEXT DEFAULT '',
                description      TEXT DEFAULT '',
                severity         TEXT DEFAULT 'medium',
                confidence       REAL DEFAULT 0.0,
                status           TEXT DEFAULT 'pending',
                observation_interval TEXT,
                affected_count   INTEGER DEFAULT 0,
                affected_area_sqm REAL,
                optical_evidence INTEGER DEFAULT 0,
                sar_evidence     INTEGER DEFAULT 0,
                user_feedback    TEXT,
                reviewed_at      TEXT,
                created_at       TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_observations_area ON observations(area_id);
            CREATE INDEX IF NOT EXISTS idx_objects_observation ON detected_objects(observation_id);
            CREATE INDEX IF NOT EXISTS idx_change_events_area ON change_events(area_id);
            CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
        """)
        conn.commit()
        logger.info("PGIL spatial memory store initialized at %s", self.db_path)

    # ── Areas ────────────────────────────────────────────────

    def store_area(self, area: Area) -> str:
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO areas
               (area_id, geometry, centroid, bounds, location_name, crs,
                metadata, observation_count, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                area.area_id,
                json.dumps(area.geometry) if area.geometry else None,
                json.dumps(area.centroid) if area.centroid else None,
                json.dumps(area.bounds) if area.bounds else None,
                area.location_name,
                area.crs,
                json.dumps(area.metadata),
                area.observation_count,
                area.created_at,
                area.updated_at,
            ),
        )
        conn.commit()
        return area.area_id

    def get_area(self, area_id: str) -> Optional[Area]:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM areas WHERE area_id = ?", (area_id,)).fetchone()
        return self._row_to_area(row) if row else None

    def get_all_areas(self) -> List[Area]:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM areas ORDER BY updated_at DESC").fetchall()
        return [self._row_to_area(r) for r in rows]

    def find_overlapping_areas(
        self, min_x: float, min_y: float, max_x: float, max_y: float, iou_threshold: float = 0.3
    ) -> List[Area]:
        """Find areas whose stored bounds overlap with the given bbox above a given IoU."""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM areas WHERE bounds IS NOT NULL").fetchall()
        results: List[Area] = []
        for row in rows:
            area = self._row_to_area(row)
            if area.bounds:
                iou = self._calculate_iou(
                    min_x, min_y, max_x, max_y,
                    area.bounds["min_x"], area.bounds["min_y"],
                    area.bounds["max_x"], area.bounds["max_y"],
                )
                if iou >= iou_threshold:
                    results.append(area)
        return results

    def increment_observation_count(self, area_id: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE areas SET observation_count = observation_count + 1, updated_at = datetime('now') WHERE area_id = ?",
            (area_id,),
        )
        conn.commit()

    # ── Observations ─────────────────────────────────────────

    def store_observation(self, obs: Observation) -> str:
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO observations
               (observation_id, area_id, acquisition_date, sensor, platform,
                resolution_m, image_path, width, height, band_count, crs, bounds,
                detected_objects_count, cloud_cover_pct, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                obs.observation_id,
                obs.area_id,
                obs.acquisition_date,
                obs.sensor,
                obs.platform,
                obs.resolution_m,
                obs.image_path,
                obs.width,
                obs.height,
                obs.band_count,
                obs.crs,
                json.dumps(obs.bounds) if obs.bounds else None,
                obs.detected_objects_count,
                obs.cloud_cover_pct,
                json.dumps(obs.metadata),
                obs.created_at,
            ),
        )
        conn.commit()
        return obs.observation_id

    def get_observations_for_area(self, area_id: str) -> List[Observation]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM observations WHERE area_id = ? ORDER BY created_at ASC",
            (area_id,),
        ).fetchall()
        return [self._row_to_observation(r) for r in rows]

    def get_observation(self, observation_id: str) -> Optional[Observation]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM observations WHERE observation_id = ?", (observation_id,)
        ).fetchone()
        return self._row_to_observation(row) if row else None

    def get_latest_observation_for_area(self, area_id: str) -> Optional[Observation]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM observations WHERE area_id = ? ORDER BY created_at DESC LIMIT 1",
            (area_id,),
        ).fetchone()
        return self._row_to_observation(row) if row else None

    def update_detected_objects_count(self, observation_id: str, count: int) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE observations SET detected_objects_count = ? WHERE observation_id = ?",
            (count, observation_id),
        )
        conn.commit()

    # ── Detected Objects ─────────────────────────────────────

    def store_objects(self, objects: List[DetectedObject]) -> int:
        if not objects:
            return 0
        conn = self._get_conn()
        for obj in objects:
            conn.execute(
                """INSERT OR REPLACE INTO detected_objects
                   (object_id, observation_id, class_name, confidence,
                    bbox, geo_bbox, geometry, area_sqm, centroid, attributes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    obj.object_id,
                    obj.observation_id,
                    obj.class_name,
                    obj.confidence,
                    json.dumps(obj.bbox) if obj.bbox else None,
                    json.dumps(obj.geo_bbox) if obj.geo_bbox else None,
                    json.dumps(obj.geometry) if obj.geometry else None,
                    obj.area_sqm,
                    json.dumps(obj.centroid) if obj.centroid else None,
                    json.dumps(obj.attributes),
                ),
            )
        conn.commit()
        return len(objects)

    def get_objects_for_observation(self, observation_id: str) -> List[DetectedObject]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM detected_objects WHERE observation_id = ?",
            (observation_id,),
        ).fetchall()
        return [self._row_to_detected_object(r) for r in rows]

    # ── Change Events ────────────────────────────────────────

    def store_change_event(self, event: ChangeEvent) -> str:
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO change_events
               (change_id, area_id, old_observation_id, new_observation_id,
                change_type, object_matches, new_objects_count, removed_objects_count,
                modified_objects_count, unchanged_objects_count, affected_area_sqm,
                change_percentage, confidence, severity, change_mask_b64,
                optical_evidence, sar_evidence, summary, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.change_id,
                event.area_id,
                event.old_observation_id,
                event.new_observation_id,
                event.change_type.value if hasattr(event.change_type, "value") else event.change_type,
                json.dumps([m.model_dump() for m in event.object_matches]),
                event.new_objects_count,
                event.removed_objects_count,
                event.modified_objects_count,
                event.unchanged_objects_count,
                event.affected_area_sqm,
                event.change_percentage,
                event.confidence,
                event.severity.value if hasattr(event.severity, "value") else event.severity,
                event.change_mask_b64,
                int(event.optical_evidence),
                int(event.sar_evidence),
                event.summary,
                json.dumps(event.metadata),
                event.created_at,
            ),
        )
        conn.commit()
        return event.change_id

    def get_change_events_for_area(self, area_id: str) -> List[ChangeEvent]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM change_events WHERE area_id = ? ORDER BY created_at DESC",
            (area_id,),
        ).fetchall()
        return [self._row_to_change_event(r) for r in rows]

    def get_change_event(self, change_id: str) -> Optional[ChangeEvent]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM change_events WHERE change_id = ?", (change_id,)
        ).fetchone()
        return self._row_to_change_event(row) if row else None

    # ── Alerts ───────────────────────────────────────────────

    def store_alert(self, alert: Alert) -> str:
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO alerts
               (alert_id, change_event_id, area_id, alert_type, title, description,
                severity, confidence, status, observation_interval, affected_count,
                affected_area_sqm, optical_evidence, sar_evidence, user_feedback,
                reviewed_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                alert.alert_id,
                alert.change_event_id,
                alert.area_id,
                alert.alert_type,
                alert.title,
                alert.description,
                alert.severity.value if hasattr(alert.severity, "value") else alert.severity,
                alert.confidence,
                alert.status.value if hasattr(alert.status, "value") else alert.status,
                alert.observation_interval,
                alert.affected_count,
                alert.affected_area_sqm,
                int(alert.optical_evidence),
                int(alert.sar_evidence),
                alert.user_feedback,
                alert.reviewed_at,
                alert.created_at,
            ),
        )
        conn.commit()
        return alert.alert_id

    def get_pending_alerts(self) -> List[Alert]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM alerts WHERE status = 'pending' ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_alert(r) for r in rows]

    def get_all_alerts(self) -> List[Alert]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM alerts ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_alert(r) for r in rows]

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM alerts WHERE alert_id = ?", (alert_id,)
        ).fetchone()
        return self._row_to_alert(row) if row else None

    def update_alert_status(
        self, alert_id: str, status: AlertStatus, feedback: Optional[str] = None
    ) -> bool:
        from datetime import datetime, timezone
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE alerts SET status = ?, user_feedback = ?, reviewed_at = ? WHERE alert_id = ?",
            (status.value, feedback, now, alert_id),
        )
        conn.commit()
        return conn.total_changes > 0

    # ── IoU helper ───────────────────────────────────────────

    @staticmethod
    def _calculate_iou(
        a_min_x: float, a_min_y: float, a_max_x: float, a_max_y: float,
        b_min_x: float, b_min_y: float, b_max_x: float, b_max_y: float,
    ) -> float:
        inter_min_x = max(a_min_x, b_min_x)
        inter_min_y = max(a_min_y, b_min_y)
        inter_max_x = min(a_max_x, b_max_x)
        inter_max_y = min(a_max_y, b_max_y)
        if inter_max_x <= inter_min_x or inter_max_y <= inter_min_y:
            return 0.0
        inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
        area_a = (a_max_x - a_min_x) * (a_max_y - a_min_y)
        area_b = (b_max_x - b_min_x) * (b_max_y - b_min_y)
        union_area = area_a + area_b - inter_area
        return float(inter_area / union_area) if union_area > 0 else 0.0

    # ── Row → Model converters ───────────────────────────────

    @staticmethod
    def _row_to_area(row: sqlite3.Row) -> Area:
        return Area(
            area_id=row["area_id"],
            geometry=json.loads(row["geometry"]) if row["geometry"] else None,
            centroid=json.loads(row["centroid"]) if row["centroid"] else None,
            bounds=json.loads(row["bounds"]) if row["bounds"] else None,
            location_name=row["location_name"],
            crs=row["crs"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            observation_count=row["observation_count"] or 0,
            created_at=row["created_at"] or "",
            updated_at=row["updated_at"] or "",
        )

    @staticmethod
    def _row_to_observation(row: sqlite3.Row) -> Observation:
        return Observation(
            observation_id=row["observation_id"],
            area_id=row["area_id"],
            acquisition_date=row["acquisition_date"],
            sensor=row["sensor"],
            platform=row["platform"],
            resolution_m=row["resolution_m"],
            image_path=row["image_path"],
            width=row["width"],
            height=row["height"],
            band_count=row["band_count"],
            crs=row["crs"],
            bounds=json.loads(row["bounds"]) if row["bounds"] else None,
            detected_objects_count=row["detected_objects_count"] or 0,
            cloud_cover_pct=row["cloud_cover_pct"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=row["created_at"] or "",
        )

    @staticmethod
    def _row_to_detected_object(row: sqlite3.Row) -> DetectedObject:
        return DetectedObject(
            object_id=row["object_id"],
            observation_id=row["observation_id"],
            class_name=row["class_name"],
            confidence=row["confidence"] or 0.0,
            bbox=json.loads(row["bbox"]) if row["bbox"] else None,
            geo_bbox=json.loads(row["geo_bbox"]) if row["geo_bbox"] else None,
            geometry=json.loads(row["geometry"]) if row["geometry"] else None,
            area_sqm=row["area_sqm"],
            centroid=json.loads(row["centroid"]) if row["centroid"] else None,
            attributes=json.loads(row["attributes"]) if row["attributes"] else {},
        )

    @staticmethod
    def _row_to_change_event(row: sqlite3.Row) -> ChangeEvent:
        from .models import ChangeCategory, ObjectMatch, Severity
        matches_raw = json.loads(row["object_matches"]) if row["object_matches"] else []
        matches = [ObjectMatch(**m) for m in matches_raw]
        return ChangeEvent(
            change_id=row["change_id"],
            area_id=row["area_id"],
            old_observation_id=row["old_observation_id"],
            new_observation_id=row["new_observation_id"],
            change_type=ChangeCategory(row["change_type"]) if row["change_type"] else ChangeCategory.GENERAL,
            object_matches=matches,
            new_objects_count=row["new_objects_count"] or 0,
            removed_objects_count=row["removed_objects_count"] or 0,
            modified_objects_count=row["modified_objects_count"] or 0,
            unchanged_objects_count=row["unchanged_objects_count"] or 0,
            affected_area_sqm=row["affected_area_sqm"],
            change_percentage=row["change_percentage"],
            confidence=row["confidence"] or 0.0,
            severity=Severity(row["severity"]) if row["severity"] else Severity.LOW,
            change_mask_b64=row["change_mask_b64"],
            optical_evidence=bool(row["optical_evidence"]),
            sar_evidence=bool(row["sar_evidence"]),
            summary=row["summary"] or "",
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=row["created_at"] or "",
        )

    @staticmethod
    def _row_to_alert(row: sqlite3.Row) -> Alert:
        from .models import AlertStatus, Severity
        return Alert(
            alert_id=row["alert_id"],
            change_event_id=row["change_event_id"],
            area_id=row["area_id"],
            alert_type=row["alert_type"] or "",
            title=row["title"] or "",
            description=row["description"] or "",
            severity=Severity(row["severity"]) if row["severity"] else Severity.MEDIUM,
            confidence=row["confidence"] or 0.0,
            status=AlertStatus(row["status"]) if row["status"] else AlertStatus.PENDING,
            observation_interval=row["observation_interval"],
            affected_count=row["affected_count"] or 0,
            affected_area_sqm=row["affected_area_sqm"],
            optical_evidence=bool(row["optical_evidence"]),
            sar_evidence=bool(row["sar_evidence"]),
            user_feedback=row["user_feedback"],
            reviewed_at=row["reviewed_at"],
            created_at=row["created_at"] or "",
        )
