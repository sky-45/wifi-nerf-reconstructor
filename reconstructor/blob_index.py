"""
Incremental SQLite Index for Telemetry Batches.
Eliminates repeated gunzip + NDJSON parsing of tens of thousands of historical
batches across restarts, while supporting explicit full rebuilds (--clean / /api/reload).
"""

import json
import os
import sqlite3
import zlib
from typing import Any, Dict, List, Optional, Tuple


class TelemetryIndex:
    """Thread-safe lightweight SQLite cache for decoded telemetry batch events."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS batch_cache (
        object_key TEXT PRIMARY KEY,
        size_bytes INTEGER NOT NULL,
        device_id TEXT,
        env_id TEXT,
        tier TEXT,
        event_count INTEGER NOT NULL DEFAULT 0,
        min_ts_ns INTEGER NOT NULL DEFAULT 0,
        max_ts_ns INTEGER NOT NULL DEFAULT 0,
        is_transit INTEGER NOT NULL DEFAULT 0,
        payload_zlib BLOB NOT NULL,
        indexed_at REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_batch_env ON batch_cache(env_id, device_id);
    """

    def __init__(self, db_path: str = ".cache/blobs/telemetry_index.sqlite"):
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(self.SCHEMA)

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM batch_cache;")

    def get_cached_batches(
        self, object_keys: List[str]
    ) -> Dict[str, Tuple[int, List[Dict[str, Any]]]]:
        """
        Fetches already-decoded event lists for the given object_keys.
        Returns {object_key: (size_bytes, events)}.
        """
        if not object_keys:
            return {}
        out: Dict[str, Tuple[int, List[Dict[str, Any]]]] = {}
        chunk_size = 900
        with self._connect() as conn:
            for i in range(0, len(object_keys), chunk_size):
                chunk = object_keys[i : i + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT object_key, size_bytes, payload_zlib FROM batch_cache WHERE object_key IN ({placeholders})",
                    chunk,
                ).fetchall()
                for key, sz, blob in rows:
                    try:
                        events = json.loads(zlib.decompress(blob).decode("utf-8"))
                        out[key] = (int(sz), events)
                    except Exception:
                        continue
        return out

    def put_batches(
        self,
        records: List[Tuple[str, int, str, str, str, int, int, int, int, List[Dict[str, Any]], float]],
    ) -> None:
        """
        Upserts decoded batch records:
        (object_key, size_bytes, device_id, env_id, tier, event_count, min_ts_ns, max_ts_ns, is_transit, events, indexed_at)
        """
        if not records:
            return
        rows = []
        for (
            key,
            sz,
            dev,
            env,
            tier,
            cnt,
            min_ts,
            max_ts,
            is_tr,
            events,
            ts_now,
        ) in records:
            blob = zlib.compress(
                json.dumps(events, separators=(",", ":")).encode("utf-8"), level=1
            )
            rows.append(
                (key, sz, dev, env, tier, cnt, min_ts, max_ts, is_tr, blob, ts_now)
            )
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO batch_cache
                (object_key, size_bytes, device_id, env_id, tier, event_count,
                 min_ts_ns, max_ts_ns, is_transit, payload_zlib, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(event_count), 0) FROM batch_cache"
            ).fetchone()
            return {
                "cached_batches": int(row[0]) if row else 0,
                "cached_events": int(row[1]) if row else 0,
                "db_path": self.db_path,
            }
