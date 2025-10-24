import sqlite3
import time
from threading import Lock
from typing import Dict, Any
import os


class MetricsCollector:
    """Simple metrics collector that stores recent metrics in-memory and persists to SQLite."""

    def __init__(self, db_path: str = "./monitoring_metrics.db"):
        self.db_path = db_path
        self._lock = Lock()
        self._metrics = []
        self._ensure_db()

    def _ensure_db(self):
        os.makedirs(os.path.dirname(self.db_path) or '.', exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                ts REAL,
                name TEXT,
                value REAL
            )
            """
        )
        conn.commit()
        conn.close()

    def record(self, name: str, value: float):
        ts = time.time()
        with self._lock:
            self._metrics.append({"ts": ts, "name": name, "value": float(value)})
        # persist immediately (small writes)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO metrics (ts, name, value) VALUES (?, ?, ?)", (ts, name, float(value)))
        conn.commit()
        conn.close()

    def latest_snapshot(self):
        with self._lock:
            return list(self._metrics[-20:])
