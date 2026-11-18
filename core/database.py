"""
Database Module
Persists driver alerts, timestamped events, vehicle telemetry, and risk states
into an SQLite database for auditing and dashboard reporting.
"""

import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import config

class AlertDatabase:
    """
    SQLite database interface for storing and retrieving safety alert events.
    """
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = str(db_path or config.DATABASE_PATH)
        self._initialize_schema()

    def _get_connection(self):
        return sqlite3.connect(self.db_path, timeout=5.0)

    def _initialize_schema(self):
        """Creates the necessary database tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    speed REAL,
                    speed_limit REAL,
                    risk_level TEXT,
                    notes TEXT
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON alerts_history(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON alerts_history(event_type)")
            conn.commit()

    def log_event(self, event_type: str, severity: str = "CAUTION",
                  speed: float = 0.0, speed_limit: float = 60.0,
                  risk_level: str = "LOW", notes: str = "") -> int:
        """
        Records an alert event with current timestamp and telemetry.
        Returns the inserted row ID.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO alerts_history (timestamp, event_type, severity, speed, speed_limit, risk_level, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, event_type.upper(), severity.upper(), round(speed, 1), round(speed_limit, 1), risk_level.upper(), notes))
            conn.commit()
            return cursor.lastrowid

    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves the most recent alerts ordered by newest first."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, event_type, severity, speed, speed_limit, risk_level, notes
                FROM alerts_history
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_summary_counts(self) -> Dict[str, int]:
        """Returns the total number of events per category."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT event_type, COUNT(*) 
                FROM alerts_history 
                GROUP BY event_type
            """)
            results = dict(cursor.fetchall())
            return results

    def clear_history(self):
        """Clears all logged history (useful for reset/demos)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM alerts_history")
            conn.commit()
