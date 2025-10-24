"""
Advanced Monitoring with Postgres Integration (Simplified)
"""
import os
import time
import psycopg2
import pandas as pd
from typing import Dict, Any, List, Optional
import sqlite3
from threading import Lock


class AdvancedMetricsCollector:
    """Advanced metrics collector with Evidently AI and Postgres support"""
    
    def __init__(self, 
                 postgres_url: Optional[str] = None,
                 sqlite_fallback: bool = True,
                 sqlite_path: str = "./monitoring_metrics.db"):
        self.postgres_url = postgres_url
        self.sqlite_fallback = sqlite_fallback
        self.sqlite_path = sqlite_path
        self._lock = Lock()
        self._metrics = []
        
        # Initialize database
        self._init_database()
        
        # Initialize Evidently
        self._init_evidently()
    
    def _init_database(self):
        """Initialize Postgres or SQLite database"""
        if self.postgres_url:
            try:
                self.conn = psycopg2.connect(self.postgres_url)
                self._create_postgres_tables()
                print("SUCCESS: Connected to Postgres database")
                return
            except Exception as e:
                print(f"WARNING: Postgres connection failed: {e}")
                if not self.sqlite_fallback:
                    raise
        
        # Fallback to SQLite
        if self.sqlite_fallback:
            os.makedirs(os.path.dirname(self.sqlite_path) or '.', exist_ok=True)
            self.conn = sqlite3.connect(self.sqlite_path)
            self._create_sqlite_tables()
            print("SUCCESS: Using SQLite database for monitoring")
    
    def _create_postgres_tables(self):
        """Create Postgres tables for monitoring (using SQLite syntax)"""
        cursor = self.conn.cursor()
        
        # Metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                name TEXT,
                value REAL,
                metadata TEXT
            )
        """)
        
        # Model performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                model_name TEXT,
                accuracy REAL,
                latency_ms REAL,
                confidence REAL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost_usd REAL
            )
        """)
        
        # Data drift table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_drift (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                feature_name TEXT,
                drift_score REAL,
                p_value REAL,
                threshold REAL
            )
        """)
        
        self.conn.commit()
    
    def _create_sqlite_tables(self):
        """Create SQLite tables for monitoring"""
        cursor = self.conn.cursor()
        
        # Metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                name TEXT,
                value REAL,
                metadata TEXT
            )
        """)
        
        # Model performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                model_name TEXT,
                accuracy REAL,
                latency_ms REAL,
                confidence REAL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost_usd REAL
            )
        """)
        
        # Data drift table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_drift (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                feature_name TEXT,
                drift_score REAL,
                p_value REAL,
                threshold REAL
            )
        """)
        
        self.conn.commit()
    
    def _init_evidently(self):
        """Initialize Evidently AI components (simplified)"""
        # Simplified implementation without complex Evidently imports
        self.column_mapping = None
        self.report = None
        print("SUCCESS: Simplified monitoring initialized")
    
    def record_metric(self, name: str, value: float, metadata: Optional[Dict] = None):
        """Record a metric with timestamp"""
        timestamp = time.time()
        
        with self._lock:
            self._metrics.append({
                "timestamp": timestamp,
                "name": name,
                "value": value,
                "metadata": metadata or {}
            })
        
        # Persist to database
        self._persist_metric(timestamp, name, value, metadata)
    
    def record_model_performance(self, 
                               model_name: str,
                               accuracy: float,
                               latency_ms: float,
                               confidence: float,
                               input_tokens: int = 0,
                               output_tokens: int = 0,
                               cost_usd: float = 0.0):
        """Record model performance metrics"""
        timestamp = time.time()
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO model_performance 
            (timestamp, model_name, accuracy, latency_ms, confidence, input_tokens, output_tokens, cost_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, model_name, accuracy, latency_ms, confidence, input_tokens, output_tokens, cost_usd))
        self.conn.commit()
    
    def record_data_drift(self, feature_name: str, drift_score: float, p_value: float, threshold: float = 0.05):
        """Record data drift metrics"""
        timestamp = time.time()
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO data_drift 
            (timestamp, feature_name, drift_score, p_value, threshold)
            VALUES (?, ?, ?, ?, ?)
        """, (timestamp, feature_name, drift_score, p_value, threshold))
        self.conn.commit()
    
    def _persist_metric(self, timestamp: float, name: str, value: float, metadata: Optional[Dict]):
        """Persist metric to database"""
        try:
            cursor = self.conn.cursor()
            metadata_str = str(metadata) if metadata else None
            
            cursor.execute("""
                INSERT INTO metrics (timestamp, name, value, metadata)
                VALUES (?, ?, ?, ?)
            """, (timestamp, name, value, metadata_str))
            self.conn.commit()
        except Exception as e:
            print(f"Failed to persist metric: {e}")
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for the last N hours"""
        cursor = self.conn.cursor()
        cutoff_time = time.time() - (hours * 3600)
        
        # Get basic metrics
        cursor.execute("""
            SELECT name, AVG(value), MIN(value), MAX(value), COUNT(*)
            FROM metrics 
            WHERE timestamp > ?
            GROUP BY name
        """, (cutoff_time,))
        
        metrics_summary = {}
        for row in cursor.fetchall():
            name, avg_val, min_val, max_val, count = row
            metrics_summary[name] = {
                "average": avg_val,
                "min": min_val,
                "max": max_val,
                "count": count
            }
        
        # Get model performance
        cursor.execute("""
            SELECT model_name, AVG(accuracy), AVG(latency_ms), AVG(confidence), COUNT(*)
            FROM model_performance 
            WHERE timestamp > ?
            GROUP BY model_name
        """, (cutoff_time,))
        
        model_performance = {}
        for row in cursor.fetchall():
            model_name, avg_acc, avg_latency, avg_conf, count = row
            model_performance[model_name] = {
                "accuracy": avg_acc,
                "latency_ms": avg_latency,
                "confidence": avg_conf,
                "requests": count
            }
        
        return {
            "metrics": metrics_summary,
            "model_performance": model_performance,
            "period_hours": hours
        }
    
    def generate_drift_report(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate drift report (simplified)"""
        try:
            # Simplified drift detection
            drift_scores = {}
            for column in reference_data.columns:
                if reference_data[column].dtype in ['float64', 'int64']:
                    ref_mean = reference_data[column].mean()
                    curr_mean = current_data[column].mean()
                    drift_score = abs(ref_mean - curr_mean) / ref_mean if ref_mean != 0 else 0
                    drift_scores[column] = drift_score
            
            return {
                "drift_scores": drift_scores,
                "overall_drift": max(drift_scores.values()) if drift_scores else 0,
                "status": "simplified_drift_detection"
            }
        except Exception as e:
            print(f"Failed to generate drift report: {e}")
            return {"error": str(e)}
    
    def latest_snapshot(self) -> List[Dict[str, Any]]:
        """Get latest metrics snapshot"""
        with self._lock:
            return list(self._metrics[-20:])
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()


# Global instance
advanced_metrics = AdvancedMetricsCollector()
