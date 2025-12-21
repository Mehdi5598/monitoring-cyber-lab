# ml_pipeline/utils/influx_client.py
import os
import pandas as pd
from datetime import datetime
from influxdb_client import InfluxDBClient, Point

class InfluxClient:
    def __init__(self, batch_size: int = 50):
        self.url = os.environ.get("INFLUX_URL", "http://influxdb:8086")
        self.token = os.environ.get("INFLUX_TOKEN", "")
        self.org = os.environ.get("INFLUX_INIT_ORG", "myorg")
        self.bucket = os.environ.get("INFLUX_BUCKET", "metrics")
        
        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        self.query_api = self.client.query_api()
        self.write_api = self.client.write_api()
        
        # Batch writing setup
        self.batch_size = batch_size
        self.write_batch = []
        self.last_flush = datetime.utcnow()
    
    def escape_key_for_flux(self, key: str) -> str:
        """
        Escape special characters in Zabbix keys for Flux queries.
        Critical: Changes [" to [\" and "] to \"]
        """
        if key.startswith('net.if.') and '["' in key and '"]' in key:
            return key.replace('["', '[\\"').replace('",', '\\",').replace('"]', '\\"]')
        return key
    
    def get_metric_window(self, metric_name: str, window_minutes: int) -> pd.DataFrame:
        """Get last N minutes of data for a metric using 'key' tag"""
        start_time = datetime.utcnow() - pd.Timedelta(minutes=window_minutes)
        
        escaped_metric = self.escape_key_for_flux(metric_name)
        
        flux_query = 'from(bucket: "{}") |> range(start: {}Z) |> filter(fn: (r) => r._measurement == "zabbix_metric") |> filter(fn: (r) => r._field == "value") |> filter(fn: (r) => r.key == "{}") |> sort(columns: ["_time"])'.format(
            self.bucket, start_time.isoformat(), escaped_metric)
        
        try:
            tables = self.query_api.query(flux_query, org=self.org)
            rows = []
            for table in tables:
                for record in table.records:
                    rows.append({
                        '_time': record.get_time(),
                        'value': float(record.get_value())
                    })
            
            if not rows:
                return None
            
            df = pd.DataFrame(rows).set_index('_time')
            df.index = pd.to_datetime(df.index)
            return df
            
        except Exception as e:
            print(f"[INFLUX ERROR] {metric_name}: {str(e)}")
            return None
    
    def write_anomaly(self, metric_name: str, timestamp, score: float, severity: str, features: dict):
        """Write anomaly with severity tagging and feature decomposition"""
        
        point = Point("anomalies") \
            .tag("metric_name", metric_name) \
            .tag("severity", severity) \
            .tag("source", "ml_pipeline") \
            .field("anomaly_score", score) \
            .time(timestamp)
        
        # Add key features for debugging
        for key, value in features.items():
            if isinstance(value, (int, float)) and not pd.isna(value):
                point.field(f"feat_{key}", value)
        
        self.write_batch.append(point)
        
        if len(self.write_batch) >= self.batch_size:
            self.flush_batch()
    
    def flush_batch(self):
        """Immediately write all pending points to InfluxDB"""
        if not self.write_batch:
            return
        
        try:
            self.write_api.write(bucket=self.bucket, org=self.org, record=self.write_batch)
            self.write_batch = []
            self.last_flush = datetime.utcnow()
        except Exception as e:
            for point in self.write_batch:
                try:
                    self.write_api.write(bucket=self.bucket, org=self.org, record=point)
                except Exception as e2:
                    pass
            self.write_batch = []
    
    def __del__(self):
        if hasattr(self, 'client'):
            self.client.close()
