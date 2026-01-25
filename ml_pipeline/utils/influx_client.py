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
        Escape Zabbix keys for Flux queries.
        
        CRITICAL: In Flux, the key value in filter is a STRING, not a regex!
        We only need to escape the double-quote character itself.
        
        Examples:
        - proc.num[] → proc.num[] (no escaping needed)
        - net.if.in["{GUID}"] → net.if.in[\"{GUID}\"] (escape quotes only)
        - vfs.fs.size[/,pused] → vfs.fs.size[/,pused] (no escaping)
        """
        # Only escape double quotes (they're inside a Flux string)
        # Backslashes need to be doubled first, then quotes
        escaped = key.replace('\\', '\\\\')  # Escape backslashes
        escaped = escaped.replace('"', '\\"')  # Escape quotes
        return escaped
    
    def get_metric_window(self, metric_name: str, window_minutes: int) -> pd.DataFrame:
        """Get last N minutes of data for a metric using 'key' tag"""
        start_time = datetime.utcnow() - pd.Timedelta(minutes=window_minutes)
        
        # FIX #2: Use robust escaping for all keys
        escaped_metric = self.escape_key_for_flux(metric_name)
        
        # Build Flux query with proper escaping
        flux_query = f'''
from(bucket: "{self.bucket}")
  |> range(start: {start_time.isoformat()}Z)
  |> filter(fn: (r) => r._measurement == "zabbix_metric")
  |> filter(fn: (r) => r._field == "value")
  |> filter(fn: (r) => r.key == "{escaped_metric}")
  |> sort(columns: ["_time"])
'''
        
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
                # Don't spam logs for metrics with no data
                return None
            
            df = pd.DataFrame(rows).set_index('_time')
            df.index = pd.to_datetime(df.index)
            return df
            
        except Exception as e:
            print(f"[INFLUX ERROR] {metric_name}: {str(e)}")
            # Print the query for debugging
            print(f"[DEBUG] Query was: {flux_query[:200]}...")
            return None
    
    def write_anomaly(self, metric_name: str, timestamp, score: float, severity: str, features: dict):
        """Write anomaly with severity tagging and feature decomposition"""
        
        point = Point("anomalies") \
            .tag("metric_name", metric_name) \
            .tag("severity", severity) \
            .tag("source", "ml_pipeline") \
            .field("anomaly_score", score) \
            .time(timestamp)
        
        # Add key features for Grafana visualization
        # Only include numeric, non-NaN values
        for key, value in features.items():
            if isinstance(value, (int, float)) and not pd.isna(value):
                # Limit to 10 most important features to avoid bloat
                if key in ['value', 'value_diff', 'value_pct_change', 'rolling_mean', 
                          'rolling_std', 'ewm', 'hour', 'weekday']:
                    point.field(f"feat_{key}", float(value))
        
        self.write_batch.append(point)
        
        # Auto-flush if batch is full
        if len(self.write_batch) >= self.batch_size:
            self.flush_batch()
    
    def flush_batch(self):
        """Immediately write all pending points to InfluxDB"""
        if not self.write_batch:
            return
        
        try:
            self.write_api.write(bucket=self.bucket, org=self.org, record=self.write_batch)
            print(f"[INFLUX] ✅ Flushed {len(self.write_batch)} anomaly records")
            self.write_batch = []
            self.last_flush = datetime.utcnow()
        except Exception as e:
            print(f"[INFLUX ERROR] Batch write failed: {e}")
            # Try writing individually as fallback
            for point in self.write_batch:
                try:
                    self.write_api.write(bucket=self.bucket, org=self.org, record=point)
                except Exception as e2:
                    print(f"[INFLUX ERROR] Individual write failed: {e2}")
            self.write_batch = []
    
    def __del__(self):
        """Cleanup: flush remaining batch and close connection"""
        if hasattr(self, 'write_batch') and self.write_batch:
            self.flush_batch()
        if hasattr(self, 'client'):
            self.client.close()
