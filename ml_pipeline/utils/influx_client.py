# ml_pipeline/utils/influx_client.py
import os
import pandas as pd
from datetime import datetime
from influxdb_client import InfluxDBClient, Point

class InfluxClient:
    def __init__(self):
        self.url = os.environ.get("INFLUX_URL", "http://influxdb:8086")
        self.token = os.environ.get("INFLUX_TOKEN", "")
        self.org = os.environ.get("INFLUX_INIT_ORG", "myorg")
        self.bucket = os.environ.get("INFLUX_BUCKET", "metrics")
        
        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        self.query_api = self.client.query_api()
        self.write_api = self.client.write_api()
    
    def escape_key_for_flux(self, key: str) -> str:
        """
        Escape special characters in Zabbix keys for Flux queries.
        Critical: Changes [" to [\" and "] to \"]
        """
        # This is the exact pattern that works in your manual query
        if key.startswith('net.if.') and '["{' in key and '}"' in key:
            return key.replace('["', '[\\"').replace('",', '\\",').replace('"]', '\\"]')
        return key
    
    def get_metric_window(self, metric_name: str, window_minutes: int) -> pd.DataFrame:
        """Get last N minutes of data for a metric using 'key' tag"""
        start_time = datetime.utcnow() - pd.Timedelta(minutes=window_minutes)
        
        # Apply Flux-specific escaping
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
                print(f"[INFLUX] No data found for key='{metric_name}'")
                return None
            
            df = pd.DataFrame(rows).set_index('_time')
            df.index = pd.to_datetime(df.index)
            print(f"[INFLUX] Found {len(df)} points for {metric_name}")
            return df
            
        except Exception as e:
            print(f"[INFLUX ERROR] {metric_name}: {str(e)}")
            print(f"[INFLUX ERROR] Query preview: {flux_query[:150]}...")
            return None
    
    def write_anomaly(self, metric_name: str, timestamp, anomaly_score: float, features: dict):
        """Write anomaly detection result with correct structure"""
        point = Point("anomalies") \
            .tag("metric_name", metric_name) \
            .tag("source", "ml_pipeline") \
            .field("anomaly_score", anomaly_score) \
            .time(timestamp)
        
        # Add key features
        for key, value in features.items():
            if isinstance(value, (int, float)) and not pd.isna(value):
                point.field(f"feat_{key}", value)
        
        self.write_api.write(bucket=self.bucket, org=self.org, record=point)
        print(f"[ANOMALY] {metric_name} at {timestamp} (score: {anomaly_score:.3f})")
