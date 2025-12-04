# ml_pipeline/pipeline.py
import os
import sys
import time
import schedule
from datetime import datetime, timedelta
import pandas as pd

# Add utils/strategies to path
sys.path.append('/app/utils')
sys.path.append('/app/strategies')

from utils.influx_client import InfluxClient
from utils.feature_engineer import FeatureEngineer
from strategies.detector import DetectorOrchestrator

class MLPipeline:
    def __init__(self):
        self.influx = InfluxClient()
        self.engineer = FeatureEngineer()
        self.orchestrator = DetectorOrchestrator()
        self.last_alert = {}  # Suppression tracking
        
        # Load scheduling config
        self.schedule_map = {
            'every_30s': 30,
            'every_1m': 60,
            'every_2m': 120,
            'every_5m': 300
        }
    
    def process_metric(self, metric_config: dict):
        """Process a single metric"""
        metric_name = metric_config['name']
        print(f"\n[{datetime.now()}] Processing: {metric_name}")
        
        # 1. Query data
        df = self.influx.get_metric_window(metric_name, metric_config['window_minutes'])
        if df is None or len(df) < 5:
            print(f"  [SKIP] Insufficient data: {len(df) if df else 0} points")
            return
        
        # 2. Engineer features
        features = self.engineer.engineer(df, metric_name)
        if features is None:
            return
        
        # 3. Detect anomalies
        predictions = self.orchestrator.detect(metric_name, features)
        if predictions is None:
            return
        
        # 4. Process results
        anomalies = features[predictions == -1]
        print(f"  → Window: {len(features)} points, Anomalies: {len(anomalies)}")
        
        for idx, row in anomalies.iterrows():
            self._handle_anomaly(metric_name, idx, row, metric_config)
    
    def _handle_anomaly(self, metric_name: str, timestamp, row: pd.Series, config: dict):
        """Write anomaly and check suppression"""
        
        # Suppression logic
        suppression_key = f"{metric_name}_{timestamp.floor('5min')}"
        if suppression_key in self.last_alert:
            if datetime.utcnow() - self.last_alert[suppression_key] < timedelta(minutes=5):
                return
        
        self.last_alert[suppression_key] = datetime.utcnow()
        
        # Write to InfluxDB
        anomaly_score = abs(row.get('value_pct_change', 1.0))
        self.influx.write_anomaly(metric_name, timestamp, anomaly_score, row.to_dict())
    
    def start(self):
        """Schedule all metrics"""
        print("🚀 ML Pipeline Starting...")
        
        for metric in self.orchestrator.config['metrics']:
            schedule_type = metric['schedule']
            interval = self.schedule_map.get(schedule_type, 30)
            
            schedule.every(interval).seconds.do(self.process_metric, metric)
            print(f"  Scheduled {metric['name']} every {interval}s")
        
        print("\n✅ Pipeline running. Press Ctrl+C to stop.\n")
        
        while True:
            schedule.run_pending()
            time.sleep(1)

if __name__ == "__main__":
    pipeline = MLPipeline()
    try:
        pipeline.start()
    except KeyboardInterrupt:
        print("\n\n🛑 Pipeline stopped by user")
    except Exception as e:
        print(f"\n💥 Pipeline crashed: {e}")
        import traceback
        traceback.print_exc()
