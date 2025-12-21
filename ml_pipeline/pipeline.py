# ml_pipeline/pipeline.py
import os
import sys
import time
import schedule
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

sys.path.append('/app/utils')
sys.path.append('/app/strategies')

from utils.influx_client import InfluxClient
from utils.feature_engineer import FeatureEngineer
from strategies.detector import DetectorOrchestrator

# Setup logging
log_dir = "/app/logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MLPipeline")

class MLPipeline:
    def __init__(self):
        logger.info("Initializing ML Pipeline...")
        
        try:
            self.influx = InfluxClient()
            logger.info("InfluxClient initialized")
        except Exception as e:
            logger.error(f"InfluxClient failed: {e}")
            raise
        
        try:
            self.engineer = FeatureEngineer()
            logger.info("FeatureEngineer initialized")
        except Exception as e:
            logger.error(f"FeatureEngineer failed: {e}")
            raise
        
        try:
            self.orchestrator = DetectorOrchestrator()
            logger.info(f"DetectorOrchestrator initialized with {len(self.orchestrator.detectors)} detectors")
        except Exception as e:
            logger.error(f"DetectorOrchestrator failed: {e}")
            raise
        
        self.last_alert = {}
        self.detection_stats = {
            "total_processed": 0,
            "total_anomalies": 0,
            "alerts_raised": 0,
            "alerts_suppressed": 0,
            "by_severity": {"normal": 0, "low": 0, "medium": 0, "high": 0, "critical": 0}
        }
        
        # Schedule map
        self.schedule_map = {
            'every_30s': 30,
            'every_1m': 60,
            'every_2m': 120,
            'every_5m': 300
        }
        
        self._write_heartbeat()
    
    def _write_heartbeat(self):
        """Write heartbeat file for Docker healthcheck"""
        try:
            with open("/app/logs/heartbeat.txt", "w") as f:
                f.write(datetime.utcnow().isoformat())
        except Exception as e:
            logger.warning(f"Failed to write heartbeat: {e}")
    
    def process_metric(self, metric_config: dict):
        """Process metric with zero-validation"""
        metric_name = metric_config['name']
        enabled = metric_config.get('enabled', True)
        
        if not enabled:
            return
        
        try:
            self.detection_stats["total_processed"] += 1
            self._write_heartbeat()
            
            # 1. Query data
            df = self.influx.get_metric_window(metric_name, metric_config['window_minutes'])
            if df is None or len(df) < 5:
                logger.debug(f"SKIP {metric_name}: Insufficient data ({len(df) if df is not None else 0} points)")
                return
            
            # NEW: Skip if too many near-zero values (VM not warmed up)
            near_zero = (df['value'].abs() < 0.01).sum()
            if near_zero / len(df) >= 0.7:
                logger.warning(f"SKIP {metric_name}: {near_zero}/{len(df)} near-zero values - skipping")
                return
            
            logger.debug(f"Processing {metric_name}: Got {len(df)} data points")
            
            # 2. Engineer features
            features = self.engineer.engineer(df, metric_name)
            if features is None or len(features) == 0:
                logger.debug(f"SKIP {metric_name}: Feature engineering failed")
                return
            
            # 3. Detect anomalies
            predictions = self.orchestrator.detect(metric_name, features)
            if predictions is None or len(predictions) == 0:
                logger.debug(f"SKIP {metric_name}: No predictions returned")
                return
            
            # 4. Process results
            anomalies = features[predictions == -1]
            num_anomalies = len(anomalies)
            
            if num_anomalies > 0:
                logger.debug(f"Detected {num_anomalies} anomalies for {metric_name}")
                self.detection_stats["total_anomalies"] += num_anomalies
                
                # Get baseline stats for scoring
                current_value = df['value'].iloc[-1]
                baseline_mean = df['value'].mean()
                baseline_std = df['value'].std() if len(df) > 1 else 1.0
                
                # Process each anomaly
                for idx, row in anomalies.iterrows():
                    self._handle_anomaly(metric_name, idx, row, metric_config, baseline_mean, baseline_std)
            else:
                logger.debug(f"Normal: All {len(features)} points for {metric_name}")
        
        except Exception as e:
            logger.error(f"Error processing {metric_name}: {e}", exc_info=True)
    
    def _handle_anomaly(self, metric_name: str, timestamp, row: pd.Series, config: dict, baseline_mean: float, baseline_std: float):
        """
        Handle anomaly: suppression → scoring → severity → alerting
        """
        
        # =========================
        # 1. ALERT SUPPRESSION
        # =========================
        suppression_key = f"{metric_name}_{timestamp.floor('30s').strftime('%Y-%m-%d %H:%M:%S')}"
        suppression_window = config.get('suppression_window_minutes', 5)
        
        last_alert = self.last_alert.get(suppression_key)
        if last_alert:
            time_diff = datetime.utcnow() - last_alert
            if time_diff < timedelta(minutes=suppression_window):
                logger.debug(f"Suppress {metric_name} - last alert {time_diff.seconds}s ago")
                return
        
        self.last_alert[suppression_key] = datetime.utcnow()

        # =========================
        # 2. ANOMALY SCORE CALCULATION
        # =========================
        pct_change = row.get("value_pct_change")
        if pd.notna(pct_change) and abs(pct_change) > 0.001:
            anomaly_score = abs(pct_change) * 2.0
        else:
            value_diff = row.get("value_diff")
            if pd.notna(value_diff) and abs(value_diff) > 0.001:
                anomaly_score = abs(value_diff) * 1.5
            else:
                anomaly_score = 1.0

        # NEW: Cap extreme scores to prevent false positives
        anomaly_score = min(anomaly_score, 50.0)

        # =========================
        # 3. SEVERITY CLASSIFICATION
        # =========================
        # NEW: Higher thresholds prevent false positives during normal operation
        if anomaly_score >= 15.0:
            severity = "critical"
        elif anomaly_score >= 8.0:
            severity = "high"
        elif anomaly_score >= 3.0:
            severity = "medium"
        else:
            severity = "low"

        # =========================
        # 4. WRITE TO INFLUXDB
        # =========================
        self.influx.write_anomaly(
            metric_name=metric_name,
            timestamp=timestamp,
            score=anomaly_score,
            severity=severity,
            features=row.to_dict()
        )

        # =========================
        # 5. LOG THE ALERT
        # =========================
        logger.warning(
            f"[{severity.upper()}] {metric_name} at {timestamp} "
            f"(score={anomaly_score:.2f})"
        )
    
    def log_stats(self):
        """Log pipeline statistics"""
        total = self.detection_stats["total_processed"]
        anomalies = self.detection_stats["total_anomalies"]
        alerts = self.detection_stats["alerts_raised"]
        suppressed = self.detection_stats["alerts_suppressed"]
        
        logger.info(
            f"\nPIPELINE STATS\n"
            f"  Metrics Processed:  {total}\n"
            f"  Anomalies Found:    {anomalies}\n"
            f"  Alerts Raised:      {alerts}\n"
            f"  Alerts Suppressed:  {suppressed}\n"
            f"  By Severity:\n"
            f"    Critical:        {self.detection_stats['by_severity']['critical']}\n"
            f"    High:            {self.detection_stats['by_severity']['high']}\n"
            f"    Medium:          {self.detection_stats['by_severity']['medium']}\n"
            f"    Low:             {self.detection_stats['by_severity']['low']}\n"
        )
    
    def start(self):
        """Start pipeline"""
        metrics = self.orchestrator.config.get('metrics', [])
        logger.info(f"Starting ML Pipeline with {len(metrics)} metrics")
        
        scheduled_count = 0
        for metric in self.orchestrator.config['metrics']:
            if not metric.get('enabled', True):
                continue
            
            schedule_type = metric['schedule']
            interval = self.schedule_map.get(schedule_type, 30)
            
            schedule.every(interval).seconds.do(self.process_metric, metric)
            scheduled_count += 1
        
        schedule.every(5).minutes.do(self.log_stats)
        
        logger.info(f"Pipeline running with {scheduled_count} metrics\n")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Pipeline stopped by user")
        except Exception as e:
            logger.error(f"Pipeline crashed: {e}", exc_info=True)
            raise
    
    def _write_heartbeat(self):
        try:
            with open("/app/logs/heartbeat.txt", "w") as f:
                f.write(datetime.utcnow().isoformat())
        except:
            pass

if __name__ == "__main__":
    pipeline = MLPipeline()
    pipeline.start()
