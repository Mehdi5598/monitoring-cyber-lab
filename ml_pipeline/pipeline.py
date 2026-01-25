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
        self.baseline_established = {}  # FIX #1: Track baseline per metric
        self.baseline_sample_count = {}  # Track samples for 500-sample retraining
        
        self.detection_stats = {
            "total_processed": 0,
            "total_anomalies": 0,
            "alerts_raised": 0,
            "alerts_suppressed": 0,
            "by_severity": {"normal": 0, "info": 0, "low": 0, "medium": 0, "high": 0, "critical": 0}
        }
        
        # Schedule map
        self.schedule_map = {
            'every_30s': 30,
            'every_1m': 60,
            'every_2m': 120,
            'every_5m': 300
        }
        
        # Discord webhook (optional)
        self.discord_webhook = os.getenv("DISCORD_WEBHOOK_URL", None)
        
        self._write_heartbeat()
    
    def _write_heartbeat(self):
        """Write heartbeat file for Docker healthcheck"""
        try:
            with open("/app/logs/heartbeat.txt", "w") as f:
                f.write(datetime.utcnow().isoformat())
        except Exception as e:
            logger.warning(f"Failed to write heartbeat: {e}")
    
    def process_metric(self, metric_config: dict):
        """Process metric with improved baseline handling"""
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
            
            # FIX #1: Improved baseline logic for demo
            near_zero = (df['value'].abs() < 0.01).sum()
            zero_ratio = near_zero / len(df)
            
            # If metric hasn't been seen before and has too many zeros, start building baseline
            if metric_name not in self.baseline_established:
                if zero_ratio >= 0.7:
                    logger.info(f"🔧 BASELINE: {metric_name} - Building baseline ({near_zero}/{len(df)} near-zero)")
                    # Mark as seen but not ready
                    self.baseline_established[metric_name] = "building"
                    return
                else:
                    # Data looks good, mark as established
                    logger.info(f"✅ BASELINE: {metric_name} - Established immediately")
                    self.baseline_established[metric_name] = "ready"
            
            # If we're still building baseline, check if we have enough varied data now
            elif self.baseline_established[metric_name] == "building":
                if zero_ratio < 0.5:  # Less strict after first check
                    logger.info(f"✅ BASELINE: {metric_name} - Now ready (varied data detected)")
                    self.baseline_established[metric_name] = "ready"
                else:
                    logger.debug(f"⏳ BASELINE: {metric_name} - Still building...")
                    return
            
            logger.debug(f"Processing {metric_name}: Got {len(df)} data points")
            
            # 2. Engineer features
            features = self.engineer.engineer(df, metric_name)
            if features is None or len(features) == 0:
                logger.debug(f"SKIP {metric_name}: Feature engineering failed")
                return
            
            # Track samples for 500-sample retraining strategy
            if metric_name not in self.baseline_sample_count:
                self.baseline_sample_count[metric_name] = 0
            self.baseline_sample_count[metric_name] += len(features)
            
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
                self.detection_stats["alerts_suppressed"] += 1
                return
        
        self.last_alert[suppression_key] = datetime.utcnow()

        # =========================
        # 2. ANOMALY SCORE CALCULATION (FIX #3)
        # =========================
        pct_change = row.get("value_pct_change")
        if pd.notna(pct_change) and abs(pct_change) > 0.001:
            raw_score = abs(pct_change) * 2.0
        else:
            value_diff = row.get("value_diff")
            if pd.notna(value_diff) and abs(value_diff) > 0.001:
                raw_score = abs(value_diff) * 1.5
            else:
                raw_score = 1.0

        # FIX #3: Logarithmic scaling instead of hard cap
        # This preserves extreme events while preventing infinity
        anomaly_score = np.log1p(raw_score) * 5.0
        anomaly_score = min(anomaly_score, 100.0)  # Soft cap at 100

        # =========================
        # 3. SEVERITY CLASSIFICATION
        # =========================
        if anomaly_score >= 20.0:
            severity = "critical"
        elif anomaly_score >= 12.0:
            severity = "high"
        elif anomaly_score >= 6.0:
            severity = "medium"
        elif anomaly_score >= 3.0:
            severity = "low"
        else:
            severity = "info"

        # Update stats
        self.detection_stats["alerts_raised"] += 1
        self.detection_stats["by_severity"][severity] += 1

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
            f"🚨 [{severity.upper()}] {metric_name} at {timestamp} "
            f"(score={anomaly_score:.2f}, raw={raw_score:.2f})"
        )
        
        # =========================
        # 6. DISCORD NOTIFICATION (HIGH/CRITICAL ONLY)
        # =========================
        logger.debug(f"Discord check: severity={severity}, webhook_set={bool(self.discord_webhook)}")
        if severity in ['high', 'critical']:
            if self.discord_webhook:
                logger.info(f"🔔 Triggering Discord alert for {metric_name}")
                self._send_discord_alert(metric_name, severity, anomaly_score, timestamp)
            else:
                logger.warning(f"⚠️ HIGH/CRITICAL alert but Discord webhook not configured!")
    
    def _send_discord_alert(self, metric_name: str, severity: str, score: float, timestamp):
        """Send alert to Discord webhook"""
        if not self.discord_webhook:
            logger.debug("Discord webhook not configured, skipping alert")
            return
            
        try:
            import requests
            
            logger.info(f"🔔 Attempting Discord alert for {metric_name} ({severity})")
            
            # Color coding
            colors = {
                'critical': 0xFF0000,  # Red
                'high': 0xFF6600,      # Orange
                'medium': 0xFFCC00,    # Yellow
                'low': 0x00FF00        # Green
            }
            
            # Format timestamp to be more readable
            time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S UTC') if hasattr(timestamp, 'strftime') else str(timestamp)
            
            embed = {
                "embeds": [{
                    "title": f"🚨 Security Alert: {severity.upper()}",
                    "description": f"**Metric**: `{metric_name}`\n**Score**: `{score:.2f}`\n**Time**: `{time_str}`",
                    "color": colors.get(severity, 0x808080),
                    "footer": {"text": "ML Security Pipeline"},
                    "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp)
                }]
            }
            
            logger.debug(f"Sending to webhook: {self.discord_webhook[:50]}...")
            response = requests.post(self.discord_webhook, json=embed, timeout=5)
            
            if response.status_code == 204:
                logger.info(f"✅ Discord alert sent successfully for {metric_name}")
            else:
                logger.error(f"❌ Discord webhook returned {response.status_code}: {response.text}")
                
        except ImportError:
            logger.error("❌ 'requests' module not found - cannot send Discord alerts")
        except requests.exceptions.Timeout:
            logger.error(f"❌ Discord webhook timeout for {metric_name}")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Discord webhook request failed: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to send Discord alert: {e}", exc_info=True)
    
    def log_stats(self):
        """Log pipeline statistics"""
        total = self.detection_stats["total_processed"]
        anomalies = self.detection_stats["total_anomalies"]
        alerts = self.detection_stats["alerts_raised"]
        suppressed = self.detection_stats["alerts_suppressed"]
        
        logger.info(
            f"\n{'='*50}\n"
            f"📊 PIPELINE STATISTICS\n"
            f"{'='*50}\n"
            f"  Metrics Processed:  {total}\n"
            f"  Anomalies Found:    {anomalies}\n"
            f"  Alerts Raised:      {alerts}\n"
            f"  Alerts Suppressed:  {suppressed}\n"
            f"\n  🎯 By Severity:\n"
            f"    🔴 Critical:  {self.detection_stats['by_severity']['critical']}\n"
            f"    🟠 High:      {self.detection_stats['by_severity']['high']}\n"
            f"    🟡 Medium:    {self.detection_stats['by_severity']['medium']}\n"
            f"    🟢 Low:       {self.detection_stats['by_severity']['low']}\n"
            f"{'='*50}\n"
        )
    
    def start(self):
        """Start pipeline"""
        metrics = self.orchestrator.config.get('metrics', [])
        logger.info(f"\n🚀 Starting ML Pipeline with {len(metrics)} metrics\n")
        
        if self.discord_webhook:
            logger.info("📢 Discord notifications ENABLED")
        
        scheduled_count = 0
        for metric in self.orchestrator.config['metrics']:
            if not metric.get('enabled', True):
                continue
            
            schedule_type = metric['schedule']
            interval = self.schedule_map.get(schedule_type, 30)
            
            schedule.every(interval).seconds.do(self.process_metric, metric)
            scheduled_count += 1
        
        schedule.every(5).minutes.do(self.log_stats)
        
        logger.info(f"✅ Pipeline running with {scheduled_count} active metrics\n")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n🛑 Pipeline stopped by user")
        except Exception as e:
            logger.error(f"💥 Pipeline crashed: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    pipeline = MLPipeline()
    pipeline.start()
