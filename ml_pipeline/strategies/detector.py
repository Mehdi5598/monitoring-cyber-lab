# ml_pipeline/strategies/detector.py
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os

class BaseDetector(ABC):
    def __init__(self, config: dict, models_dir: str = "/app/models/saved"):
        self.config = config
        self.metric_name = config['name']
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
    
    @abstractmethod
    def fit(self, X):
        pass
    
    @abstractmethod
    def predict(self, X) -> np.ndarray:
        pass
    
    def get_model_path(self):
        # Sanitize filename for filesystem safety
        safe_name = self.metric_name.replace('.', '_').replace('[', '_').replace(']', '_').replace('"', '_').replace('\\', '_').replace('/', '_')
        return os.path.join(self.models_dir, f"{safe_name}.pkl")

class IsolationForestDetector(BaseDetector):
    def __init__(self, config: dict):
        super().__init__(config)
        self.model = None
        self.samples_seen = 0
    
    def fit(self, X: pd.DataFrame):
        """Train or retrain Isolation Forest model"""
        if X is None or len(X) < 10:
            print(f"[IF] Insufficient data for {self.metric_name}: {len(X) if X else 0} points")
            return
        
        # Ensure we have numeric data
        if isinstance(X, pd.DataFrame):
            X_numeric = X.select_dtypes(include=[np.number])
        else:
            X_numeric = pd.DataFrame(X)
        
        if X_numeric.shape[1] == 0:
            print(f"[IF] No numeric features for {self.metric_name}")
            return
        
        self.samples_seen += len(X_numeric)
        retrain_threshold = self.config.get('retrain_after', 500)
        
        if self.model is None or self.samples_seen >= retrain_threshold:
            print(f"[IF] Training model for {self.metric_name} on {len(X_numeric)} samples")
            self.model = IsolationForest(
                n_estimators=self.config.get('n_estimators', 100),
                contamination=self.config.get('contamination', 0.01),
                random_state=42,
                max_samples='auto'
            )
            self.model.fit(X_numeric)
            self.samples_seen = 0
            self._save_model()
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict anomalies (1=normal, -1=anomaly)"""
        if X is None or len(X) == 0:
            return np.array([])
        
        if self.model is None:
            print(f"[IF] Model not trained for {self.metric_name}, returning all normal")
            return np.ones(len(X))
        
        # Convert to numeric DataFrame
        if isinstance(X, pd.DataFrame):
            X_numeric = X.select_dtypes(include=[np.number])
        elif isinstance(X, np.ndarray):
            X_numeric = pd.DataFrame(X)
        else:
            X_numeric = pd.DataFrame(X)
        
        if X_numeric.shape[1] == 0:
            return np.ones(len(X))
        
        try:
            predictions = self.model.predict(X_numeric)
            # IsolationForest returns -1 for anomalies, 1 for normal
            # Ensure we return numpy array
            return np.array(predictions)
        except Exception as e:
            print(f"[IF] Prediction error for {self.metric_name}: {e}")
            return np.ones(len(X))
    
    def _save_model(self):
        try:
            path = self.get_model_path()
            joblib.dump(self.model, path)
            print(f"[IF] Model saved to {path}")
        except Exception as e:
            print(f"[IF] Failed to save model: {e}")

class ZScoreBurstDetector(BaseDetector):
    """Detects sudden spikes using Z-score (no training needed)"""
    
    def fit(self, X):
        """Z-score requires no training - this is a no-op"""
        pass
    
    def predict(self, X) -> np.ndarray:
        """
        Detect bursts using Z-score method
        Accepts: pd.DataFrame, np.ndarray, or any array-like
        Returns: array of predictions (1=normal, -1=anomaly)
        """
        if X is None or len(X) == 0:
            return np.array([])
        
        # Extract values from different input types
        try:
            if isinstance(X, pd.DataFrame):
                # Try 'value' column first, then first numeric column
                if 'value' in X.columns:
                    values = X['value'].values
                else:
                    numeric_cols = X.select_dtypes(include=[np.number]).columns
                    values = X[numeric_cols[0]].values if len(numeric_cols) > 0 else np.ones(len(X))
            elif isinstance(X, np.ndarray):
                # Take first column if 2D, or flatten if 1D
                values = X[:, 0] if X.ndim == 2 else X.flatten()
            else:
                # Fallback: try to convert to numpy array
                values = np.array(X)
            
            # Ensure we have numeric data
            if not np.issubdtype(values.dtype, np.number):
                print(f"[ZS] Non-numeric data received for {self.metric_name}")
                return np.ones(len(values))
            
        except Exception as e:
            print(f"[ZS] Failed to extract values for {self.metric_name}: {e}")
            return np.ones(len(X)) if hasattr(X, '__len__') else np.array([1])
        
        # Calculate Z-scores
        mean = values.mean()
        std = values.std()
        
        # If no variance or too few samples, everything is normal
        if std == 0 or len(values) < 3:
            return np.ones(len(values))
        
        z_scores = np.abs((values - mean) / std)
        threshold = self.config.get('zscore_threshold', 3.0)
        
        # Return predictions: -1 for anomaly, 1 for normal
        predictions = np.where(z_scores > threshold, -1, 1)
        
        # Log detection
        num_anomalies = np.sum(predictions == -1)
        if num_anomalies > 0:
            print(f"[ZS] {self.metric_name}: {num_anomalies} anomalies detected (threshold: {threshold:.2f})")
        
        return predictions

class DetectorOrchestrator:
    """Manages all detectors based on configuration"""
    
    def __init__(self, config_path: str = "/app/config/metrics.yaml"):
        import yaml
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load config from {config_path}: {e}")
            self.config = {'metrics': []}
        
        self.detectors = {}
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """Initialize detectors for each configured metric"""
        if 'metrics' not in self.config:
            print("[WARN] No metrics found in configuration")
            return
        
        for metric in self.config['metrics']:
            model_type = metric.get('model')
            metric_name = metric.get('name')
            
            if not metric_name:
                print("[WARN] Skipping metric without name")
                continue
            
            try:
                if model_type == 'isolation_forest':
                    self.detectors[metric_name] = IsolationForestDetector(metric)
                    print(f"[INIT] Created Isolation Forest detector for {metric_name}")
                elif model_type == 'zscore_burst':
                    self.detectors[metric_name] = ZScoreBurstDetector(metric)
                    print(f"[INIT] Created Z-Score detector for {metric_name}")
                else:
                    print(f"[WARN] Unknown model type '{model_type}' for {metric_name}")
            except Exception as e:
                print(f"[ERROR] Failed to create detector for {metric_name}: {e}")
    
    def detect(self, metric_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Detect anomalies for a specific metric
        
        Args:
            metric_name: Name of the metric to detect
            X: DataFrame with engineered features
        
        Returns:
            np.ndarray: Array of predictions (1=normal, -1=anomaly)
        """
        if metric_name not in self.detectors:
            print(f"[WARN] No detector configured for '{metric_name}'")
            return np.ones(len(X)) if X is not None and len(X) > 0 else np.array([])
        
        detector = self.detectors[metric_name]
        
        # Training phase (only for Isolation Forest)
        if isinstance(detector, IsolationForestDetector):
            try:
                detector.fit(X)
            except Exception as e:
                print(f"[ERROR] Training failed for {metric_name}: {e}")
                return np.ones(len(X))
        
        # Prediction phase
        try:
            # Pass the DataFrame directly - let the detector handle conversion
            predictions = detector.predict(X)
            
            # Validate predictions
            if predictions is None or len(predictions) == 0:
                print(f"[WARN] No predictions returned for {metric_name}")
                return np.ones(len(X))
            
            # Ensure correct format
            predictions = np.array(predictions)
            predictions = np.where(predictions == -1, -1, 1)  # Force to ±1
            
            num_anomalies = np.sum(predictions == -1)
            if num_anomalies > 0:
                print(f"[DETECT] {metric_name}: {num_anomalies} anomalies detected")
            
            return predictions
            
        except Exception as e:
            print(f"[ERROR] Detection failed for {metric_name}: {e}")
            import traceback
            traceback.print_exc()
            # Return all normal as fail-safe
            return np.ones(len(X)) if X is not None else np.array([])
