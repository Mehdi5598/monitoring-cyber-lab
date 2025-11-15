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
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
    
    @abstractmethod
    def fit(self, X):
        pass
    
    @abstractmethod
    def predict(self, X) -> np.ndarray:
        pass
    
    def get_model_path(self, metric_name: str):
        return os.path.join(self.models_dir, f"{metric_name.replace('.', '_')}.pkl")

class IsolationForestDetector(BaseDetector):
    def __init__(self, config: dict):
        super().__init__(config)
        self.model = None
        self.samples_seen = 0
    
    def fit(self, X):
        """Fit or retrain model"""
        self.samples_seen += len(X)
        retrain_threshold = self.config.get('retrain_after', 500)
        
        if self.model is None or self.samples_seen >= retrain_threshold:
            print(f"[IF] Training model for {self.config['metric_name']}")
            self.model = IsolationForest(
                n_estimators=self.config.get('n_estimators', 100),
                contamination=self.config.get('contamination', 0.01),
                random_state=42
            )
            self.model.fit(X)
            self.samples_seen = 0
            self._save_model()
    
    def predict(self, X) -> np.ndarray:
        if self.model is None:
            return np.ones(len(X))  # All normal if not trained
        
        # -1 = anomaly, 1 = normal (scikit-learn convention)
        return self.model.predict(X.values if isinstance(X, pd.DataFrame) else X)
    
    def _save_model(self):
        path = self.get_model_path(self.config['metric_name'])
        joblib.dump(self.model, path)
        print(f"[IF] Model saved to {path}")

class ZScoreBurstDetector(BaseDetector):
    """For security events - detects sudden spikes"""
    
    def fit(self, X):
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        values = X['value'].values
        mean, std = values.mean(), values.std()
        
        if std == 0:
            return np.ones(len(values))
        
        z_scores = np.abs((values - mean) / std)
        threshold = self.config.get('zscore_threshold', 3.0)
        
        # Return -1 for anomaly, 1 for normal
        return np.where(z_scores > threshold, -1, 1)

class DetectorOrchestrator:
    """Manages all detectors based on config"""
    
    def __init__(self, config_path: str = "/app/config/metrics.yaml"):
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.detectors = {}
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        for metric in self.config['metrics']:
            model_type = metric['model']
            if model_type == 'isolation_forest':
                self.detectors[metric['name']] = IsolationForestDetector(metric)
            elif model_type == 'zscore_burst':
                self.detectors[metric['name']] = ZScoreBurstDetector(metric)
    
    def detect(self, metric_name: str, X) -> np.ndarray:
        if metric_name not in self.detectors:
            print(f"[WARN] No detector for {metric_name}")
            return None
        
        detector = self.detectors[metric_name]
        
        # Fit if needed (for stateful models)
        if hasattr(detector, 'fit'):
            detector.fit(X.values if isinstance(X, pd.DataFrame) else X)
        
        return detector.predict(X.values if isinstance(X, pd.DataFrame) else X)
