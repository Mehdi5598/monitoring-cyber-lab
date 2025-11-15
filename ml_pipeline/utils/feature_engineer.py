# ml_pipeline/utils/feature_engineer.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self):
        self.scalers = {}
    
    def engineer(self, df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
        """Transform raw metrics into ML-ready features"""
        
        if df is None or len(df) < 3:
            return None
        
        features = df.copy()
        
        # 1. Time-based features
        features['hour'] = features.index.hour
        features['weekday'] = features.index.weekday
        features['is_business_hours'] = features['hour'].between(8, 18).astype(int)
        
        # 2. Statistical features
        features['value_diff'] = features['value'].diff()
        features['value_pct_change'] = features['value'].pct_change() * 100
        
        # 3. Rolling statistics
        window = min(5, len(features) - 1)
        features['rolling_mean'] = features['value'].rolling(window).mean()
        features['rolling_std'] = features['value'].rolling(window).std()
        
        # 4. Exponential moving average
        features['ewm'] = features['value'].ewm(span=window).mean()
        
        # 5. Fill NaN values
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 6. Scale numeric features (per metric)
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        if metric_name not in self.scalers:
            self.scalers[metric_name] = StandardScaler()
        
        # Fit-transform on current data
        features[numeric_cols] = self.scalers[metric_name].fit_transform(features[numeric_cols])
        
        return features
