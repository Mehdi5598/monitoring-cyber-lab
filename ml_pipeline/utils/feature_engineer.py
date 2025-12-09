# ml_pipeline/utils/feature_engineer.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self):
        self.scalers = {}
        print("✅ FeatureEngineer initialized")  # Debug print
    
    def engineer(self, df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
        """Transform raw metrics into ML-ready features"""
        
        print(f"[FEATURE] Starting engineering for {metric_name}")  # Debug
        print(f"[FEATURE] Input shape: {df.shape}, columns: {df.columns.tolist()}")
        
        if df is None or len(df) < 3:
            print(f"[FEATURE] SKIP: too small {len(df) if df else 0}")
            return None
        
        try:
            features = df.copy()
            print(f"[FEATURE] Step 1 - Raw data: {features.head()}")
            
            # 1. Time-based features
            features['hour'] = features.index.hour
            features['weekday'] = features.index.weekday
            features['is_business_hours'] = features['hour'].between(8, 18).astype(int)
            print(f"[FEATURE] Step 2 - After time: {features.shape}")
            
            # 2. Basic statistical features
            features['value_diff'] = features['value'].diff()
            features['value_pct_change'] = features['value'].pct_change() * 100
            
            # 3. Rolling statistics
            window = min(5, len(features) - 1)
            features['rolling_mean'] = features['value'].rolling(window).mean()
            features['rolling_std'] = features['value'].rolling(window).std()
            
            # 4. Exponential moving average
            features['ewm'] = features['value'].ewm(span=window).mean()
            
            print(f"[FEATURE] Step 3 - After rolling: {features.shape}")
            
            # 5. Handle infinite and NaN values
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # 6. Scale numeric features
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            if metric_name not in self.scalers:
                self.scalers[metric_name] = StandardScaler()
            
            try:
                features[numeric_cols] = self.scalers[metric_name].fit_transform(features[numeric_cols])
                print(f"[FEATURE] Step 4 - After scaling: {features.shape}")
            except Exception as e:
                print(f"[FEATURE WARN] Scaling failed: {e}, using normalized values")
                features[numeric_cols] = (features[numeric_cols] - features[numeric_cols].min()) / (features[numeric_cols].max() - features[numeric_cols].min() + 1e-8)
            
            print(f"[FEATURE] Final shape: {features.shape}, columns: {features.columns.tolist()}")
            return features
            
        except Exception as e:
            print(f"[FEATURE ERROR] Failed to engineer features for {metric_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
