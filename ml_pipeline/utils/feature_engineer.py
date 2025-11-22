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
        
        # 5. Clean infinite and extreme values BEFORE scaling
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Cap extreme values at 5 standard deviations
        for col in ['value', 'value_diff', 'value_pct_change']:
            if col in features.columns:
                mean_val = features[col].mean()
                std_val = features[col].std()
                if pd.notna(mean_val) and pd.notna(std_val) and std_val > 0:
                    upper_bound = mean_val + 5 * std_val
                    lower_bound = mean_val - 5 * std_val
                    features[col] = features[col].clip(lower=lower_bound, upper=upper_bound)
        
        # 6. Fill NaN values
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 7. Scale numeric features (per metric)
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        if metric_name not in self.scalers:
            self.scalers[metric_name] = StandardScaler()
        
        # Check for any remaining infinite values
        if np.isinf(features[numeric_cols]).any().any():
            print(f"[WARN] Infinite values detected in {metric_name}, replacing with 0")
            features[numeric_cols] = features[numeric_cols].replace([np.inf, -np.inf], 0)
        
        # Fit-transform on current data
        try:
            features[numeric_cols] = self.scalers[metric_name].fit_transform(features[numeric_cols])
        except Exception as e:
            print(f"[ERROR] Scaling failed for {metric_name}: {e}")
            # Fallback: just normalize to 0-1 range
            features[numeric_cols] = (features[numeric_cols] - features[numeric_cols].min()) / (features[numeric_cols].max() - features[numeric_cols].min() + 1e-8)
        
        return features
