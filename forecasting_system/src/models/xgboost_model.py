import xgboost as xgb
import pandas as pd
import joblib
import logging
from src.models.base import TimeSeriesModel
from src.config import TARGET_COL, DATE_COL, GROUP_COLS

logger = logging.getLogger(__name__)

class XGBoostModel(TimeSeriesModel):
    def __init__(self, **kwargs):
        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, **kwargs)
        self.features = None

    def _prepare_data(self, df: pd.DataFrame):
        # Exclude Target, Date, and Grouping columns from features
        exclude_cols = [TARGET_COL, DATE_COL] + GROUP_COLS
        self.features = [c for c in df.columns if c not in exclude_cols]
        X = df[self.features]
        y = df[TARGET_COL]
        return X, y

    def train(self, df_train: pd.DataFrame) -> None:
        logger.info("Training XGBoost model...")
        X_train, y_train = self._prepare_data(df_train)
        self.model.fit(X_train, y_train)

    def predict(self, steps: int, future_features: pd.DataFrame = None) -> pd.DataFrame:
        logger.info(f"Predicting next {steps} steps with XGBoost...")
        if future_features is None or len(future_features) < steps:
            raise ValueError("XGBoost requires future features (lags, dates) to predict.")
        
        X_future = future_features[self.features].head(steps)
        preds = self.model.predict(X_future)
        
        result = future_features.head(steps).copy()
        result['Prediction'] = preds
        return result

    def save(self, filepath: str) -> None:
        logger.info(f"Saving XGBoost model to {filepath}")
        joblib.dump({'model': self.model, 'features': self.features}, filepath)

    def load(self, filepath: str) -> None:
        logger.info(f"Loading XGBoost model from {filepath}")
        data = joblib.load(filepath)
        self.model = data['model']
        self.features = data['features']
