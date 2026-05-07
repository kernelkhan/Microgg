import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import logging
from src.models.base import TimeSeriesModel
from src.config import TARGET_COL

logger = logging.getLogger(__name__)

class SARIMAModel(TimeSeriesModel):
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_fit = None

    def train(self, df_train: pd.DataFrame) -> None:
        logger.info(f"Training SARIMA model with order={self.order}, seasonal_order={self.seasonal_order}...")
        y = df_train[TARGET_COL].values
        
        # In a real scenario we'd handle exog features, but for simplicity we fit endogenous
        model = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order, enforce_stationarity=False)
        self.model_fit = model.fit(disp=False)

    def predict(self, steps: int, future_features: pd.DataFrame = None) -> pd.DataFrame:
        logger.info(f"Predicting next {steps} steps with SARIMA...")
        if self.model_fit is None:
            raise ValueError("Model is not trained yet.")
            
        preds = self.model_fit.forecast(steps=steps)
        
        result = pd.DataFrame({'Prediction': preds})
        if future_features is not None:
            result = pd.concat([future_features.head(steps).reset_index(drop=True), result], axis=1)
            
        return result

    def save(self, filepath: str) -> None:
        logger.info(f"Saving SARIMA model to {filepath}")
        joblib.dump({'model_fit': self.model_fit, 'order': self.order, 'seasonal_order': self.seasonal_order}, filepath)

    def load(self, filepath: str) -> None:
        logger.info(f"Loading SARIMA model from {filepath}")
        data = joblib.load(filepath)
        self.model_fit = data['model_fit']
        self.order = data['order']
        self.seasonal_order = data['seasonal_order']
