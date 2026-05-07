import pandas as pd
from prophet import Prophet
import joblib
import logging
from src.models.base import TimeSeriesModel
from src.config import TARGET_COL, DATE_COL

logger = logging.getLogger(__name__)

class ProphetModel(TimeSeriesModel):
    def __init__(self, **kwargs):
        self.model = Prophet(**kwargs)

    def train(self, df_train: pd.DataFrame) -> None:
        logger.info("Training Prophet model...")
        # Prophet requires columns to be named 'ds' and 'y'
        prophet_df = df_train[[DATE_COL, TARGET_COL]].rename(columns={DATE_COL: 'ds', TARGET_COL: 'y'})
        self.model.fit(prophet_df)

    def predict(self, steps: int, future_features: pd.DataFrame = None) -> pd.DataFrame:
        logger.info(f"Predicting next {steps} steps with Prophet...")
        if future_features is not None and DATE_COL in future_features.columns:
            future = future_features[[DATE_COL]].rename(columns={DATE_COL: 'ds'}).head(steps)
        else:
            # Generate future dates if none provided (assuming daily frequency for simplicity)
            future = self.model.make_future_dataframe(periods=steps, freq='D')
            future = future.tail(steps)
            
        forecast = self.model.predict(future)
        
        result = pd.DataFrame({'Prediction': forecast['yhat'].values})
        if future_features is not None:
            result = pd.concat([future_features.head(steps).reset_index(drop=True), result], axis=1)
            
        return result

    def save(self, filepath: str) -> None:
        logger.info(f"Saving Prophet model to {filepath}")
        joblib.dump(self.model, filepath)

    def load(self, filepath: str) -> None:
        logger.info(f"Loading Prophet model from {filepath}")
        self.model = joblib.load(filepath)
