import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import logging
from typing import Tuple, Dict, List
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import os

# Configure logging for production-readiness
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

class XGBoostForecaster:
    """
    Production-ready XGBoost time series forecasting pipeline.
    Handles data loading, feature engineering, training, evaluation, and saving.
    """
    def __init__(self, target_col: str = 'Total', date_col: str = 'Date', group_cols: List[str] = None):
        self.target_col = target_col
        self.date_col = date_col
        self.group_cols = group_cols if group_cols else ['State', 'Category']
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror', 
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
        self.features = []

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Loads the dataset from an Excel file and sorts by date."""
        logger.info(f"Loading data from {filepath}")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found at {filepath}")
            
        df = pd.read_excel(filepath)
        
        # Ensure Date column is datetime type
        if self.date_col in df.columns:
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            df = df.sort_values(by=self.date_col).reset_index(drop=True)
            
        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates required features:
        - lag_1, lag_3
        - rolling_mean_3, rolling_std_3
        - month, quarter, year
        """
        logger.info("Generating time series features...")
        df = df.copy()
        
        # 1. Date Features
        df['month'] = df[self.date_col].dt.month
        df['quarter'] = df[self.date_col].dt.quarter
        df['year'] = df[self.date_col].dt.year
        
        # Check if grouping columns exist for group-wise features
        if all(col in df.columns for col in self.group_cols):
            grouped = df.groupby(self.group_cols)[self.target_col]
        else:
            logger.warning("Grouping columns not found. Computing features across entire dataset.")
            grouped = df[self.target_col]
            
        # 2. Lag Features
        df['lag_1'] = grouped.shift(1)
        df['lag_3'] = grouped.shift(3)
        
        # 3. Rolling Features (computed on the shifted values to prevent leakage)
        # We shift by 1 first so the rolling window doesn't include the target itself
        df['rolling_mean_3'] = grouped.transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
        df['rolling_std_3'] = grouped.transform(lambda x: x.shift(1).rolling(window=3, min_periods=2).std())
        
        # Drop rows with NaN values created by lags
        initial_len = len(df)
        df = df.dropna().reset_index(drop=True)
        logger.info(f"Dropped {initial_len - len(df)} rows due to NaN values in lags.")
        
        # Define the feature columns used for training
        exclude_cols = [self.target_col, self.date_col] + self.group_cols
        self.features = [c for c in df.columns if c not in exclude_cols]
        
        return df

    def time_series_split(self, df: pd.DataFrame, test_size_weeks: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits data strictly chronologically.
        The last `test_size_weeks` weeks form the test/validation set.
        """
        logger.info(f"Performing time-series train-test split (Test size: {test_size_weeks} weeks)")
        max_date = df[self.date_col].max()
        split_date = max_date - pd.Timedelta(weeks=test_size_weeks)
        
        train_df = df[df[self.date_col] <= split_date].copy()
        test_df = df[df[self.date_col] > split_date].copy()
        
        logger.info(f"Train set size: {len(train_df)} | Test set size: {len(test_df)}")
        return train_df, test_df

    def train(self, train_df: pd.DataFrame):
        """Trains the XGBoost Regressor."""
        logger.info("Training XGBoost Regressor...")
        X_train = train_df[self.features]
        y_train = train_df[self.target_col]
        self.model.fit(X_train, y_train)
        logger.info("Training complete.")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generates predictions for a given dataset."""
        X = df[self.features]
        return self.model.predict(X)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculates RMSE, MAE, and MAPE metrics."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # Handle zeros to avoid division by zero in MAPE
        y_true_safe = np.where(y_true == 0, 1e-10, y_true)
        mape = mean_absolute_percentage_error(y_true_safe, y_pred)
        
        metrics = {"RMSE": rmse, "MAE": mae, "MAPE": mape}
        logger.info(f"Evaluation Metrics: {metrics}")
        return metrics

    def save_model(self, filepath: str):
        """Saves the trained model and feature list using joblib."""
        logger.info(f"Saving model and metadata to {filepath}")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"model": self.model, "features": self.features}, filepath)
def run_pipeline(data_path: str, model_save_path: str):
    """
    Executes the entire XGBoost forecasting pipeline.
    """
    forecaster = XGBoostForecaster()

    try:
        # 1. Load Data
        df = forecaster.load_data(data_path)

        # 2. Feature Engineering
        df_features = forecaster.create_features(df)

        # 3. Time Series Split
        train_df, test_df = forecaster.time_series_split(df_features, test_size_weeks=8)

        if len(test_df) == 0:
            logger.error("Test dataset is empty. Check data date ranges.")
            return

        # 4. Train Model
        forecaster.train(train_df)

        # 5. Predict
        logger.info("Generating predictions...")
        test_df['Predictions'] = forecaster.predict(test_df)

        # 6. Evaluate
        y_true = test_df[forecaster.target_col].values
        y_pred = test_df['Predictions'].values

        metrics = forecaster.evaluate(y_true, y_pred)

        # ==============================
        # DISPLAY SAMPLE OUTPUT
        # ==============================

        print("\n================ SAMPLE PREDICTIONS ================\n")

        results = test_df[
            [forecaster.date_col, 'State', 'Category',
             forecaster.target_col, 'Predictions']
        ].copy()

        results.rename(columns={
            forecaster.target_col: 'Actual'
        }, inplace=True)

        print(results.head(10))

        # ==============================
        # SAVE PREDICTIONS
        # ==============================

        prediction_path = "models/history/xgboost_predictions.xlsx"

        os.makedirs(os.path.dirname(prediction_path), exist_ok=True)

        results.to_excel(prediction_path, index=False)

        logger.info(f"Predictions saved to {prediction_path}")

        # ==============================
        # SAVE METRICS
        # ==============================

        metrics_df = pd.DataFrame([metrics])

        metrics_path = "models/history/xgboost_metrics.xlsx"

        metrics_df.to_excel(metrics_path, index=False)

        logger.info(f"Metrics saved to {metrics_path}")

        # ==============================
        # SAVE MODEL
        # ==============================

        forecaster.save_model(model_save_path)

        # ==============================
        # FEATURE IMPORTANCE
        # ==============================

        importance_df = pd.DataFrame({
            "Feature": forecaster.features,
            "Importance": forecaster.model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        feature_path = "models/history/feature_importance.xlsx"

        importance_df.to_excel(feature_path, index=False)

        logger.info(f"Feature importance saved to {feature_path}")

        # ==============================
        # FINAL SUCCESS MESSAGE
        # ==============================

        print("\n===================================================")
        print("XGBOOST FORECASTING PIPELINE COMPLETED SUCCESSFULLY")
        print("===================================================\n")

        print("Evaluation Metrics:")
        print(metrics)

        print("\nSaved Files:")
        print(f"1. Predictions -> {prediction_path}")
        print(f"2. Metrics -> {metrics_path}")
        print(f"3. Feature Importance -> {feature_path}")
        print(f"4. Model -> {model_save_path}")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")

if __name__ == "__main__":
    # Example usage
    data_path = "perfect_forecasting_dataset.xlsx"
    model_save_path = "models/best_model/xgboost_best.pkl"
    run_pipeline(data_path, model_save_path)
