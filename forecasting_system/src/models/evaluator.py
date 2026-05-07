import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class ModelEvaluator:
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculates standard forecasting metrics.
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # Handle zero division for MAPE by adding a small epsilon
        y_true_safe = np.where(y_true == 0, 1e-10, y_true)
        mape = mean_absolute_percentage_error(y_true_safe, y_pred)
        
        metrics = {
            "RMSE": float(rmse),
            "MAE": float(mae),
            "MAPE": float(mape)
        }
        
        logger.info(f"Evaluation Metrics: {metrics}")
        return metrics

    @staticmethod
    def select_best_model(results: Dict[str, Dict[str, float]], primary_metric: str = "RMSE") -> str:
        """
        Selects the best model name from the results dictionary based on the primary metric.
        Expected results format: {'XGBoost': {'RMSE': 10.5, ...}, 'SARIMA': {'RMSE': 12.1, ...}}
        """
        best_model = min(results.keys(), key=lambda k: results[k][primary_metric])
        logger.info(f"Best model selected: {best_model} based on {primary_metric}")
        return best_model
