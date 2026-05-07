import os
import logging
import json
import pandas as pd
from typing import Dict, Any

from src.config import BEST_MODEL_DIR, HISTORY_DIR, TARGET_COL
from src.data.loader import DataLoader
from src.data.features import FeatureEngineer
from src.data.splitter import TimeSeriesSplitter

from src.models.xgboost_model import XGBoostModel
from src.models.sarima import SARIMAModel
from src.models.prophet_model import ProphetModel
from src.models.lstm_model import LSTMModel
from src.models.evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def run_pipeline(data_path: str):
    logger.info("Starting training pipeline...")
    
    # 1. Load Data
    loader = DataLoader(data_path)
    df = loader.load_data()
    
    # 2. Feature Engineering
    engineer = FeatureEngineer()
    df_features = engineer.create_features(df)
    
    # 3. Train/Test Split
    splitter = TimeSeriesSplitter()
    train_df, test_df = splitter.split(df_features)
    
    if len(test_df) == 0:
         logger.error("Test dataset is empty. Check data date ranges and split logic.")
         return

    # 4. Initialize Models
    models = {
        "XGBoost": XGBoostModel(),
        "SARIMA": SARIMAModel(),
        "Prophet": ProphetModel(),
        "LSTM": LSTMModel(epochs=2) # low epochs for demonstration
    }
    
    results = {}
    best_model_name = None
    
    # 5. Train and Evaluate
    for name, model in models.items():
        try:
            logger.info(f"--- Processing {name} ---")
            model.train(train_df)
            
            steps = len(test_df)
            preds_df = model.predict(steps=steps, future_features=test_df)
            
            # Predict returns a dataframe with 'Prediction' column
            y_true = test_df[TARGET_COL].values
            y_pred = preds_df['Prediction'].values
            
            metrics = ModelEvaluator.evaluate(y_true, y_pred)
            results[name] = metrics
            
            # Save history
            with open(os.path.join(HISTORY_DIR, f"{name}_metrics.json"), "w") as f:
                 json.dump(metrics, f)
                 
        except Exception as e:
            logger.error(f"Failed to train/evaluate {name}: {e}")
            
    # 6. Select Best Model
    if results:
         best_model_name = ModelEvaluator.select_best_model(results, primary_metric="RMSE")
         logger.info(f"*** Best Model is {best_model_name} ***")
         
         # 7. Save Best Model
         best_model_instance = models[best_model_name]
         save_path = os.path.join(BEST_MODEL_DIR, "best_model.pkl")
         
         # For LSTM we use pt instead of pkl usually, but keeping it simple
         if best_model_name == "LSTM":
              save_path = os.path.join(BEST_MODEL_DIR, "best_model.pt")
              
         best_model_instance.save(save_path)
         
         # Save metadata
         metadata = {
              "best_model": best_model_name,
              "metrics": results[best_model_name],
              "all_results": results
         }
         with open(os.path.join(BEST_MODEL_DIR, "metadata.json"), "w") as f:
              json.dump(metadata, f)
              
    logger.info("Pipeline finished successfully.")
    return best_model_name

if __name__ == "__main__":
    # Example usage: python -m src.pipeline.train --data path/to/dataset.xlsx
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to raw dataset")
    args = parser.parse_args()
    
    run_pipeline(args.data)
