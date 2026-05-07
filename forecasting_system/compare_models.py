import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from typing import Dict, Any

# Import the forecasters we built earlier
# If any of them fail (like Prophet on Windows), the script will catch the exception and skip it.
# Import forecasting pipelines

# Import forecasting pipelines

try:
    from xgboost_pipeline import XGBoostForecaster
except:
    XGBoostForecaster = None

try:
    from sarima_pipeline import SARIMAForecaster
except:
    SARIMAForecaster = None

try:
    from prophet_pipeline import ProphetForecaster
except:
    ProphetForecaster = None

try:
    from lstm_keras_pipeline import LSTMForecaster
except:
    LSTMForecaster = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

class ModelComparator:
    """
    Production-ready pipeline to compare multiple forecasting models,
    generate visualization charts, and select/save the best model.
    """
    def __init__(self, data_path: str, output_dir: str):
        self.data_path = data_path
        self.output_dir = output_dir
        self.results = []
        self.forecasts = {}
        self.actuals = None
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "best_model"), exist_ok=True)

    def run_xgboost(self):
        logger.info("--- Running XGBoost ---")
        if XGBoostForecaster is None:
            logger.error("XGBoostForecaster module not found.")
            return
        
        forecaster = XGBoostForecaster()
        try:
            df = forecaster.load_data(self.data_path)
            df_features = forecaster.create_features(df)
            train_df, test_df = forecaster.time_series_split(df_features, test_size_weeks=3)
            
            forecaster.train(train_df)
            preds = forecaster.predict(test_df)
            
            y_true = test_df[forecaster.target_col].values
            self.actuals = y_true # Save actuals for plotting
            
            metrics = forecaster.evaluate(y_true, preds)
            self._log_result("XGBoost", metrics, forecaster, preds)
        except Exception as e:
            logger.error(f"XGBoost failed: {e}")

    def run_sarima(self):
        logger.info("--- Running SARIMA ---")
        if SARIMAForecaster is None:
            logger.error("SARIMAForecaster module not found.")
            return
            
        forecaster = SARIMAForecaster()
        try:
            series = forecaster.load_and_format_data(self.data_path)
            train_series, test_series = forecaster.time_series_split(series, test_size_weeks=3)
            
            forecaster.train(train_series)
            preds = forecaster.predict(steps=len(test_series)).values
            
            y_true = test_series.values
            if self.actuals is None:
                 self.actuals = y_true
                 
            metrics = forecaster.evaluate(y_true, preds)
            self._log_result("SARIMA", metrics, forecaster, preds)
        except Exception as e:
            logger.error(f"SARIMA failed: {e}")

    def run_prophet(self):
        logger.info("--- Running Prophet ---")
        if ProphetForecaster is None:
            logger.error("ProphetForecaster module not found.")
            return
            
        forecaster = ProphetForecaster()
        try:
            df = forecaster.load_and_format_data(self.data_path)
            train_df, test_df = forecaster.time_series_split(df, test_size_weeks=3)
            
            forecaster.train(train_df)
            forecast_df = forecaster.predict(test_df)
            preds = forecast_df['yhat'].values
            
            y_true = test_df['y'].values
            if self.actuals is None:
                 self.actuals = y_true
                 
            metrics = forecaster.evaluate(y_true, preds)
            self._log_result("Prophet", metrics, forecaster, preds)
        except Exception as e:
            logger.error(f"Prophet failed (Likely Windows C++ Compiler bug): {e}")

    def run_lstm(self):
        logger.info("--- Running LSTM ---")
        if LSTMForecaster is None:
            logger.error("LSTMForecaster module not found (Tensorflow might not be installed).")
            return
            
        forecaster = LSTMForecaster(seq_length=4)
        try:
            series = forecaster.load_and_aggregate_data(self.data_path)
            X, y, _ = forecaster.scale_and_create_sequences(series)
            X_train, y_train, X_test, y_test_scaled = forecaster.time_series_split(X, y, test_size_steps=3)
            
            forecaster.train(X_train, y_train, epochs=20, batch_size=8)
            preds = forecaster.predict(X_test).flatten()
            
            y_true = forecaster.scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
            if self.actuals is None:
                 self.actuals = y_true
                 
            metrics = forecaster.evaluate(y_true, preds)
            self._log_result("LSTM", metrics, forecaster, preds)
        except Exception as e:
            logger.error(f"LSTM failed: {e}")

    def _log_result(self, model_name: str, metrics: Dict[str, float], forecaster: Any, predictions: np.ndarray):
        self.results.append({
            "Model": model_name,
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "MAPE": metrics["MAPE"],
            "Forecaster": forecaster
        })
        self.forecasts[model_name] = predictions
        logger.info(f"{model_name} completed successfully.")

    def compare_and_select_best(self):
        if not self.results:
            logger.error("No models completed successfully. Cannot compare.")
            return

        # Create DataFrame for clean comparison
        df_results = pd.DataFrame(self.results)
        
        # Display the table
        logger.info("\n=== MODEL COMPARISON ===")
        logger.info("\n" + df_results[["Model", "RMSE", "MAE", "MAPE"]].to_string(index=False))
        
        # Save comparison to CSV
        csv_path = os.path.join(self.output_dir, "model_comparison.csv")
        df_results[["Model", "RMSE", "MAE", "MAPE"]].to_csv(csv_path, index=False)
        logger.info(f"Comparison saved to {csv_path}")

        # Automatically select best model based on RMSE
        best_row = df_results.loc[df_results['RMSE'].idxmin()]
        best_model_name = best_row['Model']
        best_forecaster = best_row['Forecaster']
        
        logger.info("\n" + "⭐" * 25)
        logger.info(f"  🏆 BEST MODEL SELECTED: {best_model_name.upper()} 🏆  ")
        logger.info("⭐" * 25 + "\n")
        
        # Save the best model
        best_model_path = os.path.join(self.output_dir, "best_model", f"{best_model_name.lower()}_best")
        
        if best_model_name == "LSTM":
            best_forecaster.save_model(f"{best_model_path}.keras", f"{best_model_path}_scaler.pkl")
        else:
            best_forecaster.save_model(f"{best_model_path}.pkl")
            
        logger.info(f"Best model saved to {best_model_path}")
        
        # Generate Visualizations
        self.generate_visualizations(df_results)

    def generate_visualizations(self, df_results: pd.DataFrame):
        logger.info("Generating visualization charts...")
        sns.set_theme(style="whitegrid")
        
        # 1. Bar chart for RMSE Comparison
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x="Model", y="RMSE", data=df_results, palette="viridis")
        plt.title("Model Comparison - RMSE (Lower is Better)", fontsize=14)
        plt.ylabel("RMSE")
        plt.xlabel("Forecasting Model")
        
        # Add values on top of bars
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points')
                        
        plot_path = os.path.join(self.output_dir, "plots", "rmse_comparison.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
        # 2. Line chart for Forecasts vs Actuals
        if self.actuals is not None and len(self.forecasts) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(self.actuals, label='Actual Sales', marker='o', color='black', linewidth=2)
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            for idx, (model_name, preds) in enumerate(self.forecasts.items()):
                plt.plot(preds, label=f'{model_name} Forecast', linestyle='--', marker='x', color=colors[idx % len(colors)])
                
            plt.title("Actuals vs. Forecasts (8-Week Validation Set)", fontsize=14)
            plt.xlabel("Weeks")
            plt.ylabel("Sales")
            plt.legend()
            
            line_plot_path = os.path.join(self.output_dir, "plots", "forecasts_vs_actuals.png")
            plt.savefig(line_plot_path, bbox_inches='tight')
            plt.close()
            
        logger.info(f"Visualizations saved to {os.path.join(self.output_dir, 'plots')}")

if __name__ == "__main__":
    DATA_FILE = "perfect_forecasting_dataset.xlsx"
    OUTPUT_FOLDER = "comparison_results"
    
    comparator = ModelComparator(data_path=DATA_FILE, output_dir=OUTPUT_FOLDER)
    
    # Run all models
    # Models are isolated in try/except blocks so if one fails (like Prophet), the others will still run and be compared!
    comparator.run_xgboost()
    comparator.run_sarima()
    comparator.run_prophet()
    comparator.run_lstm()
    
    # Compare, save best, and visualize
    comparator.compare_and_select_best()
