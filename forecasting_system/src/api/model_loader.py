import joblib
import logging
import os

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Singleton class to load and hold
    the best forecasting model in memory.
    """

    _instance = None

    model_data = None

    def __new__(cls):

        if cls._instance is None:

            cls._instance = super(
                ModelManager,
                cls
            ).__new__(cls)

            cls._instance._load_model()

        return cls._instance

    # ==========================================
    # LOAD MODEL
    # ==========================================

    def _load_model(self):

        # Primary model path (XGBoost)
        model_path = os.path.join(
            "models",
            "best_model",
            "xgboost_best.pkl"
        )

        # Fallback to SARIMA
        if not os.path.exists(model_path):

            logger.warning(
                "XGBoost model not found. "
                "Trying SARIMA..."
            )

            model_path = os.path.join(
                "models",
                "best_model",
                "sarima_best.pkl"
            )

        # Fallback to Prophet
        if not os.path.exists(model_path):

            logger.warning(
                "SARIMA model not found. "
                "Trying Prophet..."
            )

            model_path = os.path.join(
                "models",
                "best_model",
                "prophet_best.pkl"
            )

        # Fallback to LSTM
        if not os.path.exists(model_path):

            logger.warning(
                "Prophet model not found. "
                "Trying LSTM..."
            )

            model_path = os.path.join(
                "models",
                "best_model",
                "lstm_best.pkl"
            )

        # ======================================
        # LOAD MODEL
        # ======================================

        try:

            logger.info(
                f"Loading model from {model_path}..."
            )

            self.model_data = joblib.load(
                model_path
            )

            logger.info(
                "Model loaded successfully."
            )

        except Exception as e:

            logger.error(
                f"Failed to load model "
                f"from {model_path}. "
                f"Error: {e}"
            )

            self.model_data = None

    # ==========================================
    # GET MODEL
    # ==========================================

    def get_model(self):

        return self.model_data