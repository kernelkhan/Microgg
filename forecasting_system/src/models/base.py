from abc import ABC, abstractmethod
import pandas as pd
from typing import Any

class TimeSeriesModel(ABC):
    """
    Abstract Base Class for all forecasting models.
    Ensures a consistent interface for training, prediction, and saving.
    """
    
    @abstractmethod
    def train(self, df_train: pd.DataFrame) -> None:
        """Trains the model on the provided dataframe."""
        pass
        
    @abstractmethod
    def predict(self, steps: int, future_features: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generates predictions for the next `steps` periods.
        Returns a DataFrame containing at minimum the target predictions.
        """
        pass
        
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Serializes and saves the model to disk."""
        pass
        
    @abstractmethod
    def load(self, filepath: str) -> None:
        """Loads a serialized model from disk."""
        pass
