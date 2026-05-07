import pandas as pd
import logging
from src.config import DATE_COL, TARGET_COL

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_data(self) -> pd.DataFrame:
        """
        Loads dataset from Excel or CSV file.
        """
        logger.info(f"Loading data from {self.filepath}")
        if self.filepath.endswith(".xlsx") or self.filepath.endswith(".xls"):
            df = pd.read_excel(self.filepath)
        else:
            df = pd.read_csv(self.filepath)
            
        return self.clean_data(df)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic cleaning:
        - Convert Date column to datetime
        - Sort by Date
        - Fill missing numeric values with forward fill or 0
        """
        logger.info("Cleaning data: converting dates and handling missing values.")
        if DATE_COL in df.columns:
            df[DATE_COL] = pd.to_datetime(df[DATE_COL])
            df = df.sort_values(by=DATE_COL).reset_index(drop=True)
            
        # Handle missing target values if any
        if TARGET_COL in df.columns:
            # Forward fill then fillna with 0 for target
            df[TARGET_COL] = df[TARGET_COL].ffill().fillna(0)
            
        # Drop pre-existing feature columns so we can re-engineer them dynamically
        pre_engineered = ['lag_1', 'lag_3', 'rolling_mean_3', 'rolling_std_3', 'month', 'quarter', 'year', 'weekday']
        cols_to_drop = [c for c in pre_engineered if c in df.columns]
        if cols_to_drop:
            logger.info(f"Dropping pre-engineered columns: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
            
        return df
