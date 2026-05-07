import pandas as pd
import logging
from typing import Tuple
from src.config import DATE_COL, TEST_SIZE_MONTHS

logger = logging.getLogger(__name__)

class TimeSeriesSplitter:
    def __init__(self, test_size_months: int = TEST_SIZE_MONTHS):
        self.test_size_months = test_size_months

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits data chronologically to avoid data leakage.
        The last `test_size_months` months of data form the validation set.
        """
        logger.info(f"Splitting data with last {self.test_size_months} months as test set.")
        
        # Ensure it's sorted
        df = df.sort_values(by=DATE_COL)
        
        # Find the max date
        max_date = df[DATE_COL].max()
        
        # Calculate split date
        split_date = max_date - pd.DateOffset(months=self.test_size_months)
        
        train_df = df[df[DATE_COL] <= split_date].copy()
        test_df = df[df[DATE_COL] > split_date].copy()
        
        logger.info(f"Train set size: {train_df.shape}")
        logger.info(f"Test set size: {test_df.shape}")
        
        return train_df, test_df
