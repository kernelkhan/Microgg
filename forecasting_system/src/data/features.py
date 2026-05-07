import pandas as pd
import numpy as np
import logging
import holidays
from src.config import DATE_COL, TARGET_COL, GROUP_COLS, LAG_FEATURES, ROLLING_WINDOWS

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, country: str = "US"):
        self.country_holidays = holidays.country_holidays(country)

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all feature engineering steps to the dataframe.
        Assumes df is already sorted by date.
        """
        logger.info("Starting feature engineering...")
        df = df.copy()
        
        # 1. Date features
        df = self._add_date_features(df)
        
        # 2. Holiday flag
        df = self._add_holiday_flag(df)
        
        # Groupby state and category for lag and rolling features
        if all(col in df.columns for col in GROUP_COLS):
            grouped = df.groupby(GROUP_COLS)
            
            # 3. Lag features (t-1, t-7, t-30)
            df = self._add_lag_features(df, grouped)
            
            # 4. Rolling statistics
            df = self._add_rolling_features(df, grouped)
        else:
            logger.warning(f"Grouping columns {GROUP_COLS} not found. Skipping group-based lags.")
            
        # Drop NaN values created by lagging and rolling
        df = df.dropna().reset_index(drop=True)
        logger.info(f"Feature engineering completed. Shape: {df.shape}")
        
        return df

    def _add_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['day_of_week'] = df[DATE_COL].dt.dayofweek
        df['month'] = df[DATE_COL].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        return df

    def _add_holiday_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        # Check if date is in the holidays object
        df['is_holiday'] = df[DATE_COL].apply(
            lambda d: 1 if d in self.country_holidays else 0
        )
        return df

    def _add_lag_features(self, df: pd.DataFrame, grouped) -> pd.DataFrame:
        for lag in LAG_FEATURES:
            df[f'lag_{lag}'] = grouped[TARGET_COL].shift(lag)
        return df

    def _add_rolling_features(self, df: pd.DataFrame, grouped) -> pd.DataFrame:
        for window in ROLLING_WINDOWS:
            df[f'rolling_mean_{window}'] = grouped[TARGET_COL].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            df[f'rolling_std_{window}'] = grouped[TARGET_COL].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=2).std()
            )
        return df
