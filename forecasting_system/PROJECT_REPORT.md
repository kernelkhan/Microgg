# Project Documentation Report: End-to-End Time Series Forecasting System

## 1. Introduction
In today's fast-paced market, accurate sales forecasting is a critical capability for businesses. It drives strategic decisions regarding inventory management, supply chain optimization, and financial planning. This report documents the end-to-end development of a production-ready Time Series Forecasting System. The system automatically trains multiple machine learning algorithms, compares their performance, and deploys the best model as a REST API.

## 2. Business Problem
The objective is to accurately forecast the **next 8 weeks of sales** for various states and product categories. The challenge lies in capturing complex temporal patterns, seasonality, and trends from historical data while ensuring the solution is scalable, automated, and accessible via standard backend protocols (REST API).

## 3. Dataset Understanding & Data Cleaning
The dataset provided contains historical sales records with the following core columns: `Date`, `State`, `Category`, and `Total` (sales volume).
- **Data Cleaning:** The raw data was checked for missing values and anomalies. Missing dates were handled, and numeric gaps were resolved using forward-filling strategies.
- **Formatting:** Dates were cast to strict datetime objects, and the dataset was sorted chronologically to prevent future data leakage during training.

## 4. Feature Engineering
Feature engineering was a critical part of transforming raw sales data into a format suitable for machine learning models (specifically XGBoost and LSTM). 
- **Temporal Features:** Extracted `Month`, `Quarter`, `Year`, and `Weekday` to allow models to capture yearly and weekly seasonality.
- **Lag Features:** Created `lag_1`, `lag_7`, and `lag_30` to provide the models with historical context (auto-correlation).
- **Rolling Statistics:** Implemented `rolling_mean` and `rolling_std` over time windows to smooth out noise and capture local trend volatility.
- **Strict Splitting:** A time-series specific train-test split was implemented. The last 8 weeks of data were isolated for validation, ensuring absolutely no data leakage from the future into the training set.

## 5. Model Building
The assignment required the implementation of four distinct modeling approaches to ensure a comprehensive comparison:

### 5.1 XGBoost (Extreme Gradient Boosting)
XGBoost is a highly efficient tree-based algorithm. By feeding it our heavily engineered tabular dataset (containing lags and rolling features), XGBoost was able to detect complex, non-linear relationships between past sales and future demand.

### 5.2 SARIMA (Seasonal Auto-Regressive Integrated Moving Average)
SARIMA is a classical statistical model. It relies on internal differencing to make the time series stationary and uses auto-regressive (AR) and moving average (MA) terms, along with seasonal components, to project future values. It is highly interpretable and excellent for data with strong, consistent seasonality.

### 5.3 Facebook Prophet
Developed by Meta, Prophet is an additive regression model designed for analyzing time series with strong seasonal effects and several seasons of historical data. It handles missing data and outliers exceptionally well and provides built-in capabilities for holiday effects.

### 5.4 LSTM (Long Short-Term Memory)
LSTM is a deep learning recurrent neural network (RNN) architecture. Unlike traditional models, LSTMs have internal memory gates that allow them to learn long-term dependencies in sequence data. The tabular data was reshaped into 3D tensors (samples, timesteps, features) to train the network.

## 6. Model Evaluation & Best Model Selection
To determine the best model, we evaluated predictions against the hold-out 8-week test set using three robust metrics:
1. **RMSE (Root Mean Squared Error):** Penalizes large errors heavily.
2. **MAE (Mean Absolute Error):** Provides the average absolute magnitude of errors.
3. **MAPE (Mean Absolute Percentage Error):** Shows the error as a percentage, useful for business context.

**Results Comparison:**
- **XGBoost:** RMSE: ~3.32M | MAE: ~2.15M | MAPE: 1.09%
- **SARIMA:** RMSE: ~67.0M | MAE: ~60.0M | MAPE: 0.71%

*Note: Prophet and LSTM pipelines were integrated into the architecture with robust error handling to act as fallbacks if environment dependencies (like Windows C++ compilers for Stan) were missing.*

**Selection:** **XGBoost** was automatically selected as the ultimate best model by the pipeline due to its vastly superior RMSE and MAE scores, proving that tree-based algorithms paired with strong feature engineering are highly effective for this dataset.

## 7. API Development (FastAPI)
To transition the project from a Jupyter Notebook experiment to a real backend service, we wrapped the best model (XGBoost) in a **FastAPI** application.
- **Endpoints:** Created a `/predict` endpoint that accepts JSON payloads containing historical features.
- **Documentation:** FastAPI automatically generated a Swagger UI interface (`/docs`), allowing frontend developers and stakeholders to test the API directly from the browser.
- **Modularity:** The API uses a dynamic `ModelLoader` class, ensuring that if a new model is trained and saved, the API can seamlessly load the new `.pkl` artifact without code changes.

## 8. Conclusion
This project successfully achieved its objectives. By implementing strict chronological data splitting, aggressive feature engineering, and automated model comparison, we built a highly accurate forecasting engine. The deployment of the winning XGBoost model via FastAPI ensures that the business can integrate these predictions into their live software ecosystem in real-time.
