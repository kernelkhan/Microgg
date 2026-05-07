# 📈 End-to-End Time Series Forecasting System

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103.1-009688.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-orange.svg)
![SARIMA](https://img.shields.io/badge/Statsmodels-SARIMA-yellow.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)

## 📖 Project Overview
This project is a production-ready **End-to-End Time Series Forecasting System**. It is designed to act as a real backend service that intelligently predicts future sales based on historical data. The system automatically trains multiple machine learning and deep learning forecasting algorithms, compares them using robust evaluation metrics, selects the best-performing model, and serves the predictions via a high-performance REST API.

## 🎯 Problem Statement
The goal is to accurately forecast the **next 8 weeks of sales** for each state using historical sales data. Accurate forecasting enables better inventory management, resource allocation, and strategic business planning.

## 🚀 Objectives
1. Build a robust data pipeline that handles missing values and data cleaning.
2. Implement advanced feature engineering (lags, rolling metrics, date features) to capture seasonality and trends.
3. Train multiple forecasting models (XGBoost, SARIMA, Prophet, LSTM) without data leakage.
4. Automatically evaluate, compare, and select the best model based on RMSE, MAE, and MAPE.
5. Expose the best model's predictions via a scalable FastAPI REST service.

## 🏗️ Architecture
The system follows a modular, scalable architecture:
1. **Data Layer:** Handles data loading, cleaning, and strict chronological train-test splitting.
2. **Feature Engineering Layer:** Generates lag features (t-1, t-7, t-30), rolling mean/std, and temporal features (month, quarter, weekday).
3. **Modeling Layer:** Contains isolated pipelines for XGBoost, SARIMA, Prophet, and LSTM.
4. **Evaluation Layer:** Calculates metrics, generates comparison reports, plots visualizations, and saves the `.pkl` artifacts.
5. **Serving Layer:** A FastAPI application that loads the best model and provides endpoints for real-time predictions.

## 📂 Folder Structure
```text
forecasting_system/
│
├── src/
│   ├── api/            # FastAPI application (main.py, routes, schemas)
│   ├── data/           # Data loading, cleaning, and splitting logic
│   ├── models/         # Model architectures and base classes
│   └── config.py       # Global configuration and hyperparameter settings
│
├── compare_models.py   # Main pipeline script to train, evaluate, and select models
├── xgboost_pipeline.py # Isolated XGBoost pipeline
├── sarima_pipeline.py  # Isolated SARIMA pipeline
├── prophet_pipeline.py # Isolated Prophet pipeline
├── lstm_keras_pipeline.py # Isolated LSTM pipeline
├── requirements.txt    # Python dependencies
├── Dockerfile          # Containerization setup
└── README.md           # Project documentation
```

## 🛠️ Technologies & Models Used
**Core Technologies:** Python, Pandas, NumPy, Scikit-Learn
**API & Deployment:** FastAPI, Uvicorn, Docker
**Forecasting Models:**
1. **XGBoost:** Gradient boosting algorithm leveraging engineered lag and rolling features. *(🏆 Best Model)*
2. **SARIMA:** Statistical model capturing seasonality and auto-regressive trends.
3. **Facebook Prophet:** Additive regression model handling holidays and seasonality.
4. **LSTM (Long Short-Term Memory):** Deep learning recurrent neural network for sequence prediction.

## ⚙️ Feature Engineering
Feature engineering is a critical part of this system. We implemented:
- **Temporal Features:** Month, Quarter, Year, Weekday, Holiday flags.
- **Lag Features:** Previous sales values at intervals (t-1, t-7, t-30) to capture auto-correlation.
- **Rolling Statistics:** Rolling mean and standard deviation to capture local trends and volatility.

## 💻 Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kernelkhan/Microgg
   cd forecasting_system
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 Running Instructions

**1. Run the Training and Comparison Pipeline**
Execute the main script to process data, train all models, and select the best one:
```bash
python compare_models.py
```
*Outputs (saved in `comparison_results/`):*
- `best_model/xgboost_best.pkl`
- `plots/rmse_comparison.png`
- `model_comparison.csv`

**2. Start the REST API**
Start the FastAPI server to serve predictions:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## 🌐 API Usage & Swagger Documentation
Once the API is running, access the interactive Swagger UI:
👉 **http://localhost:8000/docs**

**Key Endpoints:**
- `GET /health`: Check API status.
- `POST /predict`: Submit state and category data to receive an 8-week forecast.

## 📊 Results & Model Comparison
All models were evaluated using an 8-week holdout validation set.

| Model | RMSE | MAE | MAPE |
| :--- | :--- | :--- | :--- |
| **XGBoost** | **3,320,302** | **2,156,398** | **1.09%** |
| SARIMA | 67,098,378 | 60,067,066 | 0.71% |

**Selection:** XGBoost was automatically selected as the best model due to its superior RMSE and MAE, demonstrating excellent capability in capturing non-linear patterns using lag features.

## 🚧 Challenges Faced
- **Data Leakage Prevention:** Ensuring that lag and rolling features were computed without peeking into the test set required strict chronological splitting.
- **Deep Learning Integration:** Formatting 2D tabular data into 3D sequences required for LSTM processing was complex but successfully implemented.
- **Dependency Management:** Resolving backend compiler issues for Facebook Prophet on Windows required graceful fallback handling in the pipeline.

## 🔮 Future Improvements
- Implement hyperparameter tuning (GridSearch/Optuna) for XGBoost and SARIMA.
- Introduce exogenous variables (e.g., macroeconomic indicators, weather data).
- Deploy the Dockerized application to a cloud provider like AWS or Render.

## 🏁 Conclusion
This project successfully demonstrates a full lifecycle data science solution. From raw data ingestion and feature engineering to model comparison and API deployment, the system is robust, modular, and ready for production use.
