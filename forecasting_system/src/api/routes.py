from fastapi import APIRouter, HTTPException
from src.api.schemas import PredictRequest, ForecastResponse, StateForecastResponse, ForecastItem
from src.api.model_loader import ModelManager
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Instantiate the singleton model manager
model_manager = ModelManager()

@router.get("/")
def read_root():
    """
    Root endpoint checking if the API is active.
    """
    return {
        "status": "active",
        "message": "Welcome to the Time Series Forecasting API. Go to /docs for Swagger documentation."
    }

@router.post("/predict", response_model=ForecastResponse)
def generate_forecast(request: PredictRequest):
    """
    Generates a generic forecast for the next N periods using the saved best model.
    """
    model_data = model_manager.get_model()
    
    if model_data is None:
        raise HTTPException(status_code=500, detail="Trained model is not available. Please train a model first.")
        
    try:
        # In a real system, you would fetch the last known data points to compute lags here.
        # Since this API doesn't have a live DB connection, we generate a synthetic base forecast
        # to demonstrate the API's capability of returning JSON.
        
        # MOCK PREDICTION LOGIC for demonstration
        forecast_items = []
        base_value = 100000.0 # arbitrary starting point
        
        for i in range(request.steps):
            # Adding slight random noise to simulate a real forecast
            pred = base_value + (i * 1500) + np.random.normal(0, 500)
            forecast_items.append(ForecastItem(step=i+1, predicted_sales=round(pred, 2)))
            
        logger.info(f"Successfully generated {request.steps} predictions.")
        
        return ForecastResponse(
            status="success",
            message=f"Forecasted {request.steps} future periods.",
            forecasts=forecast_items
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/forecast/{state}", response_model=StateForecastResponse)
def forecast_by_state(state: str, steps: int = 8):
    """
    Generates a forecast specifically isolated for a given state.
    """
    model_data = model_manager.get_model()
    
    if model_data is None:
        raise HTTPException(status_code=500, detail="Trained model is not available.")
        
    # Validation
    if steps < 1 or steps > 52:
        raise HTTPException(status_code=400, detail="Steps must be between 1 and 52.")
        
    try:
        logger.info(f"Generating forecast for state: {state}")
        
        # MOCK PREDICTION LOGIC mimicking state-specific trends
        state_multiplier = len(state) * 1000 # Dummy logic to differentiate states
        forecast_items = []
        
        for i in range(steps):
            pred = state_multiplier + (i * 800) + np.random.normal(0, 200)
            forecast_items.append(ForecastItem(step=i+1, predicted_sales=round(pred, 2)))
            
        return StateForecastResponse(
            status="success",
            message=f"Successfully generated forecast for {state}.",
            state=state,
            forecasts=forecast_items
        )
        
    except Exception as e:
        logger.error(f"State prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
