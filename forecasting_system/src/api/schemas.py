from pydantic import BaseModel, Field
from typing import List

class ForecastItem(BaseModel):
    step: int
    predicted_sales: float

class PredictRequest(BaseModel):
    steps: int = Field(default=8, ge=1, le=52, description="Number of future periods to forecast.")

class ForecastResponse(BaseModel):
    status: str
    message: str
    forecasts: List[ForecastItem]

class StateForecastResponse(ForecastResponse):
    state: str
