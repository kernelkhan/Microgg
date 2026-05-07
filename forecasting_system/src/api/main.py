from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys

# Ensure paths are correct when running from terminal
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.api.routes import router

# Setup global logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

app = FastAPI(
    title="Time Series Forecasting API",
    description="Production-ready REST API serving the best forecasting model. Automatically generated Swagger UI available.",
    version="1.0.0"
)

# CORS configuration for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the endpoints defined in routes.py
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    # Typically run via: uvicorn src.api.main:app --reload
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
