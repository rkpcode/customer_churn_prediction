"""
FastAPI Application for E-commerce Customer Churn Prediction
Production-ready API with automatic documentation
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import uvicorn
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from ecommerce_customer_churn.pipelines.prediction_pipeline import PredictionPipeline, CustomData
from ecommerce_customer_churn.logger import logger
from ecommerce_customer_churn.exception import ChurnPredictionException


# Initialize FastAPI app
app = FastAPI(
    title="E-commerce Customer Churn Prediction API",
    description="Production-grade API for predicting customer churn using XGBoost/CatBoost/LightGBM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize prediction pipeline
prediction_pipeline = PredictionPipeline()


# Pydantic models for request/response
class CustomerFeatures(BaseModel):
    """Customer features for churn prediction"""
    Tenure: int = Field(..., description="Number of months the customer has stayed with the company", ge=0)
    CityTier: int = Field(..., description="City tier (1, 2, or 3)", ge=1, le=3)
    WarehouseToHome: float = Field(..., description="Distance from warehouse to home in km", ge=0)
    HourSpendOnApp: float = Field(..., description="Hours spent on mobile app or website", ge=0)
    NumberOfDeviceRegistered: int = Field(..., description="Number of devices registered", ge=1)
    SatisfactionScore: int = Field(..., description="Customer satisfaction score (1-5)", ge=1, le=5)
    NumberOfAddress: int = Field(..., description="Number of addresses added", ge=1)
    Complain: int = Field(..., description="Complaint raised in last month (0 or 1)", ge=0, le=1)
    OrderAmountHikeFromlastYear: float = Field(..., description="Order amount hike from last year (%)", ge=0)
    CouponUsed: int = Field(..., description="Number of coupons used in last month", ge=0)
    OrderCount: int = Field(..., description="Number of orders placed in last month", ge=0)
    DaySinceLastOrder: int = Field(..., description="Days since last order", ge=0)
    CashbackAmount: float = Field(..., description="Average cashback amount", ge=0)
    PreferredLoginDevice: str = Field(..., description="Preferred login device")
    PreferredPaymentMode: str = Field(..., description="Preferred payment mode")
    Gender: str = Field(..., description="Customer gender")
    PreferedOrderCat: str = Field(..., description="Preferred order category")
    MaritalStatus: str = Field(..., description="Marital status")
    
    class Config:
        schema_extra = {
            "example": {
                "Tenure": 10,
                "CityTier": 1,
                "WarehouseToHome": 15.0,
                "HourSpendOnApp": 3.0,
                "NumberOfDeviceRegistered": 3,
                "SatisfactionScore": 3,
                "NumberOfAddress": 2,
                "Complain": 0,
                "OrderAmountHikeFromlastYear": 15.0,
                "CouponUsed": 5,
                "OrderCount": 10,
                "DaySinceLastOrder": 5,
                "CashbackAmount": 150.0,
                "PreferredLoginDevice": "Mobile Phone",
                "PreferredPaymentMode": "Debit Card",
                "Gender": "Male",
                "PreferedOrderCat": "Laptop & Accessory",
                "MaritalStatus": "Single"
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    customer_id: str
    churn_prediction: int
    churn_probability: float
    churn_status: str
    confidence: float


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse]


# API Endpoints

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "E-commerce Customer Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        # Check if model and preprocessor are loaded
        prediction_pipeline.load_model_and_preprocessor()
        return {
            "status": "healthy",
            "model_loaded": True,
            "preprocessor_loaded": True
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_churn(customer: CustomerFeatures):
    """
    Predict churn for a single customer
    
    Args:
        customer (CustomerFeatures): Customer features
        
    Returns:
        PredictionResponse: Prediction result
    """
    try:
        logger.info("Received prediction request for single customer")
        
        # Convert to DataFrame
        input_df = pd.DataFrame([customer.dict()])
        
        # Make prediction
        result = prediction_pipeline.predict(input_df)
        
        logger.info(f"Prediction completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(customers: List[CustomerFeatures]):
    """
    Predict churn for multiple customers
    
    Args:
        customers (List[CustomerFeatures]): List of customer features
        
    Returns:
        BatchPredictionResponse: Batch prediction results
    """
    try:
        logger.info(f"Received batch prediction request for {len(customers)} customers")
        
        # Convert to DataFrame
        input_df = pd.DataFrame([customer.dict() for customer in customers])
        
        # Make predictions
        result = prediction_pipeline.predict(input_df)
        
        logger.info(f"Batch prediction completed for {len(customers)} customers")
        return result
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_csv", tags=["Prediction"])
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Predict churn from uploaded CSV file
    
    Args:
        file (UploadFile): CSV file with customer data
        
    Returns:
        dict: Prediction results
    """
    try:
        logger.info(f"Received CSV file: {file.filename}")
        
        # Read CSV file
        df = pd.read_csv(file.file)
        logger.info(f"CSV loaded with {len(df)} rows")
        
        # Make predictions
        result = prediction_pipeline.predict(df)
        
        logger.info(f"CSV prediction completed for {len(df)} customers")
        return result
        
    except Exception as e:
        logger.error(f"CSV prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
