"""
Prediction Pipeline
Handles single and batch predictions for churn prediction
"""

import sys
from pathlib import Path
from typing import Union, List, Dict
import pandas as pd
import numpy as np

from ecommerce_customer_churn.logger import logger
from ecommerce_customer_churn.exception import ChurnPredictionException
from ecommerce_customer_churn.utils import load_object


class PredictionPipeline:
    """
    Prediction Pipeline for E-commerce Customer Churn Prediction
    """
    
    def __init__(
        self,
        model_path: Path = Path("models/best_model.pkl"),
        preprocessor_path: Path = Path("artifacts/preprocessor.pkl")
    ):
        """
        Initialize Prediction Pipeline
        
        Args:
            model_path (Path): Path to trained model
            preprocessor_path (Path): Path to preprocessor
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.preprocessor = None
        logger.info("Prediction Pipeline initialized")
    
    def load_model_and_preprocessor(self):
        """Load trained model and preprocessor"""
        try:
            if self.model is None:
                logger.info(f"Loading model from: {self.model_path}")
                self.model = load_object(self.model_path)
            
            if self.preprocessor is None:
                logger.info(f"Loading preprocessor from: {self.preprocessor_path}")
                self.preprocessor = load_object(self.preprocessor_path)
            
            logger.info("Model and preprocessor loaded successfully")
            
        except Exception as e:
            logger.error("Failed to load model or preprocessor")
            raise ChurnPredictionException(e, sys)
    
    def predict(self, input_data: Union[pd.DataFrame, Dict]) -> Dict[str, any]:
        """
        Make prediction for single or batch input
        
        Args:
            input_data (Union[pd.DataFrame, Dict]): Input data
            
        Returns:
            Dict[str, any]: Prediction results with churn probability
        """
        try:
            # Load model and preprocessor if not already loaded
            self.load_model_and_preprocessor()
            
            # Convert dict to DataFrame if necessary
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            else:
                input_df = input_data.copy()
            
            logger.info(f"Making prediction for {len(input_df)} sample(s)")
            
            # Drop CustomerID if present
            if 'CustomerID' in input_df.columns:
                customer_ids = input_df['CustomerID'].tolist()
                input_df = input_df.drop(columns=['CustomerID'])
            else:
                customer_ids = [f"Customer_{i}" for i in range(len(input_df))]
            
            # Transform input data
            input_transformed = self.preprocessor.transform(input_df)
            
            # Make predictions
            predictions = self.model.predict(input_transformed)
            probabilities = self.model.predict_proba(input_transformed)[:, 1]
            
            # Prepare results
            results = []
            for i, (customer_id, pred, prob) in enumerate(zip(customer_ids, predictions, probabilities)):
                result = {
                    "customer_id": customer_id,
                    "churn_prediction": int(pred),
                    "churn_probability": float(prob),
                    "churn_status": "Will Churn" if pred == 1 else "Will Not Churn",
                    "confidence": float(max(prob, 1 - prob))
                }
                results.append(result)
            
            logger.info(f"Predictions completed for {len(results)} sample(s)")
            
            # Return single result or list based on input
            if len(results) == 1:
                return results[0]
            else:
                return {"predictions": results}
            
        except Exception as e:
            logger.error("Prediction failed")
            raise ChurnPredictionException(e, sys)


class CustomData:
    """
    Custom data class for creating input data from individual features
    """
    
    def __init__(
        self,
        Tenure: int,
        CityTier: int,
        WarehouseToHome: float,
        HourSpendOnApp: float,
        NumberOfDeviceRegistered: int,
        SatisfactionScore: int,
        NumberOfAddress: int,
        Complain: int,
        OrderAmountHikeFromlastYear: float,
        CouponUsed: int,
        OrderCount: int,
        DaySinceLastOrder: int,
        CashbackAmount: float,
        PreferredLoginDevice: str,
        PreferredPaymentMode: str,
        Gender: str,
        PreferedOrderCat: str,
        MaritalStatus: str
    ):
        """Initialize custom data with all required features"""
        self.Tenure = Tenure
        self.CityTier = CityTier
        self.WarehouseToHome = WarehouseToHome
        self.HourSpendOnApp = HourSpendOnApp
        self.NumberOfDeviceRegistered = NumberOfDeviceRegistered
        self.SatisfactionScore = SatisfactionScore
        self.NumberOfAddress = NumberOfAddress
        self.Complain = Complain
        self.OrderAmountHikeFromlastYear = OrderAmountHikeFromlastYear
        self.CouponUsed = CouponUsed
        self.OrderCount = OrderCount
        self.DaySinceLastOrder = DaySinceLastOrder
        self.CashbackAmount = CashbackAmount
        self.PreferredLoginDevice = PreferredLoginDevice
        self.PreferredPaymentMode = PreferredPaymentMode
        self.Gender = Gender
        self.PreferedOrderCat = PreferedOrderCat
        self.MaritalStatus = MaritalStatus
    
    def get_data_as_dataframe(self) -> pd.DataFrame:
        """
        Convert custom data to DataFrame
        
        Returns:
            pd.DataFrame: Input data as DataFrame
        """
        try:
            data_dict = {
                "Tenure": [self.Tenure],
                "CityTier": [self.CityTier],
                "WarehouseToHome": [self.WarehouseToHome],
                "HourSpendOnApp": [self.HourSpendOnApp],
                "NumberOfDeviceRegistered": [self.NumberOfDeviceRegistered],
                "SatisfactionScore": [self.SatisfactionScore],
                "NumberOfAddress": [self.NumberOfAddress],
                "Complain": [self.Complain],
                "OrderAmountHikeFromlastYear": [self.OrderAmountHikeFromlastYear],
                "CouponUsed": [self.CouponUsed],
                "OrderCount": [self.OrderCount],
                "DaySinceLastOrder": [self.DaySinceLastOrder],
                "CashbackAmount": [self.CashbackAmount],
                "PreferredLoginDevice": [self.PreferredLoginDevice],
                "PreferredPaymentMode": [self.PreferredPaymentMode],
                "Gender": [self.Gender],
                "PreferedOrderCat": [self.PreferedOrderCat],
                "MaritalStatus": [self.MaritalStatus]
            }
            
            return pd.DataFrame(data_dict)
            
        except Exception as e:
            logger.error("Failed to convert custom data to DataFrame")
            raise ChurnPredictionException(e, sys)


if __name__ == "__main__":
    # Test prediction pipeline
    custom_data = CustomData(
        Tenure=10,
        CityTier=1,
        WarehouseToHome=15.0,
        HourSpendOnApp=3.0,
        NumberOfDeviceRegistered=3,
        SatisfactionScore=3,
        NumberOfAddress=2,
        Complain=0,
        OrderAmountHikeFromlastYear=15.0,
        CouponUsed=5,
        OrderCount=10,
        DaySinceLastOrder=5,
        CashbackAmount=150.0,
        PreferredLoginDevice="Mobile Phone",
        PreferredPaymentMode="Debit Card",
        Gender="Male",
        PreferedOrderCat="Laptop & Accessory",
        MaritalStatus="Single"
    )
    
    input_df = custom_data.get_data_as_dataframe()
    
    pipeline = PredictionPipeline()
    result = pipeline.predict(input_df)
    
    print(f"Prediction Result: {result}")
