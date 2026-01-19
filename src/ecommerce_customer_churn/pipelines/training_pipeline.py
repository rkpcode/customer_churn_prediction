"""
Training Pipeline
End-to-end pipeline for training the churn prediction model
"""

import sys
from pathlib import Path

from ecommerce_customer_churn.logger import logger
from ecommerce_customer_churn.exception import ChurnPredictionException
from ecommerce_customer_churn.components.data_ingestion import DataIngestion
from ecommerce_customer_churn.components.data_transformation import DataTransformation
from ecommerce_customer_churn.components.model_trainer import ModelTrainer


class TrainingPipeline:
    """
    End-to-end training pipeline for E-commerce Customer Churn Prediction
    """
    
    def __init__(self):
        """Initialize Training Pipeline"""
        logger.info("Training Pipeline initialized")
    
    def run_pipeline(self):
        """
        Execute the complete training pipeline
        
        Returns:
            str: Path to best model
        """
        try:
            logger.info("=" * 100)
            logger.info("STARTING TRAINING PIPELINE")
            logger.info("=" * 100)
            
            # Step 1: Data Ingestion
            logger.info("\n" + "=" * 100)
            logger.info("STEP 1: DATA INGESTION")
            logger.info("=" * 100)
            data_ingestion = DataIngestion()
            raw_data_path = data_ingestion.initiate_data_ingestion()
            
            # Step 2: Data Transformation
            logger.info("\n" + "=" * 100)
            logger.info("STEP 2: DATA TRANSFORMATION")
            logger.info("=" * 100)
            data_transformation = DataTransformation()
            feature_paths = data_transformation.initiate_data_transformation(raw_data_path)
            
            # Step 3: Model Training
            logger.info("\n" + "=" * 100)
            logger.info("STEP 3: MODEL TRAINING")
            logger.info("=" * 100)
            model_trainer = ModelTrainer()
            best_model_path = model_trainer.initiate_model_training(
                X_train_path=Path("data/processed/X_train_phase2.csv"),
                X_test_path=Path("data/processed/X_test_phase2.csv"),
                y_train_path=Path("data/processed/y_train.csv"),
                y_test_path=Path("data/processed/y_test.csv")
            )
            
            logger.info("\n" + "=" * 100)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 100)
            logger.info(f"Best Model saved at: {best_model_path}")
            logger.info("=" * 100)
            
            return best_model_path
            
        except Exception as e:
            logger.error("Training pipeline failed")
            raise ChurnPredictionException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    model_path = pipeline.run_pipeline()
    print(f"\n✅ Training completed successfully!")
    print(f"📦 Best Model: {model_path}")
