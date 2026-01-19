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
from ecommerce_customer_churn.components.model_evalution import ModelEvaluation


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
            tuple: Best model, model name, and metrics
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
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            
            # Step 2: Data Transformation
            logger.info("\n" + "=" * 100)
            logger.info("STEP 2: DATA TRANSFORMATION")
            logger.info("=" * 100)
            data_transformation = DataTransformation()
            X_train, y_train, X_test, y_test, preprocessor_path = data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )
            
            # Step 3: Model Training
            logger.info("\n" + "=" * 100)
            logger.info("STEP 3: MODEL TRAINING")
            logger.info("=" * 100)
            model_trainer = ModelTrainer()
            best_model, model_name, metrics = model_trainer.initiate_model_training(
                X_train, y_train, X_test, y_test
            )
            
            # Step 4: Model Evaluation
            logger.info("\n" + "=" * 100)
            logger.info("STEP 4: MODEL EVALUATION")
            logger.info("=" * 100)
            model_evaluation = ModelEvaluation()
            model_evaluation.evaluate_model(best_model, model_name, X_test, y_test)
            
            logger.info("\n" + "=" * 100)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 100)
            logger.info(f"Best Model: {model_name}")
            logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")
            logger.info("=" * 100)
            
            return best_model, model_name, metrics
            
        except Exception as e:
            logger.error("Training pipeline failed")
            raise ChurnPredictionException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    model, name, metrics = pipeline.run_pipeline()
    print(f"\nTraining completed successfully!")
    print(f"Best Model: {name}")
    print(f"Metrics: {metrics}")
