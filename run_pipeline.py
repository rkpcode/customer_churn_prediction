"""
Run Training Pipeline
Entry point for executing the training pipeline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from ecommerce_customer_churn.pipelines.training_pipeline import TrainingPipeline
from ecommerce_customer_churn.logger import logger


if __name__ == "__main__":
    try:
        logger.info("Starting training pipeline execution")
        
        pipeline = TrainingPipeline()
        model, model_name, metrics = pipeline.run_pipeline()
        
        print("\n" + "=" * 100)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 100)
        print(f"Best Model: {model_name}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print("=" * 100)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise e
