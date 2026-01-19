"""
Model Evaluation Component
Comprehensive model evaluation with plots, SHAP values, and detailed metrics
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    average_precision_score
)
import shap

from ecommerce_customer_churn.logger import logger
from ecommerce_customer_churn.exception import ChurnPredictionException
from ecommerce_customer_churn.utils import create_directories


@dataclass
class ModelEvaluationConfig:
    """Configuration for model evaluation"""
    plots_dir: Path = Path("artifacts/plots")
    reports_dir: Path = Path("artifacts/reports")


class ModelEvaluation:
    """
    Model Evaluation Component for E-commerce Customer Churn Prediction
    Generates comprehensive evaluation plots and reports
    """
    
    def __init__(self, config: ModelEvaluationConfig = ModelEvaluationConfig()):
        """
        Initialize Model Evaluation component
        
        Args:
            config (ModelEvaluationConfig): Configuration object
        """
        self.config = config
        create_directories([self.config.plots_dir, self.config.reports_dir])
        logger.info("Model Evaluation component initialized")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
        """
        Plot and save confusion matrix
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            model_name (str): Name of the model
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            save_path = self.config.plots_dir / f"{model_name}_confusion_matrix.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Confusion matrix saved to: {save_path}")
            
        except Exception as e:
            logger.error("Failed to plot confusion matrix")
            raise ChurnPredictionException(e, sys)
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, model_name: str):
        """
        Plot and save ROC curve
        
        Args:
            y_true (np.ndarray): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
            model_name (str): Name of the model
        """
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            
            save_path = self.config.plots_dir / f"{model_name}_roc_curve.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"ROC curve saved to: {save_path}")
            
        except Exception as e:
            logger.error("Failed to plot ROC curve")
            raise ChurnPredictionException(e, sys)
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, model_name: str):
        """
        Plot and save Precision-Recall curve
        
        Args:
            y_true (np.ndarray): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
            model_name (str): Name of the model
        """
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            avg_precision = average_precision_score(y_true, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {model_name}')
            plt.legend(loc="lower left")
            plt.grid(alpha=0.3)
            
            save_path = self.config.plots_dir / f"{model_name}_precision_recall_curve.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Precision-Recall curve saved to: {save_path}")
            
        except Exception as e:
            logger.error("Failed to plot Precision-Recall curve")
            raise ChurnPredictionException(e, sys)
    
    def plot_feature_importance(self, model: Any, model_name: str, top_n: int = 20):
        """
        Plot and save feature importance
        
        Args:
            model: Trained model
            model_name (str): Name of the model
            top_n (int): Number of top features to display
        """
        try:
            if not hasattr(model, 'feature_importances_'):
                logger.warning(f"{model_name} does not have feature_importances_ attribute")
                return
            
            feature_importance = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(model.feature_importances_))],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(top_n)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
            plt.title(f'Top {top_n} Feature Importance - {model_name}')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            
            save_path = self.config.plots_dir / f"{model_name}_feature_importance.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Feature importance plot saved to: {save_path}")
            
        except Exception as e:
            logger.error("Failed to plot feature importance")
            raise ChurnPredictionException(e, sys)
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
        """
        Generate and save classification report
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            model_name (str): Name of the model
        """
        try:
            report = classification_report(y_true, y_pred, target_names=['Not Churned', 'Churned'])
            
            report_path = self.config.reports_dir / f"{model_name}_classification_report.txt"
            with open(report_path, 'w') as f:
                f.write(f"Classification Report - {model_name}\n")
                f.write("=" * 60 + "\n")
                f.write(report)
            
            logger.info(f"Classification report saved to: {report_path}")
            logger.info(f"\n{report}")
            
        except Exception as e:
            logger.error("Failed to generate classification report")
            raise ChurnPredictionException(e, sys)
    
    def generate_shap_values(self, model: Any, X_test: np.ndarray, model_name: str, max_samples: int = 100):
        """
        Generate SHAP values for model interpretability
        
        Args:
            model: Trained model
            X_test (np.ndarray): Test features
            model_name (str): Name of the model
            max_samples (int): Maximum number of samples for SHAP calculation
        """
        try:
            logger.info(f"Generating SHAP values for {model_name}...")
            
            # Use a subset of data for faster computation
            X_sample = X_test[:max_samples]
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, show=False)
            save_path = self.config.plots_dir / f"{model_name}_shap_summary.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"SHAP summary plot saved to: {save_path}")
            
        except Exception as e:
            logger.warning(f"SHAP value generation failed: {e}")
            # Don't raise exception as SHAP is optional
    
    def evaluate_model(
        self,
        model: Any,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray
    ):
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            model_name (str): Name of the model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
        """
        try:
            logger.info("=" * 80)
            logger.info(f"EVALUATING MODEL: {model_name}")
            logger.info("=" * 80)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Generate all evaluation artifacts
            self.plot_confusion_matrix(y_test, y_pred, model_name)
            self.plot_roc_curve(y_test, y_pred_proba, model_name)
            self.plot_precision_recall_curve(y_test, y_pred_proba, model_name)
            self.plot_feature_importance(model, model_name)
            self.generate_classification_report(y_test, y_pred, model_name)
            self.generate_shap_values(model, X_test, model_name)
            
            logger.info("=" * 80)
            logger.info("MODEL EVALUATION COMPLETED")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error("Model evaluation failed")
            raise ChurnPredictionException(e, sys)


if __name__ == "__main__":
    # Test model evaluation
    from ecommerce_customer_churn.components.data_ingestion import DataIngestion
    from ecommerce_customer_churn.components.data_transformation import DataTransformation
    from ecommerce_customer_churn.components.model_trainer import ModelTrainer
    
    # Run pipeline
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    X_train, y_train, X_test, y_test, _ = data_transformation.initiate_data_transformation(
        train_data, test_data
    )
    
    model_trainer = ModelTrainer()
    best_model, model_name, _ = model_trainer.initiate_model_training(
        X_train, y_train, X_test, y_test
    )
    
    # Evaluate model
    model_evaluation = ModelEvaluation()
    model_evaluation.evaluate_model(best_model, model_name, X_test, y_test)
