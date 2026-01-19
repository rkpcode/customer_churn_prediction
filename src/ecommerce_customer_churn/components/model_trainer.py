"""
Model Trainer Component
Trains multiple models following industry-grade approach from notebooks
Baseline-first: Dumb → Logistic Regression → Tree Ensembles
Includes threshold tuning and comprehensive evaluation
"""

import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost

from ecommerce_customer_churn.logger import logger
from ecommerce_customer_churn.exception import ChurnPredictionException
from ecommerce_customer_churn.utils import save_object, create_directories


@dataclass
class ModelTrainerConfig:
    """Configuration for model training"""
    models_dir: Path = Path("models")
    best_model_path: Path = Path("models/best_model.pkl")
    scaler_path: Path = Path("models/scaler.pkl")
    results_path: Path = Path("models/model_results.json")
    plots_dir: Path = Path("artifacts/plots")


class ModelTrainer:
    """
    Model Trainer Component for E-commerce Customer Churn Prediction
    Implements baseline-first approach with comprehensive evaluation
    """
    
    def __init__(self, config: ModelTrainerConfig = ModelTrainerConfig()):
        """
        Initialize Model Trainer component
        
        Args:
            config (ModelTrainerConfig): Configuration object
        """
        self.config = config
        self.results = {}
        logger.info("Model Trainer component initialized")
    
    def evaluate_model(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_pred_proba: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance with multiple metrics
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (np.ndarray): Predicted probabilities (optional)
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        try:
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1_score": f1_score(y_true, y_pred, zero_division=0),
            }
            
            if y_pred_proba is not None:
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
            else:
                metrics["roc_auc"] = 0.5
            
            return metrics
            
        except Exception as e:
            logger.error("Model evaluation failed")
            raise ChurnPredictionException(e, sys)
    
    def train_dumb_baseline(self, y_test: np.ndarray) -> Dict[str, float]:
        """
        Dumb baseline: Predict majority class (No Churn) for everyone
        
        Args:
            y_test (np.ndarray): Test labels
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        try:
            logger.info("Training Dumb Baseline (Majority Class Predictor)...")
            
            # Predict 0 (No Churn) for all samples
            y_pred = np.zeros(len(y_test))
            
            metrics = self.evaluate_model(y_test, y_pred)
            
            logger.info(f"Dumb Baseline - Accuracy: {metrics['accuracy']:.4f}, Recall: {metrics['recall']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error("Dumb baseline training failed")
            raise ChurnPredictionException(e, sys)
    
    def train_logistic_regression(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Tuple[Any, StandardScaler, Dict[str, float]]:
        """
        Train Logistic Regression with StandardScaler and class weights
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            Tuple: Trained model, scaler, and metrics
        """
        try:
            logger.info("Training Logistic Regression with class_weight='balanced'...")
            
            # Scale features (LR needs scaling)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Logistic Regression
            lr = LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000,
                solver='lbfgs'
            )
            lr.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = lr.predict(X_test_scaled)
            y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
            
            # Evaluate
            metrics = self.evaluate_model(y_test, y_pred, y_pred_proba)
            
            logger.info(f"Logistic Regression - ROC-AUC: {metrics['roc_auc']:.4f}, Recall: {metrics['recall']:.4f}")
            
            return lr, scaler, metrics
            
        except Exception as e:
            logger.error("Logistic Regression training failed")
            raise ChurnPredictionException(e, sys)
    
    def train_tree_models(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, Tuple[Any, Dict[str, float]]]:
        """
        Train tree-based ensemble models with class weights
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            Dict: Dictionary of trained models and their metrics
        """
        try:
            logger.info("Training tree ensemble models...")
            
            # Calculate scale_pos_weight for XGBoost
            scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
            
            # Initialize models with class weights
            models = {
                'Random Forest': RandomForestClassifier(
                    class_weight='balanced',
                    random_state=42,
                    n_estimators=100,
                    n_jobs=-1
                ),
                'Gradient Boosting': GradientBoostingClassifier(
                    random_state=42,
                    n_estimators=100
                ),
                'XGBoost': xgb.XGBClassifier(
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    eval_metric='logloss',
                    use_label_encoder=False,
                    verbosity=0
                ),
                'LightGBM': LGBMClassifier(
                    class_weight='balanced',
                    random_state=42,
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    verbose=-1
                ),
                'CatBoost': CatBoostClassifier(
                    auto_class_weights='Balanced',
                    random_state=42,
                    iterations=200,
                    depth=6,
                    learning_rate=0.1,
                    verbose=False
                )
            }
            
            results = {}
            
            for name, model in models.items():
                logger.info(f"Training {name}...")
                
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Evaluate
                metrics = self.evaluate_model(y_test, y_pred, y_pred_proba)
                
                results[name] = (model, metrics)
                
                logger.info(f"{name} - ROC-AUC: {metrics['roc_auc']:.4f}, Recall: {metrics['recall']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error("Tree model training failed")
            raise ChurnPredictionException(e, sys)
    
    def tune_threshold(
        self, 
        y_test: np.ndarray, 
        y_pred_proba: np.ndarray, 
        strategy: str = 'top_20_pct'
    ) -> Tuple[float, Dict[str, float]]:
        """
        Business-aligned threshold tuning
        
        Args:
            y_test (np.ndarray): Test labels
            y_pred_proba (np.ndarray): Predicted probabilities
            strategy (str): Threshold selection strategy
            
        Returns:
            Tuple: Optimal threshold and metrics
        """
        try:
            logger.info(f"Tuning threshold with strategy: {strategy}")
            
            if strategy == 'top_20_pct':
                # Contact top 20% highest-risk customers
                sorted_proba = np.sort(y_pred_proba)[::-1]
                threshold = sorted_proba[int(len(sorted_proba) * 0.20)]
            else:
                # Default to 0.5
                threshold = 0.5
            
            # Apply threshold
            y_pred_tuned = (y_pred_proba >= threshold).astype(int)
            
            # Evaluate
            metrics = self.evaluate_model(y_test, y_pred_tuned, y_pred_proba)
            
            logger.info(f"Threshold: {threshold:.3f}, Recall: {metrics['recall']:.4f}, Precision: {metrics['precision']:.4f}")
            
            return threshold, metrics
            
        except Exception as e:
            logger.error("Threshold tuning failed")
            raise ChurnPredictionException(e, sys)
    
    def plot_model_comparison(self, results: Dict[str, Dict[str, float]]):
        """
        Create model comparison visualizations
        
        Args:
            results (Dict): Dictionary of model results
        """
        try:
            logger.info("Creating model comparison plots...")
            
            create_directories([self.config.plots_dir])
            
            # Create comparison dataframe
            results_df = pd.DataFrame(results).T
            results_df = results_df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]
            
            # Plot comparison
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            metrics = ['roc_auc', 'recall', 'precision', 'f1_score']
            
            for idx, metric in enumerate(metrics):
                ax = axes[idx // 2, idx % 2]
                results_df[metric].plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
                ax.set_title(f'{metric.upper()} Comparison', fontsize=14, fontweight='bold')
                ax.set_ylabel(metric.upper())
                ax.set_xlabel('Model')
                ax.grid(axis='y', alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.config.plots_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Model comparison plot saved to: {self.config.plots_dir / 'model_comparison.png'}")
            
        except Exception as e:
            logger.warning(f"Failed to create plots: {e}")
    
    def initiate_model_training(
        self, 
        X_train_path: Path, 
        X_test_path: Path, 
        y_train_path: Path, 
        y_test_path: Path
    ) -> Tuple[Any, str, Dict[str, float]]:
        """
        Main method to train all models and select the best one
        
        Args:
            X_train_path (Path): Path to training features
            X_test_path (Path): Path to test features
            y_train_path (Path): Path to training labels
            y_test_path (Path): Path to test labels
            
        Returns:
            Tuple: Best model, model name, and metrics
        """
        try:
            logger.info("=" * 80)
            logger.info("MODEL TRAINING STARTED")
            logger.info("=" * 80)
            
            # Load data
            X_train = pd.read_csv(X_train_path).values
            X_test = pd.read_csv(X_test_path).values
            y_train = pd.read_csv(y_train_path).values.ravel()
            y_test = pd.read_csv(y_test_path).values.ravel()
            
            logger.info(f"Loaded data - Train: {X_train.shape}, Test: {X_test.shape}")
            
            # 1. Dumb Baseline
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 1: DUMB BASELINE")
            logger.info("=" * 80)
            dumb_metrics = self.train_dumb_baseline(y_test)
            self.results['Dumb Baseline'] = dumb_metrics
            
            # 2. Logistic Regression
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 2: LOGISTIC REGRESSION")
            logger.info("=" * 80)
            lr_model, scaler, lr_metrics = self.train_logistic_regression(X_train, y_train, X_test, y_test)
            self.results['Logistic Regression'] = lr_metrics
            
            # 3. Tree Ensembles
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 3: TREE ENSEMBLE MODELS")
            logger.info("=" * 80)
            tree_results = self.train_tree_models(X_train, y_train, X_test, y_test)
            
            # Store all tree model results
            for name, (model, metrics) in tree_results.items():
                self.results[name] = metrics
            
            # 4. Select best model based on ROC-AUC
            logger.info("\n" + "=" * 80)
            logger.info("MODEL SELECTION")
            logger.info("=" * 80)
            
            # Exclude dumb baseline from selection
            model_scores = {k: v['roc_auc'] for k, v in self.results.items() if k != 'Dumb Baseline'}
            best_model_name = max(model_scores, key=model_scores.get)
            best_metrics = self.results[best_model_name]
            
            # Get best model object
            if best_model_name == 'Logistic Regression':
                best_model = lr_model
                best_scaler = scaler
            else:
                best_model = tree_results[best_model_name][0]
                best_scaler = None
            
            logger.info(f"BEST MODEL: {best_model_name}")
            logger.info(f"ROC-AUC: {best_metrics['roc_auc']:.4f}")
            logger.info(f"F1-Score: {best_metrics['f1_score']:.4f}")
            logger.info(f"Precision: {best_metrics['precision']:.4f}")
            logger.info(f"Recall: {best_metrics['recall']:.4f}")
            
            # 5. Threshold tuning for best model
            logger.info("\n" + "=" * 80)
            logger.info("THRESHOLD TUNING")
            logger.info("=" * 80)
            
            # Get predictions from best model
            if best_model_name == 'Logistic Regression':
                X_test_scaled = scaler.transform(X_test)
                y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            threshold, tuned_metrics = self.tune_threshold(y_test, y_pred_proba)
            self.results[f'{best_model_name} (Tuned)'] = tuned_metrics
            self.results['threshold'] = threshold
            
            # 6. Save artifacts
            create_directories([self.config.models_dir])
            
            # Save best model
            save_object(self.config.best_model_path, best_model)
            logger.info(f"Best model saved to: {self.config.best_model_path}")
            
            # Save scaler if exists
            if best_scaler is not None:
                save_object(self.config.scaler_path, best_scaler)
                logger.info(f"Scaler saved to: {self.config.scaler_path}")
            
            # Save results
            with open(self.config.results_path, 'w') as f:
                json.dump(self.results, f, indent=4)
            logger.info(f"Results saved to: {self.config.results_path}")
            
            # Create plots
            self.plot_model_comparison(self.results)
            
            logger.info("=" * 80)
            logger.info("MODEL TRAINING COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            
            return best_model, best_model_name, best_metrics
            
        except Exception as e:
            logger.error("Model training failed")
            raise ChurnPredictionException(e, sys)


if __name__ == "__main__":
    # Test model training
    from ecommerce_customer_churn.components.data_ingestion import DataIngestion
    from ecommerce_customer_churn.components.data_transformation import DataTransformation
    
    # Run data ingestion
    data_ingestion = DataIngestion()
    raw_data_path = data_ingestion.initiate_data_ingestion()
    
    # Run data transformation
    data_transformation = DataTransformation()
    paths = data_transformation.initiate_data_transformation(raw_data_path)
    
    # Run model training (using Phase 2 features)
    model_trainer = ModelTrainer()
    best_model, model_name, metrics = model_trainer.initiate_model_training(
        paths['X_train_phase2'],
        paths['X_test_phase2'],
        paths['y_train'],
        paths['y_test']
    )
    
    print(f"\nBest Model: {model_name}")
    print(f"Metrics: {metrics}")
