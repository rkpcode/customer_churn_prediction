"""
Utility Functions for E-commerce Customer Churn Prediction
Common helper functions for file I/O, configuration, and MLflow operations
"""

import os
import sys
import yaml
import json
import joblib
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
from box import ConfigBox
from ensure import ensure_annotations
import mlflow
from mlflow.tracking import MlflowClient

from ecommerce_customer_churn.logger import logger
from ecommerce_customer_churn.exception import ChurnPredictionException


@ensure_annotations
def read_yaml(file_path: Path) -> ConfigBox:
    """
    Read YAML file and return ConfigBox object
    
    Args:
        file_path (Path): Path to YAML file
        
    Returns:
        ConfigBox: Configuration as ConfigBox object
        
    Raises:
        ChurnPredictionException: If file reading fails
    """
    try:
        with open(file_path, 'r') as file:
            content = yaml.safe_load(file)
            logger.info(f"YAML file loaded successfully from: {file_path}")
            return ConfigBox(content)
    except Exception as e:
        logger.error(f"Failed to read YAML file: {file_path}")
        raise ChurnPredictionException(e, sys)


@ensure_annotations
def save_json(file_path: Path, data: dict):
    """
    Save dictionary as JSON file
    
    Args:
        file_path (Path): Path to save JSON file
        data (dict): Data to save
        
    Raises:
        ChurnPredictionException: If file saving fails
    """
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
        logger.info(f"JSON file saved successfully at: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON file: {file_path}")
        raise ChurnPredictionException(e, sys)


@ensure_annotations
def load_json(file_path: Path) -> ConfigBox:
    """
    Load JSON file and return ConfigBox object
    
    Args:
        file_path (Path): Path to JSON file
        
    Returns:
        ConfigBox: JSON content as ConfigBox object
        
    Raises:
        ChurnPredictionException: If file loading fails
    """
    try:
        with open(file_path, 'r') as file:
            content = json.load(file)
        logger.info(f"JSON file loaded successfully from: {file_path}")
        return ConfigBox(content)
    except Exception as e:
        logger.error(f"Failed to load JSON file: {file_path}")
        raise ChurnPredictionException(e, sys)


def save_object(file_path: Path, obj):
    """
    Save Python object using joblib
    
    Args:
        file_path (Path): Path to save object
        obj: Object to save
        
    Raises:
        ChurnPredictionException: If saving fails
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file:
            joblib.dump(obj, file)
        
        logger.info(f"Object saved successfully at: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save object: {file_path}")
        raise ChurnPredictionException(e, sys)


def load_object(file_path: Path):
    """
    Load Python object using joblib
    
    Args:
        file_path (Path): Path to object file
        
    Returns:
        Loaded object
        
    Raises:
        ChurnPredictionException: If loading fails
    """
    try:
        with open(file_path, 'rb') as file:
            obj = joblib.load(file)
        logger.info(f"Object loaded successfully from: {file_path}")
        return obj
    except Exception as e:
        logger.error(f"Failed to load object: {file_path}")
        raise ChurnPredictionException(e, sys)


@ensure_annotations
def create_directories(path_list: list, verbose: bool = True):
    """
    Create list of directories
    
    Args:
        path_list (list): List of directory paths
        verbose (bool): Whether to log directory creation
    """
    for path in path_list:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory: {path}")


# ============================================
# MLflow Helper Functions
# ============================================

def setup_mlflow(experiment_name: str, tracking_uri: Optional[str] = None):
    """
    Setup MLflow experiment and tracking URI
    
    Args:
        experiment_name (str): Name of the MLflow experiment
        tracking_uri (Optional[str]): MLflow tracking URI
    """
    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI set to: {tracking_uri}")
        
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set to: {experiment_name}")
    except Exception as e:
        logger.error(f"Failed to setup MLflow: {e}")
        raise ChurnPredictionException(e, sys)


def log_model_to_mlflow(model, model_name: str, signature=None, input_example=None):
    """
    Log model to MLflow
    
    Args:
        model: Trained model object
        model_name (str): Name for the model
        signature: MLflow model signature
        input_example: Example input for the model
    """
    try:
        mlflow.sklearn.log_model(
            model,
            model_name,
            signature=signature,
            input_example=input_example
        )
        logger.info(f"Model '{model_name}' logged to MLflow successfully")
    except Exception as e:
        logger.error(f"Failed to log model to MLflow: {e}")
        raise ChurnPredictionException(e, sys)


def get_best_model_from_mlflow(experiment_name: str, metric: str = "roc_auc"):
    """
    Get the best model from MLflow based on a metric
    
    Args:
        experiment_name (str): MLflow experiment name
        metric (str): Metric to optimize (default: roc_auc)
        
    Returns:
        Best model run information
    """
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            logger.warning(f"Experiment '{experiment_name}' not found")
            return None
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1
        )
        
        if runs:
            best_run = runs[0]
            logger.info(f"Best model found with {metric}: {best_run.data.metrics.get(metric)}")
            return best_run
        else:
            logger.warning("No runs found in the experiment")
            return None
            
    except Exception as e:
        logger.error(f"Failed to get best model from MLflow: {e}")
        raise ChurnPredictionException(e, sys)
