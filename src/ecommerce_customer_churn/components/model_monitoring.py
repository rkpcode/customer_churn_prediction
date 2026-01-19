"""
Model Monitoring Component using Evidently AI
Tracks data drift, model performance, and generates monitoring reports
"""

import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd
import numpy as np
from datetime import datetime

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.metrics import *

from ecommerce_customer_churn.logger import logger
from ecommerce_customer_churn.exception import ChurnPredictionException


@dataclass
class ModelMonitoringConfig:
    """Configuration for model monitoring"""
    reports_dir: Path = Path("reports")
    drift_report_path: Path = Path("reports/data_drift_report.html")
    quality_report_path: Path = Path("reports/data_quality_report.html")
    performance_report_path: Path = Path("reports/model_performance_report.html")
    drift_threshold: float = 0.5  # Threshold for drift detection


class ModelMonitoring:
    """
    Model Monitoring Component using Evidently AI
    Detects data drift, monitors data quality, and tracks model performance
    """
    
    def __init__(self, config: ModelMonitoringConfig = ModelMonitoringConfig()):
        """
        Initialize Model Monitoring component
        
        Args:
            config (ModelMonitoringConfig): Configuration object
        """
        self.config = config
        os.makedirs(self.config.reports_dir, exist_ok=True)
        logger.info("Model Monitoring component initialized")
    
    def setup_column_mapping(self, target_col: str = 'Churn', prediction_col: Optional[str] = None) -> ColumnMapping:
        """
        Setup column mapping for Evidently
        
        Args:
            target_col (str): Name of target column
            prediction_col (Optional[str]): Name of prediction column
            
        Returns:
            ColumnMapping: Evidently column mapping object
        """
        try:
            # Define numerical and categorical features
            numerical_features = [
                'Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp',
                'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
                'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
                'DaySinceLastOrder', 'CashbackAmount', 'order_frequency', 'complaint_rate'
            ]
            
            categorical_features = [
                'PreferredLoginDevice', 'PreferredPaymentMode', 'Gender',
                'PreferedOrderCat', 'MaritalStatus', 'Complain'
            ]
            
            column_mapping = ColumnMapping()
            column_mapping.target = target_col
            column_mapping.prediction = prediction_col
            column_mapping.numerical_features = numerical_features
            column_mapping.categorical_features = categorical_features
            
            return column_mapping
            
        except Exception as e:
            logger.error("Failed to setup column mapping")
            raise ChurnPredictionException(e, sys)
    
    def generate_data_drift_report(
        self, 
        reference_data: pd.DataFrame, 
        current_data: pd.DataFrame,
        target_col: str = 'Churn'
    ) -> Dict:
        """
        Generate data drift report comparing reference and current data
        
        Args:
            reference_data (pd.DataFrame): Reference dataset (training data)
            current_data (pd.DataFrame): Current dataset (new production data)
            target_col (str): Name of target column
            
        Returns:
            Dict: Drift detection results
        """
        try:
            logger.info("Generating data drift report...")
            
            # Setup column mapping
            column_mapping = self.setup_column_mapping(target_col=target_col)
            
            # Create drift report
            drift_report = Report(metrics=[
                DataDriftPreset(),
                TargetDriftPreset()
            ])
            
            # Run report
            drift_report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=column_mapping
            )
            
            # Save HTML report
            drift_report.save_html(str(self.config.drift_report_path))
            logger.info(f"Data drift report saved to: {self.config.drift_report_path}")
            
            # Extract drift metrics
            drift_results = drift_report.as_dict()
            
            # Check for drift
            dataset_drift = drift_results['metrics'][0]['result']['dataset_drift']
            drift_share = drift_results['metrics'][0]['result']['share_of_drifted_columns']
            
            drift_summary = {
                'dataset_drift_detected': dataset_drift,
                'drift_share': drift_share,
                'timestamp': datetime.now().isoformat(),
                'report_path': str(self.config.drift_report_path)
            }
            
            if dataset_drift:
                logger.warning(f"⚠️ DATA DRIFT DETECTED! Drift share: {drift_share:.2%}")
            else:
                logger.info(f"✅ No significant data drift. Drift share: {drift_share:.2%}")
            
            return drift_summary
            
        except Exception as e:
            logger.error("Failed to generate data drift report")
            raise ChurnPredictionException(e, sys)
    
    def generate_data_quality_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        target_col: str = 'Churn'
    ) -> Dict:
        """
        Generate data quality report
        
        Args:
            reference_data (pd.DataFrame): Reference dataset
            current_data (pd.DataFrame): Current dataset
            target_col (str): Name of target column
            
        Returns:
            Dict: Data quality metrics
        """
        try:
            logger.info("Generating data quality report...")
            
            # Setup column mapping
            column_mapping = self.setup_column_mapping(target_col=target_col)
            
            # Create quality report
            quality_report = Report(metrics=[
                DataQualityPreset()
            ])
            
            # Run report
            quality_report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=column_mapping
            )
            
            # Save HTML report
            quality_report.save_html(str(self.config.quality_report_path))
            logger.info(f"Data quality report saved to: {self.config.quality_report_path}")
            
            quality_summary = {
                'timestamp': datetime.now().isoformat(),
                'report_path': str(self.config.quality_report_path)
            }
            
            return quality_summary
            
        except Exception as e:
            logger.error("Failed to generate data quality report")
            raise ChurnPredictionException(e, sys)
    
    def generate_model_performance_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        target_col: str = 'Churn',
        prediction_col: str = 'prediction'
    ) -> Dict:
        """
        Generate model performance report
        
        Args:
            reference_data (pd.DataFrame): Reference dataset with predictions
            current_data (pd.DataFrame): Current dataset with predictions
            target_col (str): Name of target column
            prediction_col (str): Name of prediction column
            
        Returns:
            Dict: Performance metrics
        """
        try:
            logger.info("Generating model performance report...")
            
            # Setup column mapping
            column_mapping = self.setup_column_mapping(
                target_col=target_col,
                prediction_col=prediction_col
            )
            
            # Create performance report
            performance_report = Report(metrics=[
                ClassificationQualityMetric(),
                ClassificationClassBalance(),
                ClassificationConfusionMatrix(),
                ClassificationQualityByClass()
            ])
            
            # Run report
            performance_report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=column_mapping
            )
            
            # Save HTML report
            performance_report.save_html(str(self.config.performance_report_path))
            logger.info(f"Model performance report saved to: {self.config.performance_report_path}")
            
            performance_summary = {
                'timestamp': datetime.now().isoformat(),
                'report_path': str(self.config.performance_report_path)
            }
            
            return performance_summary
            
        except Exception as e:
            logger.error("Failed to generate model performance report")
            raise ChurnPredictionException(e, sys)
    
    def monitor_production_data(
        self,
        reference_data_path: Path,
        current_data_path: Path,
        target_col: str = 'Churn',
        prediction_col: Optional[str] = None
    ) -> Dict:
        """
        Complete monitoring pipeline for production data
        
        Args:
            reference_data_path (Path): Path to reference dataset
            current_data_path (Path): Path to current production dataset
            target_col (str): Name of target column
            prediction_col (Optional[str]): Name of prediction column
            
        Returns:
            Dict: Complete monitoring summary
        """
        try:
            logger.info("=" * 80)
            logger.info("MODEL MONITORING STARTED")
            logger.info("=" * 80)
            
            # Load data
            reference_data = pd.read_csv(reference_data_path)
            current_data = pd.read_csv(current_data_path)
            
            logger.info(f"Reference data: {reference_data.shape}")
            logger.info(f"Current data: {current_data.shape}")
            
            monitoring_results = {}
            
            # 1. Data Drift Report
            drift_summary = self.generate_data_drift_report(
                reference_data=reference_data,
                current_data=current_data,
                target_col=target_col
            )
            monitoring_results['data_drift'] = drift_summary
            
            # 2. Data Quality Report
            quality_summary = self.generate_data_quality_report(
                reference_data=reference_data,
                current_data=current_data,
                target_col=target_col
            )
            monitoring_results['data_quality'] = quality_summary
            
            # 3. Model Performance Report (if predictions available)
            if prediction_col and prediction_col in current_data.columns:
                performance_summary = self.generate_model_performance_report(
                    reference_data=reference_data,
                    current_data=current_data,
                    target_col=target_col,
                    prediction_col=prediction_col
                )
                monitoring_results['model_performance'] = performance_summary
            
            # Save monitoring summary
            summary_path = self.config.reports_dir / "monitoring_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(monitoring_results, f, indent=4)
            
            logger.info(f"Monitoring summary saved to: {summary_path}")
            
            logger.info("=" * 80)
            logger.info("MODEL MONITORING COMPLETED")
            logger.info("=" * 80)
            
            return monitoring_results
            
        except Exception as e:
            logger.error("Model monitoring failed")
            raise ChurnPredictionException(e, sys)


if __name__ == "__main__":
    # Example usage
    monitoring = ModelMonitoring()
    
    # Monitor production data
    results = monitoring.monitor_production_data(
        reference_data_path=Path("data/processed/X_train_phase2.csv"),
        current_data_path=Path("data/processed/X_test_phase2.csv"),  # Simulating new data
        target_col='Churn'
    )
    
    print("\n📊 Monitoring Results:")
    print(json.dumps(results, indent=2))
