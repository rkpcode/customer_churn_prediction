"""
Data Transformation Component
Handles train-test split, feature engineering, and preprocessing for churn prediction
Following industry-grade approach from notebooks
"""

import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ecommerce_customer_churn.logger import logger
from ecommerce_customer_churn.exception import ChurnPredictionException
from ecommerce_customer_churn.utils import create_directories


@dataclass
class DataTransformationConfig:
    """Configuration for data transformation"""
    processed_dir: Path = Path("data/processed")
    X_train_phase1_path: Path = Path("data/processed/X_train_phase1.csv")
    X_test_phase1_path: Path = Path("data/processed/X_test_phase1.csv")
    X_train_phase2_path: Path = Path("data/processed/X_train_phase2.csv")
    X_test_phase2_path: Path = Path("data/processed/X_test_phase2.csv")
    y_train_path: Path = Path("data/processed/y_train.csv")
    y_test_path: Path = Path("data/processed/y_test.csv")
    imputation_values_path: Path = Path("artifacts/imputation_values.json")
    label_encoders_path: Path = Path("artifacts/label_encoders.json")
    test_size: float = 0.2
    random_state: int = 42


class DataTransformation:
    """
    Data Transformation Component for E-commerce Customer Churn Prediction
    Performs train-test split, feature engineering, and preprocessing
    Following notebook best practices: NO SMOTE, label encoding, phase-based features
    """
    
    def __init__(self, config: DataTransformationConfig = DataTransformationConfig()):
        """
        Initialize Data Transformation component
        
        Args:
            config (DataTransformationConfig): Configuration object
        """
        self.config = config
        self.imputation_values = {}
        self.label_encoders = {}
        logger.info("Data Transformation component initialized")
    
    def perform_train_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Perform stratified train-test split FIRST (before any transformation)
        
        Args:
            df (pd.DataFrame): Raw dataframe
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        try:
            logger.info(f"Performing stratified train-test split (test_size={self.config.test_size})")
            
            # Separate features and target
            X = df.drop(['Churn', 'CustomerID'], axis=1, errors='ignore')
            y = df['Churn']
            
            # Stratified split to preserve class distribution
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                stratify=y,
                random_state=self.config.random_state
            )
            
            logger.info(f"Train set: {X_train.shape}, Churn rate: {y_train.mean()*100:.2f}%")
            logger.info(f"Test set: {X_test.shape}, Churn rate: {y_test.mean()*100:.2f}%")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error("Train-test split failed")
            raise ChurnPredictionException(e, sys)
    
    def handle_missing_values(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Handle missing values with median/mode imputation and create missing flags
        FIT on train, APPLY to test (no data leakage)
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            
        Returns:
            Tuple: Transformed X_train and X_test
        """
        try:
            logger.info("Handling missing values...")
            
            # Create copies
            X_train = X_train.copy()
            X_test = X_test.copy()
            
            # Identify columns with missing values
            numerical_missing = ['Tenure', 'HourSpendOnApp', 'OrderCount', 
                               'DaySinceLastOrder', 'OrderAmountHikeFromlastYear', 'CouponUsed']
            
            # 1. Create missing flags BEFORE imputation
            for col in numerical_missing:
                if col in X_train.columns:
                    X_train[f'{col}_was_missing'] = X_train[col].isnull().astype(int)
                    X_test[f'{col}_was_missing'] = X_test[col].isnull().astype(int)
                    logger.info(f"Created missing flag for {col}")
            
            # 2. Median imputation for numerical features (fit on train)
            numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
            for col in numerical_cols:
                if X_train[col].isnull().sum() > 0:
                    median_val = X_train[col].median()
                    self.imputation_values[col] = {'strategy': 'median', 'value': float(median_val)}
                    X_train[col].fillna(median_val, inplace=True)
                    X_test[col].fillna(median_val, inplace=True)
                    logger.info(f"Imputed {col} with median: {median_val:.2f}")
            
            # 3. Mode imputation for categorical features (fit on train)
            categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
            for col in categorical_cols:
                if X_train[col].isnull().sum() > 0:
                    mode_val = X_train[col].mode()[0]
                    self.imputation_values[col] = {'strategy': 'mode', 'value': str(mode_val)}
                    X_train[col].fillna(mode_val, inplace=True)
                    X_test[col].fillna(mode_val, inplace=True)
                    logger.info(f"Imputed {col} with mode: {mode_val}")
            
            # Verify no missing values
            logger.info(f"Train missing values: {X_train.isnull().sum().sum()}")
            logger.info(f"Test missing values: {X_test.isnull().sum().sum()}")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error("Missing value handling failed")
            raise ChurnPredictionException(e, sys)
    
    def encode_categorical_features(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Label encode categorical features (fit on train, apply to test)
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            
        Returns:
            Tuple: Encoded X_train and X_test
        """
        try:
            logger.info("Encoding categorical features...")
            
            # Create copies
            X_train = X_train.copy()
            X_test = X_test.copy()
            
            categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
            
            for col in categorical_cols:
                le = LabelEncoder()
                # Fit on train
                X_train[col] = le.fit_transform(X_train[col])
                # Apply to test (handle unseen categories)
                X_test[col] = X_test[col].map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
                # Store encoder classes for later use
                self.label_encoders[col] = le.classes_.tolist()
                logger.info(f"Encoded {col}: {len(le.classes_)} categories")
            
            logger.info(f"Total categorical features encoded: {len(categorical_cols)}")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error("Categorical encoding failed")
            raise ChurnPredictionException(e, sys)
    
    def engineer_phase2_features(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Add Phase 2 controlled features (order_frequency, complaint_rate)
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            
        Returns:
            Tuple: X_train and X_test with Phase 2 features
        """
        try:
            logger.info("Engineering Phase 2 features...")
            
            # Create copies
            X_train = X_train.copy()
            X_test = X_test.copy()
            
            # Feature 1: Order Frequency (orders per month)
            X_train['order_frequency'] = X_train['OrderCount'] / (X_train['Tenure'] + 1)
            X_test['order_frequency'] = X_test['OrderCount'] / (X_test['Tenure'] + 1)
            logger.info("Created order_frequency feature")
            
            # Feature 2: Complaint Rate (complaints per order)
            X_train['complaint_rate'] = X_train['Complain'] / (X_train['OrderCount'] + 1)
            X_test['complaint_rate'] = X_test['Complain'] / (X_test['OrderCount'] + 1)
            logger.info("Created complaint_rate feature")
            
            logger.info(f"Phase 2 features added. New shape: {X_train.shape}")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error("Phase 2 feature engineering failed")
            raise ChurnPredictionException(e, sys)
    
    def save_artifacts(self):
        """Save imputation values and label encoders as JSON"""
        try:
            create_directories([os.path.dirname(self.config.imputation_values_path)])
            
            # Save imputation values
            with open(self.config.imputation_values_path, 'w') as f:
                json.dump(self.imputation_values, f, indent=4)
            logger.info(f"Imputation values saved to: {self.config.imputation_values_path}")
            
            # Save label encoders
            with open(self.config.label_encoders_path, 'w') as f:
                json.dump(self.label_encoders, f, indent=4)
            logger.info(f"Label encoders saved to: {self.config.label_encoders_path}")
            
        except Exception as e:
            logger.error("Failed to save artifacts")
            raise ChurnPredictionException(e, sys)
    
    def initiate_data_transformation(self, raw_data_path: Path):
        """
        Main method to execute data transformation pipeline
        
        Args:
            raw_data_path (Path): Path to raw data file
            
        Returns:
            dict: Paths to all saved feature sets
        """
        try:
            logger.info("=" * 80)
            logger.info("DATA TRANSFORMATION STARTED")
            logger.info("=" * 80)
            
            # Load raw data
            df = pd.read_csv(raw_data_path)
            logger.info(f"Loaded raw data: {df.shape}")
            
            # Step 1: Train-test split FIRST
            X_train, X_test, y_train, y_test = self.perform_train_test_split(df)
            
            # Step 2: Handle missing values (fit on train, apply to test)
            X_train, X_test = self.handle_missing_values(X_train, X_test)
            
            # Step 3: Encode categorical features (fit on train, apply to test)
            X_train, X_test = self.encode_categorical_features(X_train, X_test)
            
            # Save Phase 1 features (baseline: 18 original + 6 missing flags = 24)
            logger.info("=" * 80)
            logger.info("PHASE 1: BASELINE FEATURES")
            logger.info(f"Features: {X_train.shape[1]} (18 original + 6 missing flags)")
            logger.info("=" * 80)
            
            X_train_phase1 = X_train.copy()
            X_test_phase1 = X_test.copy()
            
            # Step 4: Add Phase 2 features (controlled)
            X_train_phase2, X_test_phase2 = self.engineer_phase2_features(X_train, X_test)
            
            logger.info("=" * 80)
            logger.info("PHASE 2: CONTROLLED FEATURES")
            logger.info(f"Features: {X_train_phase2.shape[1]} (24 baseline + 2 engineered)")
            logger.info("=" * 80)
            
            # Create processed directory
            create_directories([self.config.processed_dir])
            
            # Save all feature sets
            X_train_phase1.to_csv(self.config.X_train_phase1_path, index=False)
            X_test_phase1.to_csv(self.config.X_test_phase1_path, index=False)
            X_train_phase2.to_csv(self.config.X_train_phase2_path, index=False)
            X_test_phase2.to_csv(self.config.X_test_phase2_path, index=False)
            y_train.to_csv(self.config.y_train_path, index=False, header=['Churn'])
            y_test.to_csv(self.config.y_test_path, index=False, header=['Churn'])
            
            logger.info("Saved feature sets:")
            logger.info(f"  - Phase 1 Train: {self.config.X_train_phase1_path}")
            logger.info(f"  - Phase 1 Test: {self.config.X_test_phase1_path}")
            logger.info(f"  - Phase 2 Train: {self.config.X_train_phase2_path}")
            logger.info(f"  - Phase 2 Test: {self.config.X_test_phase2_path}")
            logger.info(f"  - y_train: {self.config.y_train_path}")
            logger.info(f"  - y_test: {self.config.y_test_path}")
            
            # Save artifacts
            self.save_artifacts()
            
            logger.info("=" * 80)
            logger.info("DATA TRANSFORMATION COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            
            return {
                'X_train_phase1': self.config.X_train_phase1_path,
                'X_test_phase1': self.config.X_test_phase1_path,
                'X_train_phase2': self.config.X_train_phase2_path,
                'X_test_phase2': self.config.X_test_phase2_path,
                'y_train': self.config.y_train_path,
                'y_test': self.config.y_test_path
            }
            
        except Exception as e:
            logger.error("Data transformation failed")
            raise ChurnPredictionException(e, sys)


if __name__ == "__main__":
    # Test data transformation
    from ecommerce_customer_churn.components.data_ingestion import DataIngestion
    
    # First run data ingestion
    data_ingestion = DataIngestion()
    raw_data_path = data_ingestion.initiate_data_ingestion()
    
    # Then run data transformation
    data_transformation = DataTransformation()
    paths = data_transformation.initiate_data_transformation(raw_data_path)
    
    print("\nSaved feature sets:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
