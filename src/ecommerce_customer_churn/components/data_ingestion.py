"""
Data Ingestion Component
Downloads E-commerce Customer Churn dataset from Kaggle and performs initial validation
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

from ecommerce_customer_churn.logger import logger
from ecommerce_customer_churn.exception import ChurnPredictionException
from ecommerce_customer_churn.utils import create_directories


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion"""
    kaggle_dataset: str = "ankitverma2010/ecommerce-customer-churn-analysis-and-prediction"
    raw_data_dir: Path = Path("data/raw")
    raw_data_file: Path = Path("data/raw/ecommerce_churn.csv")


class DataIngestion:
    """
    Data Ingestion Component for E-commerce Customer Churn Prediction
    Downloads data from Kaggle and performs train-test split
    """
    
    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        """
        Initialize Data Ingestion component
        
        Args:
            config (DataIngestionConfig): Configuration object
        """
        self.config = config
        logger.info("Data Ingestion component initialized")
    
    def download_data_from_kaggle(self) -> Path:
        """
        Download dataset from Kaggle using Kaggle API
        
        Returns:
            Path: Path to downloaded data file
            
        Raises:
            ChurnPredictionException: If download fails
        """
        try:
            logger.info(f"Downloading dataset from Kaggle: {self.config.kaggle_dataset}")
            
            # Create raw data directory
            create_directories([self.config.raw_data_dir])
            
            # Initialize Kaggle API
            api = KaggleApi()
            api.authenticate()
            
            # Download dataset
            api.dataset_download_files(
                self.config.kaggle_dataset,
                path=self.config.raw_data_dir,
                unzip=True
            )
            
            logger.info(f"Dataset downloaded successfully to: {self.config.raw_data_dir}")
            
            # Find the CSV file in the downloaded directory
            csv_files = list(self.config.raw_data_dir.glob("*.csv"))
            
            if not csv_files:
                raise FileNotFoundError("No CSV file found in downloaded dataset")
            
            # Rename the first CSV file to our standard name
            downloaded_file = csv_files[0]
            if downloaded_file != self.config.raw_data_file:
                downloaded_file.rename(self.config.raw_data_file)
                logger.info(f"Renamed {downloaded_file.name} to {self.config.raw_data_file.name}")
            
            return self.config.raw_data_file
            
        except Exception as e:
            logger.error(f"Failed to download data from Kaggle: {e}")
            raise ChurnPredictionException(e, sys)
    
    def validate_data(self, data_path: Path) -> bool:
        """
        Validate downloaded data
        
        Args:
            data_path (Path): Path to data file
            
        Returns:
            bool: True if validation passes
            
        Raises:
            ChurnPredictionException: If validation fails
        """
        try:
            logger.info(f"Validating data from: {data_path}")
            
            # Check if file exists
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            # Load data
            df = pd.read_csv(data_path)
            
            # Basic validation checks
            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            # Check for minimum required columns
            required_columns = ['CustomerID', 'Churn']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Check for empty dataset
            if df.empty:
                raise ValueError("Dataset is empty")
            
            # Log basic statistics
            logger.info(f"Total records: {len(df)}")
            logger.info(f"Churn distribution:\n{df['Churn'].value_counts()}")
            logger.info(f"Missing values:\n{df.isnull().sum()}")
            
            logger.info("Data validation passed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            raise ChurnPredictionException(e, sys)
    
    def initiate_data_ingestion(self) -> Path:
        """
        Main method to execute data ingestion pipeline
        
        Returns:
            Path: Path to raw data file
        """
        try:
            logger.info("=" * 80)
            logger.info("DATA INGESTION STARTED")
            logger.info("=" * 80)
            
            # Step 1: Download data from Kaggle
            data_file = self.download_data_from_kaggle()
            
            # Step 2: Validate data
            self.validate_data(data_file)
            
            logger.info("=" * 80)
            logger.info("DATA INGESTION COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            
            return data_file
            
        except Exception as e:
            logger.error("Data ingestion failed")
            raise ChurnPredictionException(e, sys)


if __name__ == "__main__":
    # Test data ingestion
    data_ingestion = DataIngestion()
    raw_data_path = data_ingestion.initiate_data_ingestion()
    print(f"Raw data: {raw_data_path}")
