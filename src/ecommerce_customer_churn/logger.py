"""
Professional Logging Configuration for E-commerce Customer Churn Prediction
Uses loguru for advanced logging with rotation and retention
"""

import os
import sys
from pathlib import Path
from loguru import logger
from datetime import datetime

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Generate log filename with timestamp
LOG_FILE = f"churn_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE_PATH = LOGS_DIR / LOG_FILE

# Remove default logger
logger.remove()

# Add console logger with custom format
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

# Add file logger with rotation and retention
logger.add(
    LOG_FILE_PATH,
    rotation="10 MB",  # Rotate when file reaches 10 MB
    retention="30 days",  # Keep logs for 30 days
    compression="zip",  # Compress rotated logs
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG"
)

# Add error-specific logger
ERROR_LOG_FILE = LOGS_DIR / "errors.log"
logger.add(
    ERROR_LOG_FILE,
    rotation="5 MB",
    retention="60 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="ERROR"
)

logger.info(f"Logging initialized. Log file: {LOG_FILE_PATH}")


def get_logger(name: str = __name__):
    """
    Get a logger instance with the specified name
    
    Args:
        name (str): Logger name (usually __name__)
        
    Returns:
        logger: Configured logger instance
    """
    return logger.bind(name=name)
