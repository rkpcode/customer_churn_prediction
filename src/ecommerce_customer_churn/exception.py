"""
Custom Exception Handler for E-commerce Customer Churn Prediction
Provides detailed error messages with traceback information
"""

import sys
from loguru import logger


class ChurnPredictionException(Exception):
    """
    Custom exception class for churn prediction pipeline
    """
    
    def __init__(self, error_message: str, error_detail: sys):
        """
        Initialize custom exception with detailed error information
        
        Args:
            error_message (str): Error message
            error_detail (sys): System error details
        """
        super().__init__(error_message)
        self.error_message = self._get_detailed_error_message(error_message, error_detail)
        logger.error(self.error_message)
    
    @staticmethod
    def _get_detailed_error_message(error_message: str, error_detail: sys) -> str:
        """
        Generate detailed error message with file name, line number, and traceback
        
        Args:
            error_message (str): Original error message
            error_detail (sys): System error details
            
        Returns:
            str: Formatted detailed error message
        """
        _, _, exc_tb = error_detail.exc_info()
        
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            
            detailed_message = (
                f"Error occurred in script: [{file_name}] "
                f"at line number: [{line_number}] "
                f"with error message: [{error_message}]"
            )
        else:
            detailed_message = f"Error: {error_message}"
        
        return detailed_message
    
    def __str__(self):
        return self.error_message
    
    def __repr__(self):
        return f"ChurnPredictionException({self.error_message})"
