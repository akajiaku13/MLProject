import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    """
    This function returns a detailed error message.
    Args:
        error (Exception): The exception that occurred.
        error_detail (sys): The sys module to access the traceback.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in script: [{0}] at line number: [{1}] with error message: [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

class CustomException(Exception):
    """
    Custom exception class that inherits from the built-in Exception class.
    It provides a detailed error message when an exception occurs.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message