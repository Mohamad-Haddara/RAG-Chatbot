from typing import Optional, Dict, Any
from fastapi import Request
from fastapi.responses import JSONResponse
import logging


logger = logging.getLogger(__name__)


class AppException(Exception):
    """Base application exception."""


    # Default attributes to be overriden by subclasses
    status_code: int = 500
    default_message: str = "An unexpected error occured."

    def __init__(
            self,
            message: Optional[str] = None,
            status_code: Optional[int] = None,
            details: Optional[Dict[str, Any]] = None
    ) -> None:
        self.status_code = status_code if status_code is not None else self.status_code
        self.message = message if self.message is not None else self.default_message
        self.details = details or {}
        super().__init__(self.message)



# --- Subclasses with specific defualts ---


class FileTypeNotSupportedException(AppException):
    """The uploaded file type is not supported."""

    status_code = 400
    default_message = "The uploaded file type is not supported"


class PDFProcessingException(AppException):
    """Failed to extract text from a PDF"""
    status_code = 500
    default_message = "Failed to process the PDF file"