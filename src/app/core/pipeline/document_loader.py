"""Document loading - extract raw text from files (PDF, TXT, JSONL, etc.)

Source: pdfminer - text reading logic
"""

import json
from io import BytesIO
from pathlib import Path
from typing import Any


# Library to extract text from PDF
from pdfminer.high_level import extract_text as pdf_extract_text

import logging

logger = logging.getLogger(__name__)


_ALLOWED_SUFFIXES = frozenset({".text", ".md", ".pdf", ".jsonl"})


def load_document(file_path: str | Path) -> str:
    """Extract plain text from *file_path*
    
    currently supports: .txt, .md, .pdf

    Args:
        - file_path: Path to the file

    Return:
        - UTF-8 decoded text content

    Raises:
        - FileTypeNotSupportedException: If the suffix is not supported
        - PDFProcessingException: If pdfminer fails on a file
    """

