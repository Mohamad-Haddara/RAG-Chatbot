"""Document loading - extract raw text from files (PDF, TXT, JSONL, etc.)

Source: pdfminer - text reading logic
"""

import json
from io import BytesIO
from pathlib import Path
from typing import Any


# Library to extract text from PDF
from pdfminer.high_level import extract_text as pdf_extract_text

from app.utils.exceptions import PDFProcessingException, FileTypeNotSupportedException

import logging

logger = logging.getLogger(__name__)

# we use set instead of list or tuple - to get O(1) constant time
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

    # built-in tool for working with files and folders
    path = Path(file_path)
    # Extracts the file's extension and forces it into lowercase letters. (.PDF -> .pdf)
    suffix = path.suffix.lower()

    if suffix not in _ALLOWED_SUFFIXES:
        raise FileTypeNotSupportedException(f"File type {suffix} not supported")

    if suffix == ".pdf":
        try:
            return pdf_extract_text(path)

        except Exception as exc:
            logger.error(f"PDF extraction failed for {path}: {exc}")

            raise PDFProcessingException(str(exc)) from exec

    # reads the entire contents of a file and returns it as a text string - it handles two tasks: 1)It automatically opens the file, reads it, and safely closes it. 2) tells Python exactly how to translate the file's data into text.
    return path.read_text(encoding="utf-8")

