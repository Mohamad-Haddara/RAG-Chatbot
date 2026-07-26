"""Document loading - extract raw text from files (PDF, TXT, JSONL, etc.)

Source: pdfminer - text reading logic
"""

import json
from io import BytesIO # it holds raw binary zeros and ones in memory  (is ust the cup)
from pathlib import Path
from typing import Any


# Library to extract text from PDF - Designed to read files
from pdfminer.high_level import extract_text as pdf_extract_text

from utils.exceptions import PDFProcessingException, FileTypeNotSupportedException

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




def load_document_bytes(filename: str, content: bytes) -> str:
    """
    Same as function as `load_document` but accepts raw bytes.

    Useful when file has already been read into memory (e.g. from an HTTP upload)
    """

    path = Path(filename)
    suffix = path.suffix.lower()

    if suffix not in _ALLOWED_SUFFIXES:
        raise FileTypeNotSupportedException(f"File type {suffix} not supported")

    if suffix == ".pdf":
        try:
            # if our content variable is just raw binary data (e.g. downloaded the PDF from a URL, or pulled it from a database)
            # BytesIO(content) - It takes your raw bytes and tricks pdfminer into thinking it's reading a real file saved on your hard drive.
            return pdf_extract_text(BytesIO(content))

        except Exception as exc:
            logger.error(f"PDF failed for {filename}: {exc}")
            raise PDFProcessingException(str(exc)) from exc

    # Input content - This is raw binary data
    # The Action - .decode() -- This tells Python to translate that raw binary data into a readable string of text
    # 
    return content.decode("utf-8")


def load_pdf_pages(filename: str, content: bytes) -> list[tuple[int, str]]:
    """
    Iterate ``pdfminer`` extraction per page so each iteration carries its page number.

    Returns ``[(page_number_1_based, page_text), ...]``. Empty pages are skipped;
    pure-images or empty PDFs still anchor ``page=1`` so downstream has the field.
    """

    if Path(filename).suffix.lower() != ".pdf":
        raise FileTypeNotSupportedException(
            f"load_pdf_pages expects .pdf, got {filename}"
        )

    try:
        # form-feed heuristic: pdfminer uses \x0c between pages in flat extraction.
        # # When pdfminer extracts text, it automatically inserts this invisible \x0c character every time it finishes reading one page and moves to the next.
        full = pdf_extract_text(BytesIO(content))

        # Your parts variable becomes a list of strings grouped perfectly by page.
        parts = full.split("\x0c") 
        pages: list[tuple[int, str]] = []

        for idx, text in enumerate(parts, start=1):
            if text.strip():
                pages.append((idx, text))

        if not pages:
            pages = [(1, full or "")]

        return pages


    except Exception as exc:
        logger.error(f"PDF per-page extraction failed for {filename}: {exc}")
        raise PDFProcessingException(str(exc)) from exc