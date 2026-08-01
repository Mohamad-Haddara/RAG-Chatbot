"""
Text splitting utilities - split the raw text into smaller pieces called chunks


"""

from typing import TYPE_CHECKING

from langchain_text_splitters import RecursiveCharacterTextSplitter

import logging

if TYPE_CHECKING:
    from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def split_text(
        text: str,
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        model_name: str = "gpt-4"
) -> list[str]:
    """
    Split the text into overlapping chunks using tiktoken encoding.

    Args:
        text: The raw text to split.
        chunk_size: Maximum token count per chunk.
        chunk_overlap: Number of token to overlap between consecutive chunks.
        model_name: Tiktoken encoder to split use (default, "gpt-4").

    Returns:
        List of chunk strings.

    """

    # Is true only when the text is empty or contain nothing but whitespace
    if not text.split():
       
        return []

    # text splitter uses tiktoken encoder to count length and limit chunks precisely
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        model_name=model_name
    )

    # Create LangChain Document objects (for use in downstream tasks)
    docs: list[Document] = splitter.create_documents([text])
    
    return [doc.page_content for doc in docs]
