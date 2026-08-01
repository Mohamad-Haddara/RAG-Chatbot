"""
Cross-Encoder reranker - re-order retrieved chunks by relevance to the query

Using FlagReranker
"""

import gc
import time 
import torch
from FlagEmbedding import FlagReranker
from langchain_core.documents import Document # This class is used to store a piece of text and its metadata for retrieval and data processing
# key attributes - page_content, metadata

import logging

logger = logging.getLogger(__name__)