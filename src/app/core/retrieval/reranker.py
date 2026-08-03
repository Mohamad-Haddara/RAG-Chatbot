"""
Cross-Encoder reranker - re-order retrieved chunks by relevance to the query

Using FlagReranker
"""

import gc
import time 
import torch
from FlagEmbedding import FlagReranker
from app.config import settings
from utils.exceptions import RerankerException
from langchain_core.documents import Document # This class is used to store a piece of text and its metadata for retrieval and data processing
# key attributes - page_content, metadata



import logging

logger = logging.getLogger(__name__)

class CrossEncoderReRanker:
    """Lazy-loaded cross-encoder reranker using FlagEmbedding"""

    def __init__(self) -> None:
        self._model = None
        self._loaded = False


    # @property makes a method callable like an attribute — no parentheses.
    @property
    def model_name(self):
        return settings.reranker_model

    @property
    def enabled(self) -> bool:
        return settings.reranker_enabled

    @property
    def device(self) -> str:
        return settings.reranker_device


    def _load(self):
        if self._loaded:
            return

        try:
            logger.info("Loading reranker...")
            is_gpu = self.device != "cpu"
            self._model = FlagReranker(
                self.model_name,
                use_fp16=is_gpu,
                devices=self.device
            )

            logger.debug(f"Reranker init with {self.device}")

            if is_gpu:
                logger.debug("Reranker warming up...")
                self._model.compute_score(
                    [("warmup query", "warmup document")],
                    normalize=True,
                    batch_size=1,
                    max_length=32,
                )

                logger.debug("Warm-up complete")

            self._loaded = True
            logger.info("Reranker loaded")


        except Exception as e:
            logging.exception(f"Encountered unexpected error during loading: {e}")


    def rerank(
            self,
            query: str,
            documents: list[Document],
            top_k: int = 5,
    ) -> list[tuple[Document, float]]:
        """
        Return the top k ``(document, store)`` pairs ordered by cross-encoder score.

        A score of 1.0 is emitted for bypass paths (reranker disabled, no documents, or model not loaded) so downstream
        threshold filters have a well-defined value. Active reranking uses FlagReranker's normalised score (range ~ [0, 1]).
        """

        if not self.enabled or not documents:
            return [(doc, 1.0) for doc in documents[:top_k]]

        self._load()

        if self._model is None:
            raise RerankerException

        pairs = [(query, doc.page_content) for doc in documents]

        scores = self._model.compute_score(
            pairs,
            normalize = True,
            batch_size = 32,
            max_length = 512
        )

        scored = list(zip(documents,scores))

        # sort descending order (highest to lowest)
        # used when you have a list of tuples
        # sorting the scores 
        scored.sort(key=lambda x:x[1], reverse=True)


