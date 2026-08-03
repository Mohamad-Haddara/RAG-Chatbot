"""
Cross-Encoder reranker - re-order retrieved chunks by relevance to the query

Using FlagReranker

 
WHY THIS FILE EXISTS
--------------------
Vector search (a bi-encoder such as BGE-M3) embeds the query and each chunk
SEPARATELY, then compares vectors. That's fast enough to scan a whole
collection, but approximate — loosely related chunks can rank too high.
 
A cross-encoder (FlagReranker, e.g. bge-reranker-v2-m3) reads the query and a
chunk TOGETHER in one forward pass and outputs one relevance score. Far more
accurate — but far too slow to run over the entire collection.
 
Standard RAG pattern, implemented here:
    1. Vector search cheaply retrieves ~20-50 candidate chunks.
    2. The cross-encoder re-scores ONLY those candidates.
    3. Just the top_k best-scoring chunks go into the LLM prompt.
"""

# gc = garbage collector; used in close() to reclaim RAM immediately
import gc

# time = used to measure inference latency in healthcheck()
import time 

# torch = used to detect CUDA and to free cached GPU memory in close()
import torch

# FlagReranker() = BAAI's cross-encoder wrapper - the actual scoring model
#from FlagEmbedding import FlagReranker
from sentence_transformers import CrossEncoder


# settings: model name, device, and on/off flag live in config
# so behaviour changes per environment without touching this code
from src.app.config import settings

# Domain-specific exception so callers can catch "reranker broke" precisely
from src.app.utils.exceptions import RerankerException

# Document = LangChain's container for one chunk (.page_content + .metadata)
# key attributes - page_content, metadata
from langchain_core.documents import Document # This class is used to store a piece of text and its metadata for retrieval and data processing




import logging


# Module-level logger, named after this module's import path
logger = logging.getLogger(__name__)

class CrossEncoderReRanker:
    """
    Lazy-loaded cross-encoder reranker using FlagEmbedding.
    
    Lazy-loaded = the model is NOT loaded when this object is created - only the first real rank call.
    WHY: keeps app startup fast, and wastes no RAM/VRAM if reranking is disabled in config. 
    """

    def __init__(self) -> None:
        # The actual FlagReranker instance - None until first use
        self._model = None
        # Guard flag so _load() only does the expensive work once
        self._loaded = False


    # @property makes a method callable like an attribute — no parentheses.
    @property
    def model_name(self):
        # Which model to load, e.g. "BAAI/bge-reranker-v2-m3" — read live
        # from settings so config stays the single source of truth
        return settings.reranker_model

    @property
    def enabled(self) -> bool:
        # Feature flag: turn reranking on/off without any code change
        return settings.reranker_enabled

    @property
    def device(self) -> str:
        # Where the model runs: "cpu", "cuda", "cuda:0"
        return settings.reranker_device


    def _load(self):
        # Already loaded? Do nothing
        if self._loaded:
            return


        try:
            logger.info("Loading reranker...")

            # Anything that is not cpu is treatd as a GPU device
            is_gpu = self.device != "cpu"

            # Create the cross-enoder
            # use fp16 = True on GPU = half-precision weights -> half the VRAM.
            # and faster inference, with negligible accuracy loss
            # On CPU fp16 is slow/poorly supported, so it stays False there
            self._model = CrossEncoder(
                self.model_name,
                max_length=512,
                device=self.device 
            )

            logger.debug(f"Reranker init with {self.device}")

            if is_gpu:

                # The FIRST GPU inference is always slow (CUDA kernels get compiled, memory pools get allocated).
                # Run one tiny dummy scoring
                logger.debug("Reranker warming up...")
                self._model.predict(
                    # One fake (query, document) pair - content is irrelevant
                    [("warmup query", "warmup document")],
                    normalize=True, # sigmoid -> score in [0,1]
                    batch_size=1, # a single pair
                    max_length=32, # tiny token limit
                )

                logger.debug("Warm-up complete")


            # Mark done so future _load() calls return immediately
            self._loaded = True
            logger.info("Reranker loaded")


        except Exception as e:
            # Log the full traceback but DON'T crash the app here.
            # checks for that and raises RerankerException at call time.
            logging.exception(f"Encountered unexpected error during loading: {e}")


    def rerank(
            self,
            query: str,                 # the user's question
            documents: list[Document],  # candidate chunks from vector search
            top_k: int = 5,             # how many chunks to keep after scoring
    ) -> list[tuple[Document, float]]:
        """
        Return the top k ``(document, store)`` pairs ordered by cross-encoder score.

        A score of 1.0 is emitted for bypass paths (reranker disabled, no documents, or model not loaded) so downstream
        threshold filters have a well-defined value. Active reranking uses FlagReranker's normalised score (range ~ [0, 1]).
        """

        # Bypass path
        # Reranking off, or nothing to rank -> keep the original vector-search
        # order, truncated to top_k, with a fake score of 1.0.
        # WHY 1.0: downstream code may drop chunks below a score threshold;
        # 1.0 guarantees bypassed chunks always pass that filter.
        if not self.enabled or not documents:
            return [(doc, 1.0) for doc in documents[:top_k]]

        # Lazy load: first real call pays the loading cost
        self._load()

        # None → we cannot rerank → fail loudly with a domain exception
        if self._model is None:
            raise RerankerException

        pairs = [(query, doc.page_content) for doc in documents]

        scores = self._model.predict(
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

        result: list[tuple[Document, float]] = []
        for doc, score in scored[:top_k]:
            doc.metadata["rerank_score"] = score
            result.append((doc, float(score)))

        return result

    def healthcheck(self) -> dict:
        """Return a health-check result dict (for a /health endpoint or monitoring)"""
        result: dict = {
            "status": "unhealthy",
            "model": self.model_name,
            "device": self.device,
            "enabled": self.enabled,
            "loaded": self._loaded
        }

        if not self.enabled:
            result["status"] = "disabled"
            return result


        # Enabled but no model in memory (lazy load not triggered yet, or loading failed earlier) 
        if not self._loaded or self._model is None:
            result['error'] = "Model not loaded"
            return result

        try:

            # Time one tiny REAL inference — proves the model actually works, not just that the object exists
            start = time.perf_counter()
            score = self._model.predict(
                [("healthcheck","probe")],
                normalize = True,
                batch_size = 1,
                max_length = 16
            )

            latency_ms = round((time.perf_counter() - start)*100,2)

            # compute_score may return a float OR a list depending on input
            if isinstance(score, list):
                if len(score) > 0:
                    score = score[0] # Extract the first (and only) score

                else:
                    result["error"] = "Empty score list returned"
                    return result

            # Sanity check: a normalised score must be a number in [0, 1].
            # Anything else means the model returned garbage.
            if not isinstance(score, (int, float)) or not (0.0 <= score <= 1.0):
                result["error"] = f"Invalid score returned: {score}"
                return result

             # All checks passed
            result["status"] = "healthy"
            result["latency_ms"] = latency_ms
            result["warm"] = True   # loaded AND has done real inference

        except Exception as exc:
            logger.error(f"Reranker healthcheck failed: {exc}", exc_info = True)
            result["error"] = str(exc)

        return result


    def close(self) -> None:
        self._model = None
        gc.collect()

        if torch.mps.is_available():
            torch.mps.empty_cache()
            





# Query & a few chunks
query = "How many annual leave days do employees get?"

docs = [
    Document(page_content="Employees are entitled to 25 working days of annual leave per year."),
    Document(page_content="The cafeteria is open from 7am to 3pm on weekdays."),
    Document(page_content="Annual leave requests must be submitted two weeks in advance."),
    Document(page_content="Qdrant stores embeddings in named collections."),
]

# Run it (first call the model)
r = CrossEncoderReRanker()

results = r.rerank(query, docs, 2)

print(f"\n Query {query}\n")

for rank, (doc, score) in enumerate(results, 1):
    print(f"#{rank}  score={score:.4f}")
    print(f"    text:     {doc.page_content}")
    print(f"    metadata: {doc.metadata}\n")

 
print(f"raw return: {type(results).__name__} of (Document, float) tuples")
print(f"healthcheck: {r.healthcheck()}")
 
 