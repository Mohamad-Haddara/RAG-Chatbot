"""
Transformer-based NER extraction one pipeline per model, lazy loaded.

Two models: ``dslim/bert-base-NER`` (EN, CoNLL-2003) and
``cahya/bert-base-indonesian-NER`` (ID, token-level BIO).

Thread-safety note: ``transformers`` pipeline is mostly thread-safe for
inference (no graph-building after warmup). Guard lazy-init with a lock so
the first concurrent call from batch code doesn't race.


We use NER to improve query understanding, enrich metadata, and enhance retrieval precision.
It help the system find exact names, dates, and places instead of relying on general word matching.

Query Understanding: Pulls specific names, companies, or dates out of a user's prompt.

Metadata Filtering: Tags document chunks with clear labels to filter out bad search results before passing data to the LLM.

Better Ranking: Boosts search accuracy by matching specific entities rather than loose keywords.
"""

import logging
import threading
from typing import Any, cast

logger = logging.getLogger(__name__)

from transformers.pipelines import pipeline

_MODEL_REGISTERY = "gliner-community/gliner_medium-v2.5"

class NERExtrator:
    """
    Lazy-per-model NER pipeline. Load once, reuse for all chunks.

    Usage:
        - 

    """
