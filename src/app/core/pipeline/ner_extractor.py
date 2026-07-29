"""
Transformer-based NER extraction one pipeline per model, lazy loaded.

Two models: ``dslim/bert-base-NER`` (EN, CoNLL-2003) and
``cahya/bert-base-indonesian-NER`` (ID, token-level BIO).

Thread-safety note: ``transformers`` pipeline is mostly thread-safe for
inference (no graph-building after warmup). Guard lazy-init with a lock so
the first concurrent call from batch code doesn't race.
"""

