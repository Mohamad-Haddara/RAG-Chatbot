"""
Embedding step - convert text chunks into vector embeddings.

sentence-transformers adapter
"""

import asyncio

from sentence_transformers import SentenceTransformer
# pulls the model from Hugging Face (cached locally after the first download).

import logging

logger = logging.getLogger(__name__)


# Private variable - don't import this from outside.
_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# We use class becuase it hold state. The model name and the loaded weights live on instance, so I loaded once and reuse across many .encode() calls
class Embedder:
    """LangChain-compatible embeddings backed by sentence_transformers"""

    # Construction
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or _DEFAULT_MODEL # Caller can override
        # Lazy Load - Creating an Embedder costs microseconds. If it loaded weight here, 
        self._model: SentenceTransformer | None = None # Lazy Loading, construction is cheap, and the actual weights only load the first time you embed something.


    # It is called on the class itself, not an instance
    # Embedder.default_model_name() - no object, no weights
    @classmethod
    def default_model_name(cls) -> str:
        """Return the configured embedding model name without loading weights.
        
        Cheap, safe to call from debug/trace path - no SentenceTransformer
        construction, no model download.

        """
        return _DEFAULT_MODEL


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Every public method calls this first.
    # This is why the model loads once, not per request.
    def _load(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)



    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    # Two methods because LangChain requires both names. That's the "LangChain-compatible" in our docstring 
    # — any vector store expects an object with exactly these two methods, so your class drops in wherever a LangChain embeddings object is expected.

    # The real difference: one takes a list and return list of vectors (indexing our chunks)
    # The other takes one string and return one vector (user question at query time)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per text."""
        self._load() # First call load
        # This line exists purely for the type-checker.
        # _load() guarantees the model is set, but mypy can't see across the method call — it still thinks self._model might be None, and complains that None has no .encode().
        assert self._model is not None 
        embeddings = self._model.encode(texts, convert_to_numpy=True).tolist() # Without it you may get torch tensors back.
        # .tolist() converts numpy → plain Python lists, which is what Qdrant and JSON want
        return [list(e) for e in embeddings]


    def embed_query(self, text: str) -> list[float]:
        """Return a single embedding vector for query."""
        self._load()
        assert self._model is not None
        embedding = self._model.encode(text, convert_to_numpy=True) # is blocking - CPU heavy work
        return embedding.tolist()


    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Async wrapper around embed_documents."""
        return await asyncio.to_thread(self.embed_documents, texts) # This is the most important line in the file for a FastAPI app.
        # model.encode() is blocking, CPU-heavy work — it might take 300ms. In async code, blocking the event loop means every other request in your app freezes for those 300ms.
        # asyncio.to_thread moves the work to a worker thread so the event loop stays free to handle other requests.

        # I wrap rather than rewrite because there's no async version of encode() — the trick is offloading sync code, not rewriting it.
    async def aembed_query(self, text: str) -> list[float]:
        """Async wrapper around embed_query."""
        return await asyncio.to_thread(self.embed_query, text)
    

# smoke test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    e = Embedder(_DEFAULT_MODEL)
    v = e.embed_query("hello world")
    print(v)


# when app starts up 
# Eager: this single line downloads/loads the model. Your FastAPI app takes 10 seconds to boot
# Lazy: this line sets two attributes. Microseconds. The weights only load when someone actually calls embed_query().   