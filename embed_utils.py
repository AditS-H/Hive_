"""
Retrieval for reply grounding, and embedding for intent clustering.

Deliberately local + offline: sentence-transformers runs on CPU in seconds for
a corpus this size (a few thousand resolved cases per brand, after filtering
the 20k-row subsample). This keeps embeddings off the Gemini/Groq free-tier
quota entirely, so those are spent only on calls that actually need an LLM.

No vector DB: a numpy array + cosine similarity is exact and instant below
~100k vectors. Reach for FAISS/Chroma only if the corpus outgrows that --
it currently won't.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

_model = None


def get_encoder():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed(texts: list[str]) -> np.ndarray:
    vecs = get_encoder().encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


class RetrievalIndex:
    """Holds the embedded historical (customer_msg -> resolution) corpus for one brand."""

    def __init__(self, customer_msgs: list[str], resolutions: list[str]):
        assert len(customer_msgs) == len(resolutions)
        self.customer_msgs = list(customer_msgs)
        self.resolutions = list(resolutions)
        self.vectors = embed(self.customer_msgs)  # (N, D), unit-normalized

    def query(self, message: str, k: int = 3, min_similarity: float = 0.0) -> list[dict]:
        q = embed([message])[0]                 # (D,)
        sims = self.vectors @ q                  # unit vectors -> dot product == cosine sim
        top_idx = np.argsort(-sims)[:k]
        return [
            {
                "customer_msg": self.customer_msgs[i],
                "resolution": self.resolutions[i],
                "similarity": float(sims[i]),
            }
            for i in top_idx
            if sims[i] >= min_similarity
        ]
