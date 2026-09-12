"""
embed_utils.py imports `sentence_transformers` at module load time, and the
real package pulls in torch -- too large to install in this sandbox (no
route to huggingface.co for model weights either). That's a sandbox
limitation, not a reason to leave RetrievalIndex's actual logic unverified:
this stubs `sentence_transformers` in sys.modules with a small deterministic
fake encoder before importing embed_utils, so the real embed_utils.py code
(not a reimplementation of it) gets exercised -- top-k ordering, the
min_similarity cutoff, and the cosine-via-dot-product math on normalized
vectors.

On the user's machine, `pip install -r requirements.txt` installs the real
sentence-transformers and this stub is never touched -- embed_utils.py
itself is completely unmodified.
"""
import sys
import types
import hashlib
import numpy as np


def _deterministic_fake_vector(text: str, dim: int = 32) -> np.ndarray:
    """
    Hash-based bag-of-words-ish embedding: same text -> same vector, and
    texts sharing words land closer together than texts sharing none,
    which is all the RetrievalIndex tests below actually need.
    """
    vec = np.zeros(dim, dtype=np.float32)
    for word in text.lower().split():
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _install_fake_sentence_transformers():
    fake_module = types.ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            return np.stack([_deterministic_fake_vector(t) for t in texts])

    fake_module.SentenceTransformer = FakeSentenceTransformer
    sys.modules["sentence_transformers"] = fake_module


_install_fake_sentence_transformers()
import embed_utils  # noqa: E402  (must come after the stub is installed)


def test_identical_text_has_similarity_close_to_one():
    idx = embed_utils.RetrievalIndex(
        customer_msgs=["my app keeps crashing on launch"],
        resolutions=["please reinstall the app"],
    )
    results = idx.query("my app keeps crashing on launch", k=1)
    assert len(results) == 1
    assert results[0]["similarity"] > 0.99


def test_top_k_respects_k_and_is_sorted_descending():
    idx = embed_utils.RetrievalIndex(
        customer_msgs=[
            "cannot log in to my account",
            "forgot my password please help",
            "how do I change my subscription plan",
            "the app crashes every time I open it",
        ],
        resolutions=["res1", "res2", "res3", "res4"],
    )
    results = idx.query("I forgot my password and cannot log in", k=2)
    assert len(results) == 2
    assert results[0]["similarity"] >= results[1]["similarity"]


def test_min_similarity_filters_out_weak_matches():
    idx = embed_utils.RetrievalIndex(
        customer_msgs=["completely unrelated topic about gardening tomatoes"],
        resolutions=["res"],
    )
    results = idx.query("my credit card was charged twice", k=3, min_similarity=0.5)
    assert results == []


def test_query_returns_expected_shape():
    idx = embed_utils.RetrievalIndex(
        customer_msgs=["my package never arrived"],
        resolutions=["we issued a replacement"],
    )
    results = idx.query("where is my package", k=1)
    assert set(results[0].keys()) == {"customer_msg", "resolution", "similarity"}


def test_mismatched_lengths_raise():
    try:
        embed_utils.RetrievalIndex(customer_msgs=["a", "b"], resolutions=["only one"])
        assert False, "expected AssertionError for mismatched lengths"
    except AssertionError:
        pass
