"""Embedding service for TenderLens (Milestone 4).

Loads ``all-MiniLM-L6-v2`` once per :class:`EmbeddingService` instance and
exposes a single :meth:`~EmbeddingService.embed` method that batch-encodes a
list of strings into L2-normalised float32 vectors.

A module-level singleton (:data:`embedding_service`) is provided so that the
model is loaded exactly once across the application lifetime.

Example::

    from retrieval.embedding_service import embedding_service

    vecs = embedding_service.embed(["clause one", "clause two"])
    # vecs.shape == (2, 384), dtype == float32, each row unit-norm
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME: Final[str] = "all-MiniLM-L6-v2"
EMBEDDING_DIM: Final[int] = 384


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class EmbeddingService:
    """Wraps a ``sentence-transformers`` model for batch text embedding.

    Args:
        model_name: HuggingFace model identifier.  Defaults to
            :data:`MODEL_NAME` (``all-MiniLM-L6-v2``).

    The model is loaded on construction and **not** reloaded per call.
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        logger.info("Loading embedding model '%s' on CPU …", model_name)
        self._model: SentenceTransformer = SentenceTransformer(
            model_name, device="cpu"
        )
        self.model_name: str = model_name
        logger.info("Embedding model '%s' ready.", model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, texts: list[str]) -> np.ndarray:
        """Encode *texts* into L2-normalised float32 embeddings.

        Args:
            texts: Non-empty list of strings to encode.

        Returns:
            A ``float32`` NumPy array of shape ``(len(texts), 384)``.  Every
            row has unit L2-norm (cosine similarity can be computed as inner
            product).

        Raises:
            ValueError: If *texts* is empty.
        """
        if not texts:
            raise ValueError("embed() requires at least one text; got empty list.")

        raw: np.ndarray = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we normalise manually below
        ).astype(np.float32)

        # L2 normalise each row using NumPy only (no scikit-learn).
        norms: np.ndarray = np.linalg.norm(raw, axis=1, keepdims=True)
        # Avoid division by zero for any zero-vector (extremely unlikely).
        norms = np.where(norms == 0.0, 1.0, norms)
        normalised: np.ndarray = (raw / norms).astype(np.float32)

        logger.debug(
            "Embedded %d text(s) → shape %s via '%s'.",
            len(texts),
            normalised.shape,
            self.model_name,
        )
        return normalised


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

#: Shared :class:`EmbeddingService` instance.  Import and use this instead of
#: constructing a new instance so that the model is loaded only once.
embedding_service: EmbeddingService = EmbeddingService()
