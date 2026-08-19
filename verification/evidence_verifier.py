"""Evidence verifier for TenderLens (Milestone 11).

Given a factual *claim* and one or more *evidence* text chunks, this module
determines whether the evidence **supports** the claim using Natural Language
Inference (NLI).

Design contract
---------------
* :class:`VerificationResult` – structured result dataclass returned for each
  evaluation, containing at minimum ``supported``, ``label``, and ``score``.
* :class:`EvidenceVerifier` – NLI-based verifier; instantiate once and call
  :meth:`~EvidenceVerifier.verify` with a claim string and one or more
  evidence strings.

Model choice
------------
The preferred model is ``cross-encoder/nli-deberta-v3-base``.  Because it was
not present in the local HuggingFace cache at integration time, the next
smallest DeBERTa-v3 NLI model that *was* already cached —
``cross-encoder/nli-deberta-v3-small`` — is used instead.

The ``cross-encoder/nli-deberta-v3-small`` model was chosen because:

* It is already in the local cache, so no additional download is required.
* It shares the same three-class NLI head (contradiction / entailment /
  neutral) and the same label schema as the preferred *base* variant.
* It is fully local and requires no API key or internet connection.
* It is significantly more accurate than MiniLM-based alternatives while
  remaining fast on CPU.

If you wish to switch to the preferred *base* model, change
:data:`DEFAULT_MODEL_NAME` and ensure the model is downloaded to the
HuggingFace cache.

No LLM provider or external API calls are made by this module.

Typical usage::

    from verification.evidence_verifier import EvidenceVerifier

    verifier = EvidenceVerifier()
    result = verifier.verify(
        claim="The contract value is £2.4 million.",
        evidence=["The awarded contract is worth two point four million pounds."],
    )
    print(result.supported)  # True
    print(result.label)      # "entailment"
    print(result.score)      # e.g. 0.982
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model
# ---------------------------------------------------------------------------

#: The NLI CrossEncoder model loaded by default.
#:
#: ``cross-encoder/nli-deberta-v3-small`` is used because it was already
#: present in the local HuggingFace cache.  It is the smallest DeBERTa-v3
#: NLI variant available and shares the identical label schema with the
#: preferred *base* model (``cross-encoder/nli-deberta-v3-base``).
DEFAULT_MODEL_NAME: str = "cross-encoder/nli-deberta-v3-small"

# Label string expected for the entailment class.
_ENTAILMENT_LABEL: str = "entailment"

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class VerificationResult:
    """Result of verifying a single claim against a single evidence chunk.

    Attributes:
        supported: ``True`` when the NLI model returns an *entailment* label,
                   ``False`` for *contradiction* or *neutral*.
        label: The raw label string from the model (e.g. ``"entailment"``,
               ``"contradiction"``, ``"neutral"``).
        score: Confidence score in the range ``[0, 1]`` for the winning label,
               obtained by applying softmax over the model's raw logits.
        claim: The claim that was evaluated (verbatim).
        evidence: The evidence chunk the claim was evaluated against (verbatim).
    """

    supported: bool
    label: str
    score: float
    claim: str
    evidence: str


@dataclass
class MultiEvidenceVerificationResult:
    """Aggregated result when multiple evidence chunks are supplied.

    Attributes:
        best: The :class:`VerificationResult` for the evidence chunk that
              yielded the strongest result.  Priority is given to the chunk
              with the highest *entailment* confidence; if no chunk supports
              the claim, the chunk with the highest *contradiction* or
              *neutral* confidence is returned.
        all_results: All individual :class:`VerificationResult` objects, one
                     per evidence chunk, in the order they were supplied.
    """

    best: VerificationResult
    all_results: list[VerificationResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# EvidenceVerifier
# ---------------------------------------------------------------------------


class EvidenceVerifier:
    """NLI-based evidence verifier that runs fully locally.

    The verifier loads a CrossEncoder NLI model exactly once per instance and
    uses it to evaluate whether one or more evidence chunks support a given
    claim.

    The model is loaded during :meth:`__init__` and reused for all subsequent
    :meth:`verify` calls on the same instance.  No external API calls are made.

    Args:
        model_name: HuggingFace model identifier for the CrossEncoder NLI
                    model.  Defaults to :data:`DEFAULT_MODEL_NAME`.

    Raises:
        ValueError: If *claim* or any *evidence* string is empty or
                    whitespace-only.

    Example::

        verifier = EvidenceVerifier()
        result = verifier.verify(
            claim="The project deadline is 31 December 2024.",
            evidence=["Work must be completed before the end of December 2024."],
        )
        assert result.best.supported is True
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._model_name = model_name
        logger.info("EvidenceVerifier: loading NLI model %r …", model_name)
        self._model: CrossEncoder = CrossEncoder(model_name)
        # Build the label map from the model's own config so we are not
        # hard-coding index positions (robust to future model changes).
        self._id2label: dict[int, str] = dict(
            self._model.model.config.id2label
        )
        logger.info(
            "EvidenceVerifier: model loaded; labels=%s", list(self._id2label.values())
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        claim: str,
        evidence: str | Sequence[str],
    ) -> MultiEvidenceVerificationResult:
        """Verify *claim* against one or more *evidence* chunks.

        Args:
            claim: A non-empty factual statement to evaluate.
            evidence: A single evidence string or a sequence of evidence
                      strings.  Each string must be non-empty.

        Returns:
            A :class:`MultiEvidenceVerificationResult` containing the best
            result across all evidence chunks and the full list of individual
            results.

        Raises:
            ValueError: If *claim* is empty or whitespace-only.
            ValueError: If *evidence* is empty, or any evidence chunk is
                        empty or whitespace-only.
        """
        # --- Validate claim ---
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError(
                "EvidenceVerifier.verify() requires a non-empty claim string; "
                f"got: {claim!r}"
            )

        # --- Normalise evidence to a list ---
        if isinstance(evidence, str):
            evidence_list: list[str] = [evidence]
        else:
            evidence_list = list(evidence)

        if not evidence_list:
            raise ValueError(
                "EvidenceVerifier.verify() requires at least one evidence chunk; "
                "got an empty sequence."
            )

        for i, ev in enumerate(evidence_list):
            if not isinstance(ev, str) or not ev.strip():
                raise ValueError(
                    f"EvidenceVerifier.verify(): evidence chunk at index {i} is "
                    f"empty or whitespace-only; got: {ev!r}"
                )

        logger.debug(
            "EvidenceVerifier.verify: claim=%r, num_evidence=%d",
            claim[:80],
            len(evidence_list),
        )

        # --- Run batch NLI inference ---
        pairs = [[claim, ev] for ev in evidence_list]
        raw_logits: np.ndarray = self._model.predict(pairs)  # shape (N, num_labels)

        # --- Build individual results ---
        all_results: list[VerificationResult] = []
        for i, (ev, logits) in enumerate(zip(evidence_list, raw_logits)):
            result = self._build_result(claim=claim, evidence=ev, logits=logits)
            all_results.append(result)
            logger.debug(
                "EvidenceVerifier: chunk %d → label=%r score=%.4f",
                i,
                result.label,
                result.score,
            )

        # --- Select best result ---
        best = self._select_best(all_results)

        logger.debug(
            "EvidenceVerifier.verify: best label=%r supported=%s score=%.4f",
            best.label,
            best.supported,
            best.score,
        )

        return MultiEvidenceVerificationResult(best=best, all_results=all_results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_result(
        self, claim: str, evidence: str, logits: np.ndarray
    ) -> VerificationResult:
        """Convert raw NLI logits into a :class:`VerificationResult`.

        Applies softmax to convert logits to probabilities.  The winning label
        is the one with the highest probability.

        Args:
            claim: The claim string.
            evidence: The evidence string for this particular result.
            logits: 1-D array of raw model output scores (one per label).

        Returns:
            A :class:`VerificationResult` for this (claim, evidence) pair.
        """
        # Softmax over logits → probabilities
        exp_logits = np.exp(logits - np.max(logits))  # numerically stable
        probs: np.ndarray = exp_logits / exp_logits.sum()

        winning_idx: int = int(np.argmax(probs))
        label: str = self._id2label[winning_idx]
        score: float = float(probs[winning_idx])
        supported: bool = label == _ENTAILMENT_LABEL

        return VerificationResult(
            supported=supported,
            label=label,
            score=score,
            claim=claim,
            evidence=evidence,
        )

    @staticmethod
    def _select_best(results: list[VerificationResult]) -> VerificationResult:
        """Select the most relevant result from *results*.

        Selection priority:

        1. If any result has ``supported=True`` (entailment), return the one
           with the highest ``score`` among the supporting results.
        2. Otherwise, return the result with the highest ``score`` overall
           (most confident contradiction or neutral label).

        Args:
            results: A non-empty list of :class:`VerificationResult` objects.

        Returns:
            The single best :class:`VerificationResult`.
        """
        supporting = [r for r in results if r.supported]
        if supporting:
            return max(supporting, key=lambda r: r.score)
        return max(results, key=lambda r: r.score)
