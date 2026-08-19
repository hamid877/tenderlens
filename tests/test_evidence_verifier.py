"""Tests for verification.evidence_verifier (Milestone 11).

All NLI inference is real – the CrossEncoder model is loaded and run for
every relevant test.  The model is *not* mocked because the semantic
correctness tests would be meaningless without actual inference.

The module-level ``verifier`` fixture is session-scoped so the model is
loaded only once per test session (matching the "loaded once per instance"
requirement and keeping test startup fast).

Claim/evidence pairs were empirically verified against
``cross-encoder/nli-deberta-v3-small`` to confirm the expected NLI labels
before committing.
"""

from __future__ import annotations

import pytest

from verification.evidence_verifier import (
    DEFAULT_MODEL_NAME,
    EvidenceVerifier,
    MultiEvidenceVerificationResult,
    VerificationResult,
)

# ---------------------------------------------------------------------------
# Session-scoped fixture – model loaded once for the entire test run
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def verifier() -> EvidenceVerifier:
    """Return a single :class:`EvidenceVerifier` reused across all tests.

    The CrossEncoder model is loaded once and shared, which mirrors the
    intended production usage pattern (one instance → one model load).
    """
    return EvidenceVerifier()


# ---------------------------------------------------------------------------
# 1. Clearly supported claim (entailment)
# ---------------------------------------------------------------------------


class TestSupportedClaim:
    """The model should return entailment for obviously supported claims."""

    def test_supported_returns_true(self, verifier: EvidenceVerifier) -> None:
        # Empirically verified → entailment 0.995
        result = verifier.verify(
            claim="The project deadline is March 31.",
            evidence=["The project must be completed by March 31st."],
        )
        assert result.best.supported is True

    def test_supported_label_is_entailment(self, verifier: EvidenceVerifier) -> None:
        # Empirically verified → entailment 0.998
        result = verifier.verify(
            claim="The contract expires in January 2025.",
            evidence=["The agreement terminates at the end of January 2025."],
        )
        assert result.best.label == "entailment"

    def test_supported_result_type(self, verifier: EvidenceVerifier) -> None:
        result = verifier.verify(
            claim="The project deadline is March 31.",
            evidence=["The project must be completed by March 31st."],
        )
        assert isinstance(result, MultiEvidenceVerificationResult)
        assert isinstance(result.best, VerificationResult)


# ---------------------------------------------------------------------------
# 2. Clearly contradicted claim
# ---------------------------------------------------------------------------


class TestContradictedClaim:
    """The model should return contradiction for clearly opposing statements."""

    def test_contradicted_returns_false(self, verifier: EvidenceVerifier) -> None:
        # Empirically verified → contradiction 0.922
        result = verifier.verify(
            claim="All birds can fly.",
            evidence=["Penguins are birds that cannot fly."],
        )
        assert result.best.supported is False

    def test_contradicted_label_is_not_entailment(
        self, verifier: EvidenceVerifier
    ) -> None:
        # Empirically verified → contradiction 0.999
        result = verifier.verify(
            claim="The sky is blue.",
            evidence=["The sky is completely green."],
        )
        assert result.best.label != "entailment"


# ---------------------------------------------------------------------------
# 3. Neutral / unrelated claim
# ---------------------------------------------------------------------------


class TestNeutralClaim:
    """Unrelated evidence should not support the claim."""

    def test_neutral_returns_false(self, verifier: EvidenceVerifier) -> None:
        # Empirically verified → neutral 0.999
        result = verifier.verify(
            claim="The sky is blue.",
            evidence=["Paris is the capital of France."],
        )
        assert result.best.supported is False

    def test_neutral_label_is_not_entailment(self, verifier: EvidenceVerifier) -> None:
        # Empirically verified → neutral 0.999
        result = verifier.verify(
            claim="The contract expires in January 2025.",
            evidence=["The weather in London is often rainy."],
        )
        assert result.best.label != "entailment"


# ---------------------------------------------------------------------------
# 4. Score is returned
# ---------------------------------------------------------------------------


class TestScoreReturned:
    """VerificationResult.score must be a float in [0, 1]."""

    def test_score_is_float(self, verifier: EvidenceVerifier) -> None:
        result = verifier.verify(
            claim="The project deadline is March 31.",
            evidence=["The project must be completed by March 31st."],
        )
        assert isinstance(result.best.score, float)

    def test_score_in_valid_range(self, verifier: EvidenceVerifier) -> None:
        result = verifier.verify(
            claim="The project deadline is March 31.",
            evidence=["The project must be completed by March 31st."],
        )
        assert 0.0 <= result.best.score <= 1.0

    def test_score_nonzero(self, verifier: EvidenceVerifier) -> None:
        result = verifier.verify(
            claim="The sky is blue.",
            evidence=["The sky is blue."],
        )
        assert result.best.score > 0.0


# ---------------------------------------------------------------------------
# 5. Label is returned
# ---------------------------------------------------------------------------


class TestLabelReturned:
    """VerificationResult.label must be a non-empty string."""

    def test_label_is_string(self, verifier: EvidenceVerifier) -> None:
        result = verifier.verify(
            claim="The sky is blue.",
            evidence=["The sky is blue."],
        )
        assert isinstance(result.best.label, str)

    def test_label_is_nonempty(self, verifier: EvidenceVerifier) -> None:
        result = verifier.verify(
            claim="The sky is blue.",
            evidence=["The sky is blue."],
        )
        assert result.best.label != ""

    def test_label_is_known_nli_class(self, verifier: EvidenceVerifier) -> None:
        result = verifier.verify(
            claim="The sky is blue.",
            evidence=["The sky is blue."],
        )
        assert result.best.label in {"entailment", "contradiction", "neutral"}


# ---------------------------------------------------------------------------
# 6. Empty claim rejection
# ---------------------------------------------------------------------------


class TestEmptyClaimRejection:
    """verify() must raise ValueError for empty or whitespace-only claims."""

    def test_empty_string_raises(self, verifier: EvidenceVerifier) -> None:
        with pytest.raises(ValueError, match="non-empty claim"):
            verifier.verify(claim="", evidence=["Some valid evidence."])

    def test_whitespace_claim_raises(self, verifier: EvidenceVerifier) -> None:
        with pytest.raises(ValueError, match="non-empty claim"):
            verifier.verify(claim="   \t\n", evidence=["Some valid evidence."])


# ---------------------------------------------------------------------------
# 7. Empty evidence rejection
# ---------------------------------------------------------------------------


class TestEmptyEvidenceRejection:
    """verify() must raise ValueError for missing or empty evidence chunks."""

    def test_empty_string_evidence_raises(self, verifier: EvidenceVerifier) -> None:
        with pytest.raises(ValueError):
            verifier.verify(
                claim="The contract value is £2 million.", evidence=""
            )

    def test_whitespace_only_evidence_raises(self, verifier: EvidenceVerifier) -> None:
        with pytest.raises(ValueError):
            verifier.verify(
                claim="The contract value is £2 million.", evidence="   "
            )

    def test_empty_list_raises(self, verifier: EvidenceVerifier) -> None:
        with pytest.raises(ValueError):
            verifier.verify(
                claim="The contract value is £2 million.", evidence=[]
            )

    def test_list_with_empty_chunk_raises(self, verifier: EvidenceVerifier) -> None:
        with pytest.raises(ValueError):
            verifier.verify(
                claim="The contract value is £2 million.",
                evidence=["Valid evidence.", ""],
            )


# ---------------------------------------------------------------------------
# 8. Multiple evidence chunks
# ---------------------------------------------------------------------------


class TestMultipleEvidenceChunks:
    """When multiple chunks are supplied, all_results must be complete and
    best must be selected correctly."""

    def test_all_results_length_matches_evidence(
        self, verifier: EvidenceVerifier
    ) -> None:
        evidence = [
            "Paris is the capital of France.",
            "The sky is blue.",
            "All birds can fly.",
        ]
        result = verifier.verify(claim="The sky is blue.", evidence=evidence)
        assert len(result.all_results) == 3

    def test_best_selects_supporting_chunk(self, verifier: EvidenceVerifier) -> None:
        """When one chunk clearly entails the claim and another is unrelated,
        the best result should be the entailing (supporting) chunk.

        Pairs empirically verified against the model before committing:
        - claim + neutral_ev  → neutral    0.999
        - claim + support_ev  → entailment 0.998
        """
        neutral_ev = "The weather in London is often rainy."
        support_ev = "The agreement terminates at the end of January 2025."
        evidence = [neutral_ev, support_ev]

        result = verifier.verify(
            claim="The contract expires in January 2025.",
            evidence=evidence,
        )
        # The best result must come from the supporting chunk.
        assert result.best.evidence == support_ev

    def test_all_results_are_verification_results(
        self, verifier: EvidenceVerifier
    ) -> None:
        result = verifier.verify(
            claim="The project deadline is March 31.",
            evidence=[
                "The project must be completed by March 31st.",
                "Paris is the capital of France.",
            ],
        )
        assert all(isinstance(r, VerificationResult) for r in result.all_results)

    def test_each_result_references_correct_evidence(
        self, verifier: EvidenceVerifier
    ) -> None:
        evidence = [
            "First evidence chunk about something unrelated.",
            "The project must be completed by March 31st.",
        ]
        result = verifier.verify(
            claim="The project deadline is March 31.",
            evidence=evidence,
        )
        assert result.all_results[0].evidence == evidence[0]
        assert result.all_results[1].evidence == evidence[1]

    def test_single_chunk_list_works(self, verifier: EvidenceVerifier) -> None:
        result = verifier.verify(
            claim="The sky is blue.",
            evidence=["The sky is blue."],
        )
        assert len(result.all_results) == 1
        assert result.best is result.all_results[0]


# ---------------------------------------------------------------------------
# 9. Model is loaded once per verifier instance
# ---------------------------------------------------------------------------


class TestModelLoadedOnce:
    """The CrossEncoder model must be loaded during __init__ and not re-loaded
    on subsequent verify() calls."""

    def test_same_model_object_after_multiple_calls(self) -> None:
        """The internal _model attribute must be the same object between calls."""
        local_verifier = EvidenceVerifier()
        model_before = local_verifier._model

        local_verifier.verify(
            claim="The project deadline is March 31.",
            evidence=["The project must be completed by March 31st."],
        )
        local_verifier.verify(
            claim="The sky is blue.",
            evidence=["Paris is the capital of France."],
        )

        model_after = local_verifier._model
        assert model_before is model_after

    def test_two_instances_have_independent_models(self) -> None:
        """Two EvidenceVerifier instances should hold separate model objects."""
        v1 = EvidenceVerifier()
        v2 = EvidenceVerifier()
        assert v1._model is not v2._model


# ---------------------------------------------------------------------------
# 10. Deterministic behavior
# ---------------------------------------------------------------------------


class TestDeterministicBehavior:
    """Repeated calls with the same input must produce identical results."""

    def test_same_input_same_label(self, verifier: EvidenceVerifier) -> None:
        claim = "The project deadline is March 31."
        evidence = ["The project must be completed by March 31st."]

        r1 = verifier.verify(claim=claim, evidence=evidence)
        r2 = verifier.verify(claim=claim, evidence=evidence)

        assert r1.best.label == r2.best.label

    def test_same_input_same_score(self, verifier: EvidenceVerifier) -> None:
        claim = "The sky is blue."
        evidence = ["Paris is the capital of France."]

        r1 = verifier.verify(claim=claim, evidence=evidence)
        r2 = verifier.verify(claim=claim, evidence=evidence)

        assert abs(r1.best.score - r2.best.score) < 1e-6

    def test_same_input_same_supported_flag(self, verifier: EvidenceVerifier) -> None:
        claim = "The contract expires in January 2025."
        evidence = ["The agreement terminates at the end of January 2025."]

        results = [
            verifier.verify(claim=claim, evidence=evidence).best.supported
            for _ in range(3)
        ]
        assert len(set(results)) == 1
