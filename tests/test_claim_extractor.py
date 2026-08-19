"""Tests for verification.claim_extractor (Milestone 10).

All tests use synthetic answer strings.  No model download, API key, or
network call is required.
"""

from __future__ import annotations

import pytest

from verification.claim_extractor import Claim, ClaimExtractor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def extractor() -> ClaimExtractor:
    """Return a fresh :class:`ClaimExtractor` for each test."""
    return ClaimExtractor()


# ---------------------------------------------------------------------------
# 1. Single factual claim
# ---------------------------------------------------------------------------


class TestSingleFactualClaim:
    """Answers containing exactly one factual statement."""

    def test_single_sentence_single_claim(self, extractor: ClaimExtractor) -> None:
        answer = "The contract value is £2.4 million."
        claims = extractor.extract(answer)
        assert len(claims) == 1
        assert claims[0].text == "The contract value is £2.4 million."

    def test_single_claim_returns_claim_instance(self, extractor: ClaimExtractor) -> None:
        answer = "The project deadline is 31 December 2024."
        claims = extractor.extract(answer)
        assert all(isinstance(c, Claim) for c in claims)


# ---------------------------------------------------------------------------
# 2. Multiple factual claims
# ---------------------------------------------------------------------------


class TestMultipleFactualClaims:
    """Answers that yield more than one claim."""

    def test_two_claims(self, extractor: ClaimExtractor) -> None:
        answer = (
            "The contract value is £2.4 million. "
            "Work will begin in March 2024."
        )
        claims = extractor.extract(answer)
        assert len(claims) == 2
        assert claims[0].text == "The contract value is £2.4 million."
        assert claims[1].text == "Work will begin in March 2024."

    def test_three_claims(self, extractor: ClaimExtractor) -> None:
        answer = (
            "The tender reference is TEN-2024-001. "
            "The issuing authority is the Ministry of Finance. "
            "Submissions must be received by 15 February 2025."
        )
        claims = extractor.extract(answer)
        assert len(claims) == 3

    def test_order_preserved(self, extractor: ClaimExtractor) -> None:
        answer = (
            "The estimated budget is €500,000. "
            "The contract duration is 24 months. "
            "The procurement method is open tender."
        )
        claims = extractor.extract(answer)
        texts = [c.text for c in claims]
        assert texts[0].startswith("The estimated budget")
        assert texts[1].startswith("The contract duration")
        assert texts[2].startswith("The procurement method")


# ---------------------------------------------------------------------------
# 3. Mixed factual and non-factual text
# ---------------------------------------------------------------------------


class TestMixedContent:
    """Answers that mix genuine claims with conversational filler."""

    def test_greeting_stripped(self, extractor: ClaimExtractor) -> None:
        answer = (
            "Hello! "
            "The contract is valued at £1.2 million."
        )
        claims = extractor.extract(answer)
        # "Hello!" is a greeting – should be discarded
        assert len(claims) == 1
        assert claims[0].text == "The contract is valued at £1.2 million."

    def test_filler_opener_stripped(self, extractor: ClaimExtractor) -> None:
        answer = (
            "Of course, I am happy to help. "
            "The bid closing date is 30 June 2024."
        )
        claims = extractor.extract(answer)
        factual_texts = [c.text for c in claims]
        assert any("bid closing date" in t for t in factual_texts)
        assert not any("happy to help" in t for t in factual_texts)

    def test_meta_commentary_stripped(self, extractor: ClaimExtractor) -> None:
        answer = (
            "Based on the provided information, the supplier must hold ISO 9001 certification. "
            "Thank you for your question."
        )
        claims = extractor.extract(answer)
        # "Thank you" sentence should be discarded
        assert not any("Thank you" in c.text for c in claims)

    def test_closing_remark_stripped(self, extractor: ClaimExtractor) -> None:
        answer = (
            "The maximum contract value is £500,000. "
            "I hope this helps!"
        )
        claims = extractor.extract(answer)
        assert len(claims) == 1
        assert "£500,000" in claims[0].text


# ---------------------------------------------------------------------------
# 4. Empty answer rejection
# ---------------------------------------------------------------------------


class TestEmptyAnswerRejection:
    """extract() must raise ValueError for empty answers."""

    def test_empty_string_raises(self, extractor: ClaimExtractor) -> None:
        with pytest.raises(ValueError):
            extractor.extract("")

    def test_empty_string_error_message(self, extractor: ClaimExtractor) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            extractor.extract("")


# ---------------------------------------------------------------------------
# 5. Whitespace-only answer rejection
# ---------------------------------------------------------------------------


class TestWhitespaceOnlyRejection:
    """extract() must raise ValueError for whitespace-only answers."""

    def test_spaces_only_raises(self, extractor: ClaimExtractor) -> None:
        with pytest.raises(ValueError):
            extractor.extract("   ")

    def test_newlines_only_raises(self, extractor: ClaimExtractor) -> None:
        with pytest.raises(ValueError):
            extractor.extract("\n\n\t\n")

    def test_mixed_whitespace_raises(self, extractor: ClaimExtractor) -> None:
        with pytest.raises(ValueError):
            extractor.extract("  \t  \n  ")


# ---------------------------------------------------------------------------
# 6. No factual claim – returns empty list
# ---------------------------------------------------------------------------


class TestNoFactualClaimsReturnsEmpty:
    """When an answer contains only non-factual text, the result is []."""

    def test_only_greeting_returns_empty(self, extractor: ClaimExtractor) -> None:
        # "Hello!" is short and a greeting – no factual content
        claims = extractor.extract("Hello!")
        assert claims == []

    def test_only_filler_returns_empty(self, extractor: ClaimExtractor) -> None:
        claims = extractor.extract("I don't know the answer to that question.")
        assert claims == []

    def test_only_thanks_returns_empty(self, extractor: ClaimExtractor) -> None:
        claims = extractor.extract("Thank you for your question today.")
        assert claims == []

    def test_returns_list_type(self, extractor: ClaimExtractor) -> None:
        claims = extractor.extract("Hello! Sure, I hope this helps.")
        assert isinstance(claims, list)


# ---------------------------------------------------------------------------
# 7. Claim text is preserved correctly
# ---------------------------------------------------------------------------


class TestClaimTextPreservation:
    """Extracted claim text must be verbatim (stripped, but otherwise intact)."""

    def test_text_matches_sentence_verbatim(self, extractor: ClaimExtractor) -> None:
        sentence = "The supplier must provide a performance bond of 10% of the contract value."
        claims = extractor.extract(sentence)
        assert len(claims) == 1
        assert claims[0].text == sentence

    def test_numbers_preserved(self, extractor: ClaimExtractor) -> None:
        answer = "The procurement budget is USD 3,500,000 for fiscal year 2025."
        claims = extractor.extract(answer)
        assert len(claims) == 1
        assert "3,500,000" in claims[0].text

    def test_special_characters_preserved(self, extractor: ClaimExtractor) -> None:
        answer = "The contract ref. is TEN/2024/MOF/001."
        claims = extractor.extract(answer)
        assert len(claims) >= 1
        assert "TEN/2024/MOF/001" in claims[0].text

    def test_no_information_added(self, extractor: ClaimExtractor) -> None:
        """Extracted text must not include words not present in the answer."""
        answer = "The evaluation criterion is the lowest price."
        claims = extractor.extract(answer)
        assert len(claims) == 1
        for word in claims[0].text.split():
            assert word.rstrip(".,;:!?") in answer


# ---------------------------------------------------------------------------
# 8. Deterministic output
# ---------------------------------------------------------------------------


class TestDeterministicOutput:
    """Same input must always produce the same output."""

    def test_same_answer_same_claims(self, extractor: ClaimExtractor) -> None:
        answer = (
            "The project title is Urban Water Supply Improvement. "
            "The estimated cost is £4.1 million. "
            "The completion target is Q3 2026."
        )
        first = extractor.extract(answer)
        second = extractor.extract(answer)
        assert [c.text for c in first] == [c.text for c in second]

    def test_repeated_calls_same_length(self, extractor: ClaimExtractor) -> None:
        answer = "The contractor must supply quarterly progress reports."
        results = [extractor.extract(answer) for _ in range(5)]
        lengths = [len(r) for r in results]
        assert len(set(lengths)) == 1

    def test_multiple_extractors_same_result(self) -> None:
        """Two independent ClaimExtractor instances must return the same output."""
        answer = "The tender notice was published on 1 January 2024."
        e1, e2 = ClaimExtractor(), ClaimExtractor()
        assert [c.text for c in e1.extract(answer)] == [c.text for c in e2.extract(answer)]
