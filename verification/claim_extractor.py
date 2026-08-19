"""Deterministic claim extractor for TenderLens (Milestone 10).

Extracts factual, evidence-checkable statements from a free-text answer
produced by the RAG pipeline.

The extraction is intentionally **deterministic and rule-based** so that:

* no API key or model download is required;
* output is stable across runs (same input → same claims, same order);
* the class can later be augmented or replaced with an LLM-backed variant
  without changing the public interface.

Design contract
---------------
* :class:`Claim` – a simple dataclass representing a single extracted claim.
* :class:`ClaimExtractor` – stateless extractor; call :meth:`~ClaimExtractor.extract`
  with a generated answer string to obtain a :class:`list` of :class:`Claim` objects.

This module has **no** dependency on retrieval, FAISS, embeddings, or verification
logic.  It only performs text analysis on the provided answer string.

Typical usage::

    from verification.claim_extractor import ClaimExtractor

    extractor = ClaimExtractor()
    claims = extractor.extract(
        "The contract value is £2.4 million. Work will begin in March 2024."
    )
    for claim in claims:
        print(claim.text)
    # The contract value is £2.4 million.
    # Work will begin in March 2024.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic patterns – sentences that carry NO verifiable factual content
# ---------------------------------------------------------------------------

# Regex patterns that match entire sentences we want to discard.  Each entry
# is compiled once at module-load time for performance.
_NON_FACTUAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # Greetings / closings
        r"^(hello|hi|hey|good\s+(morning|afternoon|evening)|greetings)[^.!?]*[.!?]?$",
        # Pure filler openers
        r"^(of course|sure|certainly|absolutely|great|okay|ok|yes|no\s*,)[,\s][^.!?]*[.!?]?$",
        # "I / We don't know" statements
        r"^(i\s+don['']t\s+know|i\s+am\s+not\s+sure|i\s+cannot\s+(find|determine)|"
        r"there\s+is\s+no\s+(information|evidence|data)|the\s+(evidence|document)\s+does\s+not\s+(contain|mention|include|provide))[^.!?]*[.!?]?$",
        # Polite acknowledgements / meta-commentary
        r"^(thank\s+you|thanks|you're\s+welcome|i\s+hope\s+this\s+(helps|answers)|"
        r"please\s+(let\s+me\s+know|feel\s+free|note)|as\s+(requested|mentioned|stated|noted)|"
        r"based\s+on\s+the\s+(provided|given|above)\s+(information|evidence|context|document))[^.!?]*[.!?]?$",
        # Transitional / structural phrases with no claim
        r"^(in\s+(summary|conclusion|short)|to\s+summarize|to\s+conclude|"
        r"in\s+other\s+words|that\s+(said|is\s+to\s+say))[,:\s][^.!?]*[.!?]?$",
        # Questions (not claims)
        r"^[^.!?]+\?$",
    ]
]

# Minimum character length for a sentence to be considered a potential claim.
# Very short fragments ("Yes.", "No.", "OK.") are discarded.
_MIN_CLAIM_LENGTH: int = 15

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Claim:
    """A single factual claim extracted from a generated answer.

    Attributes:
        text: The verbatim claim sentence, stripped of leading/trailing
              whitespace.  The factual meaning is preserved unchanged.
    """

    text: str


# ---------------------------------------------------------------------------
# ClaimExtractor
# ---------------------------------------------------------------------------


class ClaimExtractor:
    """Stateless, deterministic extractor of factual claims from answer text.

    The extractor splits an answer into candidate sentences, then filters out
    sentences that are conversational fillers, greetings, meta-commentary, or
    questions – retaining only sentences that carry a verifiable factual
    assertion.

    The class is independent of retrieval, FAISS, embeddings, and any LLM or
    network call.  Every run with the same input produces the same output in
    the same order (deterministic).

    Raises:
        ValueError: If the supplied answer is empty or whitespace-only.

    Example::

        extractor = ClaimExtractor()
        claims = extractor.extract("The bid deadline is 30 June 2024.")
        assert claims[0].text == "The bid deadline is 30 June 2024."
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, answer: str) -> list[Claim]:
        """Extract factual claims from *answer*.

        Args:
            answer: A non-empty string produced by the RAG pipeline (or any
                    text source).  Must contain at least one non-whitespace
                    character.

        Returns:
            An ordered list of :class:`Claim` objects, one per extracted
            factual sentence.  Returns an empty list if no factual claims
            are found (e.g. the answer is entirely conversational).

        Raises:
            ValueError: If *answer* is empty or contains only whitespace.
        """
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(
                "ClaimExtractor.extract() requires a non-empty answer string; "
                f"got: {answer!r}"
            )

        logger.debug("ClaimExtractor.extract: answer length=%d", len(answer))

        sentences = self._split_sentences(answer)
        claims: list[Claim] = []

        for sentence in sentences:
            sentence = sentence.strip()
            if self._is_factual(sentence):
                claims.append(Claim(text=sentence))
                logger.debug("ClaimExtractor: accepted claim %r", sentence[:60])
            else:
                logger.debug("ClaimExtractor: discarded non-factual %r", sentence[:60])

        logger.debug(
            "ClaimExtractor.extract: %d claim(s) extracted from %d sentence(s).",
            len(claims),
            len(sentences),
        )
        return claims

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Common abbreviation suffixes that should NOT trigger a sentence split.
    # Checked against the token immediately before the period + space.
    _ABBREV_PATTERN: re.Pattern[str] = re.compile(
        r"(?i)\b("
        r"ref|no|vol|dept|est|approx|max|min|avg|std|fig|sec|"
        r"e\.g|i\.e|etc|vs|cf|mr|mrs|ms|dr|prof|sr|jr|lt|sgt|"
        r"jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec|"
        r"st|nd|rd|th"
        r")$"
    )

    @classmethod
    def _split_sentences(cls, text: str) -> list[str]:
        """Split *text* into individual sentences.

        Splits on ``.``, ``!``, or ``?`` followed by whitespace, **except**
        when the token before the period looks like a known abbreviation
        (e.g. ``ref.``, ``no.``, ``e.g.``).  Splitting on ``!`` and ``?``
        is always performed because those rarely appear in abbreviations.

        Args:
            text: Raw answer text.

        Returns:
            A list of non-empty sentence strings.
        """
        # Tokenise on any boundary that looks like end-of-sentence.
        # Strategy: find candidate split points (. ! ? followed by space or EOS)
        # and only split if the token before the delimiter is not an abbreviation.
        segments: list[str] = []
        current_start = 0

        for m in re.finditer(r"([.!?])\s+", text):
            punct = m.group(1)
            before = text[current_start : m.start()]  # text up to (excl.) punct
            after_space_start = m.end()

            # For ! and ? – always split.
            if punct in ("!", "?"):
                segments.append((text[current_start : m.start() + 1]).rstrip())
                current_start = after_space_start
                continue

            # For period – check if the token before is an abbreviation.
            last_token = before.rsplit(None, 1)[-1] if before.split() else ""
            if cls._ABBREV_PATTERN.search(last_token):
                # Do NOT split here; continue accumulating.
                continue

            segments.append(text[current_start : m.start() + 1].strip())
            current_start = after_space_start

        # Append any remaining text after the last split point.
        tail = text[current_start:].strip()
        if tail:
            segments.append(tail)

        return [s for s in segments if s]

    @staticmethod
    def _is_factual(sentence: str) -> bool:
        """Return ``True`` if *sentence* is likely a factual claim.

        A sentence is considered factual if:

        1. It meets the minimum length threshold.
        2. It does not match any of the non-factual heuristic patterns.

        Args:
            sentence: A single candidate sentence (stripped).

        Returns:
            ``True`` if the sentence should be kept as a factual claim.
        """
        if len(sentence) < _MIN_CLAIM_LENGTH:
            return False

        for pattern in _NON_FACTUAL_PATTERNS:
            if pattern.match(sentence):
                return False

        return True
