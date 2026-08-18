"""Tests for Milestone 6: context.context_builder.ContextBuilder.

Covers:
- empty evidence handling (EmptyEvidenceError raised)
- single result formatting (all fields present)
- multiple results preserve ranking / order
- source metadata is preserved (document_id, page, chunk_id, chunk_index)
- similarity score is preserved exactly
- chunk text is preserved exactly (no modification)
- maximum chunk limit works (results beyond cap are excluded)
- invalid limits are rejected (InvalidMaxChunksError raised)
- deterministic output (same input → same output every call)

All tests use synthetic SearchResult objects; no real FAISS store is needed.
"""

from __future__ import annotations

import pytest

from retrieval.vector_store import SearchResult
from context.context_builder import (
    BuiltContext,
    ContextBuilder,
    EmptyEvidenceError,
    InvalidMaxChunksError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    rank: int = 1,
    score: float = 0.90,
    chunk_id: str = "chunk-abc",
    document_id: str = "doc-001",
    page: int = 1,
    chunk_index: int = 0,
    text: str = "Sample evidence text.",
) -> SearchResult:
    """Build a synthetic :class:`SearchResult` for testing."""
    return SearchResult(
        rank=rank,
        score=score,
        chunk_id=chunk_id,
        document_id=document_id,
        page=page,
        chunk_index=chunk_index,
        text=text,
    )


def _three_results() -> list[SearchResult]:
    """Return three synthetic SearchResult objects with distinct metadata."""
    return [
        _make_result(
            rank=1,
            score=0.95,
            chunk_id="chunk-001",
            document_id="doc-A",
            page=2,
            chunk_index=3,
            text="Procurement authority requires financial statements.",
        ),
        _make_result(
            rank=2,
            score=0.80,
            chunk_id="chunk-002",
            document_id="doc-B",
            page=5,
            chunk_index=7,
            text="Bridge construction technical specifications.",
        ),
        _make_result(
            rank=3,
            score=0.65,
            chunk_id="chunk-003",
            document_id="doc-C",
            page=1,
            chunk_index=0,
            text="Environmental impact must be assessed prior to commencement.",
        ),
    ]


# ===========================================================================
# Empty evidence handling
# ===========================================================================


class TestEmptyEvidence:
    def test_empty_list_raises_empty_evidence_error(self) -> None:
        builder = ContextBuilder()
        with pytest.raises(EmptyEvidenceError):
            builder.build([])

    def test_none_raises_empty_evidence_error(self) -> None:
        """Passing None should also raise EmptyEvidenceError."""
        builder = ContextBuilder()
        with pytest.raises(EmptyEvidenceError):
            builder.build(None)  # type: ignore[arg-type]

    def test_empty_evidence_error_is_value_error_subclass(self) -> None:
        builder = ContextBuilder()
        with pytest.raises(ValueError):
            builder.build([])


# ===========================================================================
# Invalid max_chunks
# ===========================================================================


class TestInvalidMaxChunks:
    def test_zero_max_chunks_raises(self) -> None:
        with pytest.raises(InvalidMaxChunksError):
            ContextBuilder(max_chunks=0)

    def test_negative_max_chunks_raises(self) -> None:
        with pytest.raises(InvalidMaxChunksError):
            ContextBuilder(max_chunks=-5)

    def test_float_max_chunks_raises(self) -> None:
        with pytest.raises(InvalidMaxChunksError):
            ContextBuilder(max_chunks=2.5)  # type: ignore[arg-type]

    def test_string_max_chunks_raises(self) -> None:
        with pytest.raises(InvalidMaxChunksError):
            ContextBuilder(max_chunks="3")  # type: ignore[arg-type]

    def test_invalid_max_chunks_error_is_value_error_subclass(self) -> None:
        with pytest.raises(ValueError):
            ContextBuilder(max_chunks=0)

    def test_max_chunks_one_is_valid(self) -> None:
        """max_chunks=1 is the minimum valid value."""
        builder = ContextBuilder(max_chunks=1)
        result = builder.build([_make_result()])
        assert isinstance(result, BuiltContext)


# ===========================================================================
# Single result formatting
# ===========================================================================


class TestSingleResultFormatting:
    def test_returns_built_context_instance(self) -> None:
        builder = ContextBuilder()
        result = builder.build([_make_result()])
        assert isinstance(result, BuiltContext)

    def test_context_string_is_nonempty(self) -> None:
        builder = ContextBuilder()
        ctx = builder.build([_make_result()])
        assert ctx.context_string.strip() != ""

    def test_evidence_label_present(self) -> None:
        builder = ContextBuilder()
        ctx = builder.build([_make_result()])
        assert "[Evidence 1]" in ctx.context_string

    def test_document_id_in_context_string(self) -> None:
        builder = ContextBuilder()
        result = _make_result(document_id="tender-doc-xyz")
        ctx = builder.build([result])
        assert "tender-doc-xyz" in ctx.context_string

    def test_page_in_context_string(self) -> None:
        builder = ContextBuilder()
        result = _make_result(page=42)
        ctx = builder.build([result])
        assert "42" in ctx.context_string

    def test_chunk_id_in_context_string(self) -> None:
        builder = ContextBuilder()
        result = _make_result(chunk_id="abc-unique-id")
        ctx = builder.build([result])
        assert "abc-unique-id" in ctx.context_string

    def test_chunk_index_in_context_string(self) -> None:
        builder = ContextBuilder()
        result = _make_result(chunk_index=17)
        ctx = builder.build([result])
        assert "17" in ctx.context_string

    def test_rank_in_context_string(self) -> None:
        builder = ContextBuilder()
        result = _make_result(rank=1)
        ctx = builder.build([result])
        assert "1" in ctx.context_string

    def test_score_in_context_string(self) -> None:
        builder = ContextBuilder()
        result = _make_result(score=0.876543)
        ctx = builder.build([result])
        # Score should appear formatted in the string
        assert "0.876543" in ctx.context_string

    def test_text_in_context_string(self) -> None:
        builder = ContextBuilder()
        result = _make_result(text="The bidder must submit a bond.")
        ctx = builder.build([result])
        assert "The bidder must submit a bond." in ctx.context_string

    def test_chunks_used_is_one(self) -> None:
        builder = ContextBuilder()
        ctx = builder.build([_make_result()])
        assert ctx.chunks_used == 1

    def test_total_results_is_one(self) -> None:
        builder = ContextBuilder()
        ctx = builder.build([_make_result()])
        assert ctx.total_results == 1


# ===========================================================================
# Multiple results – order and ranking
# ===========================================================================


class TestMultipleResultsOrdering:
    def test_multiple_results_evidence_labels_sequential(self) -> None:
        builder = ContextBuilder()
        ctx = builder.build(_three_results())
        assert "[Evidence 1]" in ctx.context_string
        assert "[Evidence 2]" in ctx.context_string
        assert "[Evidence 3]" in ctx.context_string

    def test_retrieval_order_is_preserved(self) -> None:
        """Evidence blocks must appear in the order of the input list."""
        results = _three_results()
        builder = ContextBuilder()
        ctx = builder.build(results)
        pos1 = ctx.context_string.index("[Evidence 1]")
        pos2 = ctx.context_string.index("[Evidence 2]")
        pos3 = ctx.context_string.index("[Evidence 3]")
        assert pos1 < pos2 < pos3

    def test_chunk_texts_appear_in_order(self) -> None:
        """Text of rank-1 chunk must appear before text of rank-3 chunk."""
        results = _three_results()
        builder = ContextBuilder()
        ctx = builder.build(results)
        pos_first = ctx.context_string.index(results[0].text)
        pos_last = ctx.context_string.index(results[-1].text)
        assert pos_first < pos_last

    def test_all_document_ids_present(self) -> None:
        builder = ContextBuilder()
        ctx = builder.build(_three_results())
        for r in _three_results():
            assert r.document_id in ctx.context_string

    def test_chunks_used_equals_three(self) -> None:
        builder = ContextBuilder()
        ctx = builder.build(_three_results())
        assert ctx.chunks_used == 3

    def test_total_results_equals_three(self) -> None:
        builder = ContextBuilder()
        ctx = builder.build(_three_results())
        assert ctx.total_results == 3


# ===========================================================================
# Source metadata preservation
# ===========================================================================


class TestSourceMetadataPreservation:
    def test_document_id_preserved(self) -> None:
        result = _make_result(document_id="my-tender-doc")
        ctx = ContextBuilder().build([result])
        assert "my-tender-doc" in ctx.context_string

    def test_page_preserved(self) -> None:
        result = _make_result(page=99)
        ctx = ContextBuilder().build([result])
        assert "99" in ctx.context_string

    def test_chunk_id_preserved(self) -> None:
        result = _make_result(chunk_id="deadbeef1234")
        ctx = ContextBuilder().build([result])
        assert "deadbeef1234" in ctx.context_string

    def test_chunk_index_preserved(self) -> None:
        result = _make_result(chunk_index=55)
        ctx = ContextBuilder().build([result])
        assert "55" in ctx.context_string

    def test_rank_preserved(self) -> None:
        result = _make_result(rank=3)
        ctx = ContextBuilder().build([result])
        assert "3" in ctx.context_string

    def test_metadata_labels_present(self) -> None:
        """The formatted block should have recognisable metadata labels."""
        ctx = ContextBuilder().build([_make_result()])
        cs = ctx.context_string
        assert "Source" in cs
        assert "Page" in cs
        assert "Chunk ID" in cs
        assert "Rank" in cs
        assert "Score" in cs


# ===========================================================================
# Similarity score preservation
# ===========================================================================


class TestScorePreservation:
    def test_score_preserved_exactly(self) -> None:
        """The score must appear formatted in the output without rounding loss."""
        score = 0.123456
        result = _make_result(score=score)
        ctx = ContextBuilder().build([result])
        # We format to 6 decimal places; ensure the formatted value is present
        assert f"{score:.6f}" in ctx.context_string

    def test_high_score_preserved(self) -> None:
        result = _make_result(score=0.999999)
        ctx = ContextBuilder().build([result])
        assert "0.999999" in ctx.context_string

    def test_low_score_preserved(self) -> None:
        result = _make_result(score=0.000001)
        ctx = ContextBuilder().build([result])
        assert "0.000001" in ctx.context_string

    def test_multiple_scores_all_present(self) -> None:
        results = _three_results()
        ctx = ContextBuilder().build(results)
        for r in results:
            assert f"{r.score:.6f}" in ctx.context_string


# ===========================================================================
# Chunk text preservation
# ===========================================================================


class TestChunkTextPreservation:
    def test_text_preserved_exactly(self) -> None:
        text = "Exact text that must not be altered or trimmed."
        result = _make_result(text=text)
        ctx = ContextBuilder().build([result])
        assert text in ctx.context_string

    def test_text_with_special_characters(self) -> None:
        text = "Cost: €1,000,000.00 (VAT incl.) — see Annex A."
        result = _make_result(text=text)
        ctx = ContextBuilder().build([result])
        assert text in ctx.context_string

    def test_multiline_text_preserved(self) -> None:
        text = "Line one.\nLine two.\nLine three."
        result = _make_result(text=text)
        ctx = ContextBuilder().build([result])
        assert text in ctx.context_string

    def test_text_not_modified_or_truncated(self) -> None:
        text = "A" * 2000  # long chunk
        result = _make_result(text=text)
        ctx = ContextBuilder().build([result])
        assert text in ctx.context_string


# ===========================================================================
# Maximum chunk limit
# ===========================================================================


class TestMaxChunkLimit:
    def test_max_chunks_one_returns_one_block(self) -> None:
        builder = ContextBuilder(max_chunks=1)
        ctx = builder.build(_three_results())
        assert ctx.chunks_used == 1
        assert "[Evidence 1]" in ctx.context_string
        assert "[Evidence 2]" not in ctx.context_string

    def test_max_chunks_two_returns_two_blocks(self) -> None:
        builder = ContextBuilder(max_chunks=2)
        ctx = builder.build(_three_results())
        assert ctx.chunks_used == 2
        assert "[Evidence 2]" in ctx.context_string
        assert "[Evidence 3]" not in ctx.context_string

    def test_max_chunks_exceeds_input_returns_all(self) -> None:
        builder = ContextBuilder(max_chunks=100)
        ctx = builder.build(_three_results())
        assert ctx.chunks_used == 3

    def test_total_results_reflects_full_input(self) -> None:
        """total_results should report the full input count, not the capped count."""
        builder = ContextBuilder(max_chunks=1)
        ctx = builder.build(_three_results())
        assert ctx.total_results == 3

    def test_capped_result_text_absent(self) -> None:
        """Text from result beyond the cap must NOT appear in the context."""
        results = _three_results()
        builder = ContextBuilder(max_chunks=1)
        ctx = builder.build(results)
        # Only first result's text should be present
        assert results[0].text in ctx.context_string
        assert results[1].text not in ctx.context_string
        assert results[2].text not in ctx.context_string

    def test_first_result_is_included_under_cap(self) -> None:
        """Even with max_chunks=1, the rank-1 result must be included."""
        results = _three_results()
        builder = ContextBuilder(max_chunks=1)
        ctx = builder.build(results)
        assert results[0].document_id in ctx.context_string


# ===========================================================================
# Deterministic output
# ===========================================================================


class TestDeterministicOutput:
    def test_same_input_same_output(self) -> None:
        """Calling build() twice with the same input must produce identical output."""
        builder = ContextBuilder()
        results = _three_results()
        ctx_a = builder.build(results)
        ctx_b = builder.build(results)
        assert ctx_a.context_string == ctx_b.context_string

    def test_new_builder_same_output(self) -> None:
        """Two independent ContextBuilder instances must produce identical output."""
        results = _three_results()
        ctx_a = ContextBuilder().build(results)
        ctx_b = ContextBuilder().build(results)
        assert ctx_a.context_string == ctx_b.context_string

    def test_determinism_with_max_chunks(self) -> None:
        results = _three_results()
        ctx_a = ContextBuilder(max_chunks=2).build(results)
        ctx_b = ContextBuilder(max_chunks=2).build(results)
        assert ctx_a.context_string == ctx_b.context_string
