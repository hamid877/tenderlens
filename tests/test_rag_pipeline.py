"""Tests for Milestone 8: pipeline.rag_pipeline.RAGPipeline.

Covers:
- Successful end-to-end flow using FakeLLMProvider.
- RetrievalService is called with correct arguments.
- ContextBuilder is called with the retrieved results.
- LLMProvider receives the constructed prompt.
- query is preserved in the result.
- answer is preserved in the result.
- Retrieved sources are preserved in the result.
- top_k is passed through to RetrievalService.
- document_id is passed through to RetrievalService.
- Empty / whitespace query is rejected (EmptyQueryError).
- Retrieval errors propagate unchanged.
- Provider errors propagate unchanged.
- Deterministic behavior (same inputs → same prompt → same answer).

No FAISS indexes or real embeddings are required; all retrieval is mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from context.context_builder import BuiltContext, ContextBuilder
from generation.fake_provider import FakeLLMProvider
from generation.provider import LLMProvider
from pipeline.rag_pipeline import EmptyQueryError, RAGPipeline, RAGResult
from retrieval.retrieval_service import InvalidQueryError, RetrievalService
from retrieval.vector_store import SearchResult, StoreNotFoundError


# ---------------------------------------------------------------------------
# Helpers / shared data
# ---------------------------------------------------------------------------

def _make_result(
    rank: int = 1,
    score: float = 0.95,
    chunk_id: str = "chunk-001",
    document_id: str = "doc-A",
    page: int = 1,
    chunk_index: int = 0,
    text: str = "The procurement authority requires financial statements.",
) -> SearchResult:
    """Return a synthetic :class:`SearchResult` for use in tests."""
    return SearchResult(
        rank=rank,
        score=score,
        chunk_id=chunk_id,
        document_id=document_id,
        page=page,
        chunk_index=chunk_index,
        text=text,
    )


def _make_built_context(context_string: str = "Evidence block") -> BuiltContext:
    return BuiltContext(
        context_string=context_string,
        chunks_used=1,
        total_results=1,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_result() -> SearchResult:
    return _make_result()


@pytest.fixture()
def fake_results() -> list[SearchResult]:
    return [
        _make_result(rank=1, chunk_id="c1", text="Procurement financial statements required."),
        _make_result(rank=2, chunk_id="c2", score=0.80, text="Bridge construction specifications."),
    ]


@pytest.fixture()
def mock_retrieval(fake_results: list[SearchResult]) -> MagicMock:
    """A :class:`RetrievalService` mock that returns ``fake_results``."""
    mock = MagicMock(spec=RetrievalService)
    mock.search.return_value = fake_results
    return mock


@pytest.fixture()
def mock_builder(fake_results: list[SearchResult]) -> MagicMock:
    """A :class:`ContextBuilder` mock that returns a deterministic :class:`BuiltContext`."""
    mock = MagicMock(spec=ContextBuilder)
    mock.build.return_value = _make_built_context("Evidence block text")
    return mock


@pytest.fixture()
def fake_provider() -> FakeLLMProvider:
    return FakeLLMProvider()


@pytest.fixture()
def pipeline(
    mock_retrieval: MagicMock,
    mock_builder: MagicMock,
    fake_provider: FakeLLMProvider,
) -> RAGPipeline:
    return RAGPipeline(
        retrieval_service=mock_retrieval,
        context_builder=mock_builder,
        llm_provider=fake_provider,
    )


# ===========================================================================
# End-to-end flow
# ===========================================================================


class TestEndToEnd:
    def test_successful_flow_returns_rag_result(self, pipeline: RAGPipeline) -> None:
        result = pipeline.answer("What is the contract value?")
        assert isinstance(result, RAGResult)

    def test_answer_is_string(self, pipeline: RAGPipeline) -> None:
        result = pipeline.answer("What is the contract value?")
        assert isinstance(result.answer, str)
        assert result.answer != ""

    def test_uses_fake_provider_default_response(self, pipeline: RAGPipeline) -> None:
        result = pipeline.answer("What is the contract value?")
        assert result.answer == FakeLLMProvider.DEFAULT_RESPONSE


# ===========================================================================
# RetrievalService is called
# ===========================================================================


class TestRetrievalCalled:
    def test_retrieval_search_is_called(
        self, pipeline: RAGPipeline, mock_retrieval: MagicMock
    ) -> None:
        pipeline.answer("financial statements")
        mock_retrieval.search.assert_called_once()

    def test_retrieval_receives_query(
        self, pipeline: RAGPipeline, mock_retrieval: MagicMock
    ) -> None:
        pipeline.answer("financial statements")
        _, kwargs = mock_retrieval.search.call_args
        assert kwargs["query"] == "financial statements"

    def test_top_k_passed_to_retrieval(
        self, pipeline: RAGPipeline, mock_retrieval: MagicMock
    ) -> None:
        pipeline.answer("financial statements", top_k=3)
        _, kwargs = mock_retrieval.search.call_args
        assert kwargs["top_k"] == 3

    def test_default_top_k_is_five(
        self, pipeline: RAGPipeline, mock_retrieval: MagicMock
    ) -> None:
        pipeline.answer("financial statements")
        _, kwargs = mock_retrieval.search.call_args
        assert kwargs["top_k"] == 5

    def test_document_id_passed_to_retrieval(
        self, pipeline: RAGPipeline, mock_retrieval: MagicMock
    ) -> None:
        pipeline.answer("financial statements", document_id="doc-X")
        _, kwargs = mock_retrieval.search.call_args
        assert kwargs["document_id"] == "doc-X"

    def test_document_id_default_is_none(
        self, pipeline: RAGPipeline, mock_retrieval: MagicMock
    ) -> None:
        pipeline.answer("financial statements")
        _, kwargs = mock_retrieval.search.call_args
        assert kwargs["document_id"] is None


# ===========================================================================
# ContextBuilder is called
# ===========================================================================


class TestContextBuilderCalled:
    def test_context_builder_build_is_called(
        self, pipeline: RAGPipeline, mock_builder: MagicMock, fake_results: list[SearchResult]
    ) -> None:
        pipeline.answer("financial statements")
        mock_builder.build.assert_called_once_with(fake_results)

    def test_context_builder_receives_retrieved_results(
        self,
        mock_retrieval: MagicMock,
        mock_builder: MagicMock,
        fake_provider: FakeLLMProvider,
        fake_results: list[SearchResult],
    ) -> None:
        """ContextBuilder must receive the exact list returned by RetrievalService."""
        custom_results = [_make_result(text="custom evidence chunk")]
        mock_retrieval.search.return_value = custom_results
        mock_builder.build.return_value = _make_built_context("custom evidence")

        p = RAGPipeline(
            retrieval_service=mock_retrieval,
            context_builder=mock_builder,
            llm_provider=fake_provider,
        )
        p.answer("custom query")
        mock_builder.build.assert_called_once_with(custom_results)


# ===========================================================================
# LLMProvider receives prompt
# ===========================================================================


class TestPromptConstruction:
    def test_provider_receives_prompt(
        self, pipeline: RAGPipeline, fake_provider: FakeLLMProvider
    ) -> None:
        pipeline.answer("What is the contract value?")
        assert fake_provider.last_prompt is not None

    def test_prompt_contains_question(
        self, pipeline: RAGPipeline, fake_provider: FakeLLMProvider
    ) -> None:
        query = "What is the contract value?"
        pipeline.answer(query)
        assert fake_provider.last_prompt is not None
        assert query in fake_provider.last_prompt

    def test_prompt_contains_evidence(
        self, pipeline: RAGPipeline, fake_provider: FakeLLMProvider, mock_builder: MagicMock
    ) -> None:
        ctx_string = "Evidence block text"
        mock_builder.build.return_value = _make_built_context(ctx_string)
        pipeline.answer("What is the contract value?")
        assert fake_provider.last_prompt is not None
        assert ctx_string in fake_provider.last_prompt

    def test_prompt_contains_instruction_to_use_evidence_only(
        self, pipeline: RAGPipeline, fake_provider: FakeLLMProvider
    ) -> None:
        pipeline.answer("What is the contract value?")
        assert fake_provider.last_prompt is not None
        prompt = fake_provider.last_prompt.lower()
        # The prompt must instruct the model to use only the supplied evidence.
        assert "only" in prompt or "evidence" in prompt

    def test_prompt_is_deterministic(
        self,
        mock_retrieval: MagicMock,
        mock_builder: MagicMock,
        fake_provider: FakeLLMProvider,
    ) -> None:
        """Same inputs must produce the same prompt every time."""
        provider1 = FakeLLMProvider()
        provider2 = FakeLLMProvider()
        p1 = RAGPipeline(
            retrieval_service=mock_retrieval,
            context_builder=mock_builder,
            llm_provider=provider1,
        )
        p2 = RAGPipeline(
            retrieval_service=mock_retrieval,
            context_builder=mock_builder,
            llm_provider=provider2,
        )
        p1.answer("deterministic query")
        p2.answer("deterministic query")
        assert provider1.last_prompt is not None
        assert provider2.last_prompt is not None
        assert provider1.last_prompt == provider2.last_prompt

    def test_different_queries_produce_different_prompts(
        self,
        mock_retrieval: MagicMock,
        mock_builder: MagicMock,
    ) -> None:
        provider_a = FakeLLMProvider()
        provider_b = FakeLLMProvider()

        p1 = RAGPipeline(
            retrieval_service=mock_retrieval,
            context_builder=mock_builder,
            llm_provider=provider_a,
        )
        p2 = RAGPipeline(
            retrieval_service=mock_retrieval,
            context_builder=mock_builder,
            llm_provider=provider_b,
        )
        p1.answer("query alpha")
        p2.answer("query beta")
        assert provider_a.last_prompt != provider_b.last_prompt


# ===========================================================================
# Result fields preserved
# ===========================================================================


class TestResultPreservation:
    def test_query_preserved_in_result(self, pipeline: RAGPipeline) -> None:
        query = "What is the procurement deadline?"
        result = pipeline.answer(query)
        assert result.query == query

    def test_answer_preserved_in_result(
        self, mock_retrieval: MagicMock, mock_builder: MagicMock
    ) -> None:
        custom_response = "The deadline is 31 March 2024."
        provider = FakeLLMProvider(response=custom_response)
        p = RAGPipeline(
            retrieval_service=mock_retrieval,
            context_builder=mock_builder,
            llm_provider=provider,
        )
        result = p.answer("procurement deadline?")
        assert result.answer == custom_response

    def test_sources_preserved_in_result(
        self, pipeline: RAGPipeline, fake_results: list[SearchResult]
    ) -> None:
        result = pipeline.answer("financial statements")
        assert result.sources == fake_results

    def test_sources_metadata_preserved(
        self, pipeline: RAGPipeline, fake_results: list[SearchResult]
    ) -> None:
        result = pipeline.answer("financial statements")
        for original, returned in zip(fake_results, result.sources):
            assert returned.rank == original.rank
            assert returned.score == original.score
            assert returned.chunk_id == original.chunk_id
            assert returned.document_id == original.document_id
            assert returned.page == original.page
            assert returned.chunk_index == original.chunk_index
            assert returned.text == original.text


# ===========================================================================
# top_k pass-through
# ===========================================================================


class TestTopKPassThrough:
    def test_top_k_one(
        self, pipeline: RAGPipeline, mock_retrieval: MagicMock
    ) -> None:
        pipeline.answer("query", top_k=1)
        _, kwargs = mock_retrieval.search.call_args
        assert kwargs["top_k"] == 1

    def test_top_k_ten(
        self, pipeline: RAGPipeline, mock_retrieval: MagicMock
    ) -> None:
        pipeline.answer("query", top_k=10)
        _, kwargs = mock_retrieval.search.call_args
        assert kwargs["top_k"] == 10

    def test_top_k_hundred(
        self, pipeline: RAGPipeline, mock_retrieval: MagicMock
    ) -> None:
        pipeline.answer("query", top_k=100)
        _, kwargs = mock_retrieval.search.call_args
        assert kwargs["top_k"] == 100


# ===========================================================================
# document_id pass-through
# ===========================================================================


class TestDocumentIdPassThrough:
    def test_document_id_forwarded(
        self, pipeline: RAGPipeline, mock_retrieval: MagicMock
    ) -> None:
        pipeline.answer("query", document_id="tender-007")
        _, kwargs = mock_retrieval.search.call_args
        assert kwargs["document_id"] == "tender-007"

    def test_none_document_id_forwarded(
        self, pipeline: RAGPipeline, mock_retrieval: MagicMock
    ) -> None:
        pipeline.answer("query", document_id=None)
        _, kwargs = mock_retrieval.search.call_args
        assert kwargs["document_id"] is None


# ===========================================================================
# Empty / whitespace query rejection
# ===========================================================================


class TestEmptyQueryRejection:
    def test_empty_string_raises_empty_query_error(
        self, pipeline: RAGPipeline
    ) -> None:
        with pytest.raises(EmptyQueryError):
            pipeline.answer("")

    def test_whitespace_only_raises(self, pipeline: RAGPipeline) -> None:
        with pytest.raises(EmptyQueryError):
            pipeline.answer("   ")

    def test_tab_only_raises(self, pipeline: RAGPipeline) -> None:
        with pytest.raises(EmptyQueryError):
            pipeline.answer("\t\t")

    def test_newline_only_raises(self, pipeline: RAGPipeline) -> None:
        with pytest.raises(EmptyQueryError):
            pipeline.answer("\n\n")

    def test_empty_query_error_is_value_error_subclass(
        self, pipeline: RAGPipeline
    ) -> None:
        with pytest.raises(ValueError):
            pipeline.answer("")

    def test_empty_query_does_not_call_retrieval(
        self, pipeline: RAGPipeline, mock_retrieval: MagicMock
    ) -> None:
        with pytest.raises(EmptyQueryError):
            pipeline.answer("")
        mock_retrieval.search.assert_not_called()

    def test_whitespace_query_does_not_call_provider(
        self, pipeline: RAGPipeline, fake_provider: FakeLLMProvider
    ) -> None:
        with pytest.raises(EmptyQueryError):
            pipeline.answer("   ")
        assert fake_provider.last_prompt is None


# ===========================================================================
# Error propagation
# ===========================================================================


class TestRetrievalErrorPropagation:
    def test_store_not_found_propagates(
        self, mock_builder: MagicMock, fake_provider: FakeLLMProvider
    ) -> None:
        mock_retrieval = MagicMock(spec=RetrievalService)
        mock_retrieval.search.side_effect = StoreNotFoundError("empty store")

        p = RAGPipeline(
            retrieval_service=mock_retrieval,
            context_builder=mock_builder,
            llm_provider=fake_provider,
        )
        with pytest.raises(StoreNotFoundError, match="empty store"):
            p.answer("valid query")

    def test_invalid_query_error_from_retrieval_propagates(
        self, mock_builder: MagicMock, fake_provider: FakeLLMProvider
    ) -> None:
        mock_retrieval = MagicMock(spec=RetrievalService)
        mock_retrieval.search.side_effect = InvalidQueryError("bad query")

        p = RAGPipeline(
            retrieval_service=mock_retrieval,
            context_builder=mock_builder,
            llm_provider=fake_provider,
        )
        with pytest.raises(InvalidQueryError, match="bad query"):
            p.answer("anything")

    def test_generic_retrieval_exception_propagates(
        self, mock_builder: MagicMock, fake_provider: FakeLLMProvider
    ) -> None:
        mock_retrieval = MagicMock(spec=RetrievalService)
        mock_retrieval.search.side_effect = RuntimeError("unexpected retrieval error")

        p = RAGPipeline(
            retrieval_service=mock_retrieval,
            context_builder=mock_builder,
            llm_provider=fake_provider,
        )
        with pytest.raises(RuntimeError, match="unexpected retrieval error"):
            p.answer("valid query")

    def test_provider_not_called_when_retrieval_fails(
        self, mock_builder: MagicMock, fake_provider: FakeLLMProvider
    ) -> None:
        mock_retrieval = MagicMock(spec=RetrievalService)
        mock_retrieval.search.side_effect = StoreNotFoundError("empty")

        p = RAGPipeline(
            retrieval_service=mock_retrieval,
            context_builder=mock_builder,
            llm_provider=fake_provider,
        )
        with pytest.raises(StoreNotFoundError):
            p.answer("valid query")
        assert fake_provider.last_prompt is None


class TestProviderErrorPropagation:
    def test_provider_runtime_error_propagates(
        self, mock_retrieval: MagicMock, mock_builder: MagicMock
    ) -> None:
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.side_effect = RuntimeError("LLM unavailable")

        p = RAGPipeline(
            retrieval_service=mock_retrieval,
            context_builder=mock_builder,
            llm_provider=mock_provider,
        )
        with pytest.raises(RuntimeError, match="LLM unavailable"):
            p.answer("valid query")

    def test_provider_value_error_propagates(
        self, mock_retrieval: MagicMock, mock_builder: MagicMock
    ) -> None:
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.side_effect = ValueError("bad prompt")

        p = RAGPipeline(
            retrieval_service=mock_retrieval,
            context_builder=mock_builder,
            llm_provider=mock_provider,
        )
        with pytest.raises(ValueError, match="bad prompt"):
            p.answer("valid query")


# ===========================================================================
# Deterministic behavior
# ===========================================================================


class TestDeterminism:
    def test_same_inputs_produce_identical_results(
        self,
        mock_retrieval: MagicMock,
        mock_builder: MagicMock,
    ) -> None:
        """Two invocations with identical inputs must return identical results."""
        provider = FakeLLMProvider()
        p = RAGPipeline(
            retrieval_service=mock_retrieval,
            context_builder=mock_builder,
            llm_provider=provider,
        )
        result1 = p.answer("deterministic query", top_k=5)
        result2 = p.answer("deterministic query", top_k=5)

        assert result1.query == result2.query
        assert result1.answer == result2.answer
        # sources should be the same object reference (same mock return)
        assert result1.sources == result2.sources

    def test_same_query_produces_same_prompt(
        self,
        mock_retrieval: MagicMock,
        mock_builder: MagicMock,
    ) -> None:
        prompts: list[str] = []

        class CapturingProvider(LLMProvider):
            def _generate_impl(self, prompt: str) -> str:
                prompts.append(prompt)
                return "answer"

        p = RAGPipeline(
            retrieval_service=mock_retrieval,
            context_builder=mock_builder,
            llm_provider=CapturingProvider(),
        )
        p.answer("test query")
        p.answer("test query")
        assert prompts[0] == prompts[1]
