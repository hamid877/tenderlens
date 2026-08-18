"""Tests for generation package – Milestone 7: LLM Provider Abstraction.

Covers:
- interface rejects empty / whitespace-only prompt
- fake provider returns the expected default response
- fake provider records the last prompt received
- repeated calls are deterministic (same input → same output)
- fake provider correctly conforms to the LLMProvider interface
- configurable fake response works
- TypeError raised for non-string prompt
"""

from __future__ import annotations

import inspect

import pytest

from generation.fake_provider import FakeLLMProvider
from generation.provider import EmptyPromptError, LLMProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(response: str = FakeLLMProvider.DEFAULT_RESPONSE) -> FakeLLMProvider:
    """Return a fresh :class:`FakeLLMProvider` for each test."""
    return FakeLLMProvider(response=response)


# ---------------------------------------------------------------------------
# Interface contract tests
# ---------------------------------------------------------------------------


class TestLLMProviderInterface:
    """Verify that the abstract interface is correctly defined."""

    def test_llm_provider_is_abstract(self):
        """LLMProvider cannot be instantiated directly."""
        assert inspect.isabstract(LLMProvider)

    def test_generate_is_defined_on_interface(self):
        """LLMProvider exposes a public generate method."""
        assert callable(getattr(LLMProvider, "generate", None))

    def test_generate_impl_is_abstract(self):
        """_generate_impl is the declared abstract method."""
        assert "_generate_impl" in LLMProvider.__abstractmethods__

    def test_fake_provider_is_subclass(self):
        """FakeLLMProvider is a concrete subclass of LLMProvider."""
        assert issubclass(FakeLLMProvider, LLMProvider)

    def test_fake_provider_is_not_abstract(self):
        """FakeLLMProvider can be instantiated (it is concrete)."""
        assert not inspect.isabstract(FakeLLMProvider)


# ---------------------------------------------------------------------------
# Empty / invalid prompt rejection
# ---------------------------------------------------------------------------


class TestPromptValidation:
    """The base class must reject invalid prompts before delegation."""

    @pytest.mark.parametrize(
        "bad_prompt",
        [
            "",          # empty string
            " ",         # single space
            "\t",        # tab
            "\n",        # newline
            "   \n\t ",  # mixed whitespace
        ],
        ids=["empty", "space", "tab", "newline", "mixed-whitespace"],
    )
    def test_empty_prompt_raises(self, bad_prompt: str):
        """generate() raises EmptyPromptError for empty/whitespace prompts."""
        provider = _make_provider()
        with pytest.raises(EmptyPromptError):
            provider.generate(bad_prompt)

    def test_empty_prompt_does_not_record_last_prompt(self):
        """last_prompt must remain None when the prompt is rejected."""
        provider = _make_provider()
        with pytest.raises(EmptyPromptError):
            provider.generate("")
        assert provider.last_prompt is None

    def test_non_string_prompt_raises_type_error(self):
        """generate() raises TypeError when prompt is not a str."""
        provider = _make_provider()
        with pytest.raises(TypeError):
            provider.generate(None)  # type: ignore[arg-type]

    def test_integer_prompt_raises_type_error(self):
        """generate() raises TypeError when prompt is an int."""
        provider = _make_provider()
        with pytest.raises(TypeError):
            provider.generate(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# FakeLLMProvider behaviour
# ---------------------------------------------------------------------------


class TestFakeLLMProvider:
    """Verify the stub's functional behaviour."""

    def test_default_response(self):
        """FakeLLMProvider returns DEFAULT_RESPONSE by default."""
        provider = _make_provider()
        result = provider.generate("What is the contract value?")
        assert result == FakeLLMProvider.DEFAULT_RESPONSE

    def test_records_last_prompt(self):
        """generate() stores the prompt in last_prompt."""
        provider = _make_provider()
        prompt = "What is the procurement scope?"
        provider.generate(prompt)
        assert provider.last_prompt == prompt

    def test_last_prompt_overwritten_on_second_call(self):
        """last_prompt reflects the most-recent call, not the first."""
        provider = _make_provider()
        provider.generate("first prompt")
        provider.generate("second prompt")
        assert provider.last_prompt == "second prompt"

    def test_last_prompt_initially_none(self):
        """last_prompt is None before any call is made."""
        provider = _make_provider()
        assert provider.last_prompt is None

    # ------------------------------------------------------------------
    # Determinism
    # ------------------------------------------------------------------

    def test_repeated_calls_are_deterministic(self):
        """The same prompt always returns the same response."""
        provider = _make_provider()
        prompt = "Repeat this test ten times."
        responses = [provider.generate(prompt) for _ in range(10)]
        assert len(set(responses)) == 1, "Expected identical responses for repeated calls."

    def test_different_prompts_return_same_fixed_response(self):
        """Every prompt returns the same fixed response (stub behaviour)."""
        provider = _make_provider()
        r1 = provider.generate("prompt A")
        r2 = provider.generate("prompt B")
        assert r1 == r2 == FakeLLMProvider.DEFAULT_RESPONSE

    # ------------------------------------------------------------------
    # Configurable response
    # ------------------------------------------------------------------

    def test_configurable_response(self):
        """FakeLLMProvider accepts a custom response at construction."""
        custom_text = "custom-test-answer"
        provider = FakeLLMProvider(response=custom_text)
        assert provider.generate("any prompt") == custom_text

    def test_configurable_response_is_deterministic(self):
        """Custom response is returned consistently across multiple calls."""
        custom_text = "always-this"
        provider = FakeLLMProvider(response=custom_text)
        results = [provider.generate("q") for _ in range(5)]
        assert all(r == custom_text for r in results)

    def test_configurable_response_records_prompt(self):
        """Prompt is still recorded even when a custom response is set."""
        provider = FakeLLMProvider(response="something")
        provider.generate("verify delegation")
        assert provider.last_prompt == "verify delegation"

    # ------------------------------------------------------------------
    # Interface conformance (duck-typing / structural check)
    # ------------------------------------------------------------------

    def test_generate_method_exists(self):
        """FakeLLMProvider exposes the generate method."""
        provider = _make_provider()
        assert callable(provider.generate)

    def test_generate_returns_string(self):
        """generate() always returns a str instance."""
        provider = _make_provider()
        result = provider.generate("Is this a string?")
        assert isinstance(result, str)

    def test_valid_prompt_not_rejected(self):
        """A normal, non-empty prompt succeeds without exception."""
        provider = _make_provider()
        # Should not raise
        result = provider.generate("What are the key deliverables?")
        assert result  # non-empty response

    def test_whitespace_padded_prompt_is_accepted(self):
        """Leading/trailing whitespace is allowed as long as core is non-empty."""
        provider = _make_provider()
        # "  hello  " strips to "hello" – not empty, so must succeed
        result = provider.generate("  hello  ")
        assert isinstance(result, str)
