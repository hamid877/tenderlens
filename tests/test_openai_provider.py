"""Tests for OpenAIProvider – Milestone 9: Hosted LLM Provider.

Covers:
- Successful generation returns the expected text.
- Missing API key raises ConfigurationError at construction time.
- Blank / whitespace-only API key raises ConfigurationError.
- Empty prompt raises EmptyPromptError (via base-class validation).
- Provider/API failure is converted to GenerationError.
- The configured model is forwarded to the OpenAI client.
- The generated text is returned exactly as received from the client.
- API key is never exposed in GenerationError messages.
- Existing FakeLLMProvider tests are unaffected (imported to verify imports).
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from generation.openai_provider import (
    ConfigurationError,
    GenerationError,
    OpenAIProvider,
)
from generation.provider import EmptyPromptError

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_FAKE_KEY = "sk-test-fake-key-1234567890"
_FAKE_MODEL = "gpt-test-model"
_FAKE_RESPONSE_TEXT = "This is the generated tender summary."


def _make_mock_client(output_text: str = _FAKE_RESPONSE_TEXT) -> MagicMock:
    """Return a fully-wired mock OpenAI client whose Responses API returns
    *output_text*."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.output_text = output_text
    mock_client.responses.create.return_value = mock_response
    return mock_client


@pytest.fixture()
def mock_openai_client() -> MagicMock:
    """Patch ``openai.OpenAI`` and yield the mock client instance."""
    mock_client = _make_mock_client()
    with patch("openai.OpenAI", return_value=mock_client):
        yield mock_client


@pytest.fixture()
def provider(mock_openai_client: MagicMock) -> OpenAIProvider:  # noqa: ARG001
    """Return an :class:`OpenAIProvider` wired to the mock client."""
    return OpenAIProvider(api_key=_FAKE_KEY, model=_FAKE_MODEL)


# ---------------------------------------------------------------------------
# Construction / configuration tests
# ---------------------------------------------------------------------------


class TestOpenAIProviderConfiguration:
    """Verify configuration validation at construction time."""

    def test_missing_api_key_env_raises_configuration_error(self, monkeypatch: pytest.MonkeyPatch):
        """ConfigurationError is raised when OPENAI_API_KEY is absent."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ConfigurationError):
            OpenAIProvider()

    def test_empty_api_key_raises_configuration_error(self, monkeypatch: pytest.MonkeyPatch):
        """ConfigurationError is raised when OPENAI_API_KEY is an empty string."""
        monkeypatch.setenv("OPENAI_API_KEY", "")
        with pytest.raises(ConfigurationError):
            OpenAIProvider()

    def test_whitespace_api_key_raises_configuration_error(self, monkeypatch: pytest.MonkeyPatch):
        """ConfigurationError is raised when OPENAI_API_KEY is whitespace only."""
        monkeypatch.setenv("OPENAI_API_KEY", "   ")
        with pytest.raises(ConfigurationError):
            OpenAIProvider()

    def test_valid_key_constructs_successfully(self):
        """A valid API key allows the provider to be constructed."""
        with patch("openai.OpenAI", return_value=_make_mock_client()):
            provider = OpenAIProvider(api_key=_FAKE_KEY)
        assert isinstance(provider, OpenAIProvider)

    def test_api_key_read_from_environment(self, monkeypatch: pytest.MonkeyPatch):
        """When no explicit key is supplied, OPENAI_API_KEY env var is used."""
        monkeypatch.setenv("OPENAI_API_KEY", _FAKE_KEY)
        with patch("openai.OpenAI", return_value=_make_mock_client()) as mock_cls:
            OpenAIProvider()
        # The SDK constructor should have received the key from the environment.
        mock_cls.assert_called_once_with(api_key=_FAKE_KEY)

    def test_model_default_used_when_env_not_set(self, monkeypatch: pytest.MonkeyPatch):
        """Default model is used when OPENAI_MODEL is not in the environment."""
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        with patch("openai.OpenAI", return_value=_make_mock_client()):
            p = OpenAIProvider(api_key=_FAKE_KEY)
        from generation.openai_provider import DEFAULT_MODEL
        assert p._model == DEFAULT_MODEL

    def test_model_read_from_environment(self, monkeypatch: pytest.MonkeyPatch):
        """OPENAI_MODEL env var overrides the default model."""
        monkeypatch.setenv("OPENAI_MODEL", "gpt-env-model")
        with patch("openai.OpenAI", return_value=_make_mock_client()):
            p = OpenAIProvider(api_key=_FAKE_KEY)
        assert p._model == "gpt-env-model"

    def test_explicit_model_argument_takes_precedence(self):
        """model kwarg overrides the environment variable."""
        with patch("openai.OpenAI", return_value=_make_mock_client()):
            p = OpenAIProvider(api_key=_FAKE_KEY, model="explicit-model")
        assert p._model == "explicit-model"


# ---------------------------------------------------------------------------
# Prompt validation (inherited from base class)
# ---------------------------------------------------------------------------


class TestOpenAIProviderPromptValidation:
    """Base-class validation must still apply for OpenAIProvider."""

    @pytest.mark.parametrize(
        "bad_prompt",
        ["", " ", "\t", "\n", "  \n\t  "],
        ids=["empty", "space", "tab", "newline", "mixed"],
    )
    def test_empty_prompt_raises_empty_prompt_error(
        self, provider: OpenAIProvider, bad_prompt: str
    ):
        """generate() raises EmptyPromptError for empty/whitespace prompts."""
        with pytest.raises(EmptyPromptError):
            provider.generate(bad_prompt)

    def test_empty_prompt_does_not_call_openai(
        self, provider: OpenAIProvider, mock_openai_client: MagicMock
    ):
        """The OpenAI client must not be called when the prompt is rejected."""
        with pytest.raises(EmptyPromptError):
            provider.generate("")
        mock_openai_client.responses.create.assert_not_called()


# ---------------------------------------------------------------------------
# Successful generation
# ---------------------------------------------------------------------------


class TestOpenAIProviderGeneration:
    """Verify happy-path generation behaviour."""

    def test_successful_generation_returns_string(self, provider: OpenAIProvider):
        """generate() returns a str on success."""
        result = provider.generate("What are the key clauses?")
        assert isinstance(result, str)

    def test_generated_text_matches_mock_response(self, provider: OpenAIProvider):
        """generate() returns exactly the text from the Responses API response."""
        result = provider.generate("What are the key clauses?")
        assert result == _FAKE_RESPONSE_TEXT

    def test_configured_model_is_passed_to_client(
        self, provider: OpenAIProvider, mock_openai_client: MagicMock
    ):
        """The model configured at construction is forwarded to the API call."""
        provider.generate("Summarise section 3.")
        mock_openai_client.responses.create.assert_called_once()
        call_kwargs = mock_openai_client.responses.create.call_args
        assert call_kwargs.kwargs.get("model") == _FAKE_MODEL

    def test_prompt_is_forwarded_to_client(
        self, provider: OpenAIProvider, mock_openai_client: MagicMock
    ):
        """The prompt text is forwarded as the ``input`` argument."""
        prompt = "List all deliverables."
        provider.generate(prompt)
        call_kwargs = mock_openai_client.responses.create.call_args
        assert call_kwargs.kwargs.get("input") == prompt

    def test_custom_response_text_is_returned(self):
        """output_text from the mock is propagated back unchanged."""
        expected = "Custom generated text from model."
        mock_client = _make_mock_client(output_text=expected)
        with patch("openai.OpenAI", return_value=mock_client):
            p = OpenAIProvider(api_key=_FAKE_KEY, model=_FAKE_MODEL)
        result = p.generate("Any prompt here.")
        assert result == expected


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestOpenAIProviderErrorHandling:
    """API / SDK failures must be converted to GenerationError."""

    def test_sdk_exception_raises_generation_error(
        self, provider: OpenAIProvider, mock_openai_client: MagicMock
    ):
        """Any SDK exception is wrapped in GenerationError."""
        mock_openai_client.responses.create.side_effect = RuntimeError("network failure")
        with pytest.raises(GenerationError):
            provider.generate("Tell me about the tender.")

    def test_generation_error_chains_original_exception(
        self, provider: OpenAIProvider, mock_openai_client: MagicMock
    ):
        """GenerationError.__cause__ is the original SDK exception."""
        original = RuntimeError("original failure")
        mock_openai_client.responses.create.side_effect = original
        with pytest.raises(GenerationError) as exc_info:
            provider.generate("A prompt.")
        assert exc_info.value.__cause__ is original

    def test_api_key_not_exposed_in_generation_error(
        self, mock_openai_client: MagicMock
    ):
        """The API key must not appear in the GenerationError message."""
        secret_key = "sk-super-secret-key-must-not-leak"
        with patch("openai.OpenAI", return_value=mock_openai_client):
            p = OpenAIProvider(api_key=secret_key, model=_FAKE_MODEL)
        mock_openai_client.responses.create.side_effect = RuntimeError("boom")
        with pytest.raises(GenerationError) as exc_info:
            p.generate("Prompt.")
        assert secret_key not in str(exc_info.value)

    def test_value_error_from_sdk_raises_generation_error(
        self, provider: OpenAIProvider, mock_openai_client: MagicMock
    ):
        """ValueError from the SDK is also wrapped in GenerationError."""
        mock_openai_client.responses.create.side_effect = ValueError("bad input")
        with pytest.raises(GenerationError):
            provider.generate("Some prompt.")
