"""OpenAI hosted LLM provider (Milestone 9).

Implements :class:`~generation.provider.LLMProvider` using the official OpenAI
Python SDK and the **Responses API** (``client.responses.create``).

Configuration
-------------
The provider is driven entirely by environment variables (or a ``.env`` file
loaded upstream); it never reads hard-coded credentials.

``OPENAI_API_KEY``
    Required.  Your OpenAI secret key.  A missing or blank value raises
    :class:`ConfigurationError` at construction time so the application fails
    fast rather than leaking the absence of a key through a runtime HTTP error.

``OPENAI_MODEL``
    Optional.  The model identifier passed to the Responses API.
    Defaults to :data:`DEFAULT_MODEL` (``"gpt-4o-mini"``).

Public interface
----------------
:class:`OpenAIProvider` is a concrete :class:`~generation.provider.LLMProvider`.
Call :meth:`~generation.provider.LLMProvider.generate` with a non-empty prompt;
the base class handles validation before delegating to
:meth:`OpenAIProvider._generate_impl`.

Exceptions
----------
:class:`ConfigurationError`
    Raised at construction time if ``OPENAI_API_KEY`` is absent or blank.

:class:`GenerationError`
    Wraps any SDK / network error so callers do not need to import OpenAI
    internals.  The original exception is chained (``raise ... from exc``)
    but the API key is never included in the message or repr.

Example::

    from generation.openai_provider import OpenAIProvider

    provider = OpenAIProvider()                  # reads env vars
    answer   = provider.generate("Summarise the tender document.")
"""

from __future__ import annotations

import logging
import os

from generation.provider import LLMProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL: str = "gpt-4o-mini"
"""Fallback model used when ``OPENAI_MODEL`` is not set in the environment."""

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class ConfigurationError(Exception):
    """Raised when the provider cannot be initialised due to missing config.

    Specifically raised when ``OPENAI_API_KEY`` is absent or blank.
    """


class GenerationError(RuntimeError):
    """Raised when the OpenAI API call fails.

    Wraps SDK-specific exceptions so callers remain decoupled from the OpenAI
    SDK.  The API key is never included in the message.
    """


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------


class OpenAIProvider(LLMProvider):
    """Concrete :class:`~generation.provider.LLMProvider` backed by OpenAI.

    The provider validates configuration at construction time (fail-fast),
    builds an ``openai.OpenAI`` client, and delegates generation to the
    Responses API.

    Args:
        api_key: Override for the API key (used in tests; normally leave as
            ``None`` so the value is read from ``OPENAI_API_KEY``).
        model:   Override for the model identifier (used in tests; normally
            leave as ``None`` so the value is read from ``OPENAI_MODEL``).

    Raises:
        ConfigurationError: If the resolved API key is absent or blank.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        resolved_key: str = api_key or os.getenv("OPENAI_API_KEY", "")
        if not resolved_key.strip():
            raise ConfigurationError(
                "OPENAI_API_KEY is not set. "
                "Add it to your environment or .env file before using OpenAIProvider."
            )

        self._model: str = (
            model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        )

        # Import lazily so that tests can patch ``openai.OpenAI`` before the
        # class is constructed.  The import is intentionally kept here rather
        # than at module level to avoid import-time side-effects when the SDK
        # is not installed.
        import openai  # noqa: PLC0415

        self._client = openai.OpenAI(api_key=resolved_key)

        logger.debug(
            "OpenAIProvider initialised with model=%r.",
            self._model,
        )

    # ------------------------------------------------------------------
    # LLMProvider implementation
    # ------------------------------------------------------------------

    def _generate_impl(self, prompt: str) -> str:
        """Call the OpenAI Responses API and return the generated text.

        Args:
            prompt: A validated, non-empty prompt (guaranteed by the base
                class before this method is invoked).

        Returns:
            The model's response text as a plain string.

        Raises:
            GenerationError: If the API call raises any exception.  The
                original exception is chained for debugging but the API key
                is never included in the error message.
        """
        logger.debug(
            "OpenAIProvider._generate_impl: model=%r prompt_length=%d.",
            self._model,
            len(prompt),
        )

        try:
            response = self._client.responses.create(
                model=self._model,
                input=prompt,
            )
            text: str = response.output_text
        except Exception as exc:  # noqa: BLE001
            # Do NOT include exc in the message; it may contain the API key
            # in edge-case SDK error paths.
            raise GenerationError(
                f"OpenAI API call failed (model={self._model!r}). "
                "Check logs for details."
            ) from exc

        logger.debug(
            "OpenAIProvider._generate_impl: response_length=%d.",
            len(text),
        )
        return text
