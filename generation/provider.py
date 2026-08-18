"""Provider-independent LLM generation interface (Milestone 7).

Defines the abstract :class:`LLMProvider` contract that all concrete provider
implementations must satisfy.  The interface is intentionally free of any
reference to specific providers (OpenAI, Gemini, Anthropic, Qwen, Ollama,
etc.) and makes no network or API calls.

Usage::

    from generation.provider import LLMProvider

    class MyProvider(LLMProvider):
        def generate(self, prompt: str) -> str:
            ...

Custom exceptions
-----------------
:class:`EmptyPromptError`
    Raised when :meth:`LLMProvider.generate` receives an empty or
    whitespace-only prompt.
"""

from __future__ import annotations

import abc
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class EmptyPromptError(ValueError):
    """Raised when a prompt is empty or contains only whitespace."""


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class LLMProvider(abc.ABC):
    """Abstract base class for all LLM generation providers.

    Concrete subclasses must implement :meth:`generate`.  The base class
    handles prompt validation so that every implementation benefits from
    consistent input guards without repeating the check.

    Example::

        class EchoProvider(LLMProvider):
            def generate(self, prompt: str) -> str:
                return f"Echo: {prompt}"
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Generate a response for *prompt*.

        Validates that *prompt* is a non-empty, non-whitespace string, then
        delegates to :meth:`_generate_impl` which concrete subclasses must
        implement.

        Args:
            prompt: The input text to send to the language model.

        Returns:
            The model's response as a plain string.

        Raises:
            EmptyPromptError: If *prompt* is empty or whitespace-only.
            TypeError:        If *prompt* is not a string.
        """
        if not isinstance(prompt, str):
            raise TypeError(
                f"prompt must be a str; got {type(prompt).__name__!r}."
            )
        if not prompt.strip():
            raise EmptyPromptError(
                "prompt must not be empty or whitespace-only."
            )

        logger.debug("LLMProvider.generate: prompt length=%d.", len(prompt))
        return self._generate_impl(prompt)

    # ------------------------------------------------------------------
    # Extension point
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _generate_impl(self, prompt: str) -> str:
        """Internal generation hook for subclasses.

        Called only after the base class has validated *prompt*.  Subclasses
        must not call :meth:`generate` recursively; implement logic here
        instead.

        Args:
            prompt: A validated, non-empty prompt string.

        Returns:
            The model's response as a plain string.
        """
