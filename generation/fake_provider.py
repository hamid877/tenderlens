"""Fake/stub LLM provider for deterministic testing (Milestone 7).

:class:`FakeLLMProvider` implements :class:`~generation.provider.LLMProvider`
without making any network or API calls.  It is intended solely for use in
tests and local development.

Features
--------
- Returns a configurable, fixed response string for every call.
- Records the most-recent prompt so tests can verify delegation.
- Deterministic: the same prompt always returns the same response.

Usage::

    from generation.fake_provider import FakeLLMProvider

    provider = FakeLLMProvider()
    response = provider.generate("What is the contract value?")
    assert response == FakeLLMProvider.DEFAULT_RESPONSE
    assert provider.last_prompt == "What is the contract value?"

    # Custom response
    custom = FakeLLMProvider(response="£1 million")
    assert custom.generate("price?") == "£1 million"
"""

from __future__ import annotations

from generation.provider import LLMProvider

# ---------------------------------------------------------------------------
# FakeLLMProvider
# ---------------------------------------------------------------------------


class FakeLLMProvider(LLMProvider):
    """Deterministic stub implementation of :class:`~generation.provider.LLMProvider`.

    Args:
        response: The fixed string returned by every :meth:`generate` call.
            Defaults to :data:`DEFAULT_RESPONSE`.

    Attributes:
        DEFAULT_RESPONSE: Module-level sentinel used as the default response.
        last_prompt:      The most-recent prompt passed to :meth:`generate`,
                          or ``None`` if :meth:`generate` has not been called.
    """

    DEFAULT_RESPONSE: str = "fake-llm-response"

    def __init__(self, response: str = DEFAULT_RESPONSE) -> None:
        self._response: str = response
        self.last_prompt: str | None = None

    # ------------------------------------------------------------------
    # LLMProvider implementation
    # ------------------------------------------------------------------

    def _generate_impl(self, prompt: str) -> str:
        """Record *prompt* and return the configured fixed response.

        Args:
            prompt: Validated, non-empty prompt string (guaranteed by base
                class before this method is called).

        Returns:
            The fixed response string supplied at construction time.
        """
        self.last_prompt = prompt
        return self._response
