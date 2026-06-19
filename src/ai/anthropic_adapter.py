"""
src/ai/anthropic_adapter.py
============================
AI provider adapter for Anthropic's Claude API.

Uses the official `anthropic` Python SDK.

Supported models (set AI_MODEL in .env):
  claude-3-5-haiku-20241022    ← recommended (fast, cheap)
  claude-3-5-sonnet-20241022   ← high quality
  claude-3-opus-20240229       ← most capable
"""

from __future__ import annotations

import os

from src.ai.base import BaseAIProvider
from src.exceptions import AIError, AIProviderError, TokenLimitError


class AnthropicProvider(BaseAIProvider):
    """
    Anthropic Claude adapter wrapping the anthropic SDK.

    Automatically picks up ANTHROPIC_API_KEY from the environment.
    """

    PROVIDER_NAME = "anthropic"

    def __init__(self, model: str) -> None:
        super().__init__(model)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise AIProviderError(
                    "anthropic package not installed. Run: uv add anthropic",
                    model=self.model,
                ) from e

            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise AIProviderError(
                    "ANTHROPIC_API_KEY environment variable is not set.",
                    model=self.model,
                )
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
        return self._client

    async def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.3,
        system: str | None = None,
    ) -> str:
        """Call the Anthropic messages API and return response text."""
        try:
            import anthropic
        except ImportError as e:
            raise AIProviderError(
                "anthropic package not installed.", model=self.model
            ) from e

        client = self._get_client()

        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature

        self.log.debug("Anthropic call: model=%s max_tokens=%d", self.model, max_tokens)

        try:
            response = await client.messages.create(**kwargs)
            text = response.content[0].text if response.content else ""
            return text.strip()

        except anthropic.AuthenticationError as e:
            raise AIProviderError(
                f"Anthropic auth failed: {e}", model=self.model, status_code=401
            ) from e

        except anthropic.RateLimitError as e:
            raise AIProviderError(
                f"Anthropic rate limit: {e}", model=self.model, status_code=429
            ) from e

        except anthropic.BadRequestError as e:
            msg = str(e).lower()
            if "token" in msg or "length" in msg:
                raise TokenLimitError(
                    f"Anthropic context too long: {e}", model=self.model
                ) from e
            raise AIError(f"Anthropic bad request: {e}", model=self.model) from e

        except Exception as e:
            raise AIError(f"Anthropic unexpected error: {e}", model=self.model) from e
