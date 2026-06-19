"""
src/ai/openai_adapter.py
========================
AI provider adapter for OpenAI's API (GPT-4o, GPT-4o-mini, o1, o3-mini, etc.)

Uses the official `openai` Python SDK (v1.x async client).

Supported models (set AI_MODEL in .env):
  gpt-4o-mini     ← recommended (fast, cheap, great quality)
  gpt-4o          ← highest quality, more expensive
  o1-mini         ← reasoning model (slower)
  o3-mini         ← latest reasoning model

Error mapping:
  openai.AuthenticationError  → AIProviderError (invalid key)
  openai.RateLimitError       → AIProviderError (quota hit, treat as fatal)
  openai.APIConnectionError   → AIError (network issue, retried by tenacity)
  openai.BadRequestError      → TokenLimitError (prompt too long)
"""

from __future__ import annotations

import os

from src.ai.base import BaseAIProvider
from src.exceptions import AIError, AIProviderError, TokenLimitError


class OpenAIProvider(BaseAIProvider):
    """
    OpenAI adapter wrapping the async openai client.

    Automatically picks up OPENAI_API_KEY from the environment.
    Uses the chat completions endpoint for all calls.
    """

    PROVIDER_NAME = "openai"

    def __init__(self, model: str) -> None:
        super().__init__(model)
        self._client = None  # lazy-initialized on first call

    def _get_client(self):
        """Lazy-initialize the async OpenAI client."""
        if self._client is None:
            try:
                import openai
            except ImportError as e:
                raise AIProviderError(
                    "openai package not installed. Run: uv add openai",
                    model=self.model,
                ) from e

            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise AIProviderError(
                    "OPENAI_API_KEY environment variable is not set.",
                    model=self.model,
                )
            self._client = openai.AsyncOpenAI(api_key=api_key)
        return self._client

    async def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.3,
        system: str | None = None,
    ) -> str:
        """Call the OpenAI chat completions API and return response text."""
        try:
            import openai
        except ImportError as e:
            raise AIProviderError("openai package not installed.", model=self.model) from e

        client = self._get_client()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        self.log.debug("OpenAI call: model=%s max_tokens=%d", self.model, max_tokens)

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = response.choices[0].message.content or ""
            return text.strip()

        except openai.AuthenticationError as e:
            raise AIProviderError(
                f"OpenAI authentication failed: {e}",
                model=self.model,
                status_code=401,
            ) from e

        except openai.RateLimitError as e:
            raise AIProviderError(
                f"OpenAI rate limit / quota exceeded: {e}",
                model=self.model,
                status_code=429,
            ) from e

        except openai.BadRequestError as e:
            # Context window exceeded or invalid request
            msg = str(e).lower()
            if "context" in msg or "token" in msg or "length" in msg:
                raise TokenLimitError(
                    f"OpenAI context window exceeded: {e}",
                    model=self.model,
                ) from e
            raise AIError(f"OpenAI bad request: {e}", model=self.model) from e

        except openai.APIConnectionError as e:
            raise AIError(
                f"OpenAI connection error: {e}", model=self.model
            ) from e

        except Exception as e:
            raise AIError(
                f"OpenAI unexpected error: {e}", model=self.model
            ) from e
