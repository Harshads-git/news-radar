"""
src/ai/gemini_adapter.py
========================
AI provider adapter for Google's Gemini API.

Uses the official `google-generativeai` SDK (generativeai Python package).

Supported models (set AI_MODEL in .env):
  gemini-1.5-flash    ← recommended (fast, cheap, generous free tier)
  gemini-1.5-pro      ← highest quality
  gemini-2.0-flash    ← latest, fast
  gemini-2.0-pro      ← latest, highest quality

Error mapping:
  google.api_core.exceptions.PermissionDenied    → AIProviderError (bad key)
  google.api_core.exceptions.ResourceExhausted   → AIProviderError (quota)
  google.api_core.exceptions.InvalidArgument     → TokenLimitError (too long)
  google.api_core.exceptions.ServiceUnavailable  → AIError (retry candidate)
"""

from __future__ import annotations

import os

from src.ai.base import BaseAIProvider
from src.exceptions import AIError, AIProviderError, TokenLimitError


class GeminiProvider(BaseAIProvider):
    """
    Google Gemini adapter wrapping the google-generativeai SDK.

    Automatically picks up GEMINI_API_KEY from the environment.
    Uses generate_content() for all calls.
    """

    PROVIDER_NAME = "gemini"

    def __init__(self, model: str) -> None:
        super().__init__(model)
        self._configured = False

    def _ensure_configured(self):
        """Configure the Gemini SDK with the API key (once per process)."""
        if not self._configured:
            try:
                import google.generativeai as genai
            except ImportError as e:
                raise AIProviderError(
                    "google-generativeai package not installed. Run: uv add google-generativeai",
                    model=self.model,
                ) from e

            api_key = os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                raise AIProviderError(
                    "GEMINI_API_KEY environment variable is not set.",
                    model=self.model,
                )
            genai.configure(api_key=api_key)
            self._configured = True

    async def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.3,
        system: str | None = None,
    ) -> str:
        """Call the Gemini generate_content API and return response text."""
        try:
            import asyncio

            import google.generativeai as genai
        except ImportError as e:
            raise AIProviderError(
                "google-generativeai package not installed.", model=self.model
            ) from e

        self._ensure_configured()

        # Build the full prompt (Gemini's Python SDK doesn't use separate system messages
        # in the same way; prepend system context to the prompt if provided)
        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        self.log.debug("Gemini call: model=%s max_tokens=%d", self.model, max_tokens)

        try:
            model_client = genai.GenerativeModel(self.model)

            # The Gemini SDK's generate_content is synchronous — run in thread executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model_client.generate_content(
                    full_prompt,
                    generation_config=generation_config,
                ),
            )

            text = response.text or ""
            return text.strip()

        except Exception as e:
            # Map google.api_core exceptions to our hierarchy
            module = type(e).__module__
            class_name = type(e).__name__

            if "PermissionDenied" in class_name or "Unauthenticated" in class_name:
                raise AIProviderError(
                    f"Gemini authentication failed: {e}",
                    model=self.model,
                    status_code=401,
                ) from e

            if "ResourceExhausted" in class_name or "QuotaExceeded" in class_name:
                raise AIProviderError(
                    f"Gemini quota exceeded: {e}",
                    model=self.model,
                    status_code=429,
                ) from e

            if "InvalidArgument" in class_name:
                msg = str(e).lower()
                if "token" in msg or "length" in msg or "context" in msg:
                    raise TokenLimitError(
                        f"Gemini context window exceeded: {e}",
                        model=self.model,
                    ) from e
                raise AIError(f"Gemini invalid argument: {e}", model=self.model) from e

            if "ServiceUnavailable" in class_name or "DeadlineExceeded" in class_name:
                raise AIError(
                    f"Gemini service unavailable: {e}", model=self.model
                ) from e

            raise AIError(f"Gemini unexpected error: {e}", model=self.model) from e
