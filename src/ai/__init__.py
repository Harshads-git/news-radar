"""
src/ai/__init__.py
==================
AI package: provider adapters + scorer + summarizer.

Canonical imports:
    from src.ai import AIProviderFactory
    from src.ai.scorer import NewsScorer
    from src.ai.summarizer import NewsSummarizer   (Day 9)
"""

from src.ai.base import AIProviderFactory, BaseAIProvider

__all__ = ["BaseAIProvider", "AIProviderFactory"]
