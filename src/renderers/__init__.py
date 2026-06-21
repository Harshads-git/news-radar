"""
src/renderers/__init__.py
=========================
Renderers package — converts Briefing objects to output formats.
"""

from src.renderers.html import render_html
from src.renderers.markdown import render_markdown

__all__ = ["render_markdown", "render_html"]
