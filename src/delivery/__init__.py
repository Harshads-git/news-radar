"""
src/delivery/__init__.py
=========================
Delivery package — channels for sending the daily briefing.
"""

from src.delivery.dispatcher import DeliveryDispatcher

__all__ = ["DeliveryDispatcher"]
