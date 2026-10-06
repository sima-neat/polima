"""Minimal runtime exports for the bundled MyCobot driver."""

from .config_mycobot280 import MyCobot280M5Config
from .mycobot280 import MyCobot280M5

__all__ = ["MyCobot280M5", "MyCobot280M5Config"]
