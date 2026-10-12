"""Compatibility import for the unchanged ACT SO101 control loop.

Only the policy transport changes; do not add motor or camera operations here.
"""

from polima.robot.so101 import SO101PolicyClient as ACTSOMClient

__all__ = ["ACTSOMClient"]
