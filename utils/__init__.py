"""Shared utilities, independent of detector and training dependencies."""

from .config import load_config
from .yacs import CfgNode

__all__ = ["CfgNode", "load_config"]
