"""Load a project YAML configuration as a shared CfgNode."""

from pathlib import Path

import yaml

from .yacs import CfgNode


def load_config(path, *, freeze=False):
    """Read a YAML mapping; optionally freeze the resulting configuration."""
    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping")
    config = CfgNode(data)
    if freeze:
        config.freeze()
    return config
