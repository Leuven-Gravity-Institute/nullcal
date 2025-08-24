"""YAML read and write utilities."""

from __future__ import annotations

import yaml


def write_to_yaml(fname: str, data: dict) -> None:
    """Write a dictionary to a YAML file.

    Args:
        fname (str): File name.
        data (dict): A dictionary of data.
    """
    with open(fname, mode="w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False)


def load_from_yaml(fname: str) -> dict:
    """Load from a YAML file.

    Args:
        fname (str): File name.

    Returns:
        dict: A dictionary of data.
    """
    with open(fname, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data
