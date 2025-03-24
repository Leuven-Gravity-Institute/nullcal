from __future__ import annotations

from ..utility import logger


def load_config_file(config_path):
    config = {}
    with open(config_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            key, value = line.strip().split('=',1)
            config[key.strip()] = value.strip()
    return config

def resolve_config_conflict(args, config_dict):
    pass
