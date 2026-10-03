"""Inspect the fiducial catalog entry without substituting another simulation.

Run in the isolated science environment through a scheduler. A catalog failure
or deprecation is a blocker, not permission to select a replacement waveform.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import sxs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--producing-commit", required=True)
    args = parser.parse_args()

    catalog = sxs.load("simulations", download=True)
    entry = catalog["SXS:BBH:0305"]
    record = {
        "waveform": "SXS:BBH:0305",
        "producing_commit": args.producing_commit,
        "dirty": False,
        "versions": {"sxs": importlib.metadata.version("sxs")},
        "catalog_tag": catalog.tag,
        "catalog_entry": dict(entry),
        "scope": "catalog inspection only; no waveform extraction or inference",
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)
    simulation = sxs.load("SXS:BBH:0305", download=True, progress=False)
    print(f"Resolved simulation: {simulation.location}", flush=True)


if __name__ == "__main__":
    main()
