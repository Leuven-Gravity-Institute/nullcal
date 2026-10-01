"""Verify the isolated D1 science environment without producing a science result."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
from pathlib import Path

REQUIRED_MODULES = ("ringdown", "sxs", "ripplegw")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--producing-commit", required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    args = parser.parse_args()
    versions = {}
    for name in REQUIRED_MODULES:
        importlib.import_module(name)
        versions[name] = importlib.metadata.version(name)
        print(f"import {name}: OK ({versions[name]})", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "environment import verification; no scientific result",
        "producing_commit": args.producing_commit,
        "python": platform.python_version(),
        "system": platform.system(),
        "architecture": platform.machine(),
        "verified_imports": versions,
        "requirements_sha256": hashlib.sha256(args.requirements.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
