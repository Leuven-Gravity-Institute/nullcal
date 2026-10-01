"""Audit D1 prerequisite artifacts; this command does not run either fit.

A successful audit establishes artifact availability and provenance, not
scientific correctness or convergence. Missing inputs must remain explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

REGISTRATION = Path(__file__).resolve().parents[1] / "docs/dev/d1-control/preregistration.json"
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def sha256(path: Path) -> str:
    """Hash a file without loading a waveform into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_record(record: dict, root: Path) -> list[str]:
    reasons = []
    if not isinstance(record.get("producing_commit"), str) or not COMMIT_PATTERN.fullmatch(record["producing_commit"]):
        reasons.append("producing_commit must be a full lowercase commit SHA")
    if record.get("dirty") is not False:
        reasons.append("dirty must explicitly be false")
    if not isinstance(record.get("command"), str) or not record["command"].strip():
        reasons.append("producing command is missing")
    if not isinstance(record.get("versions"), dict) or not record["versions"]:
        reasons.append("dependency versions are missing")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        return [*reasons, "hashed output artifacts are missing"]
    for name, expected_hash in artifacts.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            reasons.append(f"artifact path escapes input directory: {name}")
        elif not path.is_file():
            reasons.append(f"missing referenced artifact: {name}")
        elif sha256(path) != expected_hash:
            reasons.append(f"SHA-256 mismatch: {name}")
    return reasons


def audit_inputs(root: Path) -> dict:
    """Check required files and their evidence records without running inference."""
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    failures = []
    hashes = {}
    for name in registration["required_artifacts"]:
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            failures.append({"artifact": name, "reason": "missing or empty required artifact"})
            continue
        hashes[name] = sha256(path)
        if path.suffix != ".json":
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as error:
            failures.append({"artifact": name, "reason": f"invalid JSON: {error}"})
            continue
        reasons = _check_record(record, root) if isinstance(record, dict) else ["record must be a JSON object"]
        if name == "waveform-manifest.json" and isinstance(record, dict):
            if record.get("waveform") != registration["injection"]["waveform"]:
                reasons.append("waveform must be the registered SXS identifier")
            if not isinstance(record.get("artifacts"), dict) or "waveform.h5" not in record["artifacts"]:
                reasons.append("waveform.h5 must be linked by SHA-256")
        if reasons:
            failures.append({"artifact": name, "reason": "; ".join(reasons)})
    return {
        "protocol": registration["protocol"],
        "registration_sha256": sha256(REGISTRATION),
        "status": "blocked" if failures else "inputs_present",
        "scope": "availability and provenance only; not a scientific validation or inference run",
        "artifact_sha256": hashes,
        "failures": failures,
        "scientific_results": None,
    }


def main(argv: list[str] | None = None) -> int:
    """Write the audit and return exit 2 when scientific prerequisites are missing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit_inputs(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 2 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
