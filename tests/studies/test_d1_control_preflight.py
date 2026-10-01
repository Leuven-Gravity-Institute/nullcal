"""Prerequisite failures must never be labelled as D1 measurements."""

import json

from d1_control_preflight import audit_inputs, main


def test_missing_inputs_names_every_required_artifact(tmp_path):
    report = audit_inputs(tmp_path)
    assert report["status"] == "blocked"
    assert report["scientific_results"] is None
    assert {item["artifact"] for item in report["failures"]} == {
        "waveform.h5",
        "waveform-manifest.json",
        "ringdown-anchor.json",
        "gr-imr-validation.json",
        "injection-prescription.json",
        "resolution-precheck.json",
    }


def test_malformed_json_is_reported_instead_of_crashing(tmp_path):
    (tmp_path / "ringdown-anchor.json").write_text("{broken", encoding="utf-8")
    report = audit_inputs(tmp_path)
    failure = next(item for item in report["failures"] if item["artifact"] == "ringdown-anchor.json")
    assert "invalid JSON" in failure["reason"]


def test_cli_returns_nonzero_and_writes_no_measurements(tmp_path):
    output = tmp_path / "audit.json"
    assert main(["--inputs", str(tmp_path), "--output", str(output)]) == 2
    report = json.loads(output.read_text())
    assert report["status"] == "blocked"
    assert report["scientific_results"] is None


def _write_fixture_records(tmp_path):
    from d1_control_preflight import REGISTRATION, sha256

    waveform = tmp_path / "waveform.h5"
    waveform.write_bytes(b"synthetic test fixture, never an SXS measurement")
    output = tmp_path / "output.npz"
    output.write_bytes(b"synthetic test fixture, never a posterior")
    record = {
        "producing_commit": "a" * 40,
        "dirty": False,
        "command": "test fixture",
        "versions": {"test": "fixture"},
        "artifacts": {"output.npz": sha256(output)},
    }
    for name in json.loads(REGISTRATION.read_text())["required_artifacts"]:
        if name.endswith(".json"):
            value = dict(record)
            if name == "waveform-manifest.json":
                value.update(waveform="SXS:BBH:0305", artifacts={"waveform.h5": sha256(waveform)})
            (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")


def test_present_artifacts_are_not_labelled_as_scientific_success(tmp_path):
    _write_fixture_records(tmp_path)
    report = audit_inputs(tmp_path)
    assert report["status"] == "inputs_present"
    assert report["scientific_results"] is None
    assert report["failures"] == []


def test_changed_waveform_fails_hash_validation(tmp_path):
    _write_fixture_records(tmp_path)
    (tmp_path / "waveform.h5").write_bytes(b"changed")
    report = audit_inputs(tmp_path)
    assert report["status"] == "blocked"
    assert any("SHA-256 mismatch: waveform.h5" in item["reason"] for item in report["failures"])


def test_artifact_path_may_not_escape_input_bundle(tmp_path):
    _write_fixture_records(tmp_path)
    path = tmp_path / "ringdown-anchor.json"
    record = json.loads(path.read_text())
    record["artifacts"] = {"../outside.npz": "a" * 64}
    path.write_text(json.dumps(record))
    report = audit_inputs(tmp_path)
    assert any("escapes input directory" in item["reason"] for item in report["failures"])


def test_unknown_provenance_and_dirty_results_are_blocked(tmp_path):
    _write_fixture_records(tmp_path)
    path = tmp_path / "ringdown-anchor.json"
    record = json.loads(path.read_text())
    record.update(producing_commit="unknown", dirty=True, command="", versions={})
    path.write_text(json.dumps(record))
    report = audit_inputs(tmp_path)
    failure = next(item for item in report["failures"] if item["artifact"] == "ringdown-anchor.json")
    assert "producing_commit" in failure["reason"]
    assert "dirty" in failure["reason"]
    assert "command" in failure["reason"]
    assert "versions" in failure["reason"]
