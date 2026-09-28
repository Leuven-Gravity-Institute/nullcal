"""Guard the cost ledger against leaking scheduler identifiers.

The ledger is published in a public repository, so it records a hardware class
(device platform, accelerator model, CPU architecture and core count) and the
scheduler job id, never a node hostname or a scheduler partition. These tests
exercise the production writer, loader and migration paths, and scan the
committed artifacts, so a regression fails here rather than shipping. Fixtures
use invented names; no real cluster identifier appears in this file.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import gpu_cost_ledger
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs" / "dev"
ARMS = DOCS / "gpu-cost-ledger-arms"


def test_find_scheduler_identifiers_flags_keys_and_values():
    payload = {"hostname": "node-a01", "note": "ran on node-a01", "partition": "gpu-part"}

    offenders = gpu_cost_ledger.find_scheduler_identifiers(payload)

    assert any("hostname" in item for item in offenders)
    assert any("partition" in item for item in offenders)
    assert any("node-a01" in item for item in offenders)
    assert gpu_cost_ledger.find_scheduler_identifiers({"hardware": {"platform": "gpu"}}) == []


@pytest.mark.parametrize("value", ["node-a01", "compute-7", "node01", "gpu12", "batch-node3"])
def test_generic_hostname_pattern_flags_a_machine_name_under_an_innocent_key(value):
    # The key is not in the denylist, so only the generic pattern can reject it.
    offenders = gpu_cost_ledger.find_scheduler_identifiers({"note": f"scheduled on {value}"})

    assert offenders, f"generic pattern missed {value!r}"


@pytest.mark.parametrize("value", ["gpu", "cpu", "x86_64", "NVIDIA A30", "2026-09-25T18:17:58.124464+00:00"])
def test_generic_hostname_pattern_does_not_flag_hardware_or_timestamps(value):
    assert gpu_cost_ledger.find_scheduler_identifiers({"note": value}) == []


@pytest.mark.parametrize(
    "payload",
    [
        {"hostname": "node-a01"},
        {"node_list": "node-a01"},
        {"partition": "gpu-part"},
        {"note": "scheduled on node-a01"},
    ],
)
def test_write_json_record_rejects_scheduler_identifiers(tmp_path, payload):
    with pytest.raises(ValueError, match="scheduler identifiers"):
        gpu_cost_ledger.write_json_record(tmp_path / "arm.json", payload, context="arm")


def test_write_json_record_accepts_a_hardware_class(tmp_path):
    path = tmp_path / "arm.json"

    gpu_cost_ledger.write_json_record(path, {"hardware": {"platform": "gpu"}}, context="arm")

    assert json.loads(path.read_text(encoding="utf-8")) == {"hardware": {"platform": "gpu"}}


def test_merge_rejects_a_dirty_arm(tmp_path):
    arms = tmp_path / "arms"
    arms.mkdir()
    (arms / "reference-gpu.json").write_text(json.dumps({"hostname": "node-a01"}), encoding="utf-8")
    arguments = SimpleNamespace(input_directory=arms, output_directory=tmp_path / "out", basename="gpu-cost-ledger")

    with pytest.raises(ValueError, match="scheduler identifiers"):
        gpu_cost_ledger.merge(arguments)


def test_redact_arm_record_replaces_identifiers_with_hardware():
    record = {
        "platform": "gpu",
        "devices": [{"platform": "gpu", "device_kind": "NVIDIA A30", "id": 0}],
        "hostname": "node-a01",
        "slurm": {"job_id": "128128", "partition": "gpu-part", "node_list": "node-a01"},
        "posterior": {"seconds_per_posterior": 6.17371466725308},
    }

    redacted = gpu_cost_ledger.redact_arm_record(
        record, cpu_architecture="x86_64", cpu_count_gpu=128, cpu_count_cpu=192
    )

    assert "hostname" not in redacted
    assert "slurm" not in redacted
    assert redacted["hardware"] == {
        "platform": "gpu",
        "accelerator": "NVIDIA A30",
        "cpu_architecture": "x86_64",
        "cpu_count": 128,
    }
    assert redacted["job"] == {"job_id": "128128"}
    assert redacted["posterior"] == record["posterior"]
    assert gpu_cost_ledger.find_scheduler_identifiers(redacted) == []
    # The migration is idempotent: a second pass leaves the record unchanged.
    assert (
        gpu_cost_ledger.redact_arm_record(redacted, cpu_architecture="x86_64", cpu_count_gpu=1, cpu_count_cpu=1)
        == redacted
    )


def test_committed_ledger_artifacts_carry_no_scheduler_identifiers():
    artifacts = [
        *sorted(ARMS.glob("*.json")),
        DOCS / "gpu-cost-ledger.json",
        DOCS / "gpu-cost-ledger.md",
        DOCS / "gpu-cost-ledger-manifest.json",
    ]
    assert artifacts

    for path in artifacts:
        text = path.read_text(encoding="utf-8")
        assert gpu_cost_ledger.find_scheduler_identifiers(text) == [], f"{path.name} leaks a scheduler identifier"
        if path.suffix == ".json":
            payload = json.loads(text)
            assert gpu_cost_ledger.find_scheduler_identifiers(payload) == [], f"{path.name} leaks a scheduler field"
