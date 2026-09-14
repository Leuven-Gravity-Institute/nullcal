"""Tests for the YAML metadata helpers.

These write the record of how an analysis was configured, so the property that matters is that a
dictionary survives the round trip unchanged -- and that the writer uses the *safe* dumper, since
the files are read back by later runs and by people.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from nullcal.metadata.yaml import load_from_yaml, write_to_yaml


@pytest.mark.unit
def test_round_trip_preserves_a_nested_dictionary(tmp_path):
    """Write then read returns an equal object, including nesting and mixed value types."""
    path = tmp_path / "metadata.yaml"
    data = {
        "duration": 4.0,
        "sampling_frequency": 512,
        "detectors": ["ET1", "ET2", "ET3"],
        "calibration": {"minimum_frequency": 20.0, "n_nodes": 10, "enabled": True},
        "label": "run-a",
        "comment": None,
    }

    write_to_yaml(str(path), data)

    assert load_from_yaml(str(path)) == data


@pytest.mark.unit
def test_written_file_is_plain_readable_yaml(tmp_path):
    """``default_flow_style=False`` gives block style, which is the point of writing YAML at all.

    A configuration record that is not readable by eye is a log file, not metadata.
    """
    path = tmp_path / "metadata.yaml"

    write_to_yaml(str(path), {"calibration": {"n_nodes": 10}})

    content = path.read_text(encoding="utf-8")
    assert "calibration:\n  n_nodes: 10\n" in content
    assert "{" not in content


@pytest.mark.unit
def test_write_uses_the_safe_dumper(tmp_path):
    """Objects with no safe representation are refused rather than serialised as Python types.

    ``yaml.safe_dump`` is what keeps the file loadable by ``safe_load`` -- and keeps arbitrary
    objects from being reconstructed on load. A numpy array is the realistic case here: it is the
    kind of value that reaches metadata by accident, and it must fail loudly at write time rather
    than produce a file that only this one Python environment can read.
    """
    path = tmp_path / "metadata.yaml"

    with pytest.raises(yaml.representer.RepresenterError):
        write_to_yaml(str(path), {"psd": np.zeros(3)})


@pytest.mark.unit
def test_load_uses_the_safe_loader(tmp_path):
    """A file containing a Python object tag must not be instantiated on load."""
    path = tmp_path / "unsafe.yaml"
    path.write_text("value: !!python/object/apply:os.system ['echo unsafe']\n", encoding="utf-8")

    with pytest.raises(yaml.constructor.ConstructorError):
        load_from_yaml(str(path))


@pytest.mark.unit
def test_write_overwrites_an_existing_file(tmp_path):
    """Mode ``w``: re-running an analysis replaces its metadata rather than appending to it."""
    path = tmp_path / "metadata.yaml"

    write_to_yaml(str(path), {"label": "first"})
    write_to_yaml(str(path), {"label": "second"})

    assert load_from_yaml(str(path)) == {"label": "second"}


@pytest.mark.unit
def test_files_are_utf8_encoded(tmp_path):
    """The encoding is pinned on both sides, so a non-ASCII label survives on any platform."""
    path = tmp_path / "metadata.yaml"
    data = {"author": "Ωmega", "note": "résumé"}

    write_to_yaml(str(path), data)

    assert path.read_bytes().decode("utf-8")
    assert load_from_yaml(str(path)) == data


@pytest.mark.unit
def test_loading_an_empty_file_returns_none(tmp_path):
    """Documented boundary: an empty document is ``None``, not an empty dict.

    Recorded because a caller writing ``load_from_yaml(...)["key"]`` gets a ``TypeError`` rather
    than a ``KeyError``, and that is worth knowing before debugging one.
    """
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    assert load_from_yaml(str(path)) is None


@pytest.mark.unit
def test_loading_a_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_from_yaml(str(tmp_path / "absent.yaml"))
