"""Keep the temporary bilby surface confined to the R5 likelihood base class."""

from pathlib import Path


def test_bilby_is_only_imported_for_the_temporary_likelihood_base_class():
    source_root = Path(__file__).parents[1] / "src"
    matches = [
        f"{path.relative_to(source_root)}:{line_number}:{line.strip()}"
        for path in source_root.rglob("*.py")
        for line_number, line in enumerate(path.read_text().splitlines(), start=1)
        if "bilby" in line.lower()
    ]

    assert matches == ["nullcal/likelihood/recalibration_likelihood.py:8:from bilby.core.likelihood import Likelihood"]
