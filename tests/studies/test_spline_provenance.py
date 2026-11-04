"""Check generated claim links using synthetic results, independently of plotting."""

import json
import re
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def test_readme_claim_links_resolve_to_run_claims(tmp_path, monkeypatch):
    # Plotting is optional and irrelevant to this metadata contract.
    plotting = ModuleType("matplotlib.pyplot")
    package = ModuleType("matplotlib")
    package.__path__ = []
    package.pyplot = plotting
    monkeypatch.setitem(sys.modules, "matplotlib", package)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", plotting)
    script = Path(__file__).parents[2] / "scripts" / "spline_resolution_study.py"
    main = runpy.run_path(script)["main"]
    namespace = main.__globals__
    results = [
        {
            "name": "submitted-10",
            "knot_count": 10,
            "knots_hz": [20.0 + 1980.0 * i / 9 for i in range(10)],
            "evaluation_minimum_hz": 20.0,
            "evaluation_maximum_hz": 512.0,
            "max_amplitude_residual_percent": 1.5,
            "max_phase_residual_degrees": 2.5,
        },
        {
            "name": "qnm-dense-19",
            "knot_count": 19,
            "knots_hz": [20.0 + 1980.0 * i / 18 for i in range(19)],
            "evaluation_minimum_hz": 20.0,
            "evaluation_maximum_hz": 2000.0,
            "max_amplitude_residual_percent": 0.5,
            "max_phase_residual_degrees": 0.25,
        },
    ]
    by_name = {row["name"]: row for row in results}
    monkeypatch.setitem(namespace, "configurations", lambda: [SimpleNamespace(name=name) for name in by_name])
    monkeypatch.setitem(namespace, "evaluate", lambda config: by_name[config.name])
    monkeypatch.setitem(namespace, "git_metadata", lambda: ("0" * 40, False))
    monkeypatch.setitem(namespace, "write_figure", lambda *_: None)
    monkeypatch.setattr(sys, "argv", [str(script), "--output-directory", str(tmp_path)])

    main()

    run = json.loads((tmp_path / "run.json").read_text())
    links = dict(
        re.findall(r"^- Claim `([^`]+)` -> result `([^`]+)`", (tmp_path / "README.md").read_text(), re.MULTILINE)
    )
    assert set(run["claims"]) == {
        "submitted_basis_not_demonstrated_over_20_2000_hz",
        "recommended_basis_resolves_bump_over_20_2000_hz",
    }
    assert links == {key: claim["supported_by"] for key, claim in run["claims"].items()}
    assert run["results"] == results
