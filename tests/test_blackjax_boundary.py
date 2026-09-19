from __future__ import annotations

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_uses_blackjax_without_bilby():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = metadata["project"]["dependencies"]
    names = {requirement.split(">=")[0].lower().replace("_", "-") for requirement in dependencies}

    assert "blackjax" in names
    assert "bilby" not in names
    assert "bilby-pipe" not in names


def test_shipped_package_has_no_bilby_imports():
    imports = []
    for path in (ROOT / "src" / "nullcal").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names if alias.name.split(".")[0] == "bilby")
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "bilby":
                imports.append(node.module)
    assert imports == []


def test_accelerator_extras_are_explicit():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = metadata["project"]["optional-dependencies"]

    assert extras["jax"] == ["jax>=0.11.1"]
    assert extras["cuda"] == ["jax[cuda13]>=0.11.1; sys_platform == 'linux'"]
