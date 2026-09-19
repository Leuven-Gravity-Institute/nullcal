"""Every third-party package ``src/nullcal`` imports must be declared in ``pyproject.toml``.

``pyyaml`` was imported by ``metadata/`` while appearing nowhere in the dependency list. It was
previously supplied by unrelated transitive dependencies that nullcal did not control.

CI cannot catch this on its own. The ``lowest-direct`` job lowers *declared* floors, and an
undeclared dependency has no floor to lower; a transitively-satisfied import looks identical to a
declared one at runtime. So the check has to be static, and this is it.

Three limitations, stated so this is not read as "no undeclared import can exist".

**Dynamic imports escape it.** ``importlib.import_module("pandas")`` and ``__import__("pandas")``
are ``ast.Call`` nodes, not ``ast.Import``/``ast.ImportFrom``, so the walk never sees them. None
exist in ``src/nullcal`` today, and any *static* addition is caught the moment it is committed, but
a future dynamic import would pass unnoticed.

**Only ``[project.dependencies]`` is read**, not ``[project.optional-dependencies]``. An import
satisfied solely by an extra would be reported as undeclared — a false positive, so it fails loudly
and in the safe direction, but it needs handling before any extra becomes importable from ``src/``.

**``rocket-fft`` is invisible by nature**: it is never imported at all. It entered the dependency
set as numba's FFT entry-point plugin and remains declared during the staged backend migration.
No static import check can decide when that transitional dependency is safe to remove.

The import check is deliberately one-directional. A separate packaging regression below pins the
reviewed removal of dependencies that have no package imports, entry points, or retained scripts.
"""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "nullcal"
PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

#: Import name -> distribution name, where the two differ. Only mismatches belong here: anything
#: absent is assumed to install under the name it imports as.
#:
#: Kept minimal on purpose. A first draft of this map also listed lal, lalsimulation,
#: configargparse and tables — none of which ``src/nullcal`` imports at all, which is precisely
#: why ``test_the_import_to_distribution_map_has_no_stale_entries`` exists and is what it caught.
IMPORT_TO_DISTRIBUTION = {
    "yaml": "pyyaml",
}

REMOVED_UNUSED_DISTRIBUTIONS = {
    "bilby-pipe",
    "configargparse",
    "healpy",
    "nptyping",
    "pycbc",
    "pyspark",
    "tables",
}


def _declared_distributions() -> set[str]:
    data = tomllib.loads(PYPROJECT.read_text())
    requirements = data["project"]["dependencies"]
    names = set()
    for requirement in requirements:
        # "package>=1.2.3" / "package[extra]>=1" / "package"
        name = requirement.split(";")[0].strip()
        for separator in (">=", "==", "<=", "~=", ">", "<", "!=", "["):
            name = name.split(separator)[0]
        names.add(name.strip().lower().replace("_", "-"))
    return names


def _all_declared_distributions() -> set[str]:
    """Return runtime and optional distributions from published package metadata."""
    data = tomllib.loads(PYPROJECT.read_text())
    requirements = list(data["project"]["dependencies"])
    for extra_requirements in data["project"].get("optional-dependencies", {}).values():
        requirements.extend(extra_requirements)

    names = set()
    for requirement in requirements:
        name = requirement.split(";")[0].strip()
        for separator in (">=", "==", "<=", "~=", ">", "<", "!=", "["):
            name = name.split(separator)[0]
        names.add(name.strip().lower().replace("_", "-"))
    return names


def _top_level_imports() -> set[str]:
    modules: set[str] = set()
    for path in SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            # node.level > 0 is a relative import: our own package, not a dependency.
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    return modules


def test_every_third_party_import_is_declared():
    """An import that resolves only through another package's dependencies is a latent break."""
    declared = _declared_distributions()
    stdlib = sys.stdlib_module_names

    undeclared = set()
    for module in sorted(_top_level_imports()):
        if module in stdlib or module == "nullcal":
            continue
        distribution = IMPORT_TO_DISTRIBUTION.get(module, module).lower().replace("_", "-")
        if distribution not in declared:
            undeclared.add(f"{module} (installs as {distribution})")

    assert not undeclared, (
        f"imported by src/nullcal but not declared in pyproject.toml: {sorted(undeclared)}. "
        "These are currently satisfied only by transitive requirements outside nullcal's control; "
        "upstream dependency changes can make the imports unavailable."
    )


def test_the_import_to_distribution_map_has_no_stale_entries():
    """Guard the guard: a mapping entry for a module nobody imports hides a rename.

    Without this, deleting the last ``import yaml`` would leave the mapping asserting a
    relationship that no longer exists, and the next person to add a ``yaml`` import would inherit
    an unverified claim.
    """
    imported = _top_level_imports()
    stale = {module for module in IMPORT_TO_DISTRIBUTION if module not in imported}
    assert not stale, f"IMPORT_TO_DISTRIBUTION maps modules that src/nullcal no longer imports: {sorted(stale)}"


def test_reviewed_unused_dependencies_stay_removed():
    """Dependencies removed after a history and entry-point audit must not drift back in."""
    unexpectedly_declared = REMOVED_UNUSED_DISTRIBUTIONS & _all_declared_distributions()
    assert not unexpectedly_declared, f"reviewed unused dependencies were reintroduced: {sorted(unexpectedly_declared)}"
