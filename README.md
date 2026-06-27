# nullcal

[![PyPI version](https://badge.fury.io/py/nullcal.svg)](https://pypi.org/project/nullcal/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Build](https://gitlab.kuleuven.be/gwc/software/nullcal/badges/main/pipeline.svg)](https://gitlab.kuleuven.be/gwc/software/nullcal/-/pipelines)
[![codecov](https://codecov.io/gh/username/package_name/branch/main/graph/badge.svg)](https://codecov.io/gh/username/package_name)
[![Python Version](https://img.shields.io/pypi/pyversions/nullcal)](https://pypi.org/project/nullcal/)
[![Security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Documentation Status](https://img.shields.io/badge/documentation-online-brightgreen)](https://groupname.gitlab.io/package_name/)
[![DOI](https://zenodo.org/badge/ID.svg)](https://doi.org/DOI)

## Pre-commit hooks

Set up Git hook scripts to automatically check files for issues before each
commit.

Install `pre-commit`:

```console
pip install pre-commit
```

Install the git hook scripts:

```console
pre-commit install
```

(Optional) it's usually a good idea to run the hooks against all of the files
when adding new hooks (usually pre-commit will only run on the changed files
during git hooks)

```console
pre-commit run --all-files
```
