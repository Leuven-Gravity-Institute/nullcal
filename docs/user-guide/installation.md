# Installation

## Requirements

- Python 3.12 or later
- Linux, macOS, or Windows

## From PyPI

Install the latest release:

```console
pip install nullcal
```

Install the explicit CPU or NVIDIA accelerator extra when selecting a JAX
backend:

```console
pip install "nullcal[jax]"       # CPU (Linux or macOS)
pip install "nullcal[cuda]"      # NVIDIA CUDA 13 (Linux)
```

## From Source

Clone and install with `uv`:

```console
git clone https://github.com/Leuven-Gravity-Institute/nullcal.git
cd nullcal
uv sync --extra jax
```

## Development Setup

Install development dependencies and pre-commit hooks:

```console
uv sync --group dev
uv run prek install
```

Verify the installation:

```console
python -c "import nullcal; print(nullcal.__version__)"
```

## Troubleshooting

- Ensure Python 3.12 or later is installed (`python --version`)
- Use a virtual environment to avoid dependency conflicts
- If you encounter issues, please open an issue on
  [GitHub](https://github.com/Leuven-Gravity-Institute/nullcal/issues)
