# GPU cost ledger — vmap over realisations

Per-posterior cost is the campaign budget. The accelerator arm runs one NUTS
chain per realisation inside a single `vmap`; the CPU arm is the same-day
baseline sampling the same realisation ensemble, so the two arms target the same
posterior family.

## How these numbers were produced

`scripts/gpu_cost_ledger.py measure` writes one `<configuration>-<platform>.json`
per arm; `merge` combines them and computes the speedups and the per-posterior
split. The accelerator arm asserts

```python
assert any(device.platform == "gpu" for device in jax.devices())
```

and exits non-zero otherwise, so a CPU-only node cannot emit GPU-labelled
numbers. Every timing call regenerates its input array, so a device-cached value
cannot be mistaken for a fresh transfer. Run the CPU arm with `JAX_PLATFORMS=cpu`.

The measured ledger is written by the `merge` subcommand; the committed copy of
that output is the table below.
