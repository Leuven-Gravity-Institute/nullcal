"""Measure the GPU cost per posterior under vmap over realisations.

Two subcommands:

``measure`` runs one backend arm (``gpu`` or ``cpu``) for one configuration and
writes ``<configuration>-<platform>.json``. The GPU arm asserts a GPU device is
present and exits non-zero otherwise, so a CPU-only node can never emit
numbers labelled GPU. The CPU arm is the same-day wall-clock baseline and must
run with ``JAX_PLATFORMS=cpu`` on the same node.

``merge`` reads the arms, computes the decomposed speedups and the campaign
pricing, and writes ``ledger.json`` plus ``ledger.md``.

See ``docs/dev/gpu-cost-ledger.md`` for how the numbers are produced and which
of them are anchored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import blackjax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from nullcal.studies import gpu_cost  # noqa: E402
from nullcal.studies.spline_resolution import augmented_log_knots  # noqa: E402

REFERENCE = "reference"
PRODUCTION = "production"
CONFIGURATIONS = (REFERENCE, PRODUCTION)
M3_PEAK_FREQUENCY_HZ = 249.43
M3_KNOT_HALF_WIDTH_HZ = 50.0
# A surrogate target that drives NUTS to near maximum tree depth makes the
# per-posterior cost unrepresentative, so flag it rather than pricing from it.
PATHOLOGICAL_STEPS_PER_SAMPLE = 30.0
# A cross-arm posterior speedup is only a hardware ratio when both arms worked
# through a comparable NUTS trajectory; beyond this factor of steps per sample
# the ratio is dominated by the surrogate target, not the device.
SPEEDUP_EQUIVALENCE_TOLERANCE = 5.0
LEDGER_BASENAME = "gpu-cost-ledger"
ARMS_DIRECTORY = "gpu-cost-ledger-arms"


def _git(arguments: list[str]) -> str:
    return subprocess.check_output(["git", *arguments], text=True).strip()  # noqa: S603, S607


def _source_is_dirty() -> bool:
    return bool(_git(["status", "--porcelain", "--", "src/nullcal", "scripts/gpu_cost_ledger.py", "pyproject.toml"]))


def _m3_knots() -> np.ndarray:
    """Return the 19-knot basis the M2 resolution study fixed for the campaign."""
    return augmented_log_knots(
        minimum_frequency=8.0,
        maximum_frequency=2048.0,
        broadband_count=10,
        peak_frequency=M3_PEAK_FREQUENCY_HZ,
        local_half_width=M3_KNOT_HALF_WIDTH_HZ,
        local_count=9,
    )


def build_configuration(name: str):
    """Return the likelihood under test for a named configuration."""
    if name == REFERENCE:
        from tests.e2e import pipeline  # noqa: PLC0415 - only the reference arm needs the frozen harness

        return pipeline.build_likelihood_from_reference_inputs()
    if name == PRODUCTION:
        return gpu_cost.build_synthetic_likelihood(
            duration=8.0,
            sampling_frequency=4096,
            minimum_frequency=20.0,
            maximum_frequency=2000.0,
            knot_frequencies=_m3_knots(),
        )
    raise ValueError(f"unknown configuration {name!r}")


def _model_record(likelihood) -> dict[str, object]:
    interferometers = likelihood.interferometers
    return {
        "detector_count": int(likelihood.parameter_shape[0]),
        "knot_count": int(likelihood.parameter_shape[1]),
        "frequency_count": int(likelihood._frequency_count),
        "selected_frequency_count": int(likelihood._frequency_indices.size),
        "duration_seconds": float(interferometers.duration),
        "sampling_frequency_hz": float(interferometers.sampling_frequency),
        "minimum_frequency_hz": float(np.min(np.asarray(likelihood._masked_frequencies))),
        "maximum_frequency_hz": float(np.max(np.asarray(likelihood._masked_frequencies))),
        "wavelet_shape": [int(value) for value in likelihood.time_frequency_transform.shape],
        "selected_time_frequency_pixels": int(np.asarray(likelihood.time_frequency_filter).sum()),
        "knot_frequencies_hz": [float(value) for value in np.asarray(likelihood.knot_frequencies)[0]],
        "strain_realisations_are_synthetic": True,
    }


def _device_record() -> list[dict[str, object]]:
    return [
        {
            "id": int(device.id),
            "platform": str(device.platform),
            "device_kind": str(device.device_kind),
        }
        for device in jax.devices()
    ]


def _slurm_record() -> dict[str, str]:
    keys = {
        "job_id": "SLURM_JOB_ID",
        "job_name": "SLURM_JOB_NAME",
        "partition": "SLURM_JOB_PARTITION",
        "node_list": "SLURM_JOB_NODELIST",
    }
    return {name: os.environ[key] for name, key in keys.items() if key in os.environ}


def measure(arguments: argparse.Namespace) -> None:
    if arguments.platform == gpu_cost.GPU_PLATFORM:
        gpu_cost.require_gpu_devices()
    elif gpu_cost.devices_with_platform(gpu_cost.GPU_PLATFORM):
        raise RuntimeError("the CPU arm sees a GPU device; run it with JAX_PLATFORMS=cpu on the same node")

    likelihood = build_configuration(arguments.configuration)
    initial_position = {
        "amplitude": jnp.zeros(likelihood.parameter_shape, dtype=jnp.float64),
        "phase": jnp.zeros(likelihood.parameter_shape, dtype=jnp.float64),
    }
    stages = gpu_cost.measure_stages(likelihood, repeats=arguments.repeats, seed=arguments.seed)
    detector_count, frequency_count = likelihood.parameter_shape[0], likelihood._frequency_count
    realisations = gpu_cost.synthetic_realisations(
        (detector_count, frequency_count), arguments.realisations, seed=arguments.seed
    )

    if arguments.platform == gpu_cost.GPU_PLATFORM:
        posterior = gpu_cost.measure_batched_posterior(
            likelihood,
            initial_position,
            realisations,
            seed=arguments.seed,
            num_chains_per_realisation=arguments.chains,
            num_warmup=arguments.warmup,
            num_samples=arguments.samples,
        )
        posterior["measurement"] = "vmap-over-realisations"
    else:
        posterior = gpu_cost.measure_posterior(
            likelihood,
            initial_position,
            realisations,
            seed=arguments.seed,
            num_chains=arguments.chains,
            num_warmup=arguments.warmup,
            num_samples=arguments.samples,
            realisation_count=arguments.cpu_realisations,
        )
        posterior["measurement"] = "single-realisation-mean"

    record = {
        "configuration": arguments.configuration,
        "platform": arguments.platform,
        "created_at": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "slurm": _slurm_record(),
        "devices": _device_record(),
        "code": {
            "repository": "https://github.com/Leuven-Gravity-Institute/nullcal",
            "commit": _git(["rev-parse", "HEAD"]),
            "dirty": _source_is_dirty(),
            "script": "scripts/gpu_cost_ledger.py",
        },
        "versions": {
            "jax": jax.__version__,
            "blackjax": blackjax.__version__,
            "numpy": np.__version__,
        },
        "model": _model_record(likelihood),
        "sampler_settings": {
            "num_chains_per_realisation": arguments.chains,
            "num_warmup": arguments.warmup,
            "num_samples": arguments.samples,
            "seed": arguments.seed,
            "timing_repeats": arguments.repeats,
        },
        "stages": stages,
        "posterior": posterior,
    }
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    output_path = arguments.output_directory / f"{arguments.configuration}-{arguments.platform}.json"
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")


def _load_arms(input_directory: Path) -> dict[str, dict[str, dict]]:
    arms: dict[str, dict[str, dict]] = {}
    for configuration in CONFIGURATIONS:
        arms[configuration] = {}
        for platform in (gpu_cost.GPU_PLATFORM, gpu_cost.CPU_PLATFORM):
            path = input_directory / f"{configuration}-{platform}.json"
            if path.exists():
                arms[configuration][platform] = json.loads(path.read_text())
    return arms


def steps_per_sample(posterior: dict) -> float:
    """Return the mean NUTS integration steps per sampling step per chain."""
    samples = posterior["num_realisations"] * posterior["chains"] * posterior["num_samples_per_chain"]
    if samples <= 0:
        raise ValueError("sampler settings must describe at least one chain sample")
    return posterior["integration_steps"] / samples


def _ledger_created_at(arms: dict[str, dict[str, dict]]) -> str:
    """Return a deterministic ledger timestamp read from the arm records.

    Using the newest arm timestamp instead of the merge clock keeps a rebuild
    byte-identical from the committed inputs.
    """
    stamps = [arm["created_at"] for configuration in arms.values() for arm in configuration.values()]
    return max(stamps) if stamps else "unavailable"


def _posterior_comparison(accelerator_arm: dict, cpu_arm: dict) -> dict[str, object]:
    """Report whether the cross-arm posterior speedup is a hardware ratio."""
    accelerator = steps_per_sample(accelerator_arm["posterior"])
    cpu = steps_per_sample(cpu_arm["posterior"])
    ratio = max(accelerator, cpu) / min(accelerator, cpu)
    equivalent = ratio <= SPEEDUP_EQUIVALENCE_TOLERANCE
    note = (
        "same realisation ensemble and a comparable NUTS trajectory length"
        if equivalent
        else (
            f"not a hardware ratio: the accelerator averaged {accelerator:.2f} and the CPU {cpu:.2f} integration "
            "steps per sample, so the ratio is dominated by the surrogate target's trajectory length"
        )
    )
    return {
        "steps_per_sample": {"accelerator": accelerator, "cpu": cpu},
        "speedup_equivalent": equivalent,
        "note": note,
    }


def _scaled_m3_estimate(ledger: dict) -> dict[str, object] | None:
    """Scale the verified reference posterior to the M2 production model.

    The factor is the production/reference ratio of single-evaluation
    likelihood-gradient wall times, which is independent of the NUTS
    integration-step count; the campaign is priced from the scaled value, not
    from the pathological production posterior.
    """
    reference = ledger["configurations"].get(REFERENCE, {})
    production = ledger["configurations"].get(PRODUCTION, {})
    if not (reference.get("complete") and production.get("complete")):
        return None
    reference_seconds = reference["posterior"]["accelerator"]["seconds_per_posterior"]
    reference_gradient = reference["stages"]["likelihood_gradient"]["accelerator_seconds"]
    production_gradient = production["stages"]["likelihood_gradient"]["accelerator_seconds"]
    ratio = production_gradient / reference_gradient
    seconds = reference_seconds * ratio
    return {
        "seconds_per_posterior": seconds,
        "derivation": (
            "reference accelerator seconds_per_posterior x (production accelerator likelihood_gradient seconds / "
            "reference accelerator likelihood_gradient seconds)"
        ),
        "inputs": {
            "reference_accelerator_seconds_per_posterior": reference_seconds,
            "reference_accelerator_likelihood_gradient_seconds": reference_gradient,
            "production_accelerator_likelihood_gradient_seconds": production_gradient,
            "source": f"docs/dev/{ARMS_DIRECTORY}/{{reference,production}}-gpu.json",
        },
        "production_over_reference_likelihood_gradient_ratio": ratio,
        "campaign_gpu_hours": {
            "one_loud_event": gpu_cost.campaign_gpu_hours(seconds, 1, 1, 1),
            "population_100x10x10": gpu_cost.campaign_gpu_hours(seconds, 100, 10, 10),
            "population_1000x10x10": gpu_cost.campaign_gpu_hours(seconds, 1000, 10, 10),
        },
        "assumptions": [
            (
                "the production/reference ratio is a single-evaluation wall-time ratio, independent of the NUTS "
                "integration-step count"
            ),
            "it assumes an M3 posterior whose NUTS trajectory length is reference-like; that is not measured here",
            (
                "the production gradient is measured with the all-ones filter (32768 pixels versus 382 on the "
                "reference), so the scaled value is an upper bound for a clustered M3 analysis"
            ),
        ],
    }


def _per_posterior_decomposition(record: dict) -> dict[str, float]:
    posterior = record["posterior"]
    gradient_seconds = record["stages"]["likelihood_gradient"]["median_seconds"]
    posterior_count = int(posterior.get("num_realisations", 1))
    # Sampling-phase integration steps are measured; one warmup step per warmup
    # iteration per chain is a lower bound because the adaptation schedule is
    # not exposed. The gradient cost is a single (unbatched) evaluation, so the
    # accelerator likelihood share is an upper bound and the sampler overhead a
    # lower bound.
    gradient_evaluations = (
        posterior["integration_steps"] / posterior_count + posterior["num_warmup"] * posterior["chains"]
    )
    return gpu_cost.decompose_per_posterior(posterior["seconds_per_posterior"], gradient_seconds, gradient_evaluations)


def merge(arguments: argparse.Namespace) -> None:
    input_directory = arguments.input_directory or arguments.output_directory
    arms = _load_arms(input_directory)
    ledger: dict[str, object] = {
        "created_at": _ledger_created_at(arms),
        "configurations": {},
        "campaign_pricing": {},
        "scaled_m3_estimate": None,
        "anchors": {},
        "unanchored": [],
    }
    for configuration in CONFIGURATIONS:
        available = arms[configuration]
        if gpu_cost.GPU_PLATFORM not in available or gpu_cost.CPU_PLATFORM not in available:
            ledger["configurations"][configuration] = {"arms": sorted(available), "complete": False}
            continue
        gpu_arm = available[gpu_cost.GPU_PLATFORM]
        cpu_arm = available[gpu_cost.CPU_PLATFORM]
        stage_speedups = {
            stage: gpu_cost.speedup(
                cpu_arm["stages"][stage]["median_seconds"], gpu_arm["stages"][stage]["median_seconds"]
            )
            for stage in gpu_arm["stages"]
        }
        posterior_speedup = gpu_cost.speedup(
            cpu_arm["posterior"]["seconds_per_posterior"], gpu_arm["posterior"]["seconds_per_posterior"]
        )
        comparison = _posterior_comparison(gpu_arm, cpu_arm)
        ledger["configurations"][configuration] = {
            "complete": True,
            "model": gpu_arm["model"],
            "accelerator": gpu_arm["devices"],
            "cpu_devices": cpu_arm["devices"],
            "code": gpu_arm["code"],
            "cpu_code": cpu_arm["code"],
            "slurm": {"gpu": gpu_arm.get("slurm", {}), "cpu": cpu_arm.get("slurm", {})},
            "sampler_settings": gpu_arm["sampler_settings"],
            "stages": {
                stage: {
                    "cpu_seconds": cpu_arm["stages"][stage]["median_seconds"],
                    "accelerator_seconds": gpu_arm["stages"][stage]["median_seconds"],
                    "speedup": stage_speedups[stage],
                }
                for stage in gpu_arm["stages"]
            },
            "posterior": {
                "cpu": cpu_arm["posterior"],
                "accelerator": gpu_arm["posterior"],
                "speedup": posterior_speedup,
                "speedup_equivalent": comparison["speedup_equivalent"],
                "speedup_note": comparison["note"],
                "steps_per_sample": comparison["steps_per_sample"],
                "cpu_decomposition_seconds": _per_posterior_decomposition(cpu_arm),
                "accelerator_decomposition_seconds": _per_posterior_decomposition(gpu_arm),
            },
        }
        pathological = steps_per_sample(gpu_arm["posterior"]) > PATHOLOGICAL_STEPS_PER_SAMPLE
        seconds_per_posterior = gpu_arm["posterior"]["seconds_per_posterior"]
        ledger["campaign_pricing"][configuration] = {
            "seconds_per_posterior": seconds_per_posterior,
            "used_for_campaign_pricing": not pathological,
            "marker": (
                "pathological surrogate upper bound (near maximum NUTS tree depth); NOT an M3 price"
                if pathological
                else "used for campaign pricing"
            ),
            "one_loud_event": gpu_cost.campaign_gpu_hours(seconds_per_posterior, 1, 1, 1),
            "population_100x10x10": gpu_cost.campaign_gpu_hours(seconds_per_posterior, 100, 10, 10),
            "population_1000x10x10": gpu_cost.campaign_gpu_hours(seconds_per_posterior, 1000, 10, 10),
            "note": "event/noise/calibration counts are illustrative; M2 fixed a population but not its size",
        }

    reference = ledger["configurations"].get(REFERENCE, {})
    production = ledger["configurations"].get(PRODUCTION, {})
    if reference.get("complete"):
        ledger["anchors"]["reference_log_likelihood"] = _reference_log_likelihood()
        ledger["anchors"]["reference_configuration"] = "frozen e2e inputs, 8 s / 2048 Hz / 20-1024 Hz / 10 knots"
    if production.get("complete"):
        ledger["unanchored"].append(
            "the timed realisations are synthetic same-shape unit-variance complex arrays, not the M3 "
            "noise-and-calibration realisations; a surrogate does not reproduce the sampler's trajectory "
            "length, so the per-posterior cost is a shape-anchored estimate rather than an M3 prediction"
        )
        ledger["unanchored"].append(
            "the production configuration selects every time-frequency pixel (all-ones filter), an upper "
            "bound on the cost of a clustered M3 analysis, which keeps only a sparse pixel set"
        )
    for configuration, record in ledger["configurations"].items():
        if not record.get("complete"):
            continue
        if steps_per_sample(record["posterior"]["accelerator"]) > PATHOLOGICAL_STEPS_PER_SAMPLE:
            ledger["unanchored"].append(
                f"the {configuration} accelerator posterior ran near maximum NUTS tree depth on the synthetic "
                "target, so its per-posterior cost is a pathological surrogate upper bound and is not used for "
                "campaign pricing; the scaled M3 estimate uses the target-independent single-evaluation ratio"
            )
    ledger["unanchored"].append(
        "the CPU baseline is one realisation draw from the same ensemble, while the accelerator arm averages over its "
        "batch; the CPU arm's own integration-step count is recorded so the trajectory lengths can be compared"
    )
    ledger["unanchored"].append("warmup gradient count is a lower bound: one gradient per warmup iteration")
    ledger["unanchored"].append(
        "the per-posterior split uses the single-evaluation likelihood-gradient cost, not the batched one, so the "
        "accelerator likelihood share is an upper bound and its sampler overhead a lower bound"
    )
    ledger["unanchored"].append(
        "the campaign size is not fixed by M2 beyond 'a population'; the pricing table is per-posterior cost times "
        "an illustrative grid"
    )
    ledger["scaled_m3_estimate"] = _scaled_m3_estimate(ledger)

    json_path = arguments.output_directory / f"{arguments.basename}.json"
    markdown_path = arguments.output_directory / f"{arguments.basename}.md"
    json_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    _write_markdown(markdown_path, ledger)
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")


def _reference_log_likelihood() -> float:
    from tests.e2e import pipeline  # noqa: PLC0415

    likelihood = pipeline.build_likelihood_from_reference_inputs()
    return float(likelihood.log_likelihood_fn(pipeline.parameter_arrays()))


def _markdown_provenance(ledger: dict) -> list[str]:
    intro = (
        "Per-posterior cost is the campaign budget. The accelerator arm runs one NUTS chain per "
        "realisation inside a single `vmap`; the CPU arm is the same-day baseline sampling the same "
        "realisation ensemble, so the two arms target the same posterior family."
    )
    method = (
        "`scripts/gpu_cost_ledger.py measure` writes one `<configuration>-<platform>.json` per arm; "
        "`merge` combines them and computes the speedups and the per-posterior split. The accelerator "
        'arm asserts `any(device.platform == "gpu" for device in jax.devices())` and exits non-zero '
        "otherwise, so a CPU-only node cannot emit GPU-labelled numbers. Every timing call regenerates "
        "its input array, so a device-cached value cannot be mistaken for a fresh transfer. Run the CPU "
        "arm with `JAX_PLATFORMS=cpu`."
    )
    lines = [
        "# GPU cost ledger — vmap over realisations",
        "",
        intro,
        "",
        "## How these numbers were produced",
        "",
        method,
        "",
        "| field | value |",
        "| --- | --- |",
        f"| ledger created | {ledger['created_at']} |",
    ]
    for configuration, record in ledger["configurations"].items():
        if not record.get("complete"):
            continue
        accelerator = ", ".join(
            f"{device['device_kind']} ({device['platform']})" for device in record.get("accelerator", [])
        )
        cpu = ", ".join(str(device["device_kind"]) for device in record.get("cpu_devices", []))
        lines.append(f"| {configuration} accelerator | {accelerator} |")
        lines.append(f"| {configuration} CPU | {cpu} |")
        lines.append(f"| {configuration} committed revision | `{record['code']['commit']}` |")
    lines.append("")
    return lines


def _markdown_configuration(configuration: str, record: dict) -> list[str]:
    lines = [f"## {configuration}", ""]
    if not record.get("complete"):
        return [*lines, f"Incomplete: arms present = {record.get('arms')}", ""]
    model = record["model"]
    lines.append(
        f"Model: {model['detector_count']} detectors, {model['selected_frequency_count']} selected bins "
        f"over {model['minimum_frequency_hz']:.0f}-{model['maximum_frequency_hz']:.0f} Hz, "
        f"{model['knot_count']} knots, transform {model['wavelet_shape']}."
    )
    lines += ["", "| stage | CPU (s) | accelerator (s) | speedup |", "| --- | ---: | ---: | ---: |"]
    for stage, values in record["stages"].items():
        lines.append(
            f"| {stage} | {values['cpu_seconds']:.6g} | {values['accelerator_seconds']:.6g} | "
            f"{values['speedup']:.2f}x |"
        )
    posterior = record["posterior"]
    summary = (
        f"Posterior: accelerator {posterior['accelerator']['seconds_per_posterior']:.6g} s/posterior "
        f"({posterior['accelerator']['measurement']}); CPU {posterior['cpu']['seconds_per_posterior']:.6g} "
        f"s/posterior; speedup {posterior['speedup']:.2f}x ({posterior['speedup_note']})."
    )
    lines += ["", summary, ""]
    for label, key in (("accelerator", "accelerator_decomposition_seconds"), ("CPU", "cpu_decomposition_seconds")):
        decomposition = posterior[key]
        split = (
            f"{label} split: likelihood {decomposition['likelihood_seconds']:.6g} s, "
            f"sampler overhead {decomposition['sampler_overhead_seconds']:.6g} s."
        )
        lines += [split, ""]
    return lines


def _markdown_rebuild() -> list[str]:
    commands = [
        "```bash",
        "uv run python scripts/gpu_cost_ledger.py merge \\",
        f"  --input-directory docs/dev/{ARMS_DIRECTORY} \\",
        f"  --output-directory docs/dev --basename {LEDGER_BASENAME}",
        f"npx --yes prettier@3.9.9 --write docs/dev/{LEDGER_BASENAME}.json docs/dev/{LEDGER_BASENAME}.md",
        "uv run python scripts/gpu_cost_ledger.py manifest \\",
        f"  --arms-directory docs/dev/{ARMS_DIRECTORY} \\",
        f"  --artifact docs/dev/{LEDGER_BASENAME}.json docs/dev/{LEDGER_BASENAME}.md \\",
        f"  --output docs/dev/{LEDGER_BASENAME}-manifest.json",
        f"npx --yes prettier@3.9.9 --write docs/dev/{LEDGER_BASENAME}-manifest.json",
        "```",
    ]
    intro = (
        f"The ledger is a deterministic function of the four arm records in `docs/dev/{ARMS_DIRECTORY}/` "
        "that are preserved in the checkout. `merge` derives `created_at` from the newest arm record, so no "
        "merge-clock value enters. Run:"
    )
    outro = (
        "A clean re-run leaves `git status --porcelain docs/dev` empty; the Prettier step applies the "
        "repository's `.prettierrc.yaml` to the generated prose."
    )
    return ["## Rebuild", "", intro, "", *commands, "", outro, ""]


def _markdown_scaled_estimate(ledger: dict) -> list[str]:
    estimate = ledger.get("scaled_m3_estimate")
    if not estimate:
        return []
    inputs = estimate["inputs"]
    hours = estimate["campaign_gpu_hours"]
    reference_seconds = inputs["reference_accelerator_seconds_per_posterior"]
    ratio_line = (
        f"= {reference_seconds:.10g} s"
        f" x ({inputs['production_accelerator_likelihood_gradient_seconds']:.10g} s"
        f" / {inputs['reference_accelerator_likelihood_gradient_seconds']:.10g} s)"
    )
    ratio_value = (
        f"= {reference_seconds:.10g} s x {estimate['production_over_reference_likelihood_gradient_ratio']:.10f}"
    )
    seconds_line = f"= {estimate['seconds_per_posterior']:.10g} s/posterior"
    scaled_row = (
        f"| scaled M3 | {hours['one_loud_event']['gpu_hours']:.4g} GPU-h "
        f"| {hours['population_100x10x10']['gpu_hours']:.4g} GPU-h "
        f"| {hours['population_1000x10x10']['gpu_hours']:.4g} GPU-h |"
    )
    prose = (
        "This is the M3 price. The production posterior above is a pathological surrogate and is **not** "
        "used for pricing. The estimate scales the verified reference per-posterior cost by the "
        "target-independent ratio of a single likelihood-gradient evaluation:"
    )
    lines = [
        "## M3 target-independent scaled estimate",
        "",
        prose,
        "",
        "```text",
        estimate["derivation"],
        ratio_line,
        ratio_value,
        seconds_line,
        "```",
        "",
        "| configuration | 1 event | 100x10x10 | 1000x10x10 |",
        "| --- | ---: | ---: | ---: |",
        scaled_row,
        "",
        "Assumptions:",
        "",
    ]
    lines.extend(f"- {item}" for item in estimate["assumptions"])
    lines += ["", f"Inputs: `{inputs['source']}`.", ""]
    return lines


def _write_markdown(path: Path, ledger: dict) -> None:
    lines = _markdown_provenance(ledger)
    for configuration, record in ledger["configurations"].items():
        lines += _markdown_configuration(configuration, record)
    pricing_prose = (
        "Per-posterior cost at the measured sampler settings. A row marked *not used* is a pathological "
        "surrogate and must not be read as campaign pricing; the M3 price is the target-independent "
        "scaled estimate below."
    )
    lines += [
        "## Campaign pricing",
        "",
        pricing_prose,
        "",
        "| configuration | 1 event | 100x10x10 | 1000x10x10 | used for pricing |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for configuration, pricing in ledger["campaign_pricing"].items():
        row = (
            f"| {configuration} | {pricing['one_loud_event']['gpu_hours']:.4g} GPU-h "
            f"| {pricing['population_100x10x10']['gpu_hours']:.4g} GPU-h "
            f"| {pricing['population_1000x10x10']['gpu_hours']:.4g} GPU-h "
            f"| {pricing['marker']} |"
        )
        lines.append(row)
    lines += ["", *_markdown_scaled_estimate(ledger), *_markdown_rebuild()]
    lines += ["## Anchors", ""]
    if ledger["anchors"]:
        lines.extend(f"- `{name}` = {value}" for name, value in sorted(ledger["anchors"].items()))
    else:
        lines.append("- none")
    lines += ["", "## Unanchored", ""]
    lines.extend(f"- {item}" for item in ledger["unanchored"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _digest(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def manifest(arguments: argparse.Namespace) -> None:
    """Record digests of the ledger inputs and of the committed artifacts.

    Nothing here is written back into the hashed artifacts, so a later checkout
    can rebuild them and byte-compare against these expected digests.
    """
    record = {
        "inputs": [_digest(path) for path in sorted(arguments.arms_directory.glob("*.json"))],
        "artifacts": [_digest(path) for path in arguments.artifact],
    }
    arguments.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"wrote {arguments.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--configuration", choices=CONFIGURATIONS, required=True)
    measure_parser.add_argument("--platform", choices=(gpu_cost.GPU_PLATFORM, gpu_cost.CPU_PLATFORM), required=True)
    measure_parser.add_argument("--output-directory", type=Path, required=True)
    measure_parser.add_argument("--realisations", type=int, default=32)
    measure_parser.add_argument("--cpu-realisations", type=int, default=1)
    measure_parser.add_argument("--chains", type=int, default=2)
    measure_parser.add_argument("--warmup", type=int, default=300)
    measure_parser.add_argument("--samples", type=int, default=300)
    measure_parser.add_argument("--repeats", type=int, default=5)
    measure_parser.add_argument("--seed", type=int, default=20260925)
    measure_parser.set_defaults(handler=measure)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--output-directory", type=Path, required=True)
    merge_parser.add_argument("--input-directory", type=Path, default=None)
    merge_parser.add_argument("--basename", default="ledger")
    merge_parser.set_defaults(handler=merge)
    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--arms-directory", type=Path, required=True)
    manifest_parser.add_argument("--artifact", type=Path, nargs="+", required=True)
    manifest_parser.add_argument("--output", type=Path, required=True)
    manifest_parser.set_defaults(handler=manifest)
    arguments = parser.parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
