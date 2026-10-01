# D1 experiment preparation

`preregistration.json` and `preregistration.md` fix the absorption criteria.
They do not contain measured absorption, calibration posteriors or spectroscopy
results. No D1 inference runner is implemented here.

`scripts/d1_control_preflight.py` checks whether the prerequisite artifact
bundle has been supplied, and checks the hashes and provenance in its JSON
records. It returns exit 2 when an input is missing or malformed. Exit 0 means
**inputs present**, not that the physics has been validated or either fit run.

```sh
uv run python scripts/d1_control_preflight.py \
  --inputs /path/to/d1-input-bundle \
  --output /path/to/prerequisite-audit.json
```

The required files are enumerated in the registration. Each JSON evidence record
must be an object containing:

- `producing_commit`: a full lowercase 40-character Git SHA;
- `dirty`: explicitly `false`;
- `command`: the actual producing command;
- `versions`: a nonempty mapping of dependency versions;
- `artifacts`: a nonempty mapping from bundle-relative output paths to SHA-256
  hashes. Absolute paths outside the bundle and escaping symlinks are rejected.

`waveform-manifest.json` must additionally identify `waveform` as `SXS:BBH:0305`
and link `waveform.h5` by hash. Other records archive the spectroscopy anchor
comparison, GR IMR source-model validation, exact injection and source-prior
prescription, and resolution pre-check. Their numerical content still requires
scientific review: the checker deliberately makes no claim about it. Test
fixtures are synthetic and cannot serve as these inputs.

Before inference, freeze and hash a single data bundle containing strain, PSD,
noise, response, masks and metadata. Every arm must reference the same hash.
Producing records must preserve the source versions and clean commit of the fit
implementation, the registration commit/hash, random seeds, environment and
scheduler job identifier. Numerical claims require actual posterior artifacts
and diagnostics; a prerequisite audit supplies none of them.

The current preparation must be followed by the selected SXS injection
construction, benchmark reproduction, conventional joint source/calibration
runner, shared-population nullcal runner, inspiral-only runner, and calibrated
`ringdown` inference with uncertainty propagation. All those fits and the
population measurement remain to be executed. Their absence must not be reported
as a negative absorption result.
