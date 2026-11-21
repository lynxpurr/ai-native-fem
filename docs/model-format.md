# Declarative models and result artifacts

The Python application API accepts schema 1.0 for static cantilever bending and
rectangular plate bending. [Model schema](../schemas/model-v1.json) ·
[Result schema](../schemas/result-v1.json)

## Run a model

```python
from pathlib import Path

from anf.application import run_analysis
from anf.artifacts import load_result, save_result
from anf.model import canonical_input_hash, validate_model

doc = {
    "schema_version": "1.0",
    "problem_type": "cantilever_bending",
    "units": {"length": "m", "force": "N", "stress": "Pa"},
    "model": {"length": 2, "young_modulus": 210e9, "second_moment": 1e-6},
    "loads": {"tip_force": 1000},
    "analysis": {"type": "static"},
}

model = validate_model(doc)
result = run_analysis(model, {"reference": "cantilever.closed-form.v1"})
directory = save_result(result, root=Path.cwd(), name="cantilever")
restored = load_result(directory)
assert restored.input_hash == canonical_input_hash(model)
print(restored.verification_status, restored.arrays["deflection"][-1])
```

`validate_model` accepts a mapping or a UTF-8 JSON string or bytes up to 1 MiB.
The returned `ModelSpec.to_dict()` includes the defaults used in the analysis.
Input errors expose `path` (a JSON Pointer), `code`, and `message` on
`ModelValidationError`. For example, `"length": "mm"` under `units` raises
`unsupported_unit` at `/units/length`. Use a fresh artifact name each time
you run the example.

All six top-level fields are required. Use the explicit SI units shown above
and `analysis.type = "static"`. Objects accept only declared fields, and JSON
keys must be unique. Numbers must be representable as finite floats.
Integer-valued mesh counts such as `8.0` normalize to integers. Each model is
self-contained data.

| Problem | Required model fields | Defaults and limits | Loads |
| --- | --- | --- | --- |
| `cantilever_bending` | `length` (m), `young_modulus` (Pa), `second_moment` (m⁴) | `elements=8`, range 1–200. Euler–Bernoulli Hermite beam with a fixed root. | `tip_force` (N) and `uniform_load` (N/m) each default to 0. |
| `rectangular_plate` | `lx`, `ly`, `thickness` (m), `young_modulus` (Pa), `poisson_ratio` (dimensionless, strictly between −1 and 0.5) | `nx=ny=8`, at least 2 each and `3*(nx+1)*(ny+1) <= 1800`. `shear_factor=5/6`, `support="clamped"`, `formulation="q4_full"`. | Required uniform `pressure` (Pa). |

Dimensions, modulus, second moment and shear factor must be positive.
`q4_full` also supports `hard_simply_supported`. Choose `mitc4_rectangle`
explicitly for an axis-aligned rectangle with `hard_simply_supported` edges.
See the [validation record](validation.md) for the tested plate domains and
the thin Q4 locking case.

## Interpret the outcome

`run_analysis` solves once. Omit `options` or set `reference` to null to
evaluate only the free-DOF residual. To compare against a reference, pass an ID
from `anf.verification.REFERENCE_REGISTRY`. The input SHA-256 matches the
existing verification report for the physical SI inputs. Key order and
equivalent default values share a fingerprint. The reference and execution
versions appear separately in the result.

| Solve status | Verification status | Meaning |
| --- | --- | --- |
| `solved` | `verification_passed` | The residual and applicable independent reference checks passed. |
| `solved` | `verification_failed` | An independent comparison failed. The result arrays and report remain available. |
| `solved` | `solved_unverified` | The residual passed. See report checks and warnings for reference availability and applicability. |
| `numerical_failure` | `not_run` | The solve or numerical gate failed. Inspect the diagnostic and report. |

The free-DOF relative residual limit is `1e-8`. See the [reference domains and
report fields](verification.md).

## Read or save a result directory

`save_result(result, root, name)` takes an existing caller-controlled directory
and a new portable name of up to 128 characters. Start the name with a letter,
digit, underscore or hyphen. Interior periods and ASCII letters, digits,
underscores and hyphens are allowed. The name must avoid reserved device
names. The writer creates the destination exclusively and publishes
`manifest.json` last. An interrupted destination can be inspected as a partial
write. Pick a fresh name for a retry. `load_result(directory)` raises
`ArtifactError` for incomplete, invalid or unsafe artifacts.

The manifest records the normalized model, input fingerprint, solve and
verification status, versions, reference ID, report and diagnostics.
Solved results include strain energy in J, the dimensionless residual,
array metadata and the SHA-256 of `arrays.npz`. Numerical failures contain
only `manifest.json` with empty array metadata and null result measurements.

| Model | Array | Shape | Unit and location |
| --- | --- | --- | --- |
| Beam | `x`, `deflection`, `slope` | `(elements+1,)` | m, m, rad at nodes |
| Beam | `root_force`, `root_moment` | `(1,)` each | N and N·m at the root |
| Plate | `x`, `y` | `(nx+1,)`, `(ny+1,)` | m along the respective coordinate axis |
| Plate | `deflection`, `beta_x`, `beta_y` | `(ny+1, nx+1)` | m, rad, rad at nodes |
| Plate | `reaction_force`, `reaction_moment_x_dof`, `reaction_moment_y_dof` | `(ny+1, nx+1)` | N, N·m, N·m, conjugate to `w`, `beta_x`, `beta_y` |

Plate grids use `[y, x]` indexing. The `beta` components follow the shear
convention `grad(w) - beta`. Reaction moments follow the associated generalized
DOFs. See the [method notes](methods.md) for signs and boundary conventions.

The reader checks model-derived array names, shapes, units and positions,
then compares member lengths and NPY headers before loading numeric data.
Arrays use C-order little-endian float64 in NPY 1.0 members of a single-volume
ZIP, with stored or deflated data and complete local headers. The manifest
limit is 1 MiB, the archive and expanded data limits are 64 MiB each, and
the reader allows at most 32 members, 1,000,000 elements per array and
10 KiB per NPY header. Members stay in memory and the loader disables pickle.
Source distributions include the JSON schemas. Installed wheels provide the
same validation and analysis through the Python modules.
