# Modal analysis

ANF solves the linear generalized eigenproblem

```text
K phi = lambda M phi,    lambda = omega^2
```

for schema 1.4 planar `truss2d` and `frame2d` models. The solver uses consistent
mass, removes fixed DOFs and calls the dense SciPy generalized eigensolver.
Install the `sparse` extra because the modal path uses SciPy:

```sh
python -m pip install ".[sparse]"
```

Modal analysis is a Python interface. The CLI and MCP do not expose it.

## Model

Schema 1.4 is a planar structure with a positive material `density` on every
member and a modal analysis block:

```json
{
  "schema_version": "1.4",
  "problem_type": "planar_structure",
  "units": {"length": "m", "force": "N", "stress": "Pa"},
  "model": {
    "nodes": {"a": [0, 0], "b": [2, 0]},
    "members": {
      "beam": {"kind": "frame2d", "i": "a", "j": "b", "elements": 8,
               "young_modulus": 210e9, "area": 0.01, "second_moment": 1e-6,
               "density": 7850}
    },
    "supports": {"a": {"ux": 0, "uy": 0, "rz": 0}}
  },
  "loads": {},
  "analysis": {"type": "modal", "modes": 6}
}
```

`loads` must be empty. Modal supports must be zero displacements; a nonzero
support is rejected with `unsupported_constraint`. The `modes` field defaults to
6 and is limited to 20. The mesh limits remain 200 nodes, 400 segments and 600
DOFs.

## Solve

```python
from pathlib import Path
from anf.application import run_modal
from anf.dynamics import assemble_mass, solve_modes

model = Path("modal-cantilever.json").read_bytes()
result = run_modal(model, {"count": 3})
print(result.frequencies_hz)
print(result.dof_labels)

mass = assemble_mass(model)
print(mass.total_mass, mass.matrix.shape)
```

`run_modal(model, options)` validates a schema 1.4 declaration and calls
`solve_modes(model, count=options["count"])`. `solve_modes` accepts a mapping or
JSON string/bytes and returns a `ModalResult`.

## Result

`ModalResult` contains:

| Field | Meaning |
| --- | --- |
| `eigenvalues` | `lambda = omega^2`, ascending |
| `omega_rad_s` | Angular frequencies, rad/s |
| `frequencies_hz` | Ordinary frequencies, Hz |
| `modes` | Mass-normalized mode shapes, one column per mode; rows follow `dof_labels` |
| `dof_labels` | Node and component labels, including generated nodes |
| `residuals` | Per-mode relative eigenpair residuals |
| `rigid_mode_count` | Number of eigenvalues inside the rigid-mode tolerance |
| `truncated` | True when the requested count exceeds the available modes |
| `diagnostics` | Structured codes such as `modal_truncated` |
| `versions` | Package, Python, NumPy and SciPy versions |
| `input_hash` | SHA-256 of the normalized physical model |

Each mode has `phi^T M phi = 1`. The sign of a mode shape is arbitrary; the
solver makes the largest-magnitude component positive only for reproducible
display. Repeated frequencies are meaningful as a subspace, not as individual
vectors.

A free planar model has three rigid modes: two translations and one rotation.
A supported model must not have a zero-energy mode. Modes are never padded with
zero frequencies; requesting more modes than the free DOF count returns the
available modes and sets `truncated`.

## Artifacts

```python
from anf.dynamics import load_modal_result, save_modal_result

directory = save_modal_result(result, Path.cwd(), "modal-result")
restored = load_modal_result(directory)
assert restored.input_hash == result.input_hash
```

A modal artifact contains `manifest.json` and `arrays.npz`. The manifest records
the normalized model, mode count, rigid-mode count, diagnostics, DOF labels,
array shapes and the SHA-256 of the array file. Loading rejects a mismatched
digest, unsupported schema, unexpected files, non-finite values, missing arrays
or a changed input hash.

## Verification

The modal tests use independent analytic anchors:

- fixed-free axial rod: `omega_n = (2n-1) pi / (2L) * sqrt(E/rho)`;
- simply supported Euler–Bernoulli beam: `omega_n = (n pi / L)^2 * sqrt(EI/(rho A))`;
- cantilever Euler–Bernoulli beam: `omega_n = (beta_n L)^2 * sqrt(EI/(rho A L^4))`;
- a free planar frame has three rigid modes and mass-normalized, orthogonal
  modes;
- renumbering nodes does not change frequencies;
- a requested count beyond the available modes is truncated, not zero-filled.

The solver rejects negative eigenvalues, rigid-mode mismatches, zero-energy
modes in supported models and eigenpairs above the residual gate. A base
installation without SciPy reports `backend_unavailable`; it can still run the
static beam, plate and structure workflows.

## Limits

The first modal release is linear, small-displacement, planar and dense. It does
not include shear deformation, independent rotary inertia, 3D frames, plates,
shells, solids, laminates, damping, time integration, buckling or nonlinear
analysis. The 600-DOF cap keeps the dense eigensolver practical; a sparse path
and plate mass matrices are separate designs.