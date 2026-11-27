# AI-Native FEM (ANF)

ANF is a Python research package for linear finite element analysis of
Euler–Bernoulli cantilevers, rectangular Reissner–Mindlin plates, and planar
trusses and frames, with consistent-mass modal, harmonic and transient dynamics
and linearized elastic buckling for the planar models. The `v0.1.0a8` research
prerelease adds schema 1.6 elastic buckling with a declared nodal preload.
[GitHub prerelease](https://github.com/lynxpurr/ai-native-fem/releases/tag/v0.1.0a8) ·
[Release notes](docs/releases/0.1.0a8.md)

## Install

Create and activate a virtual environment with Python 3.11 or newer.
Download `ai_native_fem-0.1.0a8-py3-none-any.whl` from the prerelease and
run this command from the directory containing the wheel.

```sh
python -m pip install "./ai_native_fem-0.1.0a8-py3-none-any.whl"
```

The base installation provides the Python API and the `anf` command. Pip
installs NumPy as a dependency.

## Solve a cantilever

### Python

```python
from anf import Beam, solve_cantilever

model = Beam(length=2, young_modulus=210e9, second_moment=1e-6)
result = solve_cantilever(model, tip_force=1000)
print(result.deflection[-1])  # approximately 0.0126984 m
```

### Command line

Get the example inputs from the matching source distribution on the release
page or the `v0.1.0a8` tag. Run these commands from the directory containing
`examples/`. Choose a fresh output directory each time you solve.
[Tagged cantilever input](https://github.com/lynxpurr/ai-native-fem/blob/v0.1.0a8/examples/models/cantilever.json)

```sh
anf validate examples/models/cantilever.json --json
anf solve examples/models/cantilever.json --reference cantilever.closed-form.v1 --output beam-result --json
anf inspect beam-result --json
```

Expect `verification_passed` for the analytical comparison with the bundled
cantilever reference. The result directory contains a JSON manifest with the
input fingerprint and verification report, plus the numeric arrays in
`arrays.npz`. [CLI guide](docs/cli.md) · [Model and result format](docs/model-format.md)

## Solve a planar frame

Schema 1.1 describes planar truss and frame models. This example uses the
bundled portal frame.

```python
from pathlib import Path
from anf.application import run_analysis

result = run_analysis(Path("examples/structures/portal.json").read_bytes())
print(result.solve_status, result.verification_status)
print(result.arrays["node_ux"])
```

`anf validate`, `anf solve` and `anf inspect` accept the same schema 1.1
declaration. Named load cases, combinations, end releases and rotational
springs use schema 1.2/1.3 through `anf.application.run_load_cases`; these
batch workflows are Python-only.
[Planar structure guide](docs/frame2d.md) · [Load cases](docs/load-cases.md) ·
[Frame joints](docs/frame-joints.md)

Install `ai-native-fem[sparse]` and pass `{"backend": "scipy"}` as the
`options` argument to `run_analysis` to select the optional sparse solver. The
default remains dense. [Sparse guide](docs/sparse.md)

## Run a modal analysis

Schema 1.4 adds material density and a modal analysis type. Install the
`sparse` extra for SciPy.

```python
from pathlib import Path
from anf.application import run_modal

result = run_modal(
    Path("examples/structures/modal-cantilever.json").read_bytes(),
    {"count": 3},
)
print(result.frequencies_hz)
print(result.dof_labels)
```

A free planar model has three rigid modes; a supported model must have none.
Requesting more modes than the free DOF count returns the available modes with
`truncated` set. Modal artifacts store the model fingerprint, labels, arrays and
diagnostics. [Modal guide](docs/modal.md)

## Run a forced-dynamics analysis

Schema 1.5 adds harmonic response and linear Newmark transients for the planar
models. Install the `sparse` extra for SciPy.

```python
from pathlib import Path
from anf.application import run_harmonic, run_transient

harmonic = run_harmonic(
    Path("examples/dynamics/harmonic-cantilever.json").read_bytes()
)
tip = harmonic.dof_labels.index(("input", "tip", "uy"))
print(harmonic.frequencies_hz, harmonic.magnitudes[tip], harmonic.phases_deg[tip])

transient = run_transient(
    Path("examples/dynamics/transient-cantilever.json").read_bytes()
)
print(transient.times_s[-1], transient.global_peaks["displacement"])
```

The harmonic path solves `(K - w^2 M + i w C) U = F` per frequency with
Rayleigh damping and reports complex displacement, magnitude and phase. The
transient path integrates `M a + C v + K u = f` with the average-acceleration
Newmark method on a uniform grid and stores the full `u`/`v`/`a` history, the
energy series and summary peaks. [Forced-response guide](docs/forced-response.md)

## Run a buckling analysis

Install the a8 wheel with the `sparse` extra to run schema 1.6 elastic buckling.
Use the examples from the matching source distribution or tag. The declared
nodal load is the
reference preload; the solver returns the positive load factors and modes, so
`critical_factor * reference load` is the critical physical load.

```python
from pathlib import Path
from anf.stability import solve_buckling, solve_preload

model = Path("examples/buckling/cantilever-column.json").read_bytes()
preload = solve_preload(model)
print(preload.axial_forces[:3], preload.relative_residual)

result = solve_buckling(model)
print(result.critical_factor, result.factors, result.residuals)
```

The geometric stiffness uses the axial preload only, with tension positive, so
compression lowers the stiffness and tension produces no positive factor. The
result is a linear elastic bifurcation factor, not a capacity or a post-buckling
path. [Buckling guide](docs/buckling.md)

## Mesh studies and local MCP

- Run a mesh convergence study to compare explicit grids and save a replayable
  report with the result from each solve. [Convergence guide](docs/convergence.md)
- Import native planar-member or rectangular meshes, or generate supported grids.
  [Native mesh guide](docs/mesh-io.md)
- Scan cantilever parameters or estimate local sensitivities from Python.
  [Parameter studies](docs/parameter-studies.md)
- Use a local stdio MCP client to validate models, solve them, run studies,
  and inspect artifacts. [MCP guide](docs/mcp.md)

Install the wheel with the `mcp` extra to enable the server.

```sh
python -m pip install "./ai_native_fem-0.1.0a8-py3-none-any.whl[mcp]"
```

Configure your client to launch `anf-mcp --workspace PATH`. Use an existing
writable directory for the local artifacts.

## Supported models and verification

- **Cantilever.** One-plane Euler–Bernoulli bending under a tip force, uniform
  line load, or both.
- **Plate.** Reissner–Mindlin bending of an axis-aligned rectangle under
  uniform pressure. The default `q4_full` supports clamped and hard simply
  supported edges. Select `mitc4_rectangle` explicitly for hard simple support.
- **Planar structures.** Linear elastic `truss2d` and Euler–Bernoulli
  `frame2d` members with nodal loads, uniform transverse member loads,
  prescribed supports, reactions and local member-end actions (schema 1.1).
  Schema 1.2 adds named cases, self-weight, trapezoidal member loads and
  explicit combinations. Schema 1.3 adds end releases and rotational springs.
  Batch runs are Python-only.
- **Optional sparse backend.** Install the `sparse` extra to solve schema 1.0
  to 1.1 models with SciPy. The default remains dense, and the CLI, MCP and
  schema 1.2/1.3 batch workflows stay dense.
- **Modal analysis.** Schema 1.4 planar `truss2d`/`frame2d` models with density,
  consistent mass, mass-normalized modes, frequencies and modal result
  artifacts. Requires the `sparse` extra; Python-only. [Modal guide](docs/modal.md)
- **Forced dynamics.** Schema 1.5 planar models with Rayleigh damping, complex
  harmonic loads, linear Newmark transient presets, versioned response
  artifacts and bounded frequency/step budgets. Requires the `sparse` extra;
  Python-only. [Forced-response guide](docs/forced-response.md)
- **Elastic buckling.** Schema 1.6 planar models with a nodal preload,
  axial-force geometric stiffness, Euler anchors and positive load factors.
  Requires the `sparse` extra; Python-only. [Buckling guide](docs/buckling.md)

Use SI units. Dense assembly limits model size, and fine meshes can be
poorly conditioned. The direct Python solvers raise `ArithmeticError` on
numerical failure. For CLI or declarative runs, read the solve status and
diagnostics. [Methods and limits](docs/methods.md)

Check `verification_status` separately from solve completion. In the thin
Q4 shear-locking example, the CLI solve completes with exit code `0` and
reports `verification_failed`. Automatic MITC4 reports check the equilibrium
residual and report `solved_unverified` after a successful solve. The
validation record documents the independent MITC4 comparisons and tested
ranges. [Verification guide](docs/verification.md) · [Validation record](docs/validation.md)

## Development and contribution

From a source checkout with a virtual environment active, install the test
dependencies and run the suite.

```sh
python -m pip install -e ".[test,sparse]"
python -m pytest
```

Reproduce the analytical comparisons and see the planned research.
[Runnable examples](examples/README.md) · [Roadmap](docs/roadmap.md)

The project uses the MIT license. See the contribution and security guides
for source requirements and reporting.
[License](LICENSE) · [Notice](NOTICE) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)
