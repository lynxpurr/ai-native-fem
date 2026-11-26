# Forced dynamics: harmonic response and linear Newmark transients

Schema 1.5 adds two linear forced-dynamics analyses for planar `truss2d` and
`frame2d` models:

- **harmonic response** — steady-state complex displacement at a list of
  frequencies;
- **transient response** — linear Newmark average-acceleration integration on a
  uniform time grid.

Both reuse the schema 1.4 consistent mass, zero-displacement supports and the
dense planar solver within the 600-DOF cap. They are Python interfaces: the CLI
and MCP do not expose them. Install the `sparse` extra because both paths use
SciPy:

```sh
python -m pip install ".[sparse]"
```

## Units and conventions

Inputs and outputs are SI: length m, force N, moment N·m, mass kg, time s.
Damping and phases use their own units:

| Quantity | Unit | Convention |
| --- | --- | --- |
| `alpha` (Rayleigh) | `s^-1` | `C = alpha * M + beta * K` |
| `beta` (Rayleigh) | `s` | `alpha, beta >= 0`; the damping matrix is positive semidefinite |
| `phase_deg` | degrees | relative to a cosine; positive means the component leads `t = 0` |
| harmonic response | `u(t) = Re{U exp(i omega t)}` | `omega = 2 pi f`, `f` in Hz |

Every frequency result also reports `omega_rad_s`. `phases_deg` lies in
`(-180, 180]`; a zero response has phase `0`. The transient `harmonic` preset
`amplitude * cos(2 pi f t + phase_deg)` uses the same convention on absolute
time, so transient and harmonic results can be compared directly.

## Harmonic declaration

```json
{
  "schema_version": "1.5",
  "problem_type": "planar_structure",
  "units": {"length": "m", "force": "N", "stress": "Pa"},
  "model": {
    "nodes": {"base": [0, 0], "tip": [2, 0]},
    "members": {"beam": {"kind": "frame2d", "i": "base", "j": "tip", "elements": 16,
                         "young_modulus": 210e9, "area": 0.01,
                         "second_moment": 1e-6, "density": 7850}},
    "supports": {"base": {"ux": 0, "uy": 0, "rz": 0}}
  },
  "loads": {"nodal": {"tip": {"fy": {"amplitude": 1000.0, "phase_deg": 0.0}}}},
  "analysis": {
    "type": "harmonic",
    "frequencies_hz": [0.0, 7.2358, 20.0],
    "damping": {"kind": "rayleigh", "alpha": 1.720507, "beta": 4.743437e-5}
  }
}
```

- `frequencies_hz` must be non-negative and strictly increasing, with at most
  200 points. Duplicates are rejected rather than merged.
- A nodal load is either a number (amplitude, zero phase) or
  `{"amplitude": number, "phase_deg": number}`.
- `mz` is only allowed on frame nodes; truss nodes have no rotational DOF.
- An empty `loads` object is accepted and produces a zero response.
- `f = 0` is the static limit `K u = F`; a free-body model has a singular
  stiffness there.
- The solver rejects frequencies whose scaled condition number exceeds `1e15`
  (`harmonic_singular`) and flags points above `1e11`
  (`harmonic_ill_conditioned:<index>`). An undamped model exactly at an
  eigenfrequency is rejected instead of returning a meaningless amplitude.

## Transient declaration

```json
{
  "schema_version": "1.5",
  "problem_type": "planar_structure",
  "units": {"length": "m", "force": "N", "stress": "Pa"},
  "model": {
    "nodes": {"base": [0, 0], "tip": [2, 0]},
    "members": {"beam": {"kind": "frame2d", "i": "base", "j": "tip", "elements": 16,
                         "young_modulus": 210e9, "area": 0.01,
                         "second_moment": 1e-6, "density": 7850}},
    "supports": {"base": {"ux": 0, "uy": 0, "rz": 0}}
  },
  "loads": {
    "nodal": {"tip": {"fy": {"kind": "ramp", "amplitude": 1000.0,
                             "t0_s": 0.0, "t1_s": 0.01}}}
  },
  "analysis": {
    "type": "transient",
    "time": {"start_s": 0.0, "end_s": 0.05, "steps": 500},
    "damping": {"kind": "rayleigh", "alpha": 0.0, "beta": 0.0},
    "initial_state": {"displacement": {"tip": {"uy": 0.001}}, "velocity": {}}
  }
}
```

Load presets, evaluated on absolute time:

| `kind` | Required fields | Value |
| --- | --- | --- |
| `constant` | `amplitude` | `amplitude` from the start of the window |
| `ramp` | `amplitude`, `t0_s`, `t1_s` | zero to `amplitude` on `[t0_s, t1_s]`, constant after |
| `harmonic` | `amplitude`, `frequency_hz` | `amplitude * cos(2*pi*f*t + phase_deg)`; continuous, not switched at `start_s` |

`ramp` requires `start_s <= t0_s < t1_s <= end_s`; `harmonic` requires a
positive frequency. The time grid is uniform and declared either as
`{"start_s", "end_s", "steps"}` or as an explicit `{"times_s": [...]}` array;
the array form must be strictly increasing and uniformly spaced. Initial
displacement and velocity may only act on free DOFs.

Budgets: at most 5000 steps and `n_dof * (steps + 1) <= 1_200_000`, which keeps
the five float64 history arrays under 48 MB.

## Python API

```python
from pathlib import Path
from anf.application import run_harmonic, run_transient

harmonic = run_harmonic(Path("examples/dynamics/harmonic-cantilever.json").read_bytes())
transient = run_transient(Path("examples/dynamics/transient-cantilever.json").read_bytes())
```

`anf.dynamics.harmonic_response(model)` and `anf.dynamics.transient_response(model)`
are the lower-level entry points. `HarmonicResult` exposes:

| Field | Meaning |
| --- | --- |
| `frequencies_hz`, `omega_rad_s` | the requested sweep in Hz and rad/s |
| `displacements` | complex `U` on all DOFs, rows follow `dof_labels` |
| `magnitudes`, `phases_deg` | `abs(U)` and `angle(U)` in degrees |
| `backward_errors`, `conditions` | per-frequency backward error and scaled `kappa` |
| `diagnostics`, `dof_labels`, `versions`, `input_hash` | context for the run |

`TransientResult` exposes the complete `times_s`, `displacement`, `velocity`,
`acceleration` and `energy` histories, per-DOF `peak_displacement` and
`peak_time_s`, `global_peaks` for displacement/velocity/acceleration,
`worst_backward_error` and `worst_backward_step`, `diagnostics`, `versions` and
`input_hash`. Fixed DOFs are present in the arrays and stay zero.

Energy is `1/2 v'Mv + 1/2 u'Ku`, computed from the stored arrays; it is
conserved without damping and must never increase with damping.

## Artifacts

```python
from anf.dynamics import load_harmonic_result, save_harmonic_result

directory = save_harmonic_result(harmonic, Path.cwd(), "harmonic-result")
restored = load_harmonic_result(directory)
assert restored.input_hash == harmonic.input_hash
```

`save_transient_result` / `load_transient_result` follow the same layout:
`manifest.json` plus `arrays.npz`, with the model fingerprint, labels, array
shapes, digest and diagnostics. Loading rejects a changed digest or input hash,
unexpected files, unsupported schemas, non-finite arrays and mismatched shapes.
The published manifest schemas are
[`harmonic-results-v1.5.json`](../schemas/harmonic-results-v1.5.json) and
[`transient-results-v1.5.json`](../schemas/transient-results-v1.5.json).

## Failures

Input errors raise `ModelValidationError` with a JSON Pointer and a machine
code: `unsupported_analysis`, `unsupported_damping`, `negative_damping`,
`negative_frequency`, `frequency_not_increasing`,
`frequency_budget_exceeded`, `invalid_time_grid`, `non_monotonic_time`,
`non_uniform_time`, `time_budget_exceeded`, `unsupported_load`,
`invalid_load`, `unknown_node`, `unsupported_initial_condition`,
`no_free_dofs`, `unsupported_constraint` and `backend_unavailable`.

Numerical failures raise `HarmonicSolveError` or `NewmarkSolveError`, each with
a `code`; the transient error also carries the failing `step` and `time_s`.
A failed frequency or step produces no result and no artifact, so an existing
artifact always describes a complete sweep or history.

The per-step backward error also acts as a time-resolution check. Average
acceleration is unconditionally stable, but a step that is coarse relative to
the highest structural modes leaves the recovered accelerations unresolved, and
the solver raises `newmark_not_converged` instead of returning a history that
looks complete. Refine `dt` or coarsen the mesh; a 597-DOF chain at
`dt = 1e-4` trips the gate while `dt = 1e-5` passes.

The first release does not compute support reactions or internal forces. A load
on a fixed DOF is therefore rejected with `unsupported_load`.

## Verification

The tests use independent analytic anchors:

- SDOF amplitude, phase and resonance against the closed-form complex response,
  including the zero-frequency static limit;
- a 16-element cantilever at `f = 0` against `FL^3/(3EI)`;
- a 40-mode Euler–Bernoulli cantilever series (shape functions evaluated in a
  stable exponential form) with the same Rayleigh damping;
- Newmark step response and free vibration against closed-form solutions, with
  the observed order between halvings in `[3.5, 4.5]`;
- undamped energy conservation, damped dissipation, a nonzero initial state,
  bitwise equality between summary peaks/energy and the stored arrays, and the
  steady-state amplitude against the harmonic solver.

Measured on a reference machine: step L2 errors `1.141e-1`, `2.926e-2`,
`7.357e-3`, `1.842e-3`, `4.606e-4` at `T/10` to `T/160`; undamped energy drift
below `1e-13` over 5000 steps; nonzero initial-state error `6.5e-3`; worst
backward error `4.1e-13`.

## Limits

The implementation is linear, small-displacement and planar. It does not
include nonlinear Newmark, response spectra or SRSS/CQC combination, modal
damping ratios, non-proportional or complex damping, base or multi-support
excitation, table/array force histories, variable time steps, reaction
recovery, plates, shells or 3D frames. The harmonic path solves each frequency
directly; a modal-superposition sweep is a separate design.

The cost is dominated by the per-frequency complex factorization. At the
597-free-DOF cap a factorization takes about 2.1 s on the reference machine, so
a 200-point sweep takes several minutes; the 200-point limit is an input bound,
not a runtime promise. A resolving transient run of 500 steps at that size takes
about 3.7 s. The 600-DOF cap and the declared budgets keep both paths bounded.