# Linearized elastic buckling

Schema 1.6 adds linearized elastic bifurcation analysis for planar `truss2d` and
`frame2d` models. The declared nodal load is the **reference preload**; the
solver returns the positive load factors at which the linearized stiffness
becomes singular, together with the corresponding buckling modes.
`lambda_cr * f_ref` is the critical physical load.

This is a linear elastic bifurcation analysis. It is not a capacity, a safety
factor or a post-buckling path, and it does not account for imperfections,
yielding or contact. Install the `sparse` extra because the eigensolver uses
SciPy:

```sh
python -m pip install ".[sparse]"
```

The analysis is a Python interface; the CLI and MCP do not expose it.

## Conventions

- Axial force `N` is positive in tension. The geometric stiffness is
  `K_g = N * integral H'(x)^T H'(x) dx`, so compression lowers the stiffness and
  can produce a positive critical factor; tension cannot.
- Only the axial-force contribution is included. Moment-gradient, shear and
  lateral-torsional terms are out of scope, so the result is meaningful for
  members whose preload is predominantly axial.
- Modes are normalized with `phi^T K phi = 1`, and the largest-magnitude
  component of each mode is positive for reproducible display. Repeated factors
  are meaningful as a subspace, not as individual vectors.
- The preload must be a stable linear static state. A mechanism or an
  ill-conditioned preload is rejected before any factor is reported.

## Model

Schema 1.6 uses the schema 1.1 model block (no `density`), a nodal preload and a
buckling analysis block:

```json
{
  "schema_version": "1.6",
  "problem_type": "planar_structure",
  "units": {"length": "m", "force": "N", "stress": "Pa"},
  "model": {
    "nodes": {"base": [0, 0], "tip": [2, 0]},
    "members": {"col": {"kind": "frame2d", "i": "base", "j": "tip", "elements": 16,
                        "young_modulus": 210e9, "area": 0.01, "second_moment": 1e-6}},
    "supports": {"base": {"ux": 0, "uy": 0, "rz": 0}}
  },
  "loads": {"nodal": {"tip": {"fx": -1000.0}}},
  "analysis": {"type": "buckling", "modes": 4}
}
```

- `loads` must contain nodal forces only. `uniform_transverse`, `member_line`,
  self-weight and support displacements are rejected (`unsupported_load` or
  `unsupported_constraint`).
- Supports must be zero displacements.
- End releases and connection springs (schema 1.3) are not supported in 1.6;
  the axial-force extraction needs continuous members.
- `modes` defaults to 4 and is limited to 20. The mesh limits remain 200 nodes,
  400 segments and 600 DOFs.

## Solve

```python
from pathlib import Path
from anf.stability import solve_buckling, solve_preload

model = Path("examples/buckling/cantilever-column.json").read_bytes()
preload = solve_preload(model)
print(preload.axial_forces[:3], preload.relative_residual)

result = solve_buckling(model)
print(result.critical_factor, result.factors)
print(result.modes.shape, result.residuals)
```

`solve_preload(model)` returns the preload equilibrium (`displacement`,
`reactions`, per-segment `axial_forces`, `relative_residual`).
`solve_buckling(model, modes=None)` returns:

| Field | Meaning |
| --- | --- |
| `factors` | positive load factors, ascending |
| `critical_factor` | `factors[0]` |
| `modes` | K-orthonormal modes on all DOFs, one column per factor |
| `residuals` | per-mode eigenpair residual |
| `axial_forces` | reference-preload axial forces, tension positive |
| `preload_residual` | preload static residual |
| `dof_labels`, `diagnostics`, `versions`, `input_hash` | context for the run |

The result carries no artifact; use the model declaration and `input_hash` to
reproduce a run. A `repeated_factor:<index>` diagnostic marks factors that are
equal within `1e-8` relative; only the subspace is meaningful there.

## Failures

Input errors raise `ModelValidationError`: `unsupported_schema`,
`unsupported_analysis`, `unsupported_load`, `unsupported_member`,
`unsupported_constraint`, `no_free_dofs` and `invalid_value` (mode count).

Numerical failures raise `BucklingError` with a `code`:

| Code | Meaning |
| --- | --- |
| `unstable_preload_model` | the preload static solve failed (mechanism or ill-conditioning) |
| `buckling_mechanism` | the free stiffness is singular or not positive definite |
| `buckling_no_positive_factor` | the reference load produces no compression |
| `buckling_not_converged` | an eigenpair exceeds the residual gate |

A failed solve returns no partial result. Without SciPy the call reports
`backend_unavailable`.

## Verification

Independent Euler anchors and the measured errors with the shipped solver:

| Case | Analytic load | 16-element error | 32-element error |
| --- | --- | ---: | ---: |
| fixed-free | `pi^2 E I / (4 L^2)` | 1.29e-7 | 8.11e-9 |
| pinned-pinned | `pi^2 E I / L^2` | 2.06e-6 | 1.29e-7 |
| fixed-fixed (guided) | `4 pi^2 E I / L^2` | 3.28e-5 | 2.06e-6 |

The halving error ratio is 15.6-16.0 (fourth-order convergence). Scaling the
reference load by 0.5, 1 and 2 gives factors 2:1:0.5 with the physical critical
load invariant to `1e-12`. The preload residual stays below `2.3e-14`, the
eigenpair residual below `5.1e-11`, and `K_g` is negative semidefinite in
compression and positive semidefinite in tension within `1e-10` of the matrix
scale. A tension-only model and a zero preload both raise
`buckling_no_positive_factor`.

## Limits

- Linear, small-displacement, planar `truss2d`/`frame2d` only, within the
  600-DOF cap; the eigensolve is dense and takes about 2.3 s at that size.
- Axial-force geometric stiffness only; members under significant bending
  preload or lateral-torsional buckling need a different formulation.
- No nonlinear capacity, post-buckling path, imperfection sensitivity, plates,
  shells, 3D frames, lateral-torsional buckling or empirical reduction factors.
- No result artifacts, CLI or MCP entry in the first release.
- The result is a bifurcation factor for the declared reference load, not a
  design code check.