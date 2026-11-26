# Research roadmap

The current package is a classical, isotropic beam, rectangular plate and
planar structure research baseline. The sections below record the capabilities
by version and outline future research. See
[CONTRIBUTING](../CONTRIBUTING.md) for contributions.

## Maintenance iteration

- Reproducible cantilever, thick-plate convergence and thin-plate locking examples.
- Wheel and source distribution installation tests on Linux Python 3.11/3.12
  and Windows Python 3.12.
- Dependency advisory checking and reviewed dependency/Actions update PRs.
- Investigate fine-mesh beam conditioning using analytical displacement, force
  balance and the original residual requirement. Model-size limits alone do not
  guarantee a solve.

The 0.1.0a3 prerelease includes verification reports and rectangular MITC4.
Both public distributions were downloaded, checked against the approved CI
build and installed again.
[Prerelease](https://github.com/lynxpurr/ai-native-fem/releases/tag/v0.1.0a3) ·
[Release policy](release-policy.md)

## Versioned models and results in 0.1.0a4

The [0.1.0a4 release notes](releases/0.1.0a4.md) describe the shared analysis
workflow for the existing beam and rectangular plate solvers. Use versioned
SI model files and JSON/NPZ result artifacts from
[Python](model-format.md) or the [installed command line](cli.md).
[Mesh convergence reports](convergence.md) record each mesh, reference
comparison and failure for replay. The [optional local MCP server](mcp.md)
exposes these operations over stdio.

## Planar structures and research workflow in 0.1.0a5

The [0.1.0a5 release notes](releases/0.1.0a5.md) add the R2 structural statics
and research workflow:

- Schema 1.1 planar truss and Euler–Bernoulli frame statics with dense
  assembly, prescribed displacements, reactions and local member-end actions.
- Schema 1.2 named load cases, self-weight, trapezoidal member loads and
  explicit linear combinations, with member force envelopes.
- Schema 1.3 frame end releases and rotational springs.
- An optional SciPy sparse backend with the same fingerprints and verification
  semantics as the dense solver.
- Cantilever parameter grids and local finite-difference sensitivity with
  checkpoint/resume and retained failures.
- Native mesh documents for planar members and uniform rectangular plates,
  plus deterministic line and rectangle generation. External mesh formats
  remain unsupported.

The CLI and MCP accept schema 1.0 and 1.1 single-solve declarations. Schema
1.2/1.3 batch runs, parameter studies and native mesh import are Python
interfaces. The package remains linear: 3D frames, shear deformation,
buckling, nonlinear analysis, shells and solids are not included; harmonic and
transient dynamics are Python-only (see below).

## Modal analysis in 0.1.0a6

The [0.1.0a6 release notes](releases/0.1.0a6.md) add schema 1.4 planar
consistent-mass modal analysis:

- `assemble_mass` for `truss2d`/`frame2d`, including transverse translational
  inertia and the Hermite frame mass.
- Dense generalized eigensolver within the 600-DOF planar cap, mass-normalized
  modes, frequencies, residuals and rigid-mode classification.
- Versioned modal artifacts with DOF labels and truncation diagnostics.
- Python-only access through the `sparse` extra; the CLI and MCP do not expose
  mode analysis.

Plate mass matrices, shear deformation, rotary inertia and a sparse modal path
remain separate designs.

## Forced dynamics in 0.1.0a7

The [0.1.0a7 release notes](releases/0.1.0a7.md) add planar harmonic response
and linear Newmark transients:

- complex amplitude/phase harmonic loads, direct per-frequency factorization and
  magnitude/phase results with conditioning diagnostics;
- uniform transient grids, `constant`/`ramp`/`harmonic` presets, nonzero initial
  state, complete `u`/`v`/`a` and energy histories with summary peaks;
- versioned response artifacts and explicit frequency, step and DOF-time
  budgets.

Reaction recovery, response spectra, modal damping ratios, nonlinear Newmark,
array or table force histories, variable time steps and a modal-superposition
harmonic path remain separate designs.

## Elastic buckling in develop

Schema 1.6 adds linearized elastic bifurcation for planar trusses and frames:

- the declared nodal load is the reference preload, and the solver reports
  positive load factors, K-orthonormal modes and eigenpair residuals;
- the geometric stiffness uses the axial preload only (tension positive), with
  Euler anchors for fixed-free, pinned-pinned and fixed-fixed columns;
- mechanisms, tension-only states and non-converged eigenpairs are rejected
  instead of producing a factor.

Nonlinear capacity, post-buckling paths, imperfection sensitivity, plate/shell
and lateral-torsional buckling, result artifacts and CLI/MCP entries remain
separate designs.

## Plate research

The rectangular MITC4 implementation covers hard simply supported edges under
uniform pressure. The fully integrated Q4 remains the default and provides a
locking comparison. Extending MITC4 to clamped edges or general quadrilaterals
requires separate derivation and independent verification.

Additional material models, shells and higher-order formulations require separate
source review, mathematical specification and validation before inclusion.
