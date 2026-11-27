# Prototype validation record

The initial 33 physical/validation cases were run on Linux x86-64 using installed
wheels, with Python 3.11.15 / NumPy 2.4.6 and Python 3.12.13 / NumPy 2.5.5. They
remain unchanged. Version 0.1.0a2 adds regression cases for NumPy scalar overflow,
input normalization, load-scale-independent residuals, numeric range failures
and rejection of non-finite or unbalanced solutions: 67 cases in total, passing
in both Linux environments. CI builds distributions and
runs tests against the installed wheel outside the checkout.
This record covers the tested prototype and the additions through 0.1.0a8. It
does not certify another development version of ANF. See [methods](methods.md) for
equations and restrictions.

The beam tests compare tip displacement, slope and root reactions against the
integrated differential equation for tip and uniform loads on multiple meshes.
Element stiffness is checked against the standard closed matrix and two rigid
motions. For the plate, tests check three rigid modes, positive remaining
element eigenvalues, load balance, symmetry and homogeneous boundary values.

For a square hard simply supported plate with sides 1 m, E=70 GPa, nu=0.3,
pressure 1000 Pa and k=5/6, center deflections are:

| Thickness / side | Mesh | FE deflection (m) | Continuous mode reference (m) | Relative error |
|---|---|---:|---:|---:|
| 0.2 | 4 × 4 | 8.01359278e-8 | 9.56331888e-8 | 16.20% |
| 0.2 | 8 × 8 | 9.12192845e-8 | 9.56331888e-8 | 4.62% |
| 0.2 | 16 × 16 | 9.44891705e-8 | 9.56331888e-8 | 1.20% |
| 0.005 | 8 × 8 | 5.56313372e-5 | 5.07047234e-3 | **98.90%** |

The continuous reference retains odd sine modes through index 31. The thin
plate row exposes severe shear locking. It is a negative result, not an
acceptable engineering accuracy claim. No numerical stabilization, fitted
coefficient or reference-output substitution is used to conceal it.

The [verification API](verification.md) reports the 16 × 16 thick case and
the thin locking case with their observable, reference and error.

## Rectangular MITC4

The [plate tests](../tests/test_mitc4.py) run both `q4_full` and
`mitc4_rectangle` on 1 m × 1 m and 2 m × 1 m hard simply supported plates.
Each formulation uses four thicknesses and three uniform meshes, for 48 plate
solves. The material, shear factor and pressure match the Q4 cases above.

MITC4 center-deflection errors relative to the continuous Reissner–Mindlin
series are shown below. Each axis retains odd indices through 255. Across the
eight geometry and thickness combinations, the largest relative change from
index 127 to 255 is `1.35e-7`.

| `lx` × `ly` (m) | Thickness (m) | 4 × 4 | 8 × 8 | 16 × 16 |
|---|---:|---:|---:|---:|
| 1 × 1 | 0.2 | 1.012% | 0.213% | 0.052% |
| 1 × 1 | 0.05 | 2.204% | 0.493% | 0.121% |
| 1 × 1 | 0.01 | 2.295% | 0.514% | 0.126% |
| 1 × 1 | 0.005 | 2.297% | 0.515% | 0.126% |
| 2 × 1 | 0.2 | 0.868% | 0.411% | 0.108% |
| 2 × 1 | 0.05 | 1.394% | 0.580% | 0.151% |
| 2 × 1 | 0.01 | 1.430% | 0.593% | 0.154% |
| 2 × 1 | 0.005 | 1.431% | 0.593% | 0.154% |

The MITC4 errors decrease on both refinement steps in each row and stay below
3% on 16 × 16. For the 0.005 m square on 8 × 8, the same series gives 98.90%
error for Q4 and 0.515% for MITC4. With both formulations the tests require a
free-DOF relative residual at most `1e-8`, relative vertical force imbalance
below `1e-7`, and relative difference between external work and twice the strain
energy below `1e-7`.

Square and rectangular MITC4 elements at two thicknesses have three rigid
body modes and nine positive stiffness modes. Tests check symmetry and
pressure-load conservation, and check that constant curvature produces zero
assumed shear and the exact bending energy within a relative tolerance of
`1e-12`. The energy check integrates the bending contribution directly at
Gauss points so the thin-plate result is insensitive to cancellation between
assembled shear stiffness entries.

## Planar structures, sparse and parameter studies

Version 0.1.0a5 adds planar truss/frame statics, load cases, an optional sparse
backend, cantilever parameter studies and native mesh input. The new checks use
independent closed forms and the same residual gates as the existing solvers.

- **Planar statics** ([`tests/test_frame2d.py`](../tests/test_frame2d.py)). A
  2 m frame cantilever with `EI=210 kN·m²` and a `-1000 N` tip force matches
  `w=PL³/(3EI)=-1.26984127e-2 m` and `theta=PL²/(2EI)=-9.52380952e-3 rad`,
  with 1000 N root force and 2000 N·m root moment, at relative tolerance
  `1e-8`. An axial bar chain, a simply supported beam under 500 N/m
  (`5wL⁴/(384EI)=-1/120 m` at midspan) and a portal frame integrated by the
  unit-load force method are checked at the same order. Rigid translation and
  rotation, node renumbering, nonzero prescribed displacement and explicit
  mechanism failure are separate cases. Mechanisms report
  `unstable_or_ill_conditioned` and write no solved arrays.
- **Load cases and joints** ([`tests/test_load_cases.py`](../tests/test_load_cases.py),
  [`tests/test_joints.py`](../tests/test_joints.py)). Self-weight, a
  trapezoidal member load and an explicit combination reproduce the closed-form
  beam response; `member_forces` returns the expected axial, shear and moment
  diagrams. The documented spring example gives 1125 N and 875 N vertical
  reactions, a 500 N·m spring moment and zero moment at the released end.
  Spring stiffness limits approach the released and rigid cases.
- **Sparse** ([`tests/test_sparse_parity.py`](../tests/test_sparse_parity.py)).
  Dense and SciPy results agree at relative tolerance `1e-8` (absolute `1e-6`
  for force arrays and `1e-10` for displacement and rotation), with the same
  input hash, ordering and verification status. The free-DOF relative residual
  stays at most `1e-8`. Singular systems are reported, not regularized.
- **Parameter studies** ([`tests/test_parameter_studies.py`](../tests/test_parameter_studies.py)).
  The four analytic cantilever derivatives for `P`, `E`, `I` and `L` agree
  within `5e-6` relative. Each sampled solve agrees with the analytic tip
  deflection within `2e-9` relative and keeps the `1e-8` residual gate. The
  tests execute 33 samples and four step ratios; incomplete stencils return an
  unavailable result instead of zero.
- **Native meshes** ([`tests/test_mesh_io_v1.py`](../tests/test_mesh_io_v1.py)).
  Node renumbering, rigid translation and rotation, unit conversion and sparse
  projection preserve the physical model and input hash. The rectangular MITC4
  projection stays inside the existing 1% reference gate; the 2 m × 1 m,
  0.01 m thick, 8 × 8 plate is 0.593% from the continuous reference, and the
  thin Q4 locking case remains a failed comparison. Corrupt counts, duplicate
  IDs, unknown cells, undefined sets and over-budget documents are rejected.

## Modal analysis

Schema 1.4 modal analysis is checked by
[`tests/test_modal.py`](../tests/test_modal.py),
[`tests/test_modal_mass.py`](../tests/test_modal_mass.py) and
[`tests/test_modal_artifacts.py`](../tests/test_modal_artifacts.py). The
independent frequency anchors are:

- fixed-free axial rod: `omega_n = (2n-1) pi / (2L) * sqrt(E/rho)`;
- simply supported Euler–Bernoulli beam: `omega_n = (n pi / L)^2 *
  sqrt(EI/(rho A))`;
- cantilever Euler–Bernoulli beam: `omega_n = (beta_n L)^2 *
  sqrt(EI/(rho A L^4))`.

Observed relative errors at 16 elements, E=210 GPa, rho=7850 kg/m³, A=0.01 m²,
I=1e-6 m⁴, L=2 m:

| case | mode 1 | mode 2 | mode 3 |
|---|---:|---:|---:|
| cantilever frame | 1.31e-7 | 5.11e-6 | 3.98e-5 |
| axial rod | 4.02e-4 | 3.62e-3 | 1.01e-2 |

The cantilever gate is `1e-4`; the axial-rod gate is `2e-2` at 16 elements and
the 8→16 sequence must decrease. A free planar frame has exactly three rigid
modes; the first two frequencies are zero and the third is about `2.24e-5 Hz`,
inside the `1e-3 Hz` rigid check. Elastic eigenpair residuals stay at most
`1e-9`; rigid modes use the matrix-scaled residual with a `1e-8` gate.
Renumbering preserves frequencies, a requested count beyond the available modes
sets `truncated` without zero padding, and a nonzero modal support is rejected.
Artifacts round-trip the model, labels, arrays and diagnostics; changed digests,
extra files, non-finite arrays and unsupported schemas are rejected.

For schema 1.5 forced dynamics, the SDOF complex response matches the analytic
amplitude and phase to `1e-15`, the 16-element cantilever static limit matches
`FL^3/(3EI)` to `1e-12`, and a 40-mode Euler–Bernoulli series agrees within
`2e-5` up to twice the first frequency. Newmark step and free-vibration
solutions match their closed forms with the halving ratio in `[3.5, 4.5]`;
undamped energy drift stays below `1e-10`, damped energy never increases, the
nonzero initial state matches its analytic decay, and summary peaks and energy
are bitwise identical to recomputation from the stored arrays. Undamped exact
resonance is rejected rather than reported as a finite amplitude.

For schema 1.6 buckling, the fixed-free, pinned-pinned and fixed-fixed Euler
loads converge at fourth order: 16-element errors are `1.29e-7`, `2.06e-6` and
`3.28e-5`, and the halving ratio is `15.6`–`16.0`. Scaling the reference load
leaves the physical critical load invariant to `1e-12`; the preload residual
stays below `2.3e-14` and the eigenpair residual below `5.1e-11`. `K_g` is
negative semidefinite under compression and positive semidefinite under tension,
and tension-only or mechanism states are rejected without a factor.

At integration commit `27718fc`, CI installed the wheel and source distribution
separately on Linux Python 3.11 and 3.12 and Windows Python 3.12. For each
package, the full suite reported 464 passed and 68 skipped before installing
SciPy and 546 passed with the `sparse` extra.
[CI installation logs](https://github.com/lynxpurr/ai-native-fem/actions/runs/45922487091)

Uniform-pressure rectangles, one-plane cantilever bending, planar truss/frame
statics, cantilever parameter studies, the documented native mesh projections,
schema 1.4 modal analysis, schema 1.5 harmonic and transient forced dynamics and
schema 1.6 linearized elastic buckling are the tested application families. The tests do not certify
arbitrary inputs or engineering use. Dense assembly also restricts model size;
the sparse backend does not remove the declared limits.
