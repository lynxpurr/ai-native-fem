# Prototype validation record

The initial 33 physical/validation cases were run on Linux x86-64 using installed
wheels, with Python 3.11.15 / NumPy 2.4.6 and Python 3.12.13 / NumPy 2.5.5. They
remain unchanged. Version 0.1.0a2 adds regression cases for NumPy scalar overflow,
input normalization, load-scale-independent residuals, numeric range failures
and rejection of non-finite or unbalanced solutions: 67 cases in total, passing
in both Linux environments. CI builds distributions and
runs tests against the installed wheel outside the checkout.
This record covers this small prototype, not another development
version of ANF. See [methods](methods.md) for equations and restrictions.

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

Uniform-pressure rectangles and one-plane cantilever bending are the only
tested application families. The tests do not certify arbitrary inputs or
engineering use. Dense assembly also restricts model size.
