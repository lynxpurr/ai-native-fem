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

Uniform-pressure rectangles and one-plane cantilever bending are the only
tested application families. The tests do not certify arbitrary inputs or
engineering use. Dense assembly also restricts model size.
