# ANF classical research models

Small Python implementations of Euler–Bernoulli cantilever bending and a fully
integrated rectangular Q4 Reissner–Mindlin plate. The [method notes](docs/methods.md)
give the equations, mathematical references, DOF conventions and limitations.

This is an educational research baseline. The plate element suffers from shear
locking and is unsuitable for thin-plate accuracy claims without a refinement
study. Dense assembly limits model size. No higher-order thickness or
cross-section expansion, shell, solid, laminate, nonlinear, dynamic, learned
model or service interface is included. API compatibility with other ANF
development versions is not provided.

Parameter validation rejects invalid or unrepresentable inputs. Calculations
that exceed floating-point range or fail the free-DOF residual check raise
`ArithmeticError`; see the method notes. Staying below the model-size cap does
not guarantee a sufficiently conditioned solve.

```bash
python -m venv .venv
# Activate the environment for your shell.
python -m pip install -e ".[test]"
python -m pytest
```

```python
from anf import Beam, solve_cantilever

model = Beam(length=2, young_modulus=210e9, second_moment=1e-6)
result = solve_cantilever(model, tip_force=1000)
print(result.deflection[-1])  # analytical value: 0.0126984127 m
```

Use consistent SI units. The beam API models one bending plane; it does not
include axial or torsional deformation. The plate supports uniform pressure,
with clamped or hard simply supported edges. The [validation record](docs/validation.md)
includes a thin-plate case with 98.90% error; the plate implementation is a
baseline for studying convergence and locking, not a thin-plate accuracy solution.

Project code is offered under [MIT](LICENSE). NumPy is installed separately;
see [NOTICE](NOTICE). See [CONTRIBUTING.md](CONTRIBUTING.md) for source requirements
and [SECURITY.md](SECURITY.md) for reporting.

See the [runnable examples](examples/README.md) for analytical comparisons and
the [research roadmap](docs/roadmap.md) for the maintenance and numerical work.
