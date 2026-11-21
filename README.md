# ANF classical research models

Small Python implementations of Euler–Bernoulli cantilever bending and
rectangular Reissner–Mindlin plate bending. The [method notes](docs/methods.md)
give the equations, mathematical references, DOF conventions and limitations.

This is an educational research baseline. Dense assembly limits model size.
No higher-order thickness or cross-section expansion, shell, solid, laminate,
nonlinear, dynamic, learned model or service interface is included. API
compatibility with other ANF development versions is not provided.

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

Use [verification reports](docs/verification.md) to compare a solve with a
registered reference and inspect its error, tolerance and input fingerprint.

Use the [declarative model and result format](docs/model-format.md) to run a
versioned SI model and save a result that another Python process can inspect.

Use consistent SI units. The beam API models one bending plane. The plate
supports uniform pressure on axis-aligned rectangular domains. The default
`q4_full` supports clamped and hard simply supported edges and can exhibit
severe shear locking on thin plates. For hard simple support, select
`mitc4_rectangle` explicitly.

```python
from anf import Plate, solve_plate

plate = Plate(2, 1, 0.005, 70e9, 0.3, nx=16, ny=16,
              support="hard_simply_supported", formulation="mitc4_rectangle")
result = solve_plate(plate, pressure=1000)
print(result.deflection[plate.ny // 2, plate.nx // 2])
```

See the [validation record](docs/validation.md) for the tested thicknesses,
aspect ratios and refinement results.

Project code is offered under [MIT](LICENSE). NumPy is installed separately;
see [NOTICE](NOTICE). See [CONTRIBUTING.md](CONTRIBUTING.md) for source requirements
and [SECURITY.md](SECURITY.md) for reporting.

See the [runnable examples](examples/README.md) for analytical comparisons and
the [research roadmap](docs/roadmap.md) for the maintenance and numerical work.
