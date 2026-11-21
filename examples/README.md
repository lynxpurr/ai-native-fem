# Reproducible examples

Build a wheel from this checkout with `python -m build`. Install the wheel in
a fresh virtual environment and copy the entire examples directory outside
the checkout. With that environment active, run from the copied directory.

```bash
python beam.py
python plate_convergence.py
python plate_convergence.py --formulation mitc4_rectangle
python plate_locking.py
```

The scripts import the installed `anf` package and need no pytest or plotting
dependency. CSV output uses SI units and independently calculated references.
The cantilever tip displacement is approximately 0.0126984127 m. The thick Q4
plate errors decrease from 16.20% to 4.62% to 1.20%. The thin MITC4 plate converges
below 3% on 16 × 16. The separate thin Q4 example reproduces 98.90% error from
shear locking.

The Q4 examples use odd modes through index 31. The MITC4 example uses modes
through index 255. Both series represent hard simple support. Small floating-point
differences are expected. See the [validation record](../docs/validation.md) for
the tested cases and their limits.
