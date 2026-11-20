# Reproducible examples

Install the v0.1.0a2 wheel from GitHub Releases in a fresh virtual environment.
Download the matching source archive from the maintenance commit containing these
examples (they are not included in the original a2 archives), or use a checkout
of that commit. Copy the entire examples directory outside the checkout.
With that environment active, run from the copied directory:

```bash
python beam.py
python plate_convergence.py
python plate_locking.py
```

The scripts import the installed `anf` package and need no pytest or plotting
dependency. CSV output uses SI units and independently calculated references.
The cantilever tip displacement is approximately 0.0126984127 m. The thick plate
errors decrease from 16.20% to 4.62% to 1.20%. The thin plate error remains 98.90%:
successful execution of that example confirms a known limitation.

The plate series uses odd modes through 31, as in docs/validation.md. It models
hard simple support; it is not a universal reference for other boundary data.
Small floating-point differences are expected. Each script checks its stated
behavior and exits unsuccessfully if it changes. These examples do not establish
engineering suitability.
