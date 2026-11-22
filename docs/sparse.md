# Optional sparse static solver

Install the `sparse` extra from this checkout to enable the SciPy backend.

```bash
python -m pip install ".[sparse]"
```

```python
from pathlib import Path
from anf.application import run_analysis

model = Path("examples/structures/settlement.json").read_bytes()
result = run_analysis(model, {"backend": "scipy"})
print(result.solve_status, result.relative_residual, result.execution)
```

The Python option accepts schema 1.0 beams and rectangular plates and schema
1.1 planar structures. The default Python backend and the CLI, MCP, and
schema 1.2/1.3 batch workflows use dense solvers. An explicit sparse request
without SciPy raises `backend_unavailable` at `/options/backend`.

The model fingerprint, SI units, reference checks and saved result format are
the same across backends. Saved arrays contain the results of the selected
solve. The live Python result also exposes `execution` with the installed SciPy
version, DOF count, assembled nonzeros, assembly and solve times, and available
balance metrics. Execution metrics are transient and remain outside the result
manifest.

## Assembly and numerical checks

The solver assembles element triplets in model order, sums duplicates, removes
explicit zeros and stores the stiffness in CSR. It uses CSC and SuperLU for
the free system. Nonzero prescribed displacements enter the effective load as
`f_free - K_free,fixed @ u_fixed`. Reactions come from the original global
force balance. The free-DOF relative residual limit is `1e-8`.

Planar structures use diagonal equilibration and a positive-pivot test on a
separate shifted stiffness. The solve uses the unshifted system. The bound
used for the shift can reject a marginal system before the dense eigenvalue
check would. Structures also check strain energy against applied and support
work. A failed stability or work check reports `unstable_or_ill_conditioned`.
A failed beam or plate factorization reports a numerical failure.

The absolute residual is an infinity norm in force-equivalent newtons. The
conversion divides rotational rows by a characteristic length, which is the
beam length, longest plate side or longest original structure member. The
normwise backward error divides that norm by the sum of the scaled stiffness
norm times the displacement norm and the effective load norm. Rotational
displacements are scaled by the same length. These metrics supplement the
problem's existing relative residual and independent reference checks.

## Measure scaling

```bash
python benchmarks/sparse_scaling.py
```

The benchmark runs both backends in fresh single-threaded processes. It
reports assembly, solve and total time, compressed nonzeros, process peak
resident memory and numerical parity. Error ratios above one identify runs
outside the benchmark tolerance, even if the residual passed. Timings depend
on the platform and matrix fill. The input limits are 200 beam elements, 1,800 plate DOFs, and
200 structure nodes, 400 subsegments and 600 structural DOFs.
