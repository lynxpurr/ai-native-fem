# Cantilever parameter studies

Use `anf.parameter_studies` to scan explicit parameter values or estimate local
sensitivity for a schema 1.0 Euler–Bernoulli cantilever with a tip force. The
target is signed tip deflection in metres. The mesh element count is inherited
from the base model. Changing length rescales that same mesh.

```python
from pathlib import Path
from anf.parameter_studies import ParameterGrid, finite_difference_sensitivity, run_parameter_study

path = Path("examples/parameter-studies/cantilever.json")
grid = ParameterGrid.from_dict(path.read_bytes())
scan = run_parameter_study(grid, "cantilever-scan", cache_dir="cantilever-cache")
local = finite_difference_sensitivity(grid, "cantilever-local", cache_dir="cantilever-cache")
print(local["sensitivities"]["loads.tip_force"])
```

| Axis path | Unit | Derivative unit |
| --- | --- | --- |
| `loads.tip_force` | `N` | `m/N` |
| `model.young_modulus` | `Pa` | `m/Pa` |
| `model.second_moment` | `m^4` | `m^-3` |
| `model.length` | `m` | `m/m` |

Declare each axis with a `path`, `unit`, inclusive `bounds` and ordered `values`.
The grid runs the Cartesian product with the last axis varying fastest. Local
sensitivity uses the axis paths and bounds, and takes the center from the base
model. Keep distributed load at zero. The parameter study uses the dense
Python application entry point. Its support domain is uniform, linear-elastic,
small-deflection cantilevers. Other physical models, axes and observables receive
a structured input error.

## Samples and local derivatives

Each planned sample has an ID and parameter values. Each valid physical model
has the existing ANF `input_hash`. The sample table keeps domain rejections,
invalid models, numerical failures and budget skips alongside solved results.
Solver and independent verification states are separate fields. Set the
optional `reference` to a registered cantilever reference to request its
normal applicability and verification checks. The default is `null`.

Local sensitivity estimates `(u_plus - u_minus) / (x_plus - x_minus)` at four
relative steps of `0.008`, `0.004`, `0.002` and `0.001`. All pairs must have
resolvable, solved observations. The last two relative changes in the estimate
must be at most `2.5e-5` and `1e-5`. An estimate has the finest derivative and
the absolute change from the preceding estimate in the same units. An
unavailable estimate has a `null` derivative, a machine-readable reason, and
its original stencil samples. A base deflection at or below `1e-10 m` in
magnitude gives `target_near_zero`. Step agreement describes the stability of
the local difference calculation. For the analytic cantilever anchor, the
deflection `PL^3/(3EI)` and its derivatives give independent checks.
[Cantilever derivation](https://ocw.mit.edu/courses/2-002-mechanics-and-materials-ii-spring-2004/bc25a56b5a91ad29ca5c7419616686f7_lec2.pdf)

## Resume and replay

- A study allows up to 128 planned samples, 64 reserved solver calls and 4 MiB
  of published output. `max_solves` and `max_output_bytes` can be set lower.
  Cache hits and rejected inputs use no solver reservation.
- Run into a new directory under an existing parent. To resume, call the same
  function with an identical declaration and the same directory. Committed
  rows are reused. An interrupted reservation still counts against the budget.
  If budget remains, resume can retry that sample. Each sample has one terminal
  row. A concurrent writer or an incompatible checkpoint raises `ArtifactError`.
- `report.json` is written last and includes the replay declaration, target,
  status counts, samples, attempts, environment identity, and the SHA-256 of
  `study.svg`. The SVG shows parameter bounds, units and sample counts. Failed
  and skipped samples have no plotted value. Use `ParameterGrid.from_dict`
  with `report["study"]` to replay the declaration in a fresh directory.
  `checkpoint.json` is separate bounded working data.
- Pass `cache_dir` to reuse complete solved samples across study directories.
  The cache requires a verifiable bundled linear algebra build. The report
  records whether it is available. Cache keys bind physical inputs, the
  reference, the dense solver, actual ANF and NumPy build data and the execution
  environment. A corrupted entry raises `ArtifactError`.

Run the included example with an installed package.

```sh
python examples/parameter-studies/cantilever.py cantilever-output
```
