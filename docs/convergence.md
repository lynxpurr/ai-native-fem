# Run and replay a mesh convergence study

A study keeps the SI geometry, material, loads, support and formulation fixed
and varies only the mesh. Use `tip_deflection` for a beam or
`center_deflection` for a plate. Both are measured in metres. An odd plate grid
uses bilinear interpolation at the same geometric center.

```python
import json
from pathlib import Path

from anf.reporting import write_report
from anf.studies import StudySpec, run_convergence

source = Path("examples/studies/thick-q4.json")
spec = StudySpec.from_dict(json.loads(source.read_text(encoding="utf-8")))
report_path = write_report(run_convergence(spec), "thick-study")

saved = json.loads(report_path.read_text(encoding="utf-8"))
replayed = run_convergence(StudySpec.from_dict(saved["study"]))
assert [step.input_hash for step in replayed.steps] == [step["input_hash"] for step in saved["steps"]]
```

Use explicit mesh entries with `elements` for a beam or `nx` and `ny` for a
plate. A study accepts up to 16 entries. Set `max_solves` to a smaller number
to leave later entries marked `not_run`. After a numerical failure,
`failure_policy="stop"` skips later meshes and `"record_and_continue"`
attempts them until the budget is reached. Every attempted mesh carries its
actual solve and verification state, diagnostics and input fingerprint.

`report.json` contains the normalized replay declaration, the base model
fingerprint, all steps and any estimate. `report.md` summarizes the same
observations and links to the result bundles. `convergence.svg` plots
deflection in metres by mesh step. Each attempted mesh links to its result
directory by relative path and records the SHA-256 of its manifest. Create
reports in a new directory under an existing parent. The total output defaults
to 4 MiB. Set `max_output_bytes` to request a smaller budget.

## Compare the results

- `reference_check` is the selected registry check for the observable, with
  its metric and outcome. A missing or inapplicable reference leaves the
  reference error unavailable. The thick Q4 reference applies at 16 × 16.
  The thin Q4 example retains the 8 × 8 locking result and failed comparison.
- `successive_mesh_change_m` is the absolute difference from the immediately
  preceding solved mesh in the sequence. It has units of m.
- The observed order and finest-grid discretization estimate use the last
  three meshes of a complete, consistently refined sequence. Mesh ratios must
  match across all axes and the observations must change monotonically with
  shrinking differences. Near-zero differences or reference values within
  eight floating-point spacings of the largest observation make the estimate
  unavailable. The estimate describes the observed mesh trend and remains
  separate from independent reference checks.
