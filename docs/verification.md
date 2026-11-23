# Verification reports

Use the opt-in verification API to check a classical solve against a registered
reference. The existing `solve_cantilever` and `solve_plate` functions still
return their usual result objects and raise on invalid input or numerical
failure.

```python
from anf import Beam, Plate
from anf.verification import VerificationReport, verify_cantilever, verify_plate

beam = Beam(length=2, young_modulus=210e9, second_moment=1e-6)
report = verify_cantilever(beam, tip_force=1000)
assert report.verification_status == "verification_passed"
assert VerificationReport.from_json(report.to_json()) == report

thin = Plate(1, 1, 0.005, 70e9, 0.3, nx=8, ny=8, support="hard_simply_supported")
report = verify_plate(thin, pressure=1000)
assert report.solve_status == "solved"
assert report.verification_status == "verification_failed"
print(report.checks[-1].observed, report.checks[-1].reference, report.checks[-1].error)
```

Pass a model parameter dictionary in place of a `Beam` or `Plate` if you want
constructor validation to appear in the report. Both helpers use SI values and
default to `reference="auto"`, which selects the applicable bundled reference.
Pass `reference=None` for the residual check alone, or pass a registry ID to
request a particular comparison.

To check an existing result, call
`evaluate_checks(result, reference, inputs=input_parameters(model, **loads))`.
Use the model and loads that produced the result. The evaluator checks result
dimensions, extents, finiteness and residual. The two `verify_*` helpers bind
the report directly to the inputs they pass to the solver.

## Reading schema 1.0 and 1.1

| Field | Values and meaning |
| --- | --- |
| `schema_version` | `1.0` for classical beam and plate reports; `1.1` for planar structure reports |
| `diagnostic_code` | Schema 1.1 only. The single structural failure code, or null when the solve completed |
| `input_status` | `valid` or `invalid` |
| `solve_status` | `solved`, `numerical_failure`, or `not_run` for invalid input |
| `verification_status` | `verification_passed` when every required comparison for an eligible independent reference passes, `verification_failed` when an independent comparison fails, `solved_unverified` when independent accuracy remains unestablished, or `not_run` for an unsolved case |
| `scope` | Formulation, support and unit system for a valid input |
| `input_hash` | SHA-256 of the complete SI input dictionary, serialized as compact JSON with sorted keys and finite numbers. Invalid input has no hash. |
| `checks`, `warnings`, `reference_ids` | Named check rows, diagnostics, and the IDs of referenced records |

Each check shows its observable, observed and reference values, physical unit,
metric, measured error, threshold, outcome and evidence ID. Scaled errors show
the denominator in `error_scale`. Skipped checks have outcome `not_run` or
`not_applicable` and no measured values. Solver failures include a diagnostic
and skip the comparisons. The free-DOF residual limit remains `1e-8`.

JSON preserves finite numeric precision. The hash fingerprints inputs. For
reports produced elsewhere, check the solver, inputs and provenance separately.

## Reference registry

`REFERENCE_REGISTRY` is a read-only mapping of bundled IDs to
`ReferenceRecord` values. Each record gives its kind, source, formula location,
applicability, observable units and error budget. Additions go through code and
reference review. The classification includes analytic, manufactured,
independent-numerical, external-comparison and self-regression. A
self-regression check records agreement with a prior result from this solver.

| ID | Observable and accuracy scope |
| --- | --- |
| `cantilever.closed-form.v1` | Tip deflection and slope, root force and moment for a constant-section, one-plane cantilever with a tip force and uniform line load. Relative tolerance is `1e-8`. When a reference is zero, use exact zero for zero contributions or scale absolute error by the largest contributing magnitude. |
| `structure.cantilever-tip.v1` | Tip displacement and rotation, root force and moment for one `+x` `frame2d` member with a fixed i end, a free j end and a tip `fy` only. Relative tolerance is `1e-8`. The planar residual check uses `planar_structure.relative-residual.v1`. |
| `plate.navier-31.v1` | Center deflection of the 0.2 m thick square on a 16 × 16 mesh. Relative error must be below 8%. |
| `plate.thin-q4-navier-31.v1` | The 0.005 m thick square on an 8 × 8 mesh. The same comparison exposes about 98.90% error. This negative benchmark remains ineligible for an accuracy pass if its observed error changes. |
| `plate.thin-q4-snapshot.v1` | Center deflection of the same thin case, compared with the rounded FE value in the [validation record](validation.md) at relative tolerance `1e-8`. Passing this check leaves the solve unverified. |

The plate records apply to `q4_full` and use a hard simply supported 1 m square
with Young's modulus 70 GPa, Poisson ratio 0.3, shear factor 5/6 and uniform
pressure 1000 Pa. Other Q4 inputs within the solver's domain remain unverified
by this registry.

An automatic `mitc4_rectangle` report checks the equilibrium residual and
returns `solved_unverified` after a successful solve. Explicit Q4 references
return `not_applicable` for MITC4. Its independent convergence results are in
the [validation record](validation.md).

Classical reports stay at schema 1.0. Planar structure reports use schema 1.1,
which adds the optional `diagnostic_code`; both versions round-trip through
`VerificationReport.from_json`. A structural numerical or stability failure
carries one code from `unstable_or_ill_conditioned`, `numerical_range` or
`relative_residual_exceeded`. Implicit and explicit `q4_full` produce the
original `reissner_mindlin_q4_full` fingerprint inputs. The nested `model`
dictionary omits the default formulation for those inputs. For MITC4 the
top-level identifier is `reissner_mindlin_mitc4_rectangle` and the nested model
contains `formulation="mitc4_rectangle"`.

The beam reference integrates the Euler–Bernoulli equation and uses force
and moment equilibrium. The independent plate reference minimizes continuous
bending and shear energy for each sine mode and sums odd indices through 31
in each direction. Its `reference_error_bound` is an absolute series truncation
bound in metres. Writing the modal compliances as Laplace integrals makes the
two alternating modal sums separable. Each sum is bounded by its first term
and its tail by the first omitted term. The two omitted strips are therefore
bounded by the sum of the unsigned (33, 1) and (1, 33) terms, about
`2.052e-12 m` for the thick case and `1.141e-9 m` for the thin case.
See the [mathematical basis and citations](methods.md).
