# Planar load cases and combinations

Schema 1.2 runs several linear static load sets on the same planar structure.
Use the [planar truss and frame model](frame2d.md) for nodes, members, supports
and subdivision. Declare SI units and a static analysis in the
[1.2 model schema](../schemas/model-v1.2.json).

## Python example

The [cantilever declaration](../examples/structures/load-cases.json) defines
self-weight, a service case with a tip force and a trapezoidal line load, and
one explicit combination. Run the example from the repository root.

```python
from pathlib import Path

from anf.application import run_load_cases
from anf.artifacts import load_load_case_results, save_load_case_results
from anf.recovery import member_forces

model = Path("examples/structures/load-cases.json").read_bytes()
batch = run_load_cases(model, stations=[0, 0.5, 1])
combined = batch.results["combined"]
if combined.solve_status == "solved":
    sections = member_forces(combined)
    print(sections.member_order, sections.stations, sections.moment)
    print(batch.envelope.sections["moment"].minimum_id)

directory = save_load_case_results(batch, Path.cwd(), "load-results")
restored = load_load_case_results(directory)
assert restored.request_hash == batch.request_hash
```

## Define loads and select outcomes

| Field | Input |
| --- | --- |
| `loads.cases` | Nonempty mapping of case IDs to declarations. A case can be empty. |
| `loads.cases.<id>.nodal` | Map of node IDs to global `fx`, `fy` (N) and `mz` (N·m). Use `mz` at frame nodes. |
| `loads.cases.<id>.member_line` | Map of member IDs to `basis` (`local` or `global`) and the `[qx, qy]` vectors `i` and `j` (N/m). Equal end values give a uniform load. The line varies linearly between unequal end values. Trusses accept axial lines. |
| `loads.cases.<id>.self_weight` | Global `acceleration` `[ax, ay]` (m/s²) and a `density` map with positive member densities (kg/m³). Each listed member receives the uniform line load `ρA a`. |
| `loads.combinations` | Optional mapping of combination IDs to nonempty maps of base case IDs and finite coefficients. Coefficients may be zero or negative. |

Case and combination IDs share a namespace and follow the node ID syntax.
Provide up to 32 total. Every explicit reference is checked. The JSON input
limit is 1 MiB. Use `ids=[...]` to select a nonempty subset. By default the
solver runs all declared cases and combinations, in ASCII ID order.

The solver combines nodal and local member line loads before applying the
common supports and prescribed displacements once. For a response `r`
and prescribed displacement `g`, the combination follows
`r(Σ cᵢfᵢ,g) = r(0,g) + Σ cᵢ[r(fᵢ,g) − r(0,g)]`.

## Read sections and envelopes

Stations are normalized distances from directed end `i` to end `j`.
The default stations are `0, 0.25, 0.5, 0.75, 1`. Provide up to 32
distinct stations in `[0, 1]`, in the order you want them reported.
`member_forces` returns `axial`, `shear` and `moment` arrays indexed by
`(member_order, stations)`, in N, N and N·m. The cut reports the actions
the `j` side exerts on the `i` side. Its moment is positive about local
`+z`. The section at `i` is the negative of the `i` end action, and the
section at `j` equals the `j` end action. On reversing a member, compare
complementary stations. Axial force and shear keep their signs and moment
changes sign.

The displacement envelope covers original input nodes. The section
envelope covers original members and requested stations. Each component
carries `minimum`, `maximum`, `minimum_id` and `maximum_id` arrays.
Equal values use the first ID in ASCII order. The envelope status is
`complete` when all selected results solve, `partial` when some fail and
`not_available` when none solve. Failed outcomes retain their diagnostics.
Successful outcomes have `verification_status="solved_unverified"`.

## Save and replay

The [1.2 result schema](../schemas/load-case-results-v1.2.json) describes the
`manifest.json` index. It stores the full normalized declaration, selected
IDs and stations, original and generated node ordering, software versions,
and each outcome's status and diagnostic. Each successful outcome has a
numbered `cases/NNNN/arrays.npz` keyed by its zero-based selection index.
Arrays reuse the [1.1 layout](frame2d.md) and add the three
`member_section_*` fields. The index is written last.

`batch.request_hash` covers the full declaration and the report selection.
Each result's `input_hash` covers the shared physical structure and its
effective nodal and local member loads. This lets a base case and a
combination share a physical fingerprint when their effective loads agree.

Use `run_analysis`, `save_result`, `load_result`, the CLI and the MCP tools
for single-case schemas 1.0 and 1.1. Schema 1.2 uses the Python batch API
shown above. The planar formulation assumes rigid frame connections.

The line-load integration uses linear axial and cubic Hermite displacement
shapes. Section forces follow equilibrium of the cut. [TU Delft element
loads](https://oit.tudelft.nl/CIEM5000/2026/lecture2/element_loads.html) ·
[MIT Euler–Bernoulli beam equilibrium](https://ocw.mit.edu/courses/16-001-unified-engineering-materials-and-structures-fall-2021/mit16_001_f21_lec22lec23lec24.pdf)
