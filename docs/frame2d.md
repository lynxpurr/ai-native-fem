# Planar trusses and frames

ANF solves small, linear elastic planar structures using dense NumPy matrices.
A straight, prismatic `truss2d` member carries axial force. A straight,
prismatic `frame2d` member combines axial response with Euler–Bernoulli
bending. A model has one static load set and small displacements. This
formulation assumes rigid frame joints and neglects shear deformation.

## Run a model

The [cantilever](../examples/structures/cantilever.json) has a 2 m frame
member divided into four elements and a downward 1000 N tip force.

```sh
anf validate examples/structures/cantilever.json --json
anf solve examples/structures/cantilever.json --output cantilever-result \
  --reference structure.cantilever-tip.v1 --json
anf inspect cantilever-result --json
```

The expected tip displacement is approximately `-0.0126984127 m`, the
tip rotation is `-0.00952380952 rad`, and the root reactions are
`+1000 N` vertically and `+2000 N·m` about the z axis.

```python
from pathlib import Path

from anf.application import run_analysis

model = Path("examples/structures/cantilever.json").read_bytes()
result = run_analysis(model, {"reference": "structure.cantilever-tip.v1"})
tip = result.ordering["node_order"].index(["input", "tip"])
print(result.verification_status, result.arrays["node_uy"][tip])
```

The [portal](../examples/structures/portal.json) uses a fixed base and a
vertical roller. A 1000 N horizontal load gives about 259.593 N upward
reaction at the roller. The [settlement example](../examples/structures/settlement.json)
imposes a 2 mm horizontal displacement at the right end of a guided bar
chain. The middle node moves 1 mm and the two supports develop opposing
1000 N reactions.

`anf-mcp` uses the same model declarations for `validate_model` and
`solve_model`. `inspect_artifact` reads the saved result by local artifact ID.
The tool replies return bounded summaries, with the arrays and full ordering
available in the artifact directory. Convergence studies remain available
for beam and plate declarations.

## Model schema 1.1

Use `problem_type="planar_structure"`, explicit SI units
`{"length":"m","force":"N","stress":"Pa"}`, and `analysis.type="static"`.
The [model schema](../schemas/model-v1.1.json) describes the JSON fields.

| Field | Meaning |
| --- | --- |
| `model.nodes` | Node IDs mapped to `[x,y]` in metres. |
| `model.members` | Member IDs mapped to `kind`, directed node IDs `i` and `j`, positive `young_modulus` (Pa) and `area` (m²). A frame also requires positive `second_moment` (m⁴). |
| `elements` on each member | Defaults to 1. Frames allow 1–32 equal subdivisions. Build a subdivided truss with explicit nodes, members and supports. |
| `model.supports` | Node IDs mapped to one or more prescribed global components `ux`, `uy` (m) and, for frame nodes, `rz` (rad). A component set to zero is constrained. |
| `loads.nodal` | Node IDs mapped to global `fx`, `fy` (N) or `mz` (N·m). Missing components are zero. |
| `loads.uniform_transverse` | Frame member IDs mapped to a constant local `+y` line load in N/m. |

Node IDs and member IDs use a letter followed by up to 31 ASCII letters,
digits, underscores or hyphens. Members connect only at shared explicit
nodes. Generated frame subdivisions carry translations and rotations.
At shared truss/frame nodes, the two types share the translations.
A truss-only node has no rotational DOF. Duplicate coordinates and
duplicate undirected member endpoints are rejected.

The mesh budget is 200 nodes, 400 segments and 600 active DOFs, including
generated nodes. The JSON input limit is 1 MiB. The input hash covers the
normalized physical model, including directed endpoints, constraints and
subdivision. Equivalent defaults and omitted zero loads give the same hash.

## Results and sign convention

The local x axis points from `i` to `j`. Local y is 90° counterclockwise
from local x and positive rotation is about the global +z axis.
Local member ends report actions applied by the nodes to the member,
both resolved in the same local axes. A tensile truss therefore reports
`[-N,+N]`. Reversing a member swaps the ends and flips their axial and
transverse components. Its end moments retain their sign about +z. Flip
the supplied local transverse load to preserve its global direction.

The [result schema](../schemas/result-v1.1.json) defines `manifest.json`
and the metadata for `arrays.npz`. A successful result contains:

| Arrays | Ordering and unit |
| --- | --- |
| `node_ux`, `node_uy` | `node_order`, m |
| `reaction_fx`, `reaction_fy` | `node_order`, N |
| `node_rz`, `reaction_mz` | `frame_node_order`, rad and N·m, present when there are frame DOFs |
| `member_end_axial`, `member_end_shear`, `member_end_moment` | `member_order`, columns `[i,j]`, N, N and N·m |

Original nodes use keys `["input",id]` in ASCII order. Generated nodes
follow as `["generated",member_id,k]` in member and subdivision order,
with `k` starting at 1. `frame_node_order` is the rotational subset of
`node_order`. Members use their declared IDs in ASCII order. The reported
member ends correspond to the outermost segment ends. Free DOFs report zero
reaction, and truss ends report zero shear and moment.

A mechanism or ill-conditioned free system yields
`unstable_or_ill_conditioned`. Range and residual failures use
`numerical_range` and `relative_residual_exceeded`. These outcomes write
a manifest with diagnostics and no arrays. Solved results without applicable
independent evidence have verification status `solved_unverified`.
The cantilever reference applies to one frame pointing along global +x
whose i end is fixed at zero and whose free j end has only a nonzero
vertical nodal force.

Beam and plate declarations and their result files use
[model and result schema 1.0](model-format.md). Their fingerprints stay
compatible with existing files.

## Element and solver equations

`anf.elements.Truss2D` exposes the four-DOF global stiffness and local end
actions. `anf.elements.Frame2D` exposes the six-DOF stiffness, rotation
matrix, consistent load vector and local end actions. Inputs for the element
constructors are length, axial rigidity, and direction cosine and sine.
Supply bending rigidity by keyword for a frame. For the model-level
low-level solve, pass a validated `ModelSpec` to
`anf.structures.solve_structure`. Use `run_analysis` to obtain a
versioned bundle with verification and artifact support.

For a bar with direction cosines `c,s`, the extension is
`[-c,-s,c,s] @ d`, and the axial energy is `EA/(2L)` times the square of
that extension. For a frame, the transverse field is cubic Hermite, fixed
by its endpoint deflections and slopes. Differentiating its bending energy
`½ ∫ EI(v″)² dx` with respect to the DOFs gives the bending stiffness.
Integrating the four shape functions against constant `q` gives consistent
local transverse forces `qL/2` at both ends and moments
`+qL²/12` and `-qL²/12`. The local axial and bending blocks are rotated
and assembled into the common global DOF table. Recover end actions by
subtracting each consistent local load from its local stiffness times its
local displacement vector. [TU Delft, matrix statics and frame stiffness](https://oit.tudelft.nl/CIEM5000/2026/lecture1/other_elements.html) · [Consistent element loads](https://oit.tudelft.nl/CIEM5000/2026/lecture2/element_loads.html)

With prescribed values `g`, the solver forms the effective free load
`b = p_f - K_fc g`. It solves the diagonally equilibrated free system
after checking its scaled stiffness eigenvalues. Support reactions come
from the constrained entries of `Kd - p`. The free residual uses
force-equivalent row scaling, dividing moments by the longest declared
member length before forming the relative norm. The acceptance limit is
`1e-8`. A zero effective load requires zero free residual.
[Nonzero Dirichlet conditions](https://oit.tudelft.nl/CIEM5000/2026/lecture2/direchlet.html)

The acceptance cases use the axial bar solution, the cantilever tip formulas
`PL³/(3EI)` and `PL²/(2EI)`, and static root reactions `-P` and `-PL`.
The simply supported uniform-load case checks the Hermite midpoint,
reactions, slopes and energy under refinement. The portal checks a
redundant reaction independently from force-method compatibility, using
both axial and bending flexibility. Additional cases cover mixed truss/frame
DOFs, coordinate and orientation changes, settlement work and two mechanisms.
[MIT 16.21, principle of virtual forces and unit loads](https://ocw.mit.edu/courses/16-21-techniques-for-structural-analysis-and-design-spring-2005/b0b7cf7007222223c5ef14b2d965798c_unit8_notes2.pdf)
