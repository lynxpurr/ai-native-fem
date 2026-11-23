# Native mesh input

`anf.mesh_io` reads native UTF-8 JSON for planar trusses/frames and complete,
uniform rectangular plate grids. `anf.meshing` generates straight lines and
rectangles using the same topology validator. Both are Python interfaces.

```python
from anf.meshing import generate_supported_mesh
from anf.mesh_io import import_mesh
from anf.application import run_analysis

mesh = generate_supported_mesh({
    "generator": "line", "length_unit": "mm", "property": "steel",
    "start": [0, 0], "end": [2000, 0], "segments": 1, "kind": "frame2d",
})
binding = {
    "properties": {"steel": {"young_modulus": 210e9, "area": .01, "second_moment": 1e-6}},
    "supports": [{"set": "root", "values": {"ux": 0, "uy": 0, "rz": 0}}],
    "loads": [{"set": "tip", "values": {"fy": 1000}}],
}
restored = import_mesh(mesh.to_json())
result = run_analysis(restored.to_model(binding))
print(result.solve_status, mesh.mapping)
```

Use `import_mesh(Path("mesh.json"))` for a file. String arguments are JSON text.
`validate_topology(document)` returns the same validated `MeshDocument` without
binding or solving. `to_json()` exports native bytes; `to_dict()` returns a fresh
object with `Decimal` coordinates to preserve their source precision. Use
`to_json()` when writing a mesh instead of passing that object to `json.dumps`.
Mesh and model versions are independent.

## Native document 1.0

The required fields are `mesh_version: "1.0"`, `family`, `length_unit`,
`node_count`, `cell_count`, `nodes`, `cells`, and `sets`. A member mesh can be:

```json
{
  "mesh_version": "1.0", "family": "planar_members", "length_unit": "m",
  "node_count": 2, "cell_count": 1,
  "nodes": [{"id": 9001, "xy": [0, 0]}, {"id": 11, "xy": [2, 0]}],
  "cells": [{"id": 77, "kind": "frame2d", "nodes": [9001, 11], "property": "steel"}],
  "sets": [{"name": "root", "kind": "node", "ids": [9001]},
           {"name": "tip", "kind": "node", "ids": [11]}]
}
```

Source IDs are numeric integers from 1 through 2^53−1. Node and cell ID domains
are separate. Booleans, numeric strings, duplicate IDs, repeated vertices,
duplicate coordinates, duplicate cells in either direction, isolated nodes and
unknown references are rejected. Set and property names match
`[A-Za-z][A-Za-z0-9_-]{0,31}`. Set names are unique across node/cell kinds;
members are unique declared IDs. Unreferenced defined sets survive export.
Unknown fields, duplicate JSON keys and unsupported elements are rejected.

Length units are `m` or `mm`. Decimal coordinates are scaled to SI before
conversion to binary64. Nonfinite values, underflow and coordinate collisions
are rejected. Material, section, load and displacement bindings use SI.

Member properties contain `young_modulus` and `area`; `frame2d` also requires
`second_moment`. Every cell label must have exactly one property definition,
and unused property definitions are rejected. Each cell maps to one schema 1.1
member with `elements=1`, preserving its source i→j local direction.
`supports` and `loads` reference node sets and use existing global
`ux/uy/rz` and `fx/fy/mz` components. Overlapping assignments with different
values for the same node/component are rejected. Rotate loads and supports
explicitly when moving a mesh; only constraints expressible by the current
global-component contract are supported. A valid mesh can still be unstable
or fail numerical checks: inspect the normal analysis status and diagnostics.

## Rectangles

`rectangular_grid` uses `quad4` cells and adds
`grid: {nx, ny, origin, u_corner, v_corner}`. Corner values are source node IDs;
the u edge defines the local x axis and the v edge must point along the positive
perpendicular axis. The grid must contain every uniformly spaced point and
every cell. Counterclockwise cyclic shifts are allowed. Clockwise, crossed,
sheared, nonuniform, incomplete and degenerate grids are rejected, even when
a distorted cell has positive Jacobians.

Projection onto ideal local grid points permits a maximum deviation of
`1e-10 * min(dx, dy)` metres. Normalized corner Jacobians must be within
`1e-9` of 1/4; the physical determinant must be finite and positive.
`mesh.mapping["projection"]` records the origin, axes, dimensions and maximum
correction. The resulting declaration uses the existing schema 1.0 plate
contract and its limits. It supports one uniform material/thickness, a uniform
pressure, and the existing perimeter supports and formulations. Arbitrary
nodal supports and local pressures are rejected by the binding contract.

Generate a plate with:

```python
plate = generate_supported_mesh({
    "generator": "rectangle", "length_unit": "m", "property": "plate",
    "origin": [0, 0], "u": [2, 0], "v": [0, 1], "nx": 8, "ny": 8,
})
model = plate.to_model({
    "properties": {"plate": {"young_modulus": 210e9, "poisson_ratio": .3, "thickness": .01}},
    "pressure": 1000, "support": "hard_simply_supported", "formulation": "mitc4_rectangle",
})
```

## Mapping, replay and limits

`mapping["nodes"]` and `mapping["cells"]` contain source IDs, canonical model IDs
and zero-based result indexes. Member node indexes refer to `node_order`;
`frame_result_index` separately refers to `frame_node_order` for rotations and
moments (null on truss-only nodes). Cell indexes refer to `member_order`.
Plate node indexes are `[iy, ix]`; cell indexes use row-major order.
`source_connection` retains input orientation and `model_connection` records
canonical endpoints or ideal plate corners. These mappings use the existing
result-array conventions; plate rotations and fields are in the projected
local frame. Map them through the recorded axes when comparing global fields.

Records are sorted by numeric source ID; record/key order and equivalent
integer spellings preserve the canonical model. Reassigning source IDs can
change canonical IDs and the structural fingerprint: compare results through
the source mapping. Mesh sets and provenance do not enter the existing
physical input hash.

The generators create `all_nodes`/`all_cells`, plus `root`/`tip` for lines or
`left`/`right`/`bottom`/`top` for rectangles. Optional `provenance` records
generator, version `1.0`, normalized parameters and `source_sha256` (null or
a lowercase SHA-256). Replay with
`generate_supported_mesh(provenance["parameters"], source_sha256=provenance["source_sha256"])`.
Imported metadata is a declaration, not proof of authenticity.

Input and normalized JSON are limited to 1 MiB, nesting to 8, sets to 64 and
total set references to 4096. Member meshes have at most 200 nodes, 400 cells
and 600 active DOFs. Rectangles require nx/ny from 2 to 199 and at most
1800 DOFs. Generator dimensions and coordinate expansion are checked before
enumeration. File input reads at most the budget plus one byte. File access
failures raise `ArtifactError`; invalid declarations raise
`ModelValidationError` with `path`, `code`, and `message`.

Compressed input and external formats are rejected with `unsupported_format`;
no decompression, Gmsh dependency or external generator is bundled.

The implementation uses the existing public solvers and the affine rectangular
mapping with `det(J)=dx*dy/4`. [Bathe's MIT finite element lectures](https://ocw.mit.edu/courses/2-094-finite-element-analysis-of-solids-and-fluids-ii-spring-2011/116df281a310a9ccdbbb8dd7ec7a2d6e_MIT2_094S11_2094_lectures.pdf)
describe the isoparametric mapping. Accuracy and formulation applicability
remain those of the [plate](verification.md) and [planar structure](frame2d.md)
solvers.
