# Frame end releases and rotational springs

Schema 1.3 extends the [planar load case interface](load-cases.md).
For a `frame2d` member, use the optional `end_connections` map to describe
one or both directed ends. Both translations remain shared with the node.

| Entry at `i` or `j` | Rotation |
| --- | --- |
| Omitted | The beam end and node rotate together. |
| `{"kind": "release"}` | The beam end rotates freely and transfers zero end moment. |
| `{"kind": "rotational_spring", "stiffness": 150000}` | The beam end and node may rotate differently. The spring stiffness is positive and in N·m/rad. |

Connections apply at the ends of the original member. Internal subdivisions
are rigid. A node has an external `rz` degree of freedom if any incident
frame end is rigid or spring-connected. Prescribe `rz` or apply `mz` only
at such a node. All frame ends attached by springs to that node use the
same external rotation.

The [example beam](../examples/structures/frame-joints.json) has a spring
between the left beam end and a fixed node rotation. The right end is
released. With a uniform downward load of 500 N/m, the left and right
vertical reactions are 1125 N and 875 N. The spring transmits 500 N·m
and the right end moment is zero.

```python
from pathlib import Path

from anf.application import run_load_cases
from anf.recovery import member_forces

model = Path("examples/structures/frame-joints.json").read_bytes()
result = run_load_cases(model).results["uniform"]
print(result.arrays["member_end_moment"])
print(result.arrays["frame_member_end_rotation"])
print(member_forces(result).moment)
```

In a successful result, `frame_member_end_rotation` gives the beam rotations
in radians at each original frame member's i and j ends. Its rows follow
the frame members in `batch.ordering["member_order"]`. `node_rz` and the
displacement envelope report external node rotations. The spring end
moment equals its stiffness times external rotation minus beam rotation.
The result's strain energy includes the beam and spring storage.

Use the [1.3 model schema](../schemas/model-v1.3.json) and
[1.3 result schema](../schemas/load-case-results-v1.3.json). Load case
selection, combinations, station signs, envelopes and artifact functions
follow the [load case guide](load-cases.md).
