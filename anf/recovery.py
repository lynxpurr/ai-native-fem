"""Section actions and envelopes on the original planar structure."""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ._linear import checked_arithmetic
from .structures import DISPLACEMENTS, _geometry


SECTION_UNITS = {"axial": "N", "shear": "N", "moment": "N·m"}


@checked_arithmetic
def _section_arrays(model, result, member_line, stations):
    xi = np.asarray(stations)
    values = np.empty((len(model.mesh.member_order), len(stations), 3))
    for index, name in enumerate(model.mesh.member_order):
        member = model.model["members"][name]
        nodes = model.model["nodes"]
        length, _, _ = _geometry(nodes[member["i"]], nodes[member["j"]], f"/model/members/{name}")
        s = length * xi
        first, last = member_line.get(name, ((0.0, 0.0), (0.0, 0.0)))
        axial, shear, moment = result.member_end_actions[index, 0]
        integrals = s[:, None] * ((1 - xi[:, None] / 2) * first + (xi[:, None] / 2) * last)
        values[index, :, :2] = -np.array([axial, shear]) - integrals
        load_moment = s * (((0.5 - xi / 6) * first[1] + (xi / 6) * last[1]) * s)
        values[index, :, 2] = -moment + s * shear + load_moment
    return {f"member_section_{name}": values[..., index]
            for index, name in enumerate(SECTION_UNITS)}


@dataclass(frozen=True, slots=True, eq=False)
class MemberForces:
    member_order: tuple
    stations: tuple
    axial: np.ndarray
    shear: np.ndarray
    moment: np.ndarray


def member_forces(result):
    """Return read-only N/V/M arrays over original members and requested stations."""
    if result.solve_status != "solved":
        raise ValueError("section forces are unavailable for a failed load case")
    return MemberForces(result.member_order, result.stations,
                        *(result.arrays[f"member_section_{name}"] for name in SECTION_UNITS))


@dataclass(frozen=True, slots=True, eq=False)
class Extrema:
    minimum: np.ndarray
    maximum: np.ndarray
    minimum_id: np.ndarray
    maximum_id: np.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class LoadCaseEnvelope:
    status: str
    node_order: tuple
    frame_node_order: tuple
    member_order: tuple
    stations: tuple
    displacement: MappingProxyType | None
    sections: MappingProxyType | None


def _envelope_status(results):
    solved = sum(item.solve_status == "solved" for item in results.values())
    if solved == 0:
        return "not_available"
    return "complete" if solved == len(results) else "partial"


def _extrema(ids, values):
    stack = np.stack(values)
    indices = (stack.argmin(axis=0), stack.argmax(axis=0))
    arrays = [np.take_along_axis(stack, index[None, ...], axis=0)[0] for index in indices]
    arrays.extend(ids[index] for index in indices)
    for array in arrays:
        array.setflags(write=False)
    return Extrema(*arrays)


def _envelope(batch):
    status = batch.envelope_status
    mesh = batch.model.mesh
    nodes = tuple(key[1] for key in mesh.node_order if key[0] == "input")
    frame_nodes = tuple(key[1] for key in mesh.frame_node_order if key[0] == "input")
    positions = (nodes, frame_nodes, mesh.member_order, batch.stations)
    if status == "not_available":
        return LoadCaseEnvelope(status, *positions, None, None)

    solved = [item for item in batch.results.values() if item.solve_status == "solved"]
    ids = np.asarray([item.id for item in solved])
    displacement = {}
    for component in DISPLACEMENTS:
        size = len(frame_nodes if component == "rz" else nodes)
        if size:
            # The mesh places input nodes before generated nodes in both orders.
            values = [item.arrays[f"node_{component}"][:size] for item in solved]
            displacement[component] = _extrema(ids, values)
    sections = {name: _extrema(ids, [item.arrays[f"member_section_{name}"] for item in solved])
                for name in SECTION_UNITS}
    return LoadCaseEnvelope(status, *positions, MappingProxyType(displacement), MappingProxyType(sections))
