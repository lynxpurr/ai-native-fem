"""Validation, meshing and small dense statics for planar structures."""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from ._linear import MAX_RELATIVE_RESIDUAL, checked_arithmetic, strain_energy
from .elements import Frame2D, Truss2D
from .model import _SI, _STRUCTURE, ModelValidationError, _choice, _count, _fields, _number, _pointer, _positive


DISPLACEMENTS = ("ux", "uy", "rz")
FORCES = ("fx", "fy", "mz")
_ORDER_KEYS = ("node_order", "frame_node_order", "member_order")
_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,31}")
MAX_NODES, MAX_SEGMENTS, MAX_DOFS, MAX_MEMBER_ELEMENTS = 200, 400, 600, 32


def _named(value, path, minimum=0, maximum=None):
    if not isinstance(value, Mapping):
        raise ModelValidationError(path, "invalid_type", "expected an object")
    if len(value) < minimum:
        raise ModelValidationError(path, "invalid_value", f"expected at least {minimum} entries")
    if maximum is not None and len(value) > maximum:
        raise ModelValidationError(path, "size_limit", f"expected at most {maximum} entries")
    for name in value:
        if not isinstance(name, str) or not _ID.fullmatch(name):
            raise ModelValidationError(_pointer(path, name), "invalid_id", "invalid node or member ID")
    return sorted(value.items())


def _continuous(value, path):
    number = _number(value, path)
    return 0.0 if number == 0 else number


def _nodes(entries):
    nodes = {}
    seen = set()
    for name, value in _named(entries, "/model/nodes", 2, MAX_NODES):
        path = f"/model/nodes/{name}"
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ModelValidationError(path, "invalid_type", "expected coordinates [x, y]")
        point = tuple(_continuous(component, f"{path}/{i}") for i, component in enumerate(value))
        if point in seen:
            raise ModelValidationError(path, "duplicate_node", "coordinates duplicate another node")
        nodes[name] = point
        seen.add(point)
    return nodes


def _known(name, entries, path, kind):
    if not isinstance(name, str) or name not in entries:
        raise ModelValidationError(path, f"unknown_{kind}", f"expected a declared {kind} ID")
    return name


def _members(entries, nodes):
    members = {}
    connected = set()
    frame_nodes = set()
    edges = set()
    for name, value in _named(entries, "/model/members", 1, MAX_SEGMENTS):
        path = f"/model/members/{name}"
        base = ("kind", "i", "j", "young_modulus", "area")
        value = _fields(value, path, base, {"elements": 1, "second_moment": None})
        kind = _choice(value["kind"], f"{path}/kind", ("truss2d", "frame2d"),
                       "unsupported_element")
        i, j = (_known(value[end], nodes, f"{path}/{end}", "node") for end in ("i", "j"))
        if i == j:
            raise ModelValidationError(f"{path}/j", "zero_length", "endpoints must be distinct")
        edge = frozenset((i, j))
        if edge in edges:
            raise ModelValidationError(path, "duplicate_member", "endpoints duplicate another member")
        edges.add(edge)
        connected.update(edge)
        is_frame = kind == "frame2d"
        fields = (*base, "second_moment") if is_frame else base
        value = _fields(entries[name], path, fields, {"elements": 1})
        for field in fields[3:]:
            value[field] = _positive(value[field], f"{path}/{field}")
        maximum = MAX_MEMBER_ELEMENTS if is_frame else 1
        value["elements"] = _count(value["elements"], f"{path}/elements", 1, maximum)
        members[name] = value
        if is_frame:
            frame_nodes.update(edge)
    for name in nodes:
        if name not in connected:
            raise ModelValidationError(f"/model/nodes/{name}", "isolated_node",
                                       "node must belong to a member")
    return members, frame_nodes


def _components(entries, path, nodes, frame_nodes, components, *, supports=False):
    normalized = {}
    for name, value in _named(entries, path):
        node_path = f"{path}/{name}"
        _known(name, nodes, node_path, "node")
        _fields(value, node_path, (), dict.fromkeys(components))
        if supports and not value:
            raise ModelValidationError(node_path, "invalid_value", "expected at least one constraint")
        row = {}
        for key, item in value.items():
            field = f"{node_path}/{key}"
            if key == components[2] and name not in frame_nodes:
                raise ModelValidationError(field, "inactive_dof", "node has no rotational DOF")
            number = _continuous(item, field)
            if supports or number:
                row[key] = number
        if row:
            normalized[name] = row
    return normalized


def _structure(parameters, loads):
    parameters = _fields(parameters, "/model", ("nodes", "members", "supports"))
    nodes = _nodes(parameters["nodes"])
    members, frame_nodes = _members(parameters["members"], nodes)
    supports = _components(parameters["supports"], "/model/supports", nodes, frame_nodes,
                           DISPLACEMENTS, supports=True)
    loads = _fields(loads, "/loads", (), {"nodal": {}, "uniform_transverse": {}})
    nodal = _components(loads["nodal"], "/loads/nodal", nodes, frame_nodes, FORCES)
    uniform = {}
    for name, value in _named(loads["uniform_transverse"], "/loads/uniform_transverse"):
        path = f"/loads/uniform_transverse/{name}"
        _known(name, members, path, "member")
        if members[name]["kind"] != "frame2d":
            raise ModelValidationError(path, "unsupported_load", "transverse loads require a frame")
        number = _continuous(value, path)
        if number:
            uniform[name] = number
    model = {"nodes": nodes, "members": members, "supports": supports}
    return model, {"nodal": nodal, "uniform_transverse": uniform}, _mesh(nodes, members, frame_nodes)


def _geometry(start, end, path):
    dx, dy = (b - a for a, b in zip(start, end))
    length = math.hypot(dx, dy)
    if not math.isfinite(length) or length == 0:
        raise ModelValidationError(path, "unrepresentable_geometry",
                                   "segment coordinates must give a finite positive length")
    return length, dx / length, dy / length


@dataclass(frozen=True)
class _Segment:
    member: str
    index: int
    nodes: tuple
    geometry: tuple


@dataclass(frozen=True)
class StructureMesh:
    """Deterministic node and member order, segment geometry and active DOFs."""

    node_order: tuple
    frame_node_order: tuple
    member_order: tuple
    coordinates: tuple
    node_dofs: tuple
    segments: tuple
    longest_member: float
    dof_count: int

    def ordering(self):
        return {
            key: [list(entry) if isinstance(entry, tuple) else entry for entry in getattr(self, key)]
            for key in _ORDER_KEYS
        }


def _mesh(nodes, members, frame_nodes):
    segment_count = sum(member["elements"] for member in members.values())
    generated = segment_count - len(members)
    node_count = len(nodes) + generated
    dof_count = 2 * node_count + len(frame_nodes) + generated
    if node_count > MAX_NODES or segment_count > MAX_SEGMENTS or dof_count > MAX_DOFS:
        raise ModelValidationError("/model", "size_limit",
                                   "mesh limited to 200 nodes, 400 segments and 600 DOFs")
    node_order = [("input", name) for name in nodes]
    coordinates = list(nodes.values())
    node_indices = {name: index for index, name in enumerate(nodes)}
    segments = []
    longest = 0.0
    for name, member in members.items():
        i, j = member["i"], member["j"]
        count = member["elements"]
        start, end = nodes[i], nodes[j]
        path = f"/model/members/{name}"
        longest = max(longest, _geometry(start, end, path)[0])
        indices = [node_indices[i]]
        for step in range(1, count):
            fraction = step / count
            point = tuple(a + fraction * (b - a) for a, b in zip(start, end))
            indices.append(len(coordinates))
            coordinates.append(point)
            node_order.append(("generated", name, step))
        indices.append(node_indices[j])
        for index, (left, right) in enumerate(pairwise(indices)):
            geometry = _geometry(coordinates[left], coordinates[right], path)
            segments.append(_Segment(name, index, (left, right), geometry))

    frame_order = tuple(key for key in node_order if key[0] == "generated" or key[1] in frame_nodes)
    rotations = set(frame_order)
    node_dofs = []
    offset = 0
    for key in node_order:
        size = 3 if key in rotations else 2
        node_dofs.append(tuple(range(offset, offset + size)))
        offset += size
    return StructureMesh(tuple(node_order), frame_order, tuple(members), tuple(coordinates),
                         tuple(node_dofs), tuple(segments), longest, dof_count)


class StructureSolveError(ArithmeticError):
    """A stable machine diagnostic for a failed structural solve."""

    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class StructureResult:
    mesh: StructureMesh
    displacement: np.ndarray
    reactions: np.ndarray
    member_end_actions: np.ndarray
    strain_energy: float
    relative_residual: float


def _element(member, geometry):
    length, c, s = geometry
    young = member["young_modulus"]
    axial = young * member["area"]
    if member["kind"] == "truss2d":
        return Truss2D(length, axial, c, s)
    return Frame2D(length, axial, c, s, bending_rigidity=young * member["second_moment"])


@checked_arithmetic
def _equilibrate(matrix, rhs):
    if not np.isfinite(matrix).all() or not np.isfinite(rhs).all():
        raise ArithmeticError("non-finite free stiffness or effective load")
    diagonal = np.diag(matrix)
    if np.any(diagonal <= 0):
        raise StructureSolveError("unstable_or_ill_conditioned", "non-positive free stiffness diagonal")
    scales = 1 / np.sqrt(diagonal)
    symmetric = (scales[:, None] * matrix) * scales
    try:
        eigenvalues = np.linalg.eigvalsh(symmetric)
        if not np.isfinite(eigenvalues).all():
            raise ArithmeticError("non-finite equilibrated stiffness spectrum")
        limit = 64 * np.finfo(float).eps * len(rhs) * eigenvalues[-1]
        if eigenvalues[0] <= limit:
            raise StructureSolveError(
                "unstable_or_ill_conditioned",
                f"smallest scaled stiffness eigenvalue {eigenvalues[0]:.6g} is at or below {limit:.6g}",
            )
        return scales * np.linalg.solve(symmetric, scales * rhs)
    except np.linalg.LinAlgError as error:
        raise StructureSolveError("unstable_or_ill_conditioned",
                                  "free stiffness factorization failed") from error


def _residual(matrix, displacement, rhs, rotational, length):
    gap = matrix @ displacement - rhs
    if not np.isfinite(gap).all():
        raise ArithmeticError("non-finite free force balance")
    if not np.any(rhs):
        return 0.0 if not np.any(gap) else math.inf
    # Scale each vector in force-equivalent units without squaring raw values.
    # frexp keeps the moment-to-force conversion representable until both
    # vectors have been normalized by a common power of two.
    length_fraction, length_exponent = math.frexp(length)
    scaled = []
    for vector in (gap, rhs):
        fraction, exponent = np.frexp(vector)
        fraction[rotational] /= length_fraction
        exponent[rotational] -= length_exponent
        scaled.append((fraction, exponent))
    power = max(int(np.max(exponent[fraction != 0])) for fraction, exponent in scaled
                if np.any(fraction))
    norms = [np.linalg.norm(np.ldexp(fraction, exponent - power)) for fraction, exponent in scaled]
    return float(norms[0] / norms[1]) if norms[1] else math.inf


@checked_arithmetic
def _solve(model):
    mesh = model.mesh
    stiffness = np.zeros((mesh.dof_count, mesh.dof_count))
    force = np.zeros(mesh.dof_count)
    displacement = np.zeros_like(force)
    reactions = np.zeros_like(force)
    end_actions = np.zeros((len(mesh.member_order), 2, 3))

    node_indices = {key[1]: index for index, key in enumerate(mesh.node_order) if key[0] == "input"}
    member_indices = {name: index for index, name in enumerate(mesh.member_order)}
    for name, row in model.loads["nodal"].items():
        for component, value in row.items():
            force[mesh.node_dofs[node_indices[name]][FORCES.index(component)]] = value

    contributions = []
    for segment in mesh.segments:
        member = model.model["members"][segment.member]
        element = _element(member, segment.geometry)
        is_frame = isinstance(element, Frame2D)
        width = 3 if is_frame else 2
        dofs = np.array([dof for node in segment.nodes for dof in mesh.node_dofs[node][:width]])
        transverse = model.loads["uniform_transverse"].get(segment.member, 0.0)
        stiffness[np.ix_(dofs, dofs)] += element.stiffness()
        if is_frame:
            force[dofs] += element.transformation().T @ element.local_load(transverse)
        contributions.append((segment, dofs, element, transverse))

    fixed = []
    for name, row in model.model["supports"].items():
        for component, value in row.items():
            dof = mesh.node_dofs[node_indices[name]][DISPLACEMENTS.index(component)]
            displacement[dof] = value
            fixed.append(dof)
    free = np.setdiff1d(np.arange(mesh.dof_count), fixed)
    residual = 0.0
    if free.size:
        matrix = stiffness[np.ix_(free, free)]
        rhs = force[free] - stiffness[np.ix_(free, fixed)] @ displacement[fixed]
        displacement[free] = _equilibrate(matrix, rhs)
        rotational = np.isin(free, [dofs[2] for dofs in mesh.node_dofs if len(dofs) == 3])
        residual = _residual(matrix, displacement[free], rhs, rotational, mesh.longest_member)
        if not math.isfinite(residual) or residual > MAX_RELATIVE_RESIDUAL:
            raise StructureSolveError("relative_residual_exceeded",
                                      "free-DOF relative residual exceeds 1e-8")
    reactions[fixed] = (stiffness @ displacement - force)[fixed]
    for segment, dofs, element, transverse in contributions:
        actions = (element.end_actions(displacement[dofs], transverse) if isinstance(element, Frame2D)
                   else element.end_actions(displacement[dofs]))
        index = member_indices[segment.member]
        if segment.index == 0:
            end_actions[index, 0] = actions[0]
        if segment.index == model.model["members"][segment.member]["elements"] - 1:
            end_actions[index, 1] = actions[1]

    energy = strain_energy(stiffness, displacement)
    if not all(np.isfinite(array).all() for array in (force, stiffness, displacement, reactions, end_actions)):
        raise ArithmeticError("non-finite structure result")
    return StructureResult(mesh, displacement, reactions, end_actions, energy, residual)


def solve_structure(model):
    """Solve a validated planar-structure ModelSpec with the dense backend."""
    if model.schema_version != "1.1":
        raise ValueError("solve_structure requires a planar structure declaration")
    try:
        return _solve(model)
    except StructureSolveError:
        raise
    except (ArithmeticError, ValueError) as error:
        raise StructureSolveError("numerical_range", str(error)) from error


def _structure_model_schema():
    """Return the portable declaration shape shared by the file and MCP schemas."""
    name = {"type": "string", "pattern": f"^{_ID.pattern}(?![\\s\\S])"}
    number = {"type": "number"}
    positive = {**number, "exclusiveMinimum": 0}

    def obj(properties, optional=(), **extra):
        return {"type": "object", "properties": properties, "additionalProperties": False,
                "required": [key for key in properties if key not in optional], **extra}

    def named(schema, **extra):
        return {"type": "object", "propertyNames": name, "additionalProperties": schema, **extra}

    member = {"i": name, "j": name, "young_modulus": positive, "area": positive}
    elements = {"type": "integer", "minimum": 1, "default": 1}
    truss = obj({"kind": {"const": "truss2d"}, **member,
                 "elements": {**elements, "maximum": 1}}, ("elements",))
    frame = obj({"kind": {"const": "frame2d"}, **member, "second_moment": positive,
                 "elements": {**elements, "maximum": MAX_MEMBER_ELEMENTS}}, ("elements",))
    point = {"type": "array", "minItems": 2, "maxItems": 2, "items": number}
    return obj({
        "schema_version": {"const": "1.1"},
        "problem_type": {"const": _STRUCTURE},
        "units": obj({key: {"const": value} for key, value in _SI.items()}),
        "analysis": obj({"type": {"const": "static"}}),
        "model": obj({
            "nodes": named(point, minProperties=2, maxProperties=MAX_NODES),
            "members": named({"oneOf": [truss, frame]}, minProperties=1, maxProperties=MAX_SEGMENTS),
            "supports": named(obj(dict.fromkeys(DISPLACEMENTS, number), DISPLACEMENTS, minProperties=1)),
        }),
        "loads": obj({
            "nodal": named(obj(dict.fromkeys(FORCES, number), FORCES)),
            "uniform_transverse": named(number),
        }, ("nodal", "uniform_transverse")),
    }, description="Runtime validation checks IDs, connectivity, active DOFs, geometry and mesh budgets.")
