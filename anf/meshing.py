"""Deterministic line and rectangular native meshes; no external generators."""

from decimal import Decimal, localcontext

from .mesh_io import _array, _bounded, _coordinate, _encode, _fail, _name, import_mesh
from .model import _choice, _count, _fields


def generate_supported_mesh(parameters, *, source_sha256=None):
    """Generate a bounded native mesh and record replayable parameters and version."""
    if not isinstance(parameters, dict):
        _fail("/generator", "invalid_type", "expected generation parameters")
    _bounded(parameters)
    if len(_encode(parameters).encode()) > 1024 * 1024:
        _fail("/generator", "size_limit", "generation parameters exceed 1 MiB")
    generator = _choice(parameters.get("generator"), "/generator", ("line", "rectangle"), "unsupported_generator")
    shared = ("generator", "length_unit", "property")
    fields = ("start", "end", "segments", "kind") if generator == "line" else ("origin", "u", "v", "nx", "ny")
    parameters = _fields(parameters, "/generator", (*shared, *fields))
    unit = _choice(parameters["length_unit"], "/generator/length_unit", ("m", "mm"), "unsupported_unit")
    label = _name(parameters["property"], "/generator/property")
    points = {}
    for name in (("start", "end") if generator == "line" else ("origin", "u", "v")):
        points[name] = [_coordinate(value, Decimal("0.001") if unit == "mm" else Decimal(1), f"/generator/{name}/{i}")[0]
                        for i, value in enumerate(_array(parameters[name], f"/generator/{name}", 2, 2))]
    if generator == "line":
        segments = _count(parameters["segments"], "/generator/segments", 1, 199)
        kind = _choice(parameters["kind"], "/generator/kind", ("truss2d", "frame2d"), "unsupported_element")
        nx, ny = segments, 0
        origin = points["start"]
        u = None
        v = [Decimal(0), Decimal(0)]
    else:
        nx, ny = (_count(parameters[key], f"/generator/{key}", 2, 199) for key in ("nx", "ny"))
        if 3 * (nx + 1) * (ny + 1) > 1800:
            _fail("/generator", "size_limit", "plate mesh is limited to 1800 DOFs")
        kind = "quad4"
        origin, u, v = (points[key] for key in ("origin", "u", "v"))
    nodes, cells = [], []
    # 50 significant digits makes decimal interpolation independent of the caller's context.
    with localcontext() as context:
        values = [value for point in points.values() for value in point]
        precision = max(50, max(value.adjusted() for value in values) - min(value.as_tuple().exponent for value in values) + 32)
        if 2 * (nx + 1) * (ny + 1) * precision > 1024 * 1024:
            _fail("/generator", "size_limit", "coordinate expansion exceeds generation budget")
        context.prec = precision
        if generator == "line":
            u = [b - a for a, b in zip(origin, points["end"])]
        for j in range(ny + 1):
            for i in range(nx + 1):
                point = [a + x * Decimal(i) / nx + (y * Decimal(j) / ny if ny else 0)
                         for a, x, y in zip(origin, u, v)]
                nodes.append({"id": j * (nx + 1) + i + 1, "xy": point})
    for j in range(ny if ny else 1):
        for i in range(nx):
            start = j * (nx + 1) + i + 1
            ids = [start, start + 1, start + nx + 2, start + nx + 1] if ny else [start, start + 1]
            cells.append({"id": len(cells) + 1, "kind": kind, "nodes": ids, "property": label})
    groups = {"all_nodes": list(range(1, len(nodes) + 1))}
    if ny:
        groups.update(left=[j * (nx + 1) + 1 for j in range(ny + 1)],
                      right=[(j + 1) * (nx + 1) for j in range(ny + 1)],
                      bottom=list(range(1, nx + 2)), top=list(range(ny * (nx + 1) + 1, len(nodes) + 1)))
    else:
        groups.update(root=[1], tip=[len(nodes)])
    sets = [{"name": name, "kind": "node", "ids": ids} for name, ids in groups.items()]
    sets.append({"name": "all_cells", "kind": "cell", "ids": list(range(1, len(cells) + 1))})
    normalized = {**parameters, **points}
    for key in (("segments",) if generator == "line" else ("nx", "ny")):
        normalized[key] = int(parameters[key])
    document = {"mesh_version": "1.0", "family": "rectangular_grid" if ny else "planar_members",
                "length_unit": unit, "node_count": len(nodes), "cell_count": len(cells),
                "nodes": nodes, "cells": cells, "sets": sets,
                "provenance": {"generator": generator, "version": "1.0", "parameters": normalized,
                               "source_sha256": source_sha256}}
    if ny:
        document["grid"] = {"nx": nx, "ny": ny, "origin": 1, "u_corner": nx + 1, "v_corner": ny * (nx + 1) + 1}
    return import_mesh(document)
