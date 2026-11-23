"""Bounded native meshes projected onto the existing static model contracts."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import re

from .model import ModelValidationError, _choice, _count, _decode_json, _fields, _number, validate_model

_LIMIT = 1024 * 1024
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,31}")
_SI = {"length": "m", "force": "N", "stress": "Pa"}


def _fail(path, code, message):
    raise ModelValidationError(path, code, message)


def _name(value, path):
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        _fail(path, "invalid_name", "expected a portable label of at most 32 characters")
    return value


def _id(value, path):
    return _count(value, path, 1, 2**53 - 1)


def _array(value, path, maximum, minimum=0):
    if not isinstance(value, (list, tuple)):
        _fail(path, "invalid_type", "expected an array")
    if not minimum <= len(value) <= maximum:
        _fail(path, "size_limit", f"expected {minimum} to {maximum} entries")
    return value


def _bounded(value, depth=0, budget=None):
    budget = [0] if budget is None else budget
    size = len(value) if isinstance(value, str) else (len(value.as_tuple().digits) if isinstance(value, Decimal) else 1)
    budget[0] += size + 1
    if budget[0] > _LIMIT:
        _fail("", "size_limit", "native object exceeds its input budget")
    if depth > 8:
        _fail("", "size_limit", "native document exceeds depth 8")
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            _fail("", "invalid_type", "object keys must be strings")
        for key, item in value.items():
            _bounded(key, depth + 1, budget)
            _bounded(item, depth + 1, budget)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _bounded(item, depth + 1, budget)


def _encode(value):
    # Decimal tokens preserve source coordinates across export and SI conversion.
    if isinstance(value, Decimal):
        if not value.is_finite():
            _fail("", "invalid_value", "expected finite JSON numbers")
        return str(value)
    if isinstance(value, dict):
        return "{" + ",".join(json.dumps(k) + ":" + _encode(v) for k, v in sorted(value.items())) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_encode(v) for v in value) + "]"
    try:
        return json.dumps(value, allow_nan=False, ensure_ascii=True)
    except (ValueError, TypeError) as error:
        _fail("", "invalid_value", str(error))


def _coordinate(value, factor, path):
    if type(value) not in (int, float, Decimal):
        _fail(path, "invalid_type", "expected a coordinate number")
    try:
        source = Decimal(str(value))
        if not source.is_finite():
            _fail(path, "invalid_value", "expected finite coordinates")
        # Adjust the exponent exactly, without the caller's Decimal context.
        sign, digits, exponent = source.as_tuple()
        scaled = Decimal((sign, digits, exponent + (0 if factor == 1 else -3)))
        number = _number(scaled, path)
    except (InvalidOperation, OverflowError) as error:
        _fail(path, "invalid_value", str(error))
    return (Decimal(0) if source == 0 else source), (0.0 if number == 0 else number)


def _rectangle(nodes, cells, grid):
    grid = _fields(grid, "/grid", ("nx", "ny", "origin", "u_corner", "v_corner"))
    nx, ny = (_count(grid[key], f"/grid/{key}", 2, 199) for key in ("nx", "ny"))
    if 3 * (nx + 1) * (ny + 1) > 1800:
        _fail("/grid", "size_limit", "plate mesh is limited to 1800 DOFs")
    if len(nodes) != (nx + 1) * (ny + 1) or len(cells) != nx * ny:
        _fail("/grid", "invalid_topology", "expected a complete rectangular grid")
    corners = [_id(grid[key], f"/grid/{key}") for key in ("origin", "u_corner", "v_corner")]
    if len(set(corners)) != 3 or not set(corners) <= nodes.keys():
        _fail("/grid", "unknown_node", "expected three distinct declared corner IDs")
    origin, u, v = (nodes[key] for key in corners)
    ux, uy = u[0] - origin[0], u[1] - origin[1]
    lx = math.hypot(ux, uy)
    if not math.isfinite(lx) or lx == 0:
        _fail("/grid", "unrepresentable_geometry", "u edge must have finite positive length")
    ex = (ux / lx, uy / lx)
    ey = (-ex[1], ex[0])
    vx, vy = v[0] - origin[0], v[1] - origin[1]
    ly = vx * ey[0] + vy * ey[1]
    if not math.isfinite(ly) or ly <= 0:
        _fail("/grid", "invalid_rectangle", "v must point to the positive orthogonal side")
    dx, dy = lx / nx, ly / ny
    tolerance = 1e-10 * min(dx, dy)
    if abs(vx * ex[0] + vy * ex[1]) > tolerance:
        _fail("/grid/v_corner", "invalid_rectangle", "v corner must lie on the perpendicular edge")
    if tolerance == 0 or not math.isfinite(dx * dy / 4) or dx * dy / 4 <= 0:
        _fail("/grid", "unrepresentable_geometry", "cell Jacobian must be finite and positive")
    locations, points, maximum = {}, {}, 0.0
    for source, point in nodes.items():
        x, y = point[0] - origin[0], point[1] - origin[1]
        a, b = x * ex[0] + y * ex[1], x * ey[0] + y * ey[1]
        if not math.isfinite(a) or not math.isfinite(b):
            _fail("/nodes", "unrepresentable_geometry", "local coordinates must be finite")
        if not math.isfinite(a / dx) or not math.isfinite(b / dy):
            _fail("/nodes", "invalid_rectangle", "node is outside the representable lattice")
        ix, iy = round(a / dx), round(b / dy)
        error = math.hypot(a - ix * dx, b - iy * dy)
        if not (0 <= ix <= nx and 0 <= iy <= ny) or error > tolerance:
            _fail("/nodes", "invalid_rectangle", "node is outside the uniform rectangular lattice")
        if (ix, iy) in locations.values():
            _fail("/nodes", "duplicate_node", "nodes project onto the same grid point")
        locations[source] = (ix, iy)
        points[source] = (a, b)
        maximum = max(maximum, error)
    covered = set()
    cell_rows = {}
    for cell in cells:
        ids = cell["nodes"]
        corners_xy = [locations[key] for key in ids]
        ix, iy = min(p[0] for p in corners_xy), min(p[1] for p in corners_xy)
        expected = [(ix, iy), (ix + 1, iy), (ix + 1, iy + 1), (ix, iy + 1)]
        if not any(corners_xy == expected[k:] + expected[:k] for k in range(4)):
            _fail("/cells", "invalid_connectivity", "expected one counterclockwise grid cell")
        if (ix, iy) in covered:
            _fail("/cells", "duplicate_cell", "grid cell occurs more than once")
        covered.add((ix, iy))
        # Actual coordinates, scaled by the nominal cell dimensions, at all four corners.
        anchor = points[ids[0]]
        normalized = [((points[key][0] - anchor[0]) / dx, (points[key][1] - anchor[1]) / dy) for key in ids]
        for r, s in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            signs = ((-1, -1), (1, -1), (1, 1), (-1, 1))
            dr = [a * (1 + b * s) / 4 for a, b in signs]
            ds = [b * (1 + a * r) / 4 for a, b in signs]
            ar, br = (math.fsum(p[i] * w for p, w in zip(normalized, dr)) for i in (0, 1))
            ass, bs = (math.fsum(p[i] * w for p, w in zip(normalized, ds)) for i in (0, 1))
            determinant = ar * bs - ass * br
            if not math.isfinite(determinant) or abs(determinant - 0.25) > 1e-9:
                _fail("/cells", "invalid_jacobian", "cell corner Jacobians must match a rectangle")
        cell_rows[cell["id"]] = iy * nx + ix
    return {"nx": nx, "ny": ny, "lx": lx, "ly": ly, "origin_m": origin, "u_axis": ex, "v_axis": ey,
            "max_correction_m": maximum, "jacobian_m2": dx * dy / 4}, locations, cell_rows


def _parse(document):
    _bounded(document)
    payload = _encode(document).encode("utf-8")
    if len(payload) > _LIMIT:
        _fail("", "size_limit", "native JSON is limited to 1 MiB")
    doc = _fields(document, "", ("mesh_version", "family", "length_unit", "node_count", "cell_count",
                                  "nodes", "cells", "sets"), {"grid": None, "provenance": None})
    _choice(doc["mesh_version"], "/mesh_version", ("1.0",), "unsupported_mesh_version")
    family = _choice(doc["family"], "/family", ("planar_members", "rectangular_grid"), "unsupported_family")
    unit = _choice(doc["length_unit"], "/length_unit", ("m", "mm"), "unsupported_unit")
    factor = Decimal("1" if unit == "m" else "0.001")
    max_nodes, max_cells = (200, 400) if family == "planar_members" else (600, 199 * 199)
    n = _count(doc["node_count"], "/node_count", 2, max_nodes)
    c = _count(doc["cell_count"], "/cell_count", 1, max_cells)
    raw_nodes = _array(doc["nodes"], "/nodes", max_nodes, 2)
    raw_cells = _array(doc["cells"], "/cells", max_cells, 1)
    if len(raw_nodes) != n or len(raw_cells) != c:
        _fail("", "count_mismatch", "declared node/cell counts differ from their arrays")
    nodes, source_nodes, seen = {}, [], set()
    for i, item in enumerate(raw_nodes):
        path = f"/nodes/{i}"
        item = _fields(item, path, ("id", "xy"))
        source = _id(item["id"], path + "/id")
        if source in nodes:
            _fail(path, "duplicate_id", "node ID occurs more than once")
        xy = _array(item["xy"], path + "/xy", 2, 2)
        coords = [_coordinate(v, factor, f"{path}/xy/{j}") for j, v in enumerate(xy)]
        point = tuple(v[1] for v in coords)
        if point in seen:
            _fail(path, "duplicate_node", "SI coordinates collide with another node")
        nodes[source] = point
        source_nodes.append({"id": source, "xy": [v[0] for v in coords]})
        seen.add(point)
    cells, cell_ids, edges, connected, frame_nodes = [], set(), set(), set(), set()
    for i, item in enumerate(raw_cells):
        path = f"/cells/{i}"
        item = _fields(item, path, ("id", "kind", "nodes", "property"))
        source = _id(item["id"], path + "/id")
        if source in cell_ids:
            _fail(path, "duplicate_id", "cell ID occurs more than once")
        kinds = ("truss2d", "frame2d") if family == "planar_members" else ("quad4",)
        kind = _choice(item["kind"], path + "/kind", kinds, "unsupported_element")
        size = 4 if kind == "quad4" else 2
        ids = [_id(v, path + "/nodes") for v in _array(item["nodes"], path + "/nodes", size, size)]
        if len(set(ids)) != size:
            _fail(path, "degenerate_cell", "cell vertices must be distinct")
        if not set(ids) <= nodes.keys():
            _fail(path, "unknown_node", "cell references an undeclared node")
        if frozenset(ids) in edges:
            _fail(path, "duplicate_cell", "cell duplicates the same vertices")
        if size == 2:
            a, b = (nodes[key] for key in ids)
            length = math.hypot(b[0] - a[0], b[1] - a[1])
            if not math.isfinite(length) or length == 0:
                _fail(path, "unrepresentable_geometry", "member length must be finite and positive")
        if kind == "frame2d":
            frame_nodes.update(ids)
        cells.append({"id": source, "kind": kind, "nodes": ids, "property": _name(item["property"], path + "/property")})
        cell_ids.add(source)
        edges.add(frozenset(ids))
        connected.update(ids)
    if connected != nodes.keys():
        _fail("/nodes", "isolated_node", "every node must belong to a cell")
    if family == "planar_members" and 2 * n + len(frame_nodes) > 600:
        _fail("/nodes", "size_limit", "structure is limited to 600 active DOFs")
    sets, names, references = [], set(), 0
    for i, item in enumerate(_array(doc["sets"], "/sets", 64)):
        path = f"/sets/{i}"
        item = _fields(item, path, ("name", "kind", "ids"))
        name = _name(item["name"], path + "/name")
        if name in names:
            _fail(path, "duplicate_name", "set names must be unique")
        kind = _choice(item["kind"], path + "/kind", ("node", "cell"), "unsupported_set")
        ids = [_id(v, path + "/ids") for v in _array(item["ids"], path + "/ids", 4096)]
        references += len(ids)
        if references > 4096:
            _fail("/sets", "size_limit", "sets are limited to 4096 total references")
        if len(set(ids)) != len(ids):
            _fail(path, "duplicate_id", "set members must be unique")
        if not set(ids) <= (nodes.keys() if kind == "node" else cell_ids):
            _fail(path, "unknown_entity", "set references an undeclared entity")
        sets.append({"name": name, "kind": kind, "ids": sorted(ids)})
        names.add(name)
    nodes = dict(sorted(nodes.items()))
    cells.sort(key=lambda item: item["id"])
    basis, locations, cell_rows = None, {}, {}
    if family == "rectangular_grid":
        basis, locations, cell_rows = _rectangle(nodes, cells, doc["grid"])
    elif doc["grid"] is not None:
        _fail("/grid", "unsupported_field", "member meshes do not have a rectangular grid")
    provenance = doc["provenance"]
    if provenance is not None:
        provenance = _fields(provenance, "/provenance", ("generator", "version", "parameters", "source_sha256"))
        _choice(provenance["generator"], "/provenance/generator", ("line", "rectangle"), "unsupported_generator")
        _choice(provenance["version"], "/provenance/version", ("1.0",), "unsupported_generator_version")
        if not isinstance(provenance["parameters"], dict):
            _fail("/provenance/parameters", "invalid_type", "expected generation parameters")
        generator = provenance["generator"]
        expected_family = "planar_members" if generator == "line" else "rectangular_grid"
        if family != expected_family:
            _fail("/provenance/generator", "unsupported_generator", "generator does not match mesh family")
        fields = ("start", "end", "segments", "kind") if generator == "line" else ("origin", "u", "v", "nx", "ny")
        parameters = _fields(provenance["parameters"], "/provenance/parameters",
                             ("generator", "length_unit", "property", *fields))
        _choice(parameters["generator"], "/provenance/parameters/generator", (generator,), "unsupported_generator")
        _choice(parameters["length_unit"], "/provenance/parameters/length_unit", (unit,), "unsupported_unit")
        _name(parameters["property"], "/provenance/parameters/property")
        for key in (("start", "end") if generator == "line" else ("origin", "u", "v")):
            for value in _array(parameters[key], "/provenance/parameters/" + key, 2, 2):
                _coordinate(value, factor, "/provenance/parameters/" + key)
        for key in (("segments",) if generator == "line" else ("nx", "ny")):
            _count(parameters[key], "/provenance/parameters/" + key, 1 if generator == "line" else 2, 199)
        if generator == "line":
            _choice(parameters["kind"], "/provenance/parameters/kind", ("truss2d", "frame2d"), "unsupported_element")
        value = provenance["source_sha256"]
        if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)):
            _fail("/provenance/source_sha256", "invalid_value", "expected a lowercase SHA-256 or null")
    normalized = {**doc, "node_count": n, "cell_count": c, "nodes": sorted(source_nodes, key=lambda item: item["id"]),
                  "cells": cells, "sets": sorted(sets, key=lambda item: item["name"])}
    if doc["grid"] is not None:
        normalized["grid"] = {key: int(value) for key, value in doc["grid"].items()}
    node_ids = {source: f"n{i:04d}" for i, source in enumerate(nodes, 1)}
    frame_order = {source: i for i, source in enumerate(sorted(frame_nodes))}
    mapping = {"nodes": [{"source_id": source, "model_id": node_ids[source] if basis is None else None,
                           "result_index": list(reversed(locations[source])) if basis else i,
                           "frame_result_index": frame_order.get(source) if basis is None else None}
                          for i, source in enumerate(nodes)],
               "cells": [{"source_id": cell["id"], "model_id": f"c{i+1:04d}" if basis is None else None,
                           "result_index": cell_rows[cell["id"]] if basis else i,
                           "source_connection": cell["nodes"],
                           "model_connection": [node_ids[key] for key in cell["nodes"]] if basis is None
                           else [[cell_rows[cell["id"]] // basis["nx"] + oy,
                                  cell_rows[cell["id"]] % basis["nx"] + ox]
                                 for ox, oy in ((0, 0), (1, 0), (1, 1), (0, 1))]}
                          for i, cell in enumerate(cells)], "projection": basis}
    payload = _encode(normalized).encode()
    if len(payload) > _LIMIT:
        _fail("", "size_limit", "normalized native JSON is limited to 1 MiB")
    return payload, tuple(nodes.items()), _encode(mapping).encode()


@dataclass(frozen=True, slots=True, init=False)
class MeshDocument:
    """Validated native mesh with source IDs, decimal coordinates and result mapping."""

    _payload: bytes
    _nodes: tuple
    _mapping: bytes

    def __init__(self, document):
        if isinstance(document, (str, bytes)):
            document = _decode_json(document)
        payload, nodes, mapping = _parse(document)
        object.__setattr__(self, "_payload", payload)
        object.__setattr__(self, "_nodes", nodes)
        object.__setattr__(self, "_mapping", mapping)

    def to_json(self):
        """Export native UTF-8 bytes, preserving source IDs, units and decimal coordinates."""
        return self._payload

    def to_dict(self):
        """Return an independent native object; Decimal coordinates retain source precision."""
        return _decode_json(self._payload)

    @property
    def mapping(self):
        """Return source-to-model identities and result rows (plate rows are [iy, ix])."""
        return json.loads(self._mapping)

    def to_model(self, binding):
        """Bind SI properties and loads, then use the existing single-solve validator."""
        document = self.to_dict()
        cells = document["cells"]
        if document["family"] == "rectangular_grid":
            binding = _fields(binding, "/binding", ("properties", "support", "pressure", "formulation"))
        else:
            binding = _fields(binding, "/binding", ("properties", "supports", "loads"))
        properties = binding["properties"]
        if not isinstance(properties, dict):
            _fail("/binding/properties", "invalid_type", "expected property definitions")
        for label in properties:
            _name(label, "/binding/properties")
        if set(properties) != {cell["property"] for cell in cells}:
            _fail("/binding/properties", "unknown_property", "properties must exactly cover cell labels")
        if document["family"] == "rectangular_grid":
            if len(properties) != 1:
                _fail("/binding/properties", "unsupported_property", "plate requires one uniform property")
            parameters = _fields(next(iter(properties.values())), "/binding/properties",
                                 ("young_modulus", "poisson_ratio", "thickness"), {"shear_factor": 5/6})
            basis = self.mapping["projection"]
            model = {**parameters, **{key: basis[key] for key in ("nx", "ny", "lx", "ly")},
                     "support": binding["support"], "formulation": binding["formulation"]}
            return validate_model({"schema_version": "1.0", "problem_type": "rectangular_plate", "units": _SI,
                                   "model": model, "loads": {"pressure": binding["pressure"]}, "analysis": {"type": "static"}})
        ids = {row["source_id"]: row["model_id"] for row in self.mapping["nodes"]}
        members = {}
        for index, cell in enumerate(cells, 1):
            required = ("young_modulus", "area", "second_moment") if cell["kind"] == "frame2d" else ("young_modulus", "area")
            material = _fields(properties[cell["property"]], "/binding/properties/" + cell["property"], required)
            members[f"c{index:04d}"] = {"kind": cell["kind"], "i": ids[cell["nodes"][0]],
                                       "j": ids[cell["nodes"][1]], "elements": 1, **material}
        sets = {item["name"]: item for item in document["sets"]}

        def assignments(key, components):
            result = {}
            for index, item in enumerate(_array(binding[key], f"/binding/{key}", 64)):
                path = f"/binding/{key}/{index}"
                item = _fields(item, path, ("set", "values"))
                name = _name(item["set"], path + "/set")
                if name not in sets or sets[name]["kind"] != "node":
                    _fail(path + "/set", "unknown_set", "expected a declared node set")
                values = _fields(item["values"], path + "/values", (), dict.fromkeys(components))
                values = {key: _number(value, path + "/values/" + key) for key, value in item["values"].items()}
                if not values:
                    _fail(path, "invalid_value", "expected at least one component")
                for source in sets[name]["ids"]:
                    row = result.setdefault(ids[source], {})
                    for component, value in values.items():
                        if component in row and row[component] != value:
                            _fail(path, "conflicting_assignment", "overlapping sets assign different component values")
                        row[component] = value
            return result

        supports = assignments("supports", ("ux", "uy", "rz"))
        loads = assignments("loads", ("fx", "fy", "mz"))
        return validate_model({"schema_version": "1.1", "problem_type": "planar_structure", "units": _SI,
                               "model": {"nodes": {ids[source]: list(point) for source, point in self._nodes},
                                         "members": members, "supports": supports},
                               "loads": {"nodal": loads}, "analysis": {"type": "static"}})


def import_mesh(source, *, format="native_json"):
    """Read an object, UTF-8 JSON bytes/string, or a regular .json Path (at most 1 MiB)."""
    _choice(format, "/format", ("native_json",), "unsupported_format")
    if isinstance(source, MeshDocument):
        return source
    if isinstance(source, Path):
        from .artifacts import _read_regular

        if source.suffix.lower() != ".json":
            _fail("/format", "unsupported_format", "only native .json files are supported")
        source = _read_regular(source, _LIMIT)
    if isinstance(source, bytes) and source.startswith((b"PK", b"\x1f\x8b", b"BZh", b"\xfd7zXZ")):
        _fail("/format", "unsupported_format", "compressed inputs are not supported")
    return MeshDocument(source)


def validate_topology(document):
    """Validate native topology without solving or changing the existing physical fingerprint."""
    return import_mesh(document)
