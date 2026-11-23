"""Schema 1.2 load declarations, effective loads and request identities."""

from dataclasses import dataclass
from fractions import Fraction
import math
from collections.abc import Mapping

from .model import ModelValidationError, _choice, _fields, _number, _pointer, _positive
from .structures import FORCES, _components, _continuous, _geometry, _known, _named
from .verification import _describe_case


MAX_RESULTS = 32
MAX_STATIONS = 32
DEFAULT_STATIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
_PROJECTION_TOLERANCE = 8 * math.ulp(1.0)


@dataclass(frozen=True, slots=True)
class LoadCase:
    """Effective global nodal forces and original-member local end intensities."""

    nodal: Mapping
    member_line: Mapping


def _vector(value, path):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ModelValidationError(path, "invalid_type", "expected a two-component vector")
    return tuple(_continuous(component, f"{path}/{i}") for i, component in enumerate(value))


def _load_range(path):
    return ModelValidationError(path, "load_out_of_range", "effective load is outside the finite float range")


def _local_vector(vector, basis, member, nodes, path):
    x, y = vector
    if basis == "global":
        _, c, s = _geometry(nodes[member["i"]], nodes[member["j"]], path)
        x, y = c * x + s * y, -s * x + c * y
    if not (math.isfinite(x) and math.isfinite(y)):
        raise _load_range(path)
    if member["kind"] == "truss2d":
        tolerance = _PROJECTION_TOLERANCE * max(map(abs, vector)) if basis == "global" else 0
        if abs(y) > tolerance:
            raise ModelValidationError(path, "unsupported_load", "trusses accept only axial line loads")
        y = 0.0
    return (x or 0.0, y or 0.0)


def _weight(acceleration, density, member, nodes, path):
    # Reassociate the product exactly if an intermediate product loses range.
    area = member["area"]
    mass = density * area
    vector = []
    for component in acceleration:
        value = mass * component
        if component and (not math.isfinite(value) or value == 0):
            try:
                value = float(Fraction(density) * Fraction(area) * Fraction(component))
            except OverflowError as error:
                raise _load_range(path) from error
        vector.append(value if component else 0.0)
    return _local_vector(vector, "global", member, nodes, path)


def _load_definitions(loads, nodes, members, frame_nodes):
    definitions = _fields(loads, "/loads", ("cases",), {"combinations": {}})
    entries = _named(definitions["cases"], "/loads/cases", 1, MAX_RESULTS)
    combinations = _named(definitions["combinations"], "/loads/combinations")
    if len(entries) + len(combinations) > MAX_RESULTS:
        raise ModelValidationError("/loads", "size_limit", f"expected at most {MAX_RESULTS} cases and combinations")
    cases = {}
    for name, entry in entries:
        path = f"/loads/cases/{name}"
        fields = _fields(entry, path, (), {"nodal": {}, "member_line": {}, "self_weight": None})
        nodal = _components(fields["nodal"], f"{path}/nodal", nodes, frame_nodes, FORCES)
        line = {}
        for key, load in _named(fields["member_line"], f"{path}/member_line"):
            member_path = f"{path}/member_line/{key}"
            member = members[_known(key, members, member_path, "member")]
            load = _fields(load, member_path, ("basis", "i", "j"))
            basis = _choice(load["basis"], f"{member_path}/basis", ("local", "global"), "unsupported_basis")
            for end in ("i", "j"):
                end_path = f"{member_path}/{end}"
                load[end] = _vector(load[end], end_path)
                _local_vector(load[end], basis, member, nodes, end_path)
            line[key] = load
        case = {"nodal": nodal, "member_line": line}
        if "self_weight" in entry:
            weight_path = f"{path}/self_weight"
            weight = _fields(fields["self_weight"], weight_path, ("acceleration", "density"))
            acceleration = _vector(weight["acceleration"], f"{weight_path}/acceleration")
            density = {}
            for key, value in _named(weight["density"], f"{weight_path}/density"):
                member_path = f"{weight_path}/density/{key}"
                member = members[_known(key, members, member_path, "member")]
                density[key] = _positive(value, member_path)
                _weight(acceleration, density[key], member, nodes, member_path)
            case["self_weight"] = {"acceleration": acceleration, "density": density}
        cases[name] = case
    normalized_combinations = {}
    for name, weights in combinations:
        path = f"/loads/combinations/{name}"
        if name in cases:
            raise ModelValidationError(path, "duplicate_id", "a case already uses this ID")
        normalized_combinations[name] = {
            _known(key, cases, f"{path}/{key}", "case"): _continuous(value, f"{path}/{key}")
            for key, value in _named(weights, path, 1, MAX_RESULTS)
        }
    return {"cases": cases, "combinations": normalized_combinations}


def _request(model, ids, stations):
    available = model.loads["cases"].keys() | model.loads["combinations"].keys()
    if ids is None:
        ids = tuple(available)
    if not isinstance(ids, (list, tuple)) or not 1 <= len(ids) <= MAX_RESULTS:
        raise ModelValidationError("/ids", "invalid_value", f"expected 1–{MAX_RESULTS} result IDs")
    for index, name in enumerate(ids):
        _known(name, available, f"/ids/{index}", "result")
    if len(set(ids)) != len(ids):
        raise ModelValidationError("/ids", "duplicate_id", "result IDs must be unique")
    if stations is None:
        stations = DEFAULT_STATIONS
    if not isinstance(stations, (list, tuple)) or not 1 <= len(stations) <= MAX_STATIONS:
        raise ModelValidationError("/stations", "invalid_value", f"expected 1–{MAX_STATIONS} stations")
    normalized = []
    for index, value in enumerate(stations):
        path = f"/stations/{index}"
        station = _continuous(value, path)
        if not 0 <= station <= 1 or station in normalized:
            raise ModelValidationError(path, "invalid_value", "expected a unique station in [0, 1]")
        normalized.append(station)
    return tuple(sorted(ids)), tuple(normalized)


def _request_hash(model, ids, stations):
    inputs = {**model.to_dict(), "projection": "anf.planar_load_case_request.v1",
              "selected_ids": ids, "stations": stations}
    return _describe_case(inputs)[0]


def _case_hash(model, effective):
    inputs = model.to_dict()
    del inputs["schema_version"]
    inputs["projection"] = "anf.planar_load_case.v1"
    inputs["loads"] = {"nodal": effective.nodal, "member_line": effective.member_line}
    return _describe_case(inputs)[0]


def _weighted_sum(terms, path):
    try:
        value = math.fsum(coefficient * load for coefficient, load in terms)
    except (OverflowError, ValueError):
        value = math.nan
    if not math.isfinite(value):
        try:
            value = float(sum(Fraction(c) * Fraction(load) for c, load in terms))
        except OverflowError as error:
            raise _load_range(path) from error
    return value or 0.0


def _combine(cases, path):
    nodal_terms, line_terms = {}, {}
    for coefficient, case in cases:
        if not coefficient:
            continue
        for name, row in case.nodal.items():
            target = nodal_terms.setdefault(name, {})
            for component, load in row.items():
                target.setdefault(component, []).append((coefficient, load))
        for name, ends in case.member_line.items():
            target = line_terms.setdefault(name, (([], []), ([], [])))
            for end_terms, end in zip(target, ends, strict=True):
                for terms, load in zip(end_terms, end, strict=True):
                    terms.append((coefficient, load))
    nodal, line = {}, {}
    for name, row in nodal_terms.items():
        values = {component: value for component, terms in row.items()
                  if (value := _weighted_sum(terms, path))}
        if values:
            nodal[name] = values
    for name, ends in line_terms.items():
        values = tuple(tuple(_weighted_sum(terms, path) for terms in end) for end in ends)
        if any(value for end in values for value in end):
            line[name] = values
    return LoadCase(nodal, line)


def _base_case(model, name):
    nodes, members = (model.model[key] for key in ("nodes", "members"))
    case = model.loads["cases"][name]
    path = f"/loads/cases/{name}"
    line = {
        key: tuple(_local_vector(load[end], load["basis"], members[key], nodes,
                                 f"{path}/member_line/{key}/{end}") for end in ("i", "j"))
        for key, load in case["member_line"].items()
    }
    weight_line = {}
    if weight := case.get("self_weight"):
        for key, density in weight["density"].items():
            load = _weight(weight["acceleration"], density, members[key], nodes, f"{path}/self_weight/density/{key}")
            weight_line[key] = (load, load)
    return _combine(((1, LoadCase(case["nodal"], line)), (1, LoadCase({}, weight_line))), path)


def _resolve_cases(model, ids):
    combinations = model.loads["combinations"]
    base_ids = {case for name in ids for case in (combinations[name] if name in combinations else (name,))}
    base = {name: _base_case(model, name) for name in sorted(base_ids)}
    for name in ids:
        if name in combinations:
            terms = ((factor, base[case]) for case, factor in combinations[name].items())
            yield name, "combination", _combine(terms, _pointer("/loads/combinations", name))
        else:
            yield name, "case", base[name]
