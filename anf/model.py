"""Versioned SI declarations for the classical static solvers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import json
import math
from types import MappingProxyType
from typing import TYPE_CHECKING

from .beam import Beam
from .plate import Plate
from .verification import _describe_case, input_parameters

if TYPE_CHECKING:
    from .structures import StructureMesh

_JSON_LIMIT = 1024 * 1024
_SI = {"length": "m", "force": "N", "stress": "Pa"}
_BEAM = "cantilever_bending"
_PLATE = "rectangular_plate"
_STRUCTURE = "planar_structure"


class ModelValidationError(ValueError):
    """Invalid declaration with a JSON Pointer, machine code and message."""

    def __init__(self, path, code, message):
        self.path = path
        self.code = code
        self.message = message
        super().__init__(f"{path or '/'}: {code}: {message}")


def _pointer(path, key):
    return path + "/" + str(key).replace("~", "~0").replace("/", "~1")


class _ObjectPairs(list):
    pass


def _decode_json(payload):
    try:
        if len(payload) > _JSON_LIMIT:
            raise ModelValidationError("", "size_limit", "JSON is limited to 1 MiB")
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        if len(payload) > _JSON_LIMIT:
            raise ModelValidationError("", "size_limit", "JSON is limited to 1 MiB")
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_ObjectPairs,
                           parse_int=Decimal, parse_float=Decimal, parse_constant=Decimal)
        return _unique_objects(value)
    except (UnicodeError, RecursionError, InvalidOperation, ValueError) as error:
        if isinstance(error, ModelValidationError):
            raise
        raise ModelValidationError("", "invalid_json", "expected a UTF-8 JSON document") from error


def _unique_objects(value, path=""):
    if isinstance(value, _ObjectPairs):
        obj = {}
        for key, item in value:
            field = _pointer(path, key)
            if key in obj:
                raise ModelValidationError(field, "duplicate_field", "field occurs more than once")
            obj[key] = _unique_objects(item, field)
        return obj
    if isinstance(value, list):
        return [_unique_objects(item, _pointer(path, i)) for i, item in enumerate(value)]
    return value


def _fields(value, path, required, defaults=None):
    if not isinstance(value, Mapping):
        raise ModelValidationError(path, "invalid_type", "expected an object")
    defaults = {} if defaults is None else defaults
    for key in value:
        if key not in required and key not in defaults:
            raise ModelValidationError(_pointer(path, key), "unknown_field", "field is not supported")
    for key in required:
        if key not in value:
            raise ModelValidationError(_pointer(path, key), "missing_field", "field is required")
    return {**defaults, **value}


def _choice(value, path, choices, code):
    if not isinstance(value, str):
        raise ModelValidationError(path, "invalid_type", "expected a string")
    if value not in choices:
        raise ModelValidationError(path, code, f"expected one of {', '.join(choices)}")
    return value


def _number(value, path):
    if type(value) not in (int, float, Decimal):
        raise ModelValidationError(path, "invalid_type", "expected a number")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        number = math.inf
    if not math.isfinite(number) or (number == 0 and value != 0):
        raise ModelValidationError(path, "invalid_value", "expected a representable finite number")
    return number


def _positive(value, path):
    number = _number(value, path)
    if number <= 0:
        raise ModelValidationError(path, "invalid_value", "expected a positive number")
    return number


def _count(value, path, minimum, maximum):
    number = _number(value, path)
    if not number.is_integer() or value != int(number):
        raise ModelValidationError(path, "invalid_value", "expected an integer")
    if not minimum <= number <= maximum:
        raise ModelValidationError(path, "invalid_value", f"expected an integer from {minimum} to {maximum}")
    return int(number)


def _beam(parameters, loads):
    positive_fields = ("length", "young_modulus", "second_moment")
    parameters = _fields(parameters, "/model", positive_fields, {"elements": 8})
    for name in positive_fields:
        parameters[name] = _positive(parameters[name], f"/model/{name}")
    parameters["elements"] = _count(parameters["elements"], "/model/elements", 1, 200)
    loads = _fields(loads, "/loads", (), {"tip_force": 0.0, "uniform_load": 0.0})
    return Beam(**parameters), loads


def _plate(parameters, loads):
    positive_fields = ("lx", "ly", "thickness", "young_modulus")
    parameters = _fields(parameters, "/model", (*positive_fields, "poisson_ratio"), {
        "nx": 8, "ny": 8, "support": "clamped", "shear_factor": 5/6, "formulation": "q4_full",
    })
    for name in (*positive_fields, "shear_factor"):
        parameters[name] = _positive(parameters[name], f"/model/{name}")
    nu = _number(parameters["poisson_ratio"], "/model/poisson_ratio")
    if not -1 < nu < 0.5:
        raise ModelValidationError("/model/poisson_ratio", "invalid_value", "expected -1 < nu < 0.5")
    parameters["poisson_ratio"] = nu
    for name in ("nx", "ny"):
        parameters[name] = _count(parameters[name], f"/model/{name}", 2, 199)
    if 3 * (parameters["nx"] + 1) * (parameters["ny"] + 1) > 1800:
        raise ModelValidationError("/model", "size_limit", "plate mesh is limited to 1800 DOFs")
    formulation = _choice(parameters["formulation"], "/model/formulation",
                          ("q4_full", "mitc4_rectangle"), "unsupported_formulation")
    support = _choice(parameters["support"], "/model/support",
                      ("clamped", "hard_simply_supported"), "unsupported_support")
    if formulation == "mitc4_rectangle" and support != "hard_simply_supported":
        raise ModelValidationError("/model/support", "unsupported_support",
                                   "mitc4_rectangle requires hard_simply_supported edges")
    return Plate(**parameters), _fields(loads, "/loads", ("pressure",))


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    return value


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return list(value)
    return value


@dataclass(frozen=True, slots=True, init=False)
class ModelSpec:
    """An immutable, validated static declaration with explicit defaults."""

    schema_version: str
    problem_type: str
    model: Beam | Plate | Mapping
    loads: Mapping
    mesh: StructureMesh | None

    def __init__(self, doc):
        if isinstance(doc, (str, bytes)):
            doc = _decode_json(doc)
        doc = _fields(doc, "", ("schema_version", "problem_type", "units", "model", "loads", "analysis"))
        version = _choice(doc["schema_version"], "/schema_version", ("1.0", "1.1"), "unsupported_schema")
        kinds = (_STRUCTURE,) if version == "1.1" else (_BEAM, _PLATE)
        kind = _choice(doc["problem_type"], "/problem_type", kinds, "unsupported_problem")
        units = _fields(doc["units"], "/units", _SI)
        for name, expected in _SI.items():
            _choice(units[name], f"/units/{name}", (expected,), "unsupported_unit")
        analysis = _fields(doc["analysis"], "/analysis", ("type",))
        _choice(analysis["type"], "/analysis/type", ("static",), "unsupported_analysis")
        mesh = None
        if kind == _STRUCTURE:
            from .structures import _structure

            model, loads, mesh = _structure(doc["model"], doc["loads"])
        else:
            model, loads = (_beam if kind == _BEAM else _plate)(doc["model"], doc["loads"])
            loads = {key: _number(value, f"/loads/{key}") for key, value in loads.items()}
        object.__setattr__(self, "schema_version", version)
        object.__setattr__(self, "problem_type", kind)
        object.__setattr__(self, "model", _freeze(model))
        object.__setattr__(self, "loads", _freeze(loads))
        object.__setattr__(self, "mesh", mesh)

    def to_dict(self):
        """Return the normalized JSON-compatible declaration."""
        return {
            "schema_version": self.schema_version, "problem_type": self.problem_type, "units": dict(_SI),
            "model": _plain(self.model) if self.mesh is not None else asdict(self.model),
            "loads": _plain(self.loads), "analysis": {"type": "static"},
        }


def validate_model(doc):
    """Validate a mapping or UTF-8 JSON string/bytes, or accept a ModelSpec."""
    return doc if type(doc) is ModelSpec else ModelSpec(doc)


def _physical_inputs(model):
    if model.mesh is None:
        return input_parameters(model.model, **model.loads)
    payload = model.to_dict()
    del payload["schema_version"]
    payload["projection"] = "anf.planar_structure.v1"
    return payload


def canonical_input_hash(model):
    """Return the verification report's SHA-256 for the validated physical inputs."""
    return _describe_case(_physical_inputs(validate_model(model)))[0]
