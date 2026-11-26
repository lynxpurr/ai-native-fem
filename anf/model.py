"""Versioned SI declarations for the classical static solvers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal, InvalidOperation
import json
import math
from types import MappingProxyType
from typing import TYPE_CHECKING

from .beam import Beam
from .plate import Plate
from .verification import _describe_case, _json, input_parameters

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


def _decode_json(payload, limit=_JSON_LIMIT):
    try:
        if len(payload) > limit:
            raise ModelValidationError("", "size_limit", f"JSON is limited to {limit // _JSON_LIMIT} MiB")
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        if len(payload) > limit:
            raise ModelValidationError("", "size_limit", f"JSON is limited to {limit // _JSON_LIMIT} MiB")
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


def _damping(value, path="/analysis/damping"):
    row = _fields(value, path, (), {"kind": _RAYLEIGH, "alpha": 0.0, "beta": 0.0})
    _choice(row["kind"], _pointer(path, "kind"), (_RAYLEIGH,), "unsupported_damping")
    coefficients = []
    for name in ("alpha", "beta"):
        number = _number(row[name], _pointer(path, name))
        if number < 0:
            raise ModelValidationError(_pointer(path, name), "negative_damping",
                                       "Rayleigh coefficients must be non-negative")
        coefficients.append(number)
    return Damping(*coefficients)


def _frequencies(value, path="/analysis/frequencies_hz"):
    if not isinstance(value, (list, tuple)):
        raise ModelValidationError(path, "invalid_type", "expected an array of frequencies")
    if not value:
        raise ModelValidationError(path, "invalid_value", "expected at least one frequency")
    if len(value) > _FREQUENCY_LIMIT:
        raise ModelValidationError(path, "frequency_budget_exceeded",
                                   f"at most {_FREQUENCY_LIMIT} frequencies are supported")
    numbers = [_number(item, f"{path}/{index}") for index, item in enumerate(value)]
    for index, number in enumerate(numbers):
        if number < 0:
            raise ModelValidationError(f"{path}/{index}", "negative_frequency",
                                       "frequencies must be non-negative")
        if index and number <= numbers[index - 1]:
            raise ModelValidationError(f"{path}/{index}", "frequency_not_increasing",
                                       "frequencies must be strictly increasing")
    return tuple(numbers)


def _time_grid(value, path="/analysis/time"):
    row = _fields(value, path, (),
                  {"start_s": None, "end_s": None, "steps": None, "times_s": None})
    scalar = [row[key] is not None for key in ("start_s", "end_s", "steps")]
    if row["times_s"] is not None:
        if any(scalar):
            raise ModelValidationError(path, "invalid_time_grid",
                                       "expected either start_s/end_s/steps or times_s")
        times = row["times_s"]
        if not isinstance(times, (list, tuple)):
            raise ModelValidationError(_pointer(path, "times_s"), "invalid_type",
                                       "expected an array of times")
        if len(times) < 2:
            raise ModelValidationError(_pointer(path, "times_s"), "invalid_time_grid",
                                       "expected at least two time points")
        if len(times) > _TIME_POINT_LIMIT:
            raise ModelValidationError(_pointer(path, "times_s"), "time_budget_exceeded",
                                       f"at most {_TIME_POINT_LIMIT} time points are supported")
        base = _pointer(path, "times_s")
        numbers = [_number(item, f"{base}/{index}") for index, item in enumerate(times)]
        if numbers[0] < 0:
            raise ModelValidationError(f"{base}/0", "invalid_time_grid",
                                       "start time must be non-negative")
        intervals = []
        for index in range(1, len(numbers)):
            step = numbers[index] - numbers[index - 1]
            if not step > 0:
                raise ModelValidationError(f"{base}/{index}", "non_monotonic_time",
                                           "times must be strictly increasing")
            intervals.append(step)
        for index, step in enumerate(intervals, start=1):
            if abs(step - intervals[0]) > 1e-9 * intervals[0]:
                raise ModelValidationError(f"{base}/{index}", "non_uniform_time",
                                           "times must be uniformly spaced")
        return TimeGrid(numbers[0], numbers[-1], len(numbers) - 1, intervals[0],
                        tuple(numbers))
    if not all(scalar):
        missing = next(key for key in ("start_s", "end_s", "steps") if row[key] is None)
        raise ModelValidationError(_pointer(path, missing), "missing_field", "field is required")
    start = _number(row["start_s"], _pointer(path, "start_s"))
    end = _number(row["end_s"], _pointer(path, "end_s"))
    steps_path = _pointer(path, "steps")
    count = _number(row["steps"], steps_path)
    if count.is_integer() and count > _STEP_LIMIT:
        raise ModelValidationError(steps_path, "time_budget_exceeded",
                                   f"at most {_STEP_LIMIT} steps are supported")
    steps = _count(row["steps"], steps_path, 1, _STEP_LIMIT)
    if start < 0:
        raise ModelValidationError(_pointer(path, "start_s"), "invalid_time_grid",
                                   "start time must be non-negative")
    if end <= start:
        raise ModelValidationError(_pointer(path, "end_s"), "invalid_time_grid",
                                   "end time must be greater than the start time")
    step = (end - start) / steps
    if not math.isfinite(step) or step <= 0:
        raise ModelValidationError(_pointer(path, "steps"), "invalid_time_grid",
                                   "the time step must be finite and positive")
    return TimeGrid(start, end, steps, step,
                    tuple(start + index * step for index in range(steps + 1)))


def _dynamics_analysis(analysis):
    if not isinstance(analysis, Mapping):
        raise ModelValidationError("/analysis", "invalid_type", "expected an object")
    kind = _choice(analysis.get("type"), "/analysis/type", (_HARMONIC, _TRANSIENT),
                   "unsupported_analysis")
    if kind == _HARMONIC:
        row = _fields(analysis, "/analysis", ("type", "frequencies_hz"), {"damping": {}})
        options = DynamicsOptions(kind, _damping(row["damping"]),
                                  frequencies_hz=_frequencies(row["frequencies_hz"]))
        return kind, options, None
    row = _fields(analysis, "/analysis", ("type", "time"),
                  {"damping": {}, "initial_state": {}})
    options = DynamicsOptions(kind, _damping(row["damping"]), time=_time_grid(row["time"]))
    return kind, options, row["initial_state"]


def _initial_state(value, model):
    from .structures import DISPLACEMENTS, _components

    block = _fields(value, "/analysis/initial_state", (), {"displacement": {}, "velocity": {}})
    frame_nodes = {node for member in model["members"].values()
                   if member["kind"] == "frame2d" for node in (member["i"], member["j"])}
    supports = model["supports"]

    def parse(entries, path):
        try:
            row = _components(entries, path, model["nodes"], frame_nodes, DISPLACEMENTS)
        except ModelValidationError as error:
            if error.code in ("inactive_dof", "unknown_field", "invalid_value"):
                raise ModelValidationError(error.path, "unsupported_initial_condition",
                                           error.message) from error
            raise
        for name, components in row.items():
            for component in components:
                if component in supports.get(name, {}):
                    raise ModelValidationError(
                        f"{path}/{name}/{component}", "unsupported_initial_condition",
                        "initial state on a fixed DOF is not supported")
        return _freeze(row)

    return (parse(block["displacement"], "/analysis/initial_state/displacement"),
            parse(block["velocity"], "/analysis/initial_state/velocity"))


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


_RAYLEIGH = "rayleigh"
_HARMONIC = "harmonic"
_TRANSIENT = "transient"
_FREQUENCY_LIMIT = 200
_STEP_LIMIT = 5000
_TIME_POINT_LIMIT = _STEP_LIMIT + 1
_STATE_LIMIT = 1_200_000


@dataclass(frozen=True, slots=True)
class Damping:
    """Rayleigh damping coefficients in their SI units."""

    alpha: float
    beta: float

    def to_dict(self):
        return {"kind": _RAYLEIGH, "alpha": self.alpha, "beta": self.beta}


@dataclass(frozen=True, slots=True)
class TimeGrid:
    """A validated uniform time grid with its materialized samples."""

    start_s: float
    end_s: float
    steps: int
    dt_s: float
    times_s: tuple

    def to_dict(self):
        return {"start_s": self.start_s, "end_s": self.end_s, "steps": self.steps}


@dataclass(frozen=True, slots=True)
class DynamicsOptions:
    """Validated harmonic or transient options, without any solver state."""

    kind: str
    damping: Damping
    frequencies_hz: tuple = ()
    time: TimeGrid | None = None
    initial_displacement: Mapping = field(
        default_factory=lambda: MappingProxyType({}))
    initial_velocity: Mapping = field(
        default_factory=lambda: MappingProxyType({}))

    def to_analysis_dict(self):
        if self.kind == _HARMONIC:
            return {"type": _HARMONIC,
                    "frequencies_hz": list(self.frequencies_hz),
                    "damping": self.damping.to_dict()}
        return {"type": _TRANSIENT, "time": self.time.to_dict(),
                "damping": self.damping.to_dict(),
                "initial_state": {"displacement": _plain(self.initial_displacement),
                                  "velocity": _plain(self.initial_velocity)}}


@dataclass(frozen=True, slots=True, init=False)
class ModelSpec:
    """An immutable, validated static declaration with explicit defaults."""

    schema_version: str
    problem_type: str
    model: Beam | Plate | Mapping
    loads: Mapping
    mesh: StructureMesh | None
    analysis_type: str
    modes: int | None
    dynamics: DynamicsOptions | None

    def __init__(self, doc, *, _versions=("1.0", "1.1")):
        if isinstance(doc, (str, bytes)):
            doc = _decode_json(doc)
        doc = _fields(doc, "", ("schema_version", "problem_type", "units", "model", "loads", "analysis"))
        version = _choice(doc["schema_version"], "/schema_version", _versions,
                          "unsupported_schema")
        kinds = (_BEAM, _PLATE) if version == "1.0" else (_STRUCTURE,)
        kind = _choice(doc["problem_type"], "/problem_type", kinds, "unsupported_problem")
        units = _fields(doc["units"], "/units", _SI)
        for name, expected in _SI.items():
            _choice(units[name], f"/units/{name}", (expected,), "unsupported_unit")
        dynamics = None
        state_block = None
        if version == "1.4":
            analysis = _fields(doc["analysis"], "/analysis", ("type",), {"modes": 6})
            _choice(analysis["type"], "/analysis/type", ("modal",), "unsupported_analysis")
            analysis_type, modes = "modal", _count(analysis["modes"], "/analysis/modes", 1, 20)
        elif version == "1.5":
            analysis_type, dynamics, state_block = _dynamics_analysis(doc["analysis"])
            modes = None
        else:
            analysis = _fields(doc["analysis"], "/analysis", ("type",))
            _choice(analysis["type"], "/analysis/type", ("static",), "unsupported_analysis")
            analysis_type, modes = "static", None
        mesh = None
        if kind == _STRUCTURE:
            from .structures import _structure

            options = (None if dynamics is None else
                       {"kind": dynamics.kind, "time": dynamics.time})
            model, loads, mesh = _structure(doc["model"], doc["loads"], version,
                                            dynamics=options)
        else:
            model, loads = (_beam if kind == _BEAM else _plate)(doc["model"], doc["loads"])
            loads = {key: _number(value, f"/loads/{key}") for key, value in loads.items()}
        if dynamics is not None and dynamics.kind == _TRANSIENT:
            displacement, velocity = _initial_state(state_block, model)
            dynamics = replace(dynamics, initial_displacement=displacement,
                               initial_velocity=velocity)
            points = mesh.dof_count * (dynamics.time.steps + 1)
            if points > _STATE_LIMIT:
                raise ModelValidationError("/analysis/time", "time_budget_exceeded",
                                           f"at most {_STATE_LIMIT} DOF-time points are supported")
        object.__setattr__(self, "schema_version", version)
        object.__setattr__(self, "problem_type", kind)
        object.__setattr__(self, "model", _freeze(model))
        object.__setattr__(self, "loads", _freeze(loads))
        object.__setattr__(self, "mesh", mesh)
        object.__setattr__(self, "analysis_type", analysis_type)
        object.__setattr__(self, "modes", modes)
        object.__setattr__(self, "dynamics", dynamics)
        if version in ("1.2", "1.3") and len(_json(self.to_dict()).encode("utf-8")) > _JSON_LIMIT:
            raise ModelValidationError("", "size_limit", "normalized JSON is limited to 1 MiB")

    def to_dict(self):
        """Return the normalized JSON-compatible declaration."""
        if self.dynamics is not None:
            analysis = self.dynamics.to_analysis_dict()
        else:
            analysis = {"type": self.analysis_type,
                        **({"modes": self.modes} if self.modes is not None else {})}
        return {
            "schema_version": self.schema_version, "problem_type": self.problem_type, "units": dict(_SI),
            "model": _plain(self.model) if self.mesh is not None else asdict(self.model),
            "loads": _plain(self.loads),
            "analysis": analysis,
        }


def validate_model(doc):
    """Validate a schema 1.0 or 1.1 single-solve declaration."""
    return _validate_model(doc, ("1.0", "1.1"))


def validate_load_case_model(doc):
    """Validate a schema 1.2 or 1.3 declaration without solving the cases."""
    return _validate_model(doc, ("1.2", "1.3"))


def validate_modal_model(doc):
    """Validate a schema 1.4 modal declaration without assembling mass."""
    return _validate_model(doc, ("1.4",))


def validate_dynamics_model(doc):
    """Validate a schema 1.4 modal or 1.5 forced-dynamics declaration."""
    return _validate_model(doc, ("1.4", "1.5"))


def _validate_model(doc, versions):
    if type(doc) is not ModelSpec:
        return ModelSpec(doc, _versions=versions)
    _choice(doc.schema_version, "/schema_version", versions, "unsupported_schema")
    return doc


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


def canonical_modal_input_hash(model):
    """Return the SHA-256 for a validated schema 1.4 modal declaration."""
    return _describe_case(_physical_inputs(validate_modal_model(model)))[0]


def canonical_dynamics_input_hash(model):
    """Return the SHA-256 for a validated schema 1.4 or 1.5 declaration."""
    return _describe_case(_physical_inputs(validate_dynamics_model(model)))[0]
