"""Replayable parameter scans and local sensitivity for a static cantilever."""

from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import file_digest, sha256
from itertools import product
import json
import math
import os
from pathlib import Path
import platform
from tempfile import NamedTemporaryFile

import numpy as np

from ._linear import MAX_RELATIVE_RESIDUAL
from .application import _versions, run_analysis
from .artifacts import ArtifactError, _flush, _read_regular, _regular, _require_directory, _safe_name
from .model import (ModelSpec, ModelValidationError, _choice, _count, _decode_json, _fields,
                    _number, _ObjectPairs, _unique_objects, canonical_input_hash, validate_model)
from .studies import _deflection
from .verification import _CLASSICAL_REFERENCES


_PARAMETER_UNITS = {
    "loads.tip_force": "N", "model.young_modulus": "Pa", "model.second_moment": "m^4", "model.length": "m",
}
_DERIVATIVE_UNITS = dict(zip(_PARAMETER_UNITS, ("m/N", "m/Pa", "m^-3", "m/m"), strict=True))
_RATIOS = (0.008, 0.004, 0.002, 0.001)
_NEAR_ZERO_M = 1e-10
_MAX_SAMPLES = 128
_MAX_SOLVES = 64
_MAX_BYTES = 4 * 1024 * 1024
_STATUSES = ("solved", "numerical_failure", "out_of_domain", "invalid_input", "not_run")
_VERIFICATION = ("verification_passed", "verification_failed", "solved_unverified")
_OUTCOME_FIELDS = {"status", "solve_status", "verification_status", "observed_m", "relative_residual",
                   "reference_check", "diagnostics"}


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value):
    return sha256(_encode(value)).hexdigest()


def _value(value, path):
    number = _number(value, path)
    return 0.0 if number == 0 else number


@dataclass(frozen=True, slots=True)
class _Axis:
    path: str
    unit: str
    bounds: tuple[float, float]
    values: tuple[float, ...]

    @classmethod
    def parse(cls, item, index):
        path = f"/study/axes/{index}"
        item = _fields(item, path, ("path", "unit", "bounds", "values"))
        parameter = _choice(item["path"], f"{path}/path", tuple(_PARAMETER_UNITS), "unsupported_parameter")
        unit = _choice(item["unit"], f"{path}/unit", (_PARAMETER_UNITS[parameter],), "unsupported_unit")
        bounds = item["bounds"]
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ModelValidationError(f"{path}/bounds", "invalid_value", "expected two bounds")
        lo, hi = (_value(bound, f"{path}/bounds/{i}") for i, bound in enumerate(bounds))
        if lo >= hi or (parameter.startswith("model.") and lo <= 0):
            raise ModelValidationError(f"{path}/bounds", "invalid_value", "expected increasing physical bounds")
        values = item["values"]
        if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= _MAX_SAMPLES:
            raise ModelValidationError(f"{path}/values", "invalid_value", f"expected 1–{_MAX_SAMPLES} values")
        values = tuple(_value(value, f"{path}/values/{i}") for i, value in enumerate(values))
        if len(set(values)) != len(values):
            raise ModelValidationError(f"{path}/values", "duplicate_value", "axis values must be distinct")
        return cls(parameter, unit, (lo, hi), values)


@dataclass(frozen=True, slots=True, init=False)
class ParameterGrid:
    """A validated cantilever declaration, ordered parameter axes and study budgets."""

    model: ModelSpec
    axes: tuple[_Axis, ...]
    observable: str
    reference: str | None
    max_solves: int
    max_output_bytes: int

    def __init__(self, model, axes, *, observable="tip_deflection", reference=None,
                 max_solves=_MAX_SOLVES, max_output_bytes=_MAX_BYTES):
        model = validate_model(model)
        if model.problem_type != "cantilever_bending":
            raise ModelValidationError("/study/model/problem_type", "unsupported_problem", "expected a cantilever")
        if model.loads["uniform_load"] != 0.0:
            raise ModelValidationError("/study/model/loads/uniform_load", "unsupported_load", "expected zero")
        _choice(observable, "/study/observable", ("tip_deflection",), "unsupported_observable")
        if reference is not None:
            beam_references = tuple(key for key, record in _CLASSICAL_REFERENCES.items()
                                    if (observable, "m") in record.observable_units)
            _choice(reference, "/study/reference", beam_references, "unsupported_reference")
        if not isinstance(axes, (list, tuple)) or not 1 <= len(axes) <= len(_PARAMETER_UNITS):
            raise ModelValidationError("/study/axes", "invalid_value", "expected one to four axes")
        axes = tuple(_Axis.parse(axis, i) for i, axis in enumerate(axes))
        if len({axis.path for axis in axes}) != len(axes):
            raise ModelValidationError("/study/axes", "duplicate_axis", "parameter paths must be distinct")
        if math.prod(len(axis.values) for axis in axes) > _MAX_SAMPLES:
            raise ModelValidationError("/study/axes", "size_limit", f"expected at most {_MAX_SAMPLES} samples")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "observable", observable)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "max_solves", _count(max_solves, "/study/max_solves", 1, _MAX_SOLVES))
        object.__setattr__(self, "max_output_bytes",
                           _count(max_output_bytes, "/study/max_output_bytes", 1, _MAX_BYTES))

    @classmethod
    def from_dict(cls, doc):
        """Read the normalized study declaration saved in a report, or its JSON bytes."""
        if isinstance(doc, (str, bytes)):
            doc = _decode_json(doc)
        return cls(**_fields(doc, "/study", ("model", "axes"), {
            "observable": "tip_deflection", "reference": None,
            "max_solves": _MAX_SOLVES, "max_output_bytes": _MAX_BYTES,
        }))

    def to_dict(self):
        return {"model": self.model.to_dict(), "axes": [asdict(axis) for axis in self.axes],
                "observable": self.observable, "reference": self.reference, "max_solves": self.max_solves,
                "max_output_bytes": self.max_output_bytes}


def _assignments(grid, sensitivity):
    paths = tuple(axis.path for axis in grid.axes)
    if not sensitivity:
        for values in product(*(axis.values for axis in grid.axes)):
            yield dict(zip(paths, values, strict=True)), {}, None
        return
    document = grid.model.to_dict()
    center = {}
    for path in paths:
        section, field = path.split(".")
        center[path] = document[section][field]
    yield center, {"role": "center"}, None
    for path, value in center.items():
        for ratio in _RATIOS:
            for sign, side in ((-1, "minus"), (1, "plus")):
                changed = value + sign * (ratio * abs(value))
                reason = None
                if not math.isfinite(changed) or changed == value:
                    reason = "perturbation_unresolved"
                yield {**center, path: changed if math.isfinite(changed) else None}, {
                    "role": "perturbation", "axis": path, "ratio": ratio, "side": side,
                }, reason


def _diagnostic(code, path, message):
    return {"code": code, "path": path, "message": message}


def _sample_model(grid, parameters, unresolved):
    if any(value is None for value in parameters.values()):
        return None, "invalid_input", _diagnostic(unresolved, "/study", "perturbation is not representable")
    document = grid.model.to_dict()
    for path, value in parameters.items():
        section, field = path.split(".")
        document[section][field] = value
    outside = [axis for axis in grid.axes if not axis.bounds[0] <= parameters[axis.path] <= axis.bounds[1]]
    try:
        model = validate_model(document)
    except ModelValidationError as error:
        model = None
        if not outside:
            return None, "invalid_input", _diagnostic(error.code, error.path, error.message)
    if outside:
        path = "/" + outside[0].path.replace(".", "/")
        return model, "out_of_domain", _diagnostic("out_of_domain", path, "sample exceeds its declared bounds")
    if unresolved:
        return model, "invalid_input", _diagnostic(unresolved, "/study", "perturbation equals its base value")
    return model, None, None


def _plan(grid, request_hash, sensitivity):
    samples, models = [], []
    for index, (parameters, role, unresolved) in enumerate(_assignments(grid, sensitivity), 1):
        model, status, diagnostic = _sample_model(grid, parameters, unresolved)
        samples.append({"id": f"{request_hash}:{index}", "index": index, "parameters": parameters, **role,
                        "input_hash": canonical_input_hash(model) if model is not None else None,
                        "preflight_status": status, "preflight_diagnostic": diagnostic})
        models.append(model)
    return samples, models


def _file_hash(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _execution_identity():
    root = Path(__file__).resolve().parent
    build = {p.relative_to(root).as_posix(): _file_hash(p) for p in sorted(root.rglob("*.py"))}
    versions = {**_versions(), "backend": "dense", "anf_build": _digest(build)}
    bundled = sorted((Path(np.__file__).resolve().parent.parent / "numpy.libs").glob("*"))
    blas = {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "host": platform.node(), "settings": {key: value for key, value in os.environ.items()
                if key.startswith(("OMP_", "OPENBLAS_", "MKL_", "BLIS_", "VECLIB_", "NPY_", "LD_", "DYLD_"))}}
    verified = False
    try:
        lapack = Path(np.linalg._umath_linalg.__file__)
        blas["build"] = np.show_config(mode="dicts")
        blas["lapack"] = _file_hash(lapack)
        blas["bundled"] = {p.name: _file_hash(p) for p in bundled if p.is_file()}
        verified = any("openblas" in name.lower() for name in blas["bundled"]) and not any(
            name in blas["settings"] for name in ("LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES")
        )
    except (AttributeError, OSError, TypeError):
        pass  # Resume still binds the locally observable environment. Shared cache stays disabled.
    return {**versions, "linear_algebra": _digest(blas), "shared_cache_available": verified}


def _directory(requested):
    requested = Path(requested)
    _safe_name(requested.name)
    directory = requested.parent.resolve(strict=True) / requested.name
    directory.mkdir(mode=0o700, exist_ok=True)
    _require_directory(directory)
    return directory


@contextmanager
def _exclusive(directory):
    path = directory / "study.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags, 0o600), "r+b") as stream:
        info = os.fstat(stream.fileno())
        if not _regular(info) or not os.path.samestat(path.lstat(), info):
            raise ArtifactError("invalid study lock file")
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ArtifactError("study directory is already in use") from error
        yield


def _write_atomic(path, payload):
    with NamedTemporaryFile(dir=path.parent, prefix=".anf-", delete=False) as stream:
        staging = Path(stream.name)
        try:
            stream.write(payload)
            _flush(stream)
        except BaseException:
            stream.close()
            staging.unlink(missing_ok=True)
            raise
    try:
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)


def _write_record(path, body):
    payload = _encode({"digest": _digest(body), "body": body})
    if len(payload) > _MAX_BYTES:
        raise ArtifactError(f"study record exceeds its byte budget: {path.name}")
    _write_atomic(path, payload)


def _read_record(path):
    try:
        record = _unique_objects(json.loads(_read_regular(path, _MAX_BYTES), object_pairs_hook=_ObjectPairs))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, RecursionError) as error:
        raise ArtifactError(f"invalid study record: {path.name}") from error
    try:
        if set(record) == {"digest", "body"} and isinstance(record["body"], dict):
            if record["digest"] == _digest(record["body"]):
                return record["body"]
    except (TypeError, ValueError, RecursionError):
        pass
    raise ArtifactError(f"invalid study record digest: {path.name}")


def _skipped(status, diagnostic):
    return {"status": status, "solve_status": "not_run", "verification_status": "not_run", "observed_m": None,
            "relative_residual": None, "reference_check": None, "diagnostics": [diagnostic]}


def _solve(model, reference):
    result = run_analysis(model, {"backend": "dense", "reference": reference})
    check = next((asdict(item) for item in result.report.checks
                  if item.evidence_id == reference and item.observable == "tip_deflection"), None)
    return {"status": result.solve_status, "solve_status": result.solve_status,
            "verification_status": result.verification_status,
            "observed_m": _deflection(result) if result.solve_status == "solved" else None,
            "relative_residual": result.relative_residual, "reference_check": check,
            "diagnostics": list(result.diagnostics)}


def _valid_outcome(outcome):
    if (not isinstance(outcome, dict) or outcome.keys() != _OUTCOME_FIELDS
            or outcome["status"] not in _STATUSES or not isinstance(outcome["diagnostics"], list)
            or any(not isinstance(item, dict) or not isinstance(item.get("code"), str)
                   or not isinstance(item.get("message"), str) for item in outcome["diagnostics"])):
        return False
    observed, residual = outcome["observed_m"], outcome["relative_residual"]
    if outcome["status"] == "solved":
        return (outcome["solve_status"] == "solved" and outcome["verification_status"] in _VERIFICATION
                and type(observed) in (int, float) and type(residual) in (int, float)
                and 0 <= residual <= MAX_RELATIVE_RESIDUAL)
    expected = "numerical_failure" if outcome["status"] == "numerical_failure" else "not_run"
    return (outcome["solve_status"] == expected and outcome["verification_status"] == "not_run"
            and observed is None and residual is None and bool(outcome["diagnostics"]))


def _load_checkpoint(path, expected, max_solves):
    state = _read_record(path)
    if state is None:
        for name in ("report.json", "study.svg"):
            if os.path.lexists(path.parent / name):
                raise ArtifactError("existing study output has no checkpoint")
        state = {**expected, "rows": {}, "attempts": []}
        _write_record(path, state)
        return state
    if any(state.get(key) != value for key, value in expected.items()):
        raise ArtifactError("checkpoint request, plan or execution identity differs")
    rows, attempts = state.get("rows"), state.get("attempts")
    ids = {sample["id"] for sample in expected["plan"]}
    if (not isinstance(rows, dict) or not rows.keys() <= ids or not isinstance(attempts, list)
            or len(attempts) > max_solves or any(
                not isinstance(item, dict) or not isinstance(item.get("id"), str) or item["id"] not in ids
                or item.get("status") not in ("finished", "interrupted", "started") for item in attempts
            ) or any(item["status"] == "started" for item in attempts[:-1])):
        raise ArtifactError("invalid study checkpoint state")
    origins = {}
    for sample_id, row in rows.items():
        if (not isinstance(row, dict) or row.get("origin") not in ("none", "solve", "cache")
                or not _valid_outcome({key: value for key, value in row.items() if key != "origin"})):
            raise ArtifactError("invalid study checkpoint outcome")
        origins[sample_id] = row["origin"]
    finished = [item["id"] for item in attempts if item["status"] == "finished"]
    if len(set(finished)) != len(finished) or set(finished) != {key for key in origins if origins[key] == "solve"}:
        raise ArtifactError("study checkpoint outcomes do not match the reservations")
    if attempts and attempts[-1]["status"] == "started":
        if attempts[-1]["id"] in rows:
            raise ArtifactError("study checkpoint has a committed row with an active reservation")
        attempts[-1]["status"] = "interrupted"
        _write_record(path, state)
    return state


def _cache_path(cache_dir, sample, execution, reference):
    if cache_dir is None:
        return None
    key = _digest({"input_hash": sample["input_hash"], "execution": execution, "reference": reference})
    return cache_dir / f"{key}.json"


def _evaluate(path, cache_path, grid, model, state, sample, stop_reason):
    if stop_reason is not None:
        return _skipped("not_run", stop_reason), "none"
    if sample["preflight_status"] is not None:
        return _skipped(sample["preflight_status"], sample["preflight_diagnostic"]), "none"
    cached = _read_record(cache_path) if cache_path is not None else None
    if cached is not None:
        outcome = cached.get("outcome")
        if cached.get("key") != cache_path.stem or not _valid_outcome(outcome) or outcome["status"] != "solved":
            raise ArtifactError("invalid cached study sample")
        return outcome, "cache"
    attempts = state["attempts"]
    if len(attempts) >= grid.max_solves:
        return _skipped("not_run", _diagnostic("solve_budget_exhausted", "/study/max_solves",
                                              "no solve reservations remain")), "none"
    reservation = {"id": sample["id"], "status": "started"}
    attempts.append(reservation)
    _write_record(path, state)
    outcome = _solve(model, grid.reference)
    reservation["status"] = "finished"
    return outcome, "solve"


def _sequence_level(minus, plus, path):
    level = {"ratio": minus["ratio"], "minus_id": minus["id"], "plus_id": plus["id"],
             "derivative": None, "reason": None}
    for row in (minus, plus):
        if row["status"] != "solved":
            level["reason"] = row["diagnostics"][0]["code"] if row["diagnostics"] else row["status"]
            return level
    below, above = minus["observed_m"], plus["observed_m"]
    difference = above - below
    width = plus["parameters"][path] - minus["parameters"][path]
    if (not math.isfinite(width) or width <= 0 or not math.isfinite(difference)
            or abs(difference) <= 8 * math.ulp(max(abs(below), abs(above)))):
        level["reason"] = "perturbation_unresolved"
    else:
        derivative = difference / width
        if not math.isfinite(derivative) or derivative == 0.0:
            level["reason"] = "perturbation_unresolved"
        else:
            level["derivative"] = derivative
    return level


def _center_reason(row):
    if row["status"] != "solved":
        return _diagnostic("base_unavailable", "/study/model", "base sample did not solve")
    if abs(row["observed_m"]) <= _NEAR_ZERO_M:
        return _diagnostic("target_near_zero", "/study/observable", "base deflection is at or below 1e-10 m")
    return None


def _sensitivities(grid, rows):
    center_reason = _center_reason(rows[0])
    results = {}
    for i, axis in enumerate(grid.axes):
        start = 1 + i * 2 * len(_RATIOS)
        sides = rows[start:start + 2 * len(_RATIOS)]
        levels = [_sequence_level(minus, plus, axis.path) for minus, plus in zip(sides[::2], sides[1::2], strict=True)]
        reason = center_reason["code"] if center_reason else next((level["reason"] for level in levels
                                                                   if level["reason"]), None)
        result = {"status": "unavailable", "reason": reason, "derivative": None, "unit": _DERIVATIVE_UNITS[axis.path],
                  "last_change": None, "levels": levels}
        if reason is None:
            a, b, c = (level["derivative"] for level in levels[-3:])
            relative = [abs(x / max(abs(x), abs(y)) - y / max(abs(x), abs(y))) for x, y in ((a, b), (b, c))]
            if relative[0] <= 2.5e-5 and relative[1] <= 1e-5:
                result.update(status="estimated", derivative=c, last_change=abs(c - b))
            else:
                result["reason"] = "unstable_sequence"
        results[axis.path] = result
    return results


def _plot(report):
    rows = report["samples"]
    counts = report["counts"]
    axes = report["study"]["axes"]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 780 350" role="img">',
           '<title>Tip deflection by sample</title><rect width="780" height="350" fill="white"/>',
           '<g fill="#253044" font-family="sans-serif" font-size="12">',
           '<text x="12" y="19">Tip deflection (m) by sample index</text>',
           f'<text x="12" y="39">{len(rows)} samples · ' + " · ".join(f"{n} {s}" for s, n in counts.items()) + '</text>']
    for i, axis in enumerate(axes):
        lo, hi = axis["bounds"]
        svg.append(f'<text x="12" y="{58 + i*16}">{axis["path"]} [{lo:.7g}, {hi:.7g}] {axis["unit"]}</text>')
    states = dict(zip(_VERIFICATION, ("#12704d", "#a63426", "#245b98"), strict=True))
    for i, (state, color) in enumerate(states.items()):
        svg.append(f'<text x="{12 + i*220}" y="135" fill="{color}">{state}</text>')
    svg.append('<path d="M 104 154 V 310 H 760" fill="none" stroke="#344054"/>')
    values = [row["observed_m"] for row in rows if row["status"] == "solved"]
    if values:
        lo, hi = min(values), max(values)
        scale = max(abs(hi), abs(lo)) or 1.0
        span = hi / scale - lo / scale
        svg.append(f'<text x="98" y="167" text-anchor="end">{hi:.5g}</text>')
        if span:
            svg.append(f'<text x="98" y="304" text-anchor="end">{lo:.5g}</text>')
        for row in rows:
            if row["status"] == "solved":
                fraction = (row["observed_m"] / scale - lo / scale) / span if span else 0.5
                x = 108 + 648 * (row["index"] - 1) / max(1, len(rows) - 1)
                color = states[row["verification_status"]]
                svg.append(f'<circle cx="{x:.2f}" cy="{302 - 134*fraction:.2f}" r="3" fill="{color}"/>')
    else:
        svg.append('<text x="340" y="236">No solved samples</text>')
    svg.append('<text x="108" y="330" text-anchor="middle">1</text>')
    if len(rows) > 1:
        svg.append(f'<text x="756" y="330" text-anchor="middle">{len(rows)}</text>')
    return ("\n".join([*svg, "</g></svg>"]) + "\n").encode("utf-8")


def _publish(directory, report, limit):
    plot = _plot(report)
    report["plot"] = {"path": "study.svg", "sha256": sha256(plot).hexdigest()}
    payload = (json.dumps(report, allow_nan=False, indent=2) + "\n").encode("utf-8")
    if len(payload) + len(plot) > limit:
        raise ArtifactError("study report exceeds its byte budget")
    _write_atomic(directory / "study.svg", plot)
    _write_atomic(directory / "report.json", payload)


def _run(grid, output_dir, cache_dir, *, sensitivity):
    grid = grid if isinstance(grid, ParameterGrid) else ParameterGrid.from_dict(grid)
    study = grid.to_dict()
    # Use the same JSON types in the returned declaration and in its on-disk replay.
    study = json.loads(_encode(study))
    request = {"schema_version": "1.0", "kind": "sensitivity" if sensitivity else "grid", "study": study,
               "step_ratios": list(_RATIOS) if sensitivity else [], "backend": "dense"}
    request_hash = _digest(request)
    plan, models = _plan(grid, request_hash, sensitivity)
    execution = _execution_identity()
    try:
        directory = _directory(output_dir)
        cache = _directory(cache_dir) if cache_dir is not None and execution["shared_cache_available"] else None
        checkpoint = directory / "checkpoint.json"
        with _exclusive(directory):
            state = _load_checkpoint(checkpoint, {"request_hash": request_hash, "execution": execution, "plan": plan},
                                     grid.max_solves)
            rows, stop = [], None
            for sample, model in zip(plan, models, strict=True):
                sample_id = sample["id"]
                row = state["rows"].get(sample_id)
                if row is None:
                    cached = _cache_path(cache, sample, execution, grid.reference)
                    outcome, origin = _evaluate(checkpoint, cached, grid, model, state, sample, stop)
                    row = {**outcome, "origin": origin}
                    state["rows"][sample_id] = row
                    _write_record(checkpoint, state)
                    if cached is not None and origin == "solve" and outcome["status"] == "solved":
                        _write_record(cached, {"key": cached.stem, "outcome": outcome})
                rows.append({**sample, **row})
                if sensitivity and sample["index"] == 1:
                    stop = _center_reason(row)
            counts = Counter(row["status"] for row in rows)
            verified = Counter(row["verification_status"] for row in rows if row["status"] == "solved")
            report = {**request, "request_hash": request_hash, "base_model_hash": canonical_input_hash(grid.model),
                      "execution": execution, "status": "complete" if counts["solved"] == len(rows) else "partial",
                      "observable_unit": "m", "samples": rows, "attempts": state["attempts"],
                      "counts": {status: counts[status] for status in _STATUSES},
                      "verification_counts": {status: verified[status] for status in _VERIFICATION}}
            if sensitivity:
                report["sensitivities"] = _sensitivities(grid, rows)
                if any(row["status"] == "unavailable" for row in report["sensitivities"].values()):
                    report["status"] = "partial"
            _publish(directory, report, grid.max_output_bytes)
            return report
    except OSError as error:
        raise ArtifactError(f"could not write parameter study: {error}") from error


def run_parameter_study(grid, output_dir, *, cache_dir=None):
    """Enumerate the grid, checkpoint each outcome, and return its saved report."""
    return _run(grid, output_dir, cache_dir, sensitivity=False)


def finite_difference_sensitivity(grid, output_dir, *, cache_dir=None):
    """Perturb the base declaration along each axis and report stable local derivatives."""
    return _run(grid, output_dir, cache_dir, sensitivity=True)
