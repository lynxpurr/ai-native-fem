"""Opt-in verification reports for the classical beam and plate solvers."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from hashlib import sha256
import json
import math
import re
from types import MappingProxyType

import numpy as np

from ._linear import MAX_RELATIVE_RESIDUAL, finite
from .beam import Beam, BeamResult, solve_cantilever
from .plate import Plate, PlateResult, solve_plate


class ReferenceKind(StrEnum):
    ANALYTIC = "analytic"
    MANUFACTURED = "manufactured"
    INDEPENDENT_NUMERICAL = "independent-numerical"
    EXTERNAL_COMPARISON = "external-comparison"
    SELF_REGRESSION = "self-regression"


_INDEPENDENT = {
    ReferenceKind.ANALYTIC, ReferenceKind.MANUFACTURED, ReferenceKind.INDEPENDENT_NUMERICAL,
}


@dataclass(frozen=True)
class ReferenceRecord:
    """Metadata for a reference. Eligibility comes from the bundled registry."""

    id: str
    kind: ReferenceKind
    source: str
    formula_location: str
    scope: str
    observable_units: tuple[tuple[str, str], ...]
    threshold: float
    error_budget: str
    negative_case: bool = False

    def __post_init__(self):
        object.__setattr__(self, "kind", ReferenceKind(self.kind))
        for name in ("id", "source", "formula_location", "scope", "error_budget"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if not isinstance(self.observable_units, tuple) or not self.observable_units:
            raise ValueError("observable_units must be a nonempty tuple")
        for pair in self.observable_units:
            if (not isinstance(pair, tuple) or len(pair) != 2
                    or any(not isinstance(item, str) or not item.strip() for item in pair)):
                raise ValueError("each observable must have a name and unit")
        if len(dict(self.observable_units)) != len(self.observable_units):
            raise ValueError("reference observables must be unique")
        threshold = finite(self.threshold, "threshold")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        object.__setattr__(self, "threshold", threshold)
        if type(self.negative_case) is not bool:
            raise ValueError("negative_case must be a boolean")


_BEAM = ReferenceRecord(
    "cantilever.closed-form.v1", ReferenceKind.ANALYTIC,
    "D. M. Kochmann, Introduction to Finite Element Analysis (ETH Zurich, 2025-02-14)",
    "Equations (12.29) and (12.36); see docs/methods.md#beam for the integrated solution",
    "Constant-section, one-plane Euler–Bernoulli cantilever, tip force and uniform line load",
    (("tip_deflection", "m"), ("tip_slope", "rad"), ("root_force", "N"), ("root_moment", "N·m")),
    1e-8, "Closed form; no discretization. Exact zero or contribution scaling for zero reference.",
)
_PLATE = ReferenceRecord(
    "plate.navier-31.v1", ReferenceKind.INDEPENDENT_NUMERICAL,
    "J. Schöberl, Interactive Finite Elements (2024), Reissner Mindlin plates",
    "Chapter 60, total energy; see docs/methods.md#plate for the modal compliance derivation",
    "Uniform 1000 Pa on a hard simply supported 1 m square, E=70 GPa, nu=0.3, k=5/6; "
    "0.2 m on 16×16",
    (("center_deflection", "m"),), 0.08,
    "Odd modes through 31 on each axis. Each check reports an absolute series tail bound in m.",
)
_THIN = replace(_PLATE, id="plate.thin-q4-navier-31.v1",
                scope=_PLATE.scope.replace("0.2 m on 16×16", "0.005 m on 8×8"),
                negative_case=True)
_NAVIER_IDS = {_PLATE.id, _THIN.id}
_SNAPSHOT = ReferenceRecord(
    "plate.thin-q4-snapshot.v1", ReferenceKind.SELF_REGRESSION,
    "ANF prototype validation record",
    "docs/validation.md, 0.005 thickness/side, 8×8, FE deflection column",
    _THIN.scope,
    (("center_deflection", "m"),), 1e-8, "Tolerance covers the published FE value's rounding.",
)
_STRUCTURE_REFERENCE = ReferenceRecord(
    "structure.cantilever-tip.v1", ReferenceKind.ANALYTIC,
    "R. Radovitzky, MIT 16.21, Principle of Virtual Forces (2005-03-18)",
    "Equations (12)–(17); see docs/frame2d.md for the cantilever observables",
    "One +x frame member, fixed i end, free j end, nonzero tip fy only",
    (("tip_uy", "m"), ("tip_rz", "rad"), ("root_fy", "N"), ("root_mz", "N·m")),
    1e-8, "Closed form from elastic bending and static equilibrium.",
)

_CLASSICAL_REFERENCES = MappingProxyType({
    record.id: record for record in (_BEAM, _PLATE, _THIN, _SNAPSHOT)
})
REFERENCE_REGISTRY = MappingProxyType({**_CLASSICAL_REFERENCES, _STRUCTURE_REFERENCE.id: _STRUCTURE_REFERENCE})
_RESIDUAL_ID = "free_dof_relative_residual"
_RESIDUAL_EVIDENCE = "static.relative-residual.v1"
_STRUCTURE_RESIDUAL = "planar_structure.relative-residual.v1"
_STRUCTURE_FAILURES = ("unstable_or_ill_conditioned", "numerical_range", "relative_residual_exceeded")


@dataclass(frozen=True)
class VerificationCheck:
    id: str
    observable: str
    observed: float | None
    reference: float | None
    unit: str
    metric: str
    threshold: float
    outcome: str
    evidence_id: str
    error: float | None = None
    error_scale: float | None = None
    reference_error_bound: float | None = None

    def __post_init__(self):
        if self.outcome not in ("pass", "fail", "not_applicable", "not_run"):
            raise ValueError("unknown check outcome")
        for name in ("id", "observable", "unit", "metric", "evidence_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("observed", "reference", "threshold", "error", "error_scale",
                     "reference_error_bound"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, finite(value, name))
        performed = self.outcome in ("pass", "fail")
        measurements = (self.observed, self.reference, self.error)
        if self.threshold is None or self.threshold < 0:
            raise ValueError("a check requires a nonnegative threshold")
        if performed:
            if any(value is None for value in measurements) or self.error < 0:
                raise ValueError("a performed check requires finite measurements")
            strict = self.evidence_id in _NAVIER_IDS
            passed = self.error < self.threshold if strict else self.error <= self.threshold
            if passed != (self.outcome == "pass"):
                raise ValueError("check outcome contradicts its error and threshold")
        elif any(value is not None for value in measurements):
            raise ValueError("an unperformed check cannot contain measurements")
        if self.error_scale is not None and self.error_scale <= 0:
            raise ValueError("error_scale must be positive")
        if self.reference_error_bound is not None and self.reference_error_bound < 0:
            raise ValueError("reference_error_bound must be nonnegative")


@dataclass(frozen=True)
class VerificationReport:
    input_hash: str | None
    input_status: str
    solve_status: str
    checks: tuple[VerificationCheck, ...]
    warnings: tuple[str, ...]
    scope: str
    schema_version: str = "1.0"
    diagnostic_code: str | None = None

    @property
    def reference_ids(self):
        return tuple(dict.fromkeys(
            check.evidence_id for check in self.checks if check.evidence_id in REFERENCE_REGISTRY
        ))

    @property
    def verification_status(self):
        if self.solve_status != "solved":
            return "not_run"
        comparisons = [check for check in self.checks
                       if check.evidence_id in REFERENCE_REGISTRY
                       and REFERENCE_REGISTRY[check.evidence_id].kind in _INDEPENDENT]
        if any(check.outcome == "fail" for check in comparisons):
            return "verification_failed"
        for reference_id in self.reference_ids:
            record = REFERENCE_REGISTRY[reference_id]
            required = set(dict(record.observable_units))
            passed = {check.observable for check in comparisons
                      if check.evidence_id == reference_id and check.outcome == "pass"}
            if not record.negative_case and required == passed:
                return "verification_passed"
        return "solved_unverified"

    def __post_init__(self):
        if self.schema_version not in ("1.0", "1.1"):
            raise ValueError("unsupported verification report schema")
        structural = self.schema_version == "1.1"
        if structural and self.solve_status == "numerical_failure":
            if self.diagnostic_code not in _STRUCTURE_FAILURES or len(self.warnings) != 1:
                raise ValueError("structural failure requires one diagnostic")
        elif self.diagnostic_code is not None:
            raise ValueError("unexpected diagnostic code")
        if self.input_status == "valid":
            if (self.solve_status not in ("solved", "numerical_failure")
                    or not isinstance(self.input_hash, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", self.input_hash)):
                raise ValueError("valid inputs require a hash and a solve outcome")
        elif (self.input_status != "invalid" or self.solve_status != "not_run"
              or self.input_hash is not None):
            raise ValueError("invalid inputs require a skipped solve and no input hash")

        if not isinstance(self.scope, str) or not self.scope.strip():
            raise ValueError("scope must be a nonempty string")
        if not isinstance(self.warnings, tuple) or any(not isinstance(w, str) for w in self.warnings):
            raise ValueError("warnings must be a tuple of strings")
        if not isinstance(self.checks, tuple) or not all(
            isinstance(check, VerificationCheck) for check in self.checks
        ):
            raise ValueError("checks must be a tuple of VerificationCheck values")
        if len({check.id for check in self.checks}) != len(self.checks):
            raise ValueError("check IDs must be unique")
        residuals = []
        residual_evidence = _STRUCTURE_RESIDUAL if structural else _RESIDUAL_EVIDENCE
        for check in self.checks:
            if check.evidence_id == residual_evidence:
                if check != _residual(check.observed, evidence=residual_evidence):
                    raise ValueError("invalid residual check")
                residuals.append(check)
            else:
                references = REFERENCE_REGISTRY if structural else _CLASSICAL_REFERENCES
                record = references.get(check.evidence_id)
                if (record is None or dict(record.observable_units).get(check.observable) != check.unit
                        or (structural and record is not _STRUCTURE_REFERENCE)
                        or check.id != f"{record.id}.{check.observable}"):
                    raise ValueError("check does not match its registered evidence")
                if check.outcome in ("pass", "fail"):
                    expected_check = _compare(record, check.observable, check.unit, check.observed,
                                              check.reference, zero_scale=check.error_scale or 0.0,
                                              bound=check.reference_error_bound)
                    if check != expected_check:
                        raise ValueError("check does not match its registered metric and threshold")
                elif check.threshold != record.threshold:
                    raise ValueError("unperformed check has the wrong reference threshold")
        expected = "pass" if self.solve_status == "solved" else "not_run"
        if len(residuals) != 1 or residuals[0].outcome != expected:
            raise ValueError("report requires a residual check consistent with the solve status")
        if expected == "not_run" and any(check.outcome != "not_run" for check in self.checks):
            raise ValueError("unsolved cases cannot have performed checks")

    def to_json(self):
        """Serialize the complete report using strict JSON."""
        data = asdict(self)
        if self.schema_version == "1.0":
            del data["diagnostic_code"]
        return _json({**data, "verification_status": self.verification_status,
                      "reference_ids": self.reference_ids})

    @classmethod
    def from_json(cls, payload):
        """Read a versioned report serialized by to_json."""
        if not isinstance(payload, str) or len(payload) > 65536:
            raise ValueError("expected a JSON report of at most 65536 characters")
        try:
            data = json.loads(payload, parse_constant=_reject_constant, object_pairs_hook=_unique_keys)
            if not isinstance(data, dict) or data.get("schema_version") not in ("1.0", "1.1"):
                raise ValueError("unsupported verification report schema")
            if ("diagnostic_code" in data) != (data["schema_version"] == "1.1"):
                raise ValueError("diagnostic code does not match the report schema")
            status = data.pop("verification_status")
            reference_ids = data.pop("reference_ids")
            if not all(isinstance(items, list) for items in (data["checks"], data["warnings"], reference_ids)):
                raise ValueError("checks, warnings and reference_ids must be JSON arrays")
            data["checks"] = tuple(VerificationCheck(**check) for check in data["checks"])
            data["warnings"] = tuple(data["warnings"])
            report = cls(**data)
        except (TypeError, KeyError, RecursionError) as error:
            raise ValueError("malformed verification report") from error
        if status != report.verification_status or reference_ids != list(report.reference_ids):
            raise ValueError("report status or references contradict its checks")
        return report


def _json(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _reject_constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


def _unique_keys(items):
    data = dict(items)
    if len(data) != len(items):
        raise ValueError("duplicate JSON keys")
    return data


def input_parameters(model, **loads):
    """Return the SI input dictionary used to fingerprint a beam or plate run."""
    if type(model) is Beam:
        normalized_loads = {"tip_force": 0.0, "uniform_load": 0.0}
        formulation, support = "euler_bernoulli_hermite", "cantilever"
    elif type(model) is Plate:
        normalized_loads = {"pressure": None}
        formulation, support = f"reissner_mindlin_{model.formulation}", model.support
    else:
        raise TypeError("model must be a Beam or Plate")
    if loads.keys() - normalized_loads.keys():
        raise ValueError("unknown load parameters")
    normalized_loads.update(loads)
    model_parameters = asdict(model)
    if type(model) is Plate and model.formulation == "q4_full":
        # Keep the canonical schema 1.0 Q4 input and its existing SHA-256.
        del model_parameters["formulation"]
    return {
        "formulation": formulation, "support": support, "units": "SI", "model": model_parameters,
        "loads": {name: finite(value, name) for name, value in normalized_loads.items()},
    }


def _reference(reference):
    if reference is None:
        return None
    key = reference.id if isinstance(reference, ReferenceRecord) else reference
    registered = _CLASSICAL_REFERENCES.get(key) if isinstance(key, str) else None
    if registered is None or (isinstance(reference, ReferenceRecord) and reference != registered):
        raise ValueError("reference must be an unchanged record or ID from REFERENCE_REGISTRY")
    return registered


def _plate_case(model, pressure):
    if model.formulation != "q4_full":
        return None
    if (model.lx, model.ly, model.young_modulus, model.poisson_ratio, model.shear_factor,
        model.support, pressure) != (1, 1, 70e9, 0.3, 5/6, "hard_simply_supported", 1000):
        return None
    return {(0.2, 16, 16): "thick", (0.005, 8, 8): "thin"}.get(
        (model.thickness, model.nx, model.ny)
    )


def _residual(observed=None, *, evidence=_RESIDUAL_EVIDENCE):
    return VerificationCheck(
        _RESIDUAL_ID, _RESIDUAL_ID, observed, 0.0 if observed is not None else None,
        "1", "absolute", MAX_RELATIVE_RESIDUAL,
        "pass" if observed is not None else "not_run", evidence, error=observed,
    )


def _unperformed(record, outcome):
    if record is None:
        return ()
    return tuple(
        VerificationCheck(f"{record.id}.{name}", name, None, None, unit,
                          "relative", record.threshold, outcome, record.id)
        for name, unit in record.observable_units
    )


def _compare(record, name, unit, observed, expected, *, zero_scale=0.0, bound=None):
    if expected:
        scale, metric = abs(expected), "relative"
    elif zero_scale:
        scale, metric = zero_scale, "scaled_absolute"
    else:
        scale, metric = None, "absolute"
    difference = abs(float(observed) - expected)
    error = difference if scale is None else difference / scale
    threshold = 0.0 if scale is None else record.threshold
    passed = error < threshold if record.id in _NAVIER_IDS else error <= threshold
    return VerificationCheck(f"{record.id}.{name}", name, observed, expected, unit, metric,
                             threshold, "pass" if passed else "fail", record.id,
                             error, scale, bound)


def _beam_checks(model, result, tip_force, uniform_load):
    length = model.length
    rigidity = model.young_modulus * model.second_moment
    tip, line = tip_force, uniform_load
    components = (
        (tip * length**3 / (3 * rigidity), line * length**4 / (8 * rigidity)),
        (tip * length**2 / (2 * rigidity), line * length**3 / (6 * rigidity)),
        (-tip, -line * length),
        (-tip * length, -line * length**2 / 2),
    )
    observations = (result.deflection[-1], result.slope[-1], *result.root_reaction)
    checks = []
    for (name, unit), observed, terms in zip(_BEAM.observable_units, observations, components, strict=True):
        if (tip and terms[0] == 0) or (line and terms[1] == 0):
            raise ArithmeticError("a nonzero beam reference term underflowed")
        checks.append(_compare(_BEAM, name, unit, observed, math.fsum(terms),
                               zero_scale=max(map(abs, terms))))
    return tuple(checks)


def _navier_center(model, pressure, modes=31):
    thickness, nu, modulus = model.thickness, model.poisson_ratio, model.young_modulus
    bending = modulus * thickness**3 / (12 * (1 - nu**2))
    shear = model.shear_factor * modulus * thickness / (2 * (1 + nu))

    def term(m, n):
        wave = (m * math.pi / model.lx)**2 + (n * math.pi / model.ly)**2
        return 16 * pressure / (math.pi**2 * m * n) * (1 / (bending * wave**2) + 1 / (shear * wave))

    odd = range(1, modes + 1, 2)
    value = math.fsum((-1)**((m + n - 2)//2) * term(m, n) for m in odd for n in odd)
    return value, abs(term(modes + 2, 1)) + abs(term(1, modes + 2))


def _check_result(model, result):
    if type(model) is Beam:
        points = (model.elements + 1,)
        expected = BeamResult
        shapes = {"x": points, "deflection": points, "slope": points, "root_reaction": (2,)}
    else:
        points = (model.ny + 1, model.nx + 1)
        expected = PlateResult
        shapes = {"x": (model.nx + 1,), "y": (model.ny + 1,), "deflection": points,
                  "rotations": (*points, 2), "reactions": (*points, 3)}
    if not isinstance(result, expected):
        raise TypeError(f"expected a {expected.__name__} for these inputs")
    for name, shape in shapes.items():
        array = getattr(result, name)
        if (not isinstance(array, np.ndarray) or array.shape != shape
                or array.dtype.kind not in "fiu" or not np.isfinite(array).all()):
            raise ArithmeticError(f"invalid result array: {name}")
    if result.x[0] != 0 or result.x[-1] != (model.length if type(model) is Beam else model.lx):
        raise ArithmeticError("result x coordinates do not match the input extent")
    if type(model) is Plate and (result.y[0] != 0 or result.y[-1] != model.ly):
        raise ArithmeticError("result y coordinates do not match the input extent")
    finite(result.strain_energy, "strain_energy")
    residual = finite(result.relative_residual, "relative_residual")
    if not 0 <= residual <= MAX_RELATIVE_RESIDUAL:
        raise ArithmeticError("free-DOF relative residual is outside the accepted range")
    return residual


def _evaluate(model, result, inputs, record):
    input_hash, scope = _describe_case(inputs)
    try:
        residual = _check_result(model, result)
    except (ArithmeticError, ValueError) as error:
        return _failed(input_hash, scope, record, error)

    warnings = []
    case = _plate_case(model, inputs["loads"]["pressure"]) if type(model) is Plate else None
    if case == "thin":
        warnings.append("Known shear locking in the fully integrated thin Q4 plate case.")
    if record is None:
        warnings.append("No independent reference was evaluated.")
        checks = ()
    else:
        try:
            checks = _compare_result(model, result, inputs["loads"], case, record)
        except (ArithmeticError, ValueError) as error:
            warnings.append(f"Reference check could not be evaluated: {error}")
            checks = _unperformed(record, "not_run")
        if checks[0].outcome == "not_applicable":
            warnings.append("Reference is outside its applicability domain for these inputs.")
        if record.kind not in _INDEPENDENT:
            warnings.append("This comparison does not establish independent accuracy.")
    if record is _THIN and checks[0].outcome == "pass":
        warnings.append("The thin Q4 negative baseline changed; review the locking evidence.")
    return VerificationReport(input_hash, "valid", "solved", (_residual(residual), *checks),
                              tuple(warnings), scope)


def _compare_result(model, result, loads, case, record):
    if record is _BEAM and type(model) is Beam:
        return _beam_checks(model, result, **loads)
    if (record is _PLATE and case == "thick") or (record in (_THIN, _SNAPSHOT) and case == "thin"):
        observed = result.deflection[model.ny//2, model.nx//2]
        expected, bound = (_navier_center(model, loads["pressure"]) if record.id in _NAVIER_IDS
                           else (5.56313372e-5, None))
        return (_compare(record, "center_deflection", "m", observed, expected, bound=bound),)
    return _unperformed(record, "not_applicable")


def _failed(input_hash, scope, record, error):
    return VerificationReport(input_hash, "valid", "numerical_failure",
                              (_residual(), *_unperformed(record, "not_run")),
                              (str(error),), scope)


def _describe_case(inputs):
    if inputs.get("problem_type") == "planar_structure":
        scope = "planar_structure, linear elastic statics, SI"
    else:
        scope = f"{inputs['formulation']}, {inputs['support']}, {inputs['units']}"
    return sha256(_json(inputs).encode("utf-8")).hexdigest(), scope


def _structure_cantilever(model, result):
    members = model.model["members"]
    if len(members) != 1:
        return _unperformed(_STRUCTURE_REFERENCE, "not_applicable")
    member, = members.values()
    root, tip = member["i"], member["j"]
    start, end = (model.model["nodes"][node] for node in (root, tip))
    nodal = model.loads["nodal"]
    if (member["kind"] != "frame2d" or end[0] <= start[0] or end[1] != start[1]
            or model.model["supports"] != {root: {"ux": 0, "uy": 0, "rz": 0}}
            or nodal.keys() != {tip} or nodal[tip].keys() != {"fy"}
            or model.loads["uniform_transverse"]):
        return _unperformed(_STRUCTURE_REFERENCE, "not_applicable")
    root_dofs, tip_dofs = (model.mesh.node_dofs[model.mesh.node_order.index(("input", node))]
                           for node in (root, tip))
    length = end[0] - start[0]
    force = nodal[tip]["fy"]
    rigidity = member["young_modulus"] * member["second_moment"]
    slope = (force / rigidity) * length * length / 2
    expected = (slope * length * (2/3), slope, -force, -force * length)
    if not all(math.isfinite(value) and value != 0 for value in expected):
        raise ArithmeticError("cantilever reference is outside the representable numerical range")
    observed = (result.displacement[tip_dofs[1]], result.displacement[tip_dofs[2]],
                result.reactions[root_dofs[1]], result.reactions[root_dofs[2]])
    return tuple(_compare(_STRUCTURE_REFERENCE, name, unit, actual, target)
                 for (name, unit), actual, target in
                 zip(_STRUCTURE_REFERENCE.observable_units, observed, expected, strict=True))


def _structure_report(model, inputs, reference, *, result=None, failure=None):
    input_hash, scope = _describe_case(inputs)
    record = _STRUCTURE_REFERENCE if reference is not None else None
    if failure is not None:
        return VerificationReport(
            input_hash, "valid", "numerical_failure",
            (_residual(evidence=_STRUCTURE_RESIDUAL), *_unperformed(record, "not_run")),
            (str(failure),), scope, "1.1", failure.code,
        )
    checks = ()
    warnings = ()
    if record is None:
        warnings = ("No independent reference was evaluated.",)
    else:
        try:
            checks = _structure_cantilever(model, result)
        except (ArithmeticError, ValueError) as error:
            checks = _unperformed(record, "not_run")
            warnings = (f"Reference check could not be evaluated: {error}",)
        if checks[0].outcome == "not_applicable":
            warnings = ("Reference is outside its applicability domain for these inputs.",)
    return VerificationReport(
        input_hash, "valid", "solved",
        (_residual(result.relative_residual, evidence=_STRUCTURE_RESIDUAL), *checks),
        warnings, scope, "1.1",
    )


def evaluate_checks(result, reference=None, *, inputs):
    """Check an existing BeamResult or PlateResult against its actual SI inputs."""
    record = _reference(reference)
    if isinstance(result, BeamResult):
        model_type = Beam
    elif isinstance(result, PlateResult):
        model_type = Plate
    else:
        raise TypeError("result must be a BeamResult or PlateResult")
    try:
        model = model_type(**inputs["model"])
        normalized = input_parameters(model, **inputs["loads"])
        if normalized != inputs:
            raise ValueError("inputs must contain the complete SI case from input_parameters")
    except (KeyError, TypeError, ValueError) as error:
        return _invalid(model_type, record, error)
    return _evaluate(model, result, normalized, record)


def _invalid(model_type, record, error):
    return VerificationReport(None, "invalid", "not_run",
                              (_residual(), *_unperformed(record, "not_run")),
                              (str(error),), model_type.__name__)


def _verify(model, model_type, solve, loads, reference):
    automatic = isinstance(reference, str) and reference == "auto"
    record = None if automatic else _reference(reference)
    try:
        if isinstance(model, Mapping):
            model = model_type(**model)
        if type(model) is not model_type:
            raise TypeError(f"model must be a {model_type.__name__} or its parameter mapping")
        inputs = input_parameters(model, **loads)
    except (TypeError, ValueError) as error:
        return _invalid(model_type, record, error)

    if automatic:
        if model_type is Beam:
            record = _BEAM
        else:
            case = _plate_case(model, inputs["loads"]["pressure"])
            record = {"thick": _PLATE, "thin": _THIN}.get(case)
    try:
        result = solve(model, **inputs["loads"])
    except (ArithmeticError, ValueError) as error:
        input_hash, scope = _describe_case(inputs)
        return _failed(input_hash, scope, record, error)
    return _evaluate(model, result, inputs, record)


def verify_cantilever(model, *, tip_force=0.0, uniform_load=0.0, reference="auto"):
    """Solve the cantilever and return a verification report, including failures."""
    return _verify(model, Beam, solve_cantilever,
                   {"tip_force": tip_force, "uniform_load": uniform_load}, reference)


def verify_plate(model, *, pressure, reference="auto"):
    """Solve a plate and return a verification report, including failures."""
    return _verify(model, Plate, solve_plate, {"pressure": pressure}, reference)
