"""Bounded mesh refinement studies for declarative static models."""

from dataclasses import asdict, dataclass
from fractions import Fraction
import math
from types import MappingProxyType

from .application import run_analysis
from .artifacts import ResultBundle
from .beam import Beam
from .model import ModelSpec, ModelValidationError, _count, _fields, canonical_input_hash, validate_model
from .verification import _CLASSICAL_REFERENCES


_MAX_MESHES = 16
_MAX_OUTPUT_BYTES = 4 * 1024 * 1024
_OBSERVABLES = {"cantilever_bending": "tip_deflection", "rectangular_plate": "center_deflection"}


@dataclass(frozen=True, slots=True, init=False)
class StudySpec:
    """A fixed physical model with an explicit sequence of meshes and budgets."""

    model: ModelSpec
    cases: tuple[ModelSpec, ...]
    observable: str
    reference: str | None
    failure_policy: str
    max_solves: int
    max_output_bytes: int

    def __init__(self, model, meshes, observable, *, reference=None, failure_policy="stop",
                 max_solves=_MAX_MESHES, max_output_bytes=_MAX_OUTPUT_BYTES, base_model_hash=None):
        model = validate_model(model)
        if model.problem_type not in _OBSERVABLES:
            raise ModelValidationError("/study/model/problem_type", "unsupported_problem",
                                       "convergence studies support only beam and plate models")
        if base_model_hash is not None and base_model_hash != canonical_input_hash(model):
            raise ModelValidationError("/study/base_model_hash", "hash_mismatch", "base model fingerprint differs")
        if observable != _OBSERVABLES[model.problem_type]:
            raise ModelValidationError("/study/observable", "unsupported_observable",
                                       f"expected {_OBSERVABLES[model.problem_type]}")
        if reference is not None and (not isinstance(reference, str) or reference not in _CLASSICAL_REFERENCES):
            raise ModelValidationError("/study/reference", "unsupported_reference", "expected a registered ID")
        if failure_policy not in ("stop", "record_and_continue"):
            raise ModelValidationError("/study/failure_policy", "invalid_value", "expected stop or record_and_continue")
        if not isinstance(meshes, (tuple, list)) or not 1 <= len(meshes) <= _MAX_MESHES:
            raise ModelValidationError("/study/meshes", "invalid_value", f"expected 1–{_MAX_MESHES} explicit meshes")
        max_solves = _count(max_solves, "/study/max_solves", 1, _MAX_MESHES)
        max_output_bytes = _count(max_output_bytes, "/study/max_output_bytes", 1, _MAX_OUTPUT_BYTES)
        mesh_fields = ("elements",) if type(model.model) is Beam else ("nx", "ny")
        cases = []
        for index, mesh in enumerate(meshes):
            path = f"/study/meshes/{index}"
            doc = model.to_dict()
            doc["model"].update(_fields(mesh, path, mesh_fields))
            try:
                cases.append(validate_model(doc))
            except ModelValidationError as error:
                raise ModelValidationError(path + error.path.removeprefix("/model"),
                                           error.code, error.message) from error

        object.__setattr__(self, "model", model)
        object.__setattr__(self, "cases", tuple(cases))
        object.__setattr__(self, "observable", observable)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "failure_policy", failure_policy)
        object.__setattr__(self, "max_solves", max_solves)
        object.__setattr__(self, "max_output_bytes", max_output_bytes)

    @classmethod
    def from_dict(cls, doc):
        """Validate a study declaration, including one read from a saved report."""
        return cls(**_fields(doc, "/study", ("model", "meshes", "observable"), {
            "reference": None, "failure_policy": "stop", "max_solves": _MAX_MESHES,
            "max_output_bytes": _MAX_OUTPUT_BYTES, "base_model_hash": None,
        }))

    @property
    def meshes(self):
        mesh_fields = ("elements",) if type(self.model.model) is Beam else ("nx", "ny")
        return tuple({name: getattr(case.model, name) for name in mesh_fields} for case in self.cases)

    def to_dict(self):
        """Return the complete executable study declaration."""
        return {"model": self.model.to_dict(), "base_model_hash": canonical_input_hash(self.model),
                "meshes": list(self.meshes), "observable": self.observable, "reference": self.reference,
                "failure_policy": self.failure_policy, "max_solves": self.max_solves,
                "max_output_bytes": self.max_output_bytes}


@dataclass(frozen=True, slots=True)
class _Step:
    input_hash: str
    result: ResultBundle | None
    observed_m: float | None
    successive_mesh_change_m: float | None
    skipped_reason: str | None = None

    @property
    def solve_status(self):
        return self.result.solve_status if self.result is not None else "not_run"

    def reference_check(self, observable):
        if self.result is None or self.result.reference is None:
            return None
        return next((check for check in self.result.report.checks
                     if check.evidence_id == self.result.reference and check.observable == observable), None)

    def to_dict(self, mesh, observable):
        check = self.reference_check(observable)
        result = self.result
        diagnostics = [{"code": self.skipped_reason, "message": "This mesh was not attempted."}]
        if result is not None:
            diagnostics = list(result.diagnostics)
        return {
            "mesh": mesh, "input_hash": self.input_hash, "solve_status": self.solve_status,
            "verification_status": result.verification_status if result is not None else "not_run",
            "observed_m": self.observed_m, "successive_mesh_change_m": self.successive_mesh_change_m,
            "relative_residual": result.relative_residual if result is not None else None,
            "reference_check": asdict(check) if check is not None else None,
            "reference_kind": _CLASSICAL_REFERENCES[check.evidence_id].kind.value if check is not None else None,
            "diagnostics": diagnostics,
        }


@dataclass(frozen=True, slots=True)
class StudyResult:
    """Recorded outcomes and the conditional estimate for one declared study."""

    spec: StudySpec
    steps: tuple[_Step, ...]
    estimate: MappingProxyType

    @property
    def status(self):
        return "complete" if all(step.solve_status == "solved" for step in self.steps) else "partial"

    def to_dict(self):
        return {"schema_version": "1.0", "study": self.spec.to_dict(), "status": self.status,
                "steps": [step.to_dict(mesh, self.spec.observable)
                          for step, mesh in zip(self.steps, self.spec.meshes, strict=True)],
                "estimate": dict(self.estimate)}


def _deflection(result):
    values = result.arrays["deflection"]
    if values.ndim == 1:
        return float(values[-1])
    ny, nx = result.model.model.ny, result.model.model.nx
    center = values[ny // 2:ny // 2 + 1 + ny % 2, nx // 2:nx // 2 + 1 + nx % 2]
    return math.fsum(float(value) / center.size for value in center.flat)


def _estimate(spec, steps):
    def unavailable(reason):
        return {"status": "unavailable", "reason": reason, "observed_order": None,
                "finest_discretization_estimate_m": None, "refinement_ratio": None}

    if len(steps) < 3:
        return unavailable("insufficient_meshes")
    if any(step.observed_m is None for step in steps):
        return unavailable("incomplete_sequence")

    meshes = spec.meshes
    ratios = {Fraction(after[name], before[name])
              for before, after in zip(meshes, meshes[1:]) for name in before}
    if len(ratios) != 1 or min(ratios) <= 1:
        return unavailable("inconsistent_refinement")

    values = [step.observed_m for step in steps]
    scale = max(map(abs, values))
    if scale == 0:
        return unavailable("near_zero_change")
    minimum = 8 * math.ulp(scale) / scale
    for step in steps:
        check = step.reference_check(spec.observable)
        if check is not None and check.reference is not None and abs(check.reference) <= minimum * scale:
            return unavailable("near_zero_reference")
    changes = [b / scale - a / scale for a, b in zip(values, values[1:])]
    if any(abs(change) <= minimum for change in changes):
        return unavailable("near_zero_change")
    if min(changes) < 0 < max(changes):
        return unavailable("non_monotone_sequence")
    if any(abs(after) >= abs(before) for before, after in zip(changes, changes[1:])):
        return unavailable("non_decreasing_change")

    refinement = float(ratios.pop())
    coarse, fine = abs(changes[-2]), abs(changes[-1])
    ratio = coarse / fine
    if ratio <= 1:
        return unavailable("non_decreasing_change")
    order = math.log(ratio) / math.log(refinement)
    error = scale * (fine / (ratio - 1))
    if not math.isfinite(error):
        return unavailable("out_of_range")
    return {"status": "estimated", "reason": None, "observed_order": order,
            "finest_discretization_estimate_m": error, "refinement_ratio": refinement}


def run_convergence(model, meshes=None, observable=None, **options):
    """Solve each selected mesh once and keep skipped and failed steps in order."""
    if type(model) is StudySpec:
        if meshes is not None or observable is not None or options:
            raise TypeError("a StudySpec already declares meshes, observable and options")
        spec = model
    else:
        spec = StudySpec(model, meshes, observable, **options)
    steps = []
    previous = None
    stopped = False
    for index, case in enumerate(spec.cases):
        if stopped or index >= spec.max_solves:
            reason = "previous_failure" if stopped else "solve_budget"
            steps.append(_Step(canonical_input_hash(case), None, None, None, reason))
            continue
        result = run_analysis(case, {"reference": spec.reference})
        if result.solve_status == "numerical_failure":
            steps.append(_Step(result.input_hash, result, None, None))
            stopped = spec.failure_policy == "stop"
            previous = None
            continue
        observed = _deflection(result)
        change = abs(observed - previous) if previous is not None else None
        if change is not None and not math.isfinite(change):
            change = None
        steps.append(_Step(result.input_hash, result, observed, change))
        previous = observed
    return StudyResult(spec, tuple(steps), MappingProxyType(_estimate(spec, steps)))
