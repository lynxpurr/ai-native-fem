"""Single-solve application entry point for declarative static analysis."""

from importlib.metadata import PackageNotFoundError, version
import platform
from types import MappingProxyType

import numpy as np

from .artifacts import LoadCaseResults, ResultBundle, _DTYPE, _case_result, _result_arrays
from .beam import Beam, _solve_cantilever, solve_cantilever
from .dynamics import harmonic_response, solve_modes
from .loads import _case_hash, _request, _resolve_cases
from .model import (ModelValidationError, _choice, _fields, _physical_inputs,
                    validate_load_case_model, validate_modal_model, validate_model)
from .plate import _solve_plate, solve_plate
from .recovery import _section_arrays
from .structures import StructureSolveError, _solve_structure, solve_structure
from .verification import (_CLASSICAL_REFERENCES, _STRUCTURE_REFERENCE, _describe_case, _failed,
                           _structure_report, evaluate_checks, input_parameters)


def run_analysis(model, options=None):
    """Solve a declaration once and return its result, diagnostics and verification report."""
    model = validate_model(model)
    options = _fields({} if options is None else options, "/options", (),
                      {"reference": None, "backend": "dense"})
    reference = options["reference"]
    backend = _choice(options["backend"], "/options/backend", ("dense", "scipy"), "unsupported_backend")
    structural = model.schema_version == "1.1"
    references = (_STRUCTURE_REFERENCE.id,) if structural else _CLASSICAL_REFERENCES
    if reference is not None and (not isinstance(reference, str) or reference not in references):
        raise ModelValidationError("/options/reference", "unsupported_reference",
                                   "expected null or a registered ID for this model schema"
                                   if structural else "expected null or an ID from REFERENCE_REGISTRY")

    sparse = _sparse_backend() if backend == "scipy" else None
    if structural:
        inputs = _physical_inputs(model)
        try:
            result = (solve_structure(model) if sparse is None else
                      _solve_structure(model, model.loads["nodal"], sparse=sparse))
        except StructureSolveError as error:
            result = None
            report = _structure_report(model, inputs, reference, failure=error)
        else:
            report = _structure_report(model, inputs, reference, result=result)
    else:
        result, report = _run_classical(model, reference, sparse=sparse)

    versions = _versions()
    execution = None if sparse is None else sparse.metrics
    if report.solve_status == "numerical_failure":
        return ResultBundle(model, report, reference, versions, {}, execution=execution)

    return ResultBundle(model, report, reference, versions, _result_arrays(result),
                        result.strain_energy, result.relative_residual, execution=execution)


def run_modal(model, options=None):
    """Solve the modal problem for a schema 1.4 declaration."""
    options = _fields({} if options is None else options, "/options", (), {"count": None})
    return solve_modes(validate_modal_model(model), count=options["count"])


def run_harmonic(model, options=None):
    """Solve the steady-state harmonic response for a schema 1.5 declaration."""
    _fields({} if options is None else options, "/options", ())
    return harmonic_response(model)


def run_load_cases(model, *, ids=None, stations=None):
    """Solve selected static cases and combinations in a schema 1.2 or 1.3 declaration."""
    model = validate_load_case_model(model)
    ids, stations = _request(model, ids, stations)
    cases = [(name, kind, effective, _case_hash(model, effective))
             for name, kind, effective in _resolve_cases(model, ids)]
    results = {}
    for name, kind, effective, fingerprint in cases:
        diagnostic = None
        arrays, energy, residual = {}, None, None
        try:
            value = _solve_structure(model, effective.nodal, effective.member_line)
            sections = _section_arrays(model, value, effective.member_line, stations)
        except StructureSolveError as error:
            diagnostic = (error.code, str(error))
        except ArithmeticError as error:
            diagnostic = ("numerical_range", str(error))
        else:
            arrays = _result_arrays(value)
            arrays.update({key: np.asarray(array, dtype=_DTYPE) for key, array in sections.items()})
            energy, residual = value.strain_energy, value.relative_residual
        results[name] = _case_result(model, stations, name, kind, fingerprint, arrays, energy,
                                     residual, diagnostic)
    return LoadCaseResults(model, stations, MappingProxyType(_versions()), MappingProxyType(results))


def _versions():
    try:
        package = version("ai-native-fem")
    except PackageNotFoundError:
        package = "uninstalled"
    return {"anf": package, "python": platform.python_version(), "numpy": np.__version__}


def _sparse_backend():
    try:
        from .backends.scipy import SparseSolve
    except ImportError as error:
        raise ModelValidationError(
            "/options/backend", "backend_unavailable",
            "the scipy backend requires the ai-native-fem[sparse] extra",
        ) from error
    return SparseSolve()


def _run_classical(model, reference, *, sparse=None):
    inputs = input_parameters(model.model, **model.loads)
    solver = solve_cantilever if type(model.model) is Beam else solve_plate
    try:
        if sparse is None:
            result = solver(model.model, **model.loads)
        else:
            sparse_solver = _solve_cantilever if type(model.model) is Beam else _solve_plate
            result = sparse_solver(model.model, **model.loads, sparse=sparse)
    except (ArithmeticError, ValueError) as error:
        input_hash, scope = _describe_case(inputs)
        result = None
        record = None if reference is None else _CLASSICAL_REFERENCES[reference]
        report = _failed(input_hash, scope, record, error)
    else:
        report = evaluate_checks(result, reference, inputs=inputs)

    return result, report
