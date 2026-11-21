"""Single-solve application entry point for declarative static analysis."""

from importlib.metadata import PackageNotFoundError, version
import platform

import numpy as np

from .artifacts import ResultBundle, _result_arrays
from .beam import Beam, solve_cantilever
from .model import ModelValidationError, _fields, validate_model
from .plate import solve_plate
from .verification import REFERENCE_REGISTRY, _describe_case, _failed, evaluate_checks, input_parameters


def run_analysis(model, options=None):
    """Solve a declaration once and return its result, diagnostics and verification report."""
    model = validate_model(model)
    options = _fields({} if options is None else options, "/options", (), {"reference": None})
    reference = options["reference"]
    if reference is not None and (not isinstance(reference, str) or reference not in REFERENCE_REGISTRY):
        raise ModelValidationError("/options/reference", "unsupported_reference",
                                   "expected null or an ID from REFERENCE_REGISTRY")

    inputs = input_parameters(model.model, **model.loads)
    solver = solve_cantilever if type(model.model) is Beam else solve_plate
    try:
        result = solver(model.model, **model.loads)
    except (ArithmeticError, ValueError) as error:
        input_hash, scope = _describe_case(inputs)
        record = None if reference is None else REFERENCE_REGISTRY[reference]
        report = _failed(input_hash, scope, record, error)
    else:
        report = evaluate_checks(result, reference, inputs=inputs)

    try:
        package = version("ai-native-fem")
    except PackageNotFoundError:
        package = "uninstalled"
    versions = {"anf": package, "python": platform.python_version(), "numpy": np.__version__}
    if report.solve_status == "numerical_failure":
        return ResultBundle(model, report, reference, versions, {})

    return ResultBundle(model, report, reference, versions, _result_arrays(result),
                        result.strain_energy, result.relative_residual)
