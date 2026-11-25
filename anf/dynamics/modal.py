"""Dense generalized modal analysis for schema 1.4 planar structures."""

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import math
import platform

import numpy as np

from ..model import (ModelValidationError, _fields, canonical_modal_input_hash,
                     validate_modal_model)
from .mass import assemble_mass
from .system import assemble_stiffness, free_dofs


class ModalSolveError(ArithmeticError):
    """A stable machine diagnostic for a failed modal solve."""

    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ModalResult:
    """Mass-normalized modes, frequencies and residuals for a schema 1.4 model."""

    model: object
    eigenvalues: np.ndarray
    omega_rad_s: np.ndarray
    frequencies_hz: np.ndarray
    modes: np.ndarray
    dof_labels: tuple
    residuals: np.ndarray
    rigid_mode_count: int
    truncated: bool
    diagnostics: tuple
    versions: dict
    input_hash: str

    def __post_init__(self):
        count = len(self.eigenvalues)
        if self.modes.shape != (len(self.dof_labels), count):
            raise ValueError("mode shape dimensions do not match the DOF labels")
        if not all(len(values) == count for values in
                   (self.omega_rad_s, self.frequencies_hz, self.residuals)):
            raise ValueError("modal result arrays must have the same length")
        if not all(np.isfinite(values).all() for values in
                   (self.eigenvalues, self.omega_rad_s, self.frequencies_hz,
                    self.modes, self.residuals)):
            raise ValueError("modal result values must be finite")


def _free_dofs(spec, mesh):
    """Modal wrapper that keeps the historical ModalSolveError contract."""
    try:
        return free_dofs(spec, mesh)
    except ModelValidationError as error:
        raise ModalSolveError(error.code, error.message) from error


def _versions():
    try:
        package = version("ai-native-fem")
    except PackageNotFoundError:
        package = "uninstalled"
    try:
        import scipy
    except ImportError:  # pragma: no cover - solve_modes imports SciPy first
        scipy_version = "uninstalled"
    else:
        scipy_version = scipy.__version__
    return {"anf": package, "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy_version}


def solve_modes(model, count=None, options=None):
    """Solve K phi = lambda M phi for a schema 1.4 planar model."""
    _fields({} if options is None else options, "/options", ())
    spec = validate_modal_model(model)
    mesh = spec.mesh
    if mesh is None:
        raise TypeError("modal model did not produce a structure mesh")
    if count is None:
        count = spec.modes
    if type(count) is not int or not 1 <= count <= 20:
        raise ModelValidationError("/count", "invalid_value",
                                   "expected an integer from 1 to 20")

    try:
        from scipy.linalg import eigh
    except (ImportError, ModuleNotFoundError) as error:
        raise ModalSolveError(
            "backend_unavailable",
            "modal analysis requires the ai-native-fem[sparse] extra (SciPy)",
        ) from error

    free, supported = _free_dofs(spec, mesh)
    if not free.size:
        raise ModalSolveError("modal_no_free_dofs", "the model has no free degrees of freedom")
    stiffness = assemble_stiffness(spec)
    mass_matrix = assemble_mass(spec)
    stiffness_free = stiffness[np.ix_(free, free)]
    mass_free = mass_matrix.matrix[np.ix_(free, free)]
    try:
        eigenvalues, modes = eigh(stiffness_free, mass_free)
    except np.linalg.LinAlgError as error:
        raise ModalSolveError("modal_not_converged",
                              "generalized eigenproblem did not converge") from error

    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    rigid_limit = 1e-10 * scale
    if np.any(eigenvalues < -rigid_limit):
        raise ModalSolveError("unstable_or_ill_conditioned",
                              "the generalized stiffness has a negative eigenvalue")
    rigid_count = int(np.sum(np.abs(eigenvalues) <= rigid_limit))
    if not supported and rigid_count != 3:
        raise ModalSolveError(
            "rigid_mode_mismatch",
            f"a free planar model requires three rigid modes, found {rigid_count}",
        )
    if supported and rigid_count:
        raise ModalSolveError("unstable_or_ill_conditioned",
                              "a supported model has a zero-energy mode")

    truncated = count > eigenvalues.size
    take = min(count, eigenvalues.size)
    eigenvalues = eigenvalues[:take]
    modes = modes[:, :take]
    stiffness_norm = np.linalg.norm(stiffness_free)
    residuals = []
    for index in range(take):
        stiffness_mode = stiffness_free @ modes[:, index]
        mass_mode = mass_free @ modes[:, index]
        if abs(eigenvalues[index]) <= rigid_limit:
            scale_value = stiffness_norm * np.linalg.norm(modes[:, index])
            residual = np.linalg.norm(stiffness_mode) / scale_value if scale_value else 0.0
            limit = 1e-8
        else:
            scale_value = (np.linalg.norm(stiffness_mode)
                           + abs(eigenvalues[index]) * np.linalg.norm(mass_mode))
            residual = (np.linalg.norm(stiffness_mode - eigenvalues[index] * mass_mode) / scale_value
                        if scale_value else 0.0)
            limit = 1e-9
        residuals.append(float(residual))
        if residual > limit:
            raise ModalSolveError("modal_not_converged",
                                  "one or more eigenpairs exceed the relative residual gate")
    residuals = np.asarray(residuals)
    for index in range(take):
        largest = int(np.argmax(np.abs(modes[:, index])))
        if modes[largest, index] < 0:
            modes[:, index] *= -1
    orthogonality = modes.T @ mass_free @ modes - np.eye(take)
    if np.max(np.abs(orthogonality)) > 1e-8:
        raise ModalSolveError("modal_not_converged",
                              "mode shapes are not mass-normalized")
    omega = np.sqrt(np.maximum(eigenvalues, 0.0))
    labels = tuple(mass_matrix.dof_labels[index] for index in free)
    for array in (eigenvalues, omega, modes, residuals):
        array.setflags(write=False)
    return ModalResult(
        model=spec,
        eigenvalues=eigenvalues,
        omega_rad_s=omega,
        frequencies_hz=omega / (2 * math.pi),
        modes=modes,
        dof_labels=labels,
        residuals=residuals,
        rigid_mode_count=rigid_count,
        truncated=truncated,
        diagnostics=("modal_truncated",) if truncated else (),
        versions=_versions(),
        input_hash=canonical_modal_input_hash(spec),
    )


__all__ = ["ModalResult", "ModalSolveError", "solve_modes"]