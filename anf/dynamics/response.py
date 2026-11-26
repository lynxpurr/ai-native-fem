"""Dense steady-state harmonic response for schema 1.5 planar structures."""

from dataclasses import dataclass
import math

import numpy as np

from ..model import (ModelValidationError, canonical_dynamics_input_hash,
                     validate_dynamics_model)
from .mass import assemble_mass
from .system import assemble_stiffness, dof_labels, free_dofs, versions

_COMPONENTS = ("fx", "fy", "mz")
_ILL_CONDITIONED = 1e11
_SINGULAR = 1e15
_MAX_BACKWARD_ERROR = 1e-12


class HarmonicSolveError(ArithmeticError):
    """A stable machine diagnostic for a failed harmonic solve."""

    def __init__(self, code, message, *, index=None, frequency_hz=None, kappa=None):
        self.code = code
        self.index = index
        self.frequency_hz = frequency_hz
        self.kappa = kappa
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class HarmonicResult:
    """Complex steady-state response, magnitudes and phases for schema 1.5."""

    model: object
    frequencies_hz: np.ndarray
    omega_rad_s: np.ndarray
    displacements: np.ndarray
    magnitudes: np.ndarray
    phases_deg: np.ndarray
    dof_labels: tuple
    backward_errors: np.ndarray
    conditions: np.ndarray
    diagnostics: tuple
    versions: dict
    input_hash: str

    def __post_init__(self):
        count = len(self.frequencies_hz)
        expected = (len(self.dof_labels), count)
        if self.displacements.shape != expected:
            raise ValueError("displacement dimensions do not match the DOF labels")
        for values in (self.magnitudes, self.phases_deg):
            if values.shape != expected:
                raise ValueError("magnitude and phase dimensions must match the displacements")
        for values in (self.omega_rad_s, self.backward_errors, self.conditions):
            if values.shape != (count,):
                raise ValueError("per-frequency arrays must match the frequency count")
        if not np.isfinite(self.displacements).all():
            raise ValueError("harmonic response values must be finite")
        for values in (self.omega_rad_s, self.magnitudes, self.phases_deg,
                       self.backward_errors, self.conditions):
            if not np.isfinite(values).all():
                raise ValueError("harmonic response values must be finite")


def _linalg():
    try:
        from scipy.linalg import lapack, lu_factor, lu_solve
    except (ImportError, ModuleNotFoundError) as error:
        raise ModelValidationError(
            "/analysis", "backend_unavailable",
            "harmonic response requires the ai-native-fem[sparse] extra") from error
    return lapack, lu_factor, lu_solve


def _force(spec, mesh, free):
    position = {dof: index for index, dof in enumerate(free)}
    nodes = {key[1]: index for index, key in enumerate(mesh.node_order)
             if key[0] == "input"}
    force = np.zeros(len(free), dtype=complex)
    for name, row in spec.loads["nodal"].items():
        index = nodes[name]
        for component, entry in row.items():
            dof = mesh.node_dofs[index][_COMPONENTS.index(component)]
            if dof not in position:
                raise ModelValidationError(
                    f"/loads/nodal/{name}/{component}", "unsupported_load",
                    "loads on fixed DOFs are not supported because reactions are not computed")
            phase = math.radians(entry["phase_deg"])
            force[position[dof]] += entry["amplitude"] * complex(math.cos(phase),
                                                                 math.sin(phase))
    return force


def _condition(matrix, stiffness, mass, damping, omega, lu, lapack):
    anorm = float(np.linalg.norm(matrix, 1))
    if not math.isfinite(anorm) or anorm == 0.0:
        return math.inf
    rcond, info = lapack.zgecon(lu, anorm)
    if info != 0 or not math.isfinite(rcond) or rcond <= 0.0:
        return math.inf
    inverse = 1.0 / (rcond * anorm)
    scale = (float(np.linalg.norm(stiffness, 1))
             + omega**2 * float(np.linalg.norm(mass, 1))
             + omega * float(np.linalg.norm(damping, 1)))
    return inverse * scale


def _solve(spec, lapack, lu_factor, lu_solve):
    mesh = spec.mesh
    free, _ = free_dofs(spec, mesh)
    if not len(free):
        raise ModelValidationError("/model/supports", "no_free_dofs",
                                   "the model has no free degrees of freedom")
    stiffness = assemble_stiffness(spec)[np.ix_(free, free)]
    mass = assemble_mass(spec).matrix[np.ix_(free, free)]
    coefficients = spec.dynamics.damping
    damping = coefficients.alpha * mass + coefficients.beta * stiffness
    force = _force(spec, mesh, free)

    frequencies = np.asarray(spec.dynamics.frequencies_hz, dtype=float)
    count = len(frequencies)
    displacements = np.zeros((mesh.dof_count, count), dtype=complex)
    backward_errors = np.zeros(count)
    conditions = np.zeros(count)
    diagnostics = []
    for index, frequency in enumerate(frequencies):
        omega = 2.0 * math.pi * frequency
        matrix = stiffness - omega**2 * mass + 1j * omega * damping
        try:
            lu, piv = lu_factor(matrix)
            response = lu_solve((lu, piv), force)
        except (np.linalg.LinAlgError, ValueError) as error:
            raise HarmonicSolveError(
                "harmonic_singular", f"frequency {frequency} Hz could not be solved",
                index=index, frequency_hz=float(frequency)) from error
        kappa = _condition(matrix, stiffness, mass, damping, omega, lu, lapack)
        if kappa > _SINGULAR:
            raise HarmonicSolveError(
                "harmonic_singular",
                f"frequency {frequency} Hz is numerically singular (kappa {kappa:.3e})",
                index=index, frequency_hz=float(frequency), kappa=kappa)
        denominator = (float(np.linalg.norm(matrix, 1)) * float(np.linalg.norm(response))
                       + float(np.linalg.norm(force)))
        error = (float(np.linalg.norm(matrix @ response - force)) / denominator
                 if denominator else 0.0)
        if not np.isfinite(response).all() or error > _MAX_BACKWARD_ERROR:
            raise HarmonicSolveError(
                "harmonic_not_converged",
                f"frequency {frequency} Hz exceeds the backward-error gate",
                index=index, frequency_hz=float(frequency), kappa=kappa)
        if kappa > _ILL_CONDITIONED:
            diagnostics.append(f"harmonic_ill_conditioned:{index}")
        displacements[free, index] = response
        backward_errors[index] = error
        conditions[index] = kappa

    magnitudes = np.abs(displacements)
    return HarmonicResult(
        model=spec,
        frequencies_hz=frequencies,
        omega_rad_s=2.0 * math.pi * frequencies,
        displacements=displacements,
        magnitudes=magnitudes,
        phases_deg=np.degrees(np.angle(displacements)),
        dof_labels=dof_labels(mesh),
        backward_errors=backward_errors,
        conditions=conditions,
        diagnostics=tuple(diagnostics),
        versions=versions(),
        input_hash=canonical_dynamics_input_hash(spec),
    )


def harmonic_response(model):
    """Solve the steady-state response of a schema 1.5 harmonic declaration."""
    spec = validate_dynamics_model(model)
    if spec.analysis_type != "harmonic":
        raise ModelValidationError("/analysis/type", "unsupported_analysis",
                                   "harmonic_response requires a harmonic analysis")
    lapack, lu_factor, lu_solve = _linalg()
    return _solve(spec, lapack, lu_factor, lu_solve)


__all__ = ["HarmonicResult", "HarmonicSolveError", "harmonic_response"]