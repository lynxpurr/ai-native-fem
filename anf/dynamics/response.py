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
_MAX_NEWMARK_ERROR = 1e-10


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


__all__ = ["HarmonicResult", "HarmonicSolveError", "NewmarkSolveError",
           "TransientResult", "harmonic_response", "transient_response"]

@dataclass(frozen=True, slots=True)
class TransientResult:
    """Linear Newmark history, energy and summary peaks for schema 1.5."""

    model: object
    times_s: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    energy: np.ndarray
    dof_labels: tuple
    start_s: float
    dt_s: float
    peak_displacement: np.ndarray
    peak_time_s: np.ndarray
    global_peaks: dict
    worst_backward_error: float
    worst_backward_step: int
    diagnostics: tuple
    versions: dict
    input_hash: str

    def __post_init__(self):
        count = len(self.times_s)
        expected = (count, len(self.dof_labels))
        for values in (self.displacement, self.velocity, self.acceleration):
            if values.shape != expected:
                raise ValueError("history dimensions must match the time points and DOF labels")
        if self.energy.shape != (count,):
            raise ValueError("energy must match the time points")
        for values in (self.peak_displacement, self.peak_time_s):
            if values.shape != (len(self.dof_labels),):
                raise ValueError("peak arrays must match the DOF labels")
        for values in (self.times_s, self.displacement, self.velocity,
                       self.acceleration, self.energy, self.peak_displacement,
                       self.peak_time_s):
            if not np.isfinite(values).all():
                raise ValueError("transient result values must be finite")
        if not isinstance(self.dt_s, float) or not math.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError("the time step must be finite and positive")


class NewmarkSolveError(ArithmeticError):
    """A stable machine diagnostic for a failed transient step."""

    def __init__(self, code, message, *, step=None, time_s=None):
        self.code = code
        self.step = step
        self.time_s = time_s
        super().__init__(message)


def _linalg_cholesky():
    try:
        from scipy.linalg import cho_factor, cho_solve
    except (ImportError, ModuleNotFoundError) as error:
        raise ModelValidationError(
            "/analysis", "backend_unavailable",
            "transient response requires the ai-native-fem[sparse] extra") from error
    return cho_factor, cho_solve


def _transient_presets(spec, mesh, free):
    position = {dof: index for index, dof in enumerate(free)}
    nodes = {key[1]: index for index, key in enumerate(mesh.node_order)
             if key[0] == "input"}
    entries = []
    for name, row in spec.loads["nodal"].items():
        index = nodes[name]
        for component, preset in row.items():
            dof = mesh.node_dofs[index][_COMPONENTS.index(component)]
            if dof not in position:
                raise ModelValidationError(
                    f"/loads/nodal/{name}/{component}", "unsupported_load",
                    "loads on fixed DOFs are not supported because reactions are not computed")
            entries.append((position[dof], preset))
    return entries


def _preset_value(preset, time):
    kind = preset["kind"]
    amplitude = preset["amplitude"]
    if kind == "constant":
        return amplitude
    if kind == "ramp":
        start, end = preset["t0_s"], preset["t1_s"]
        if time <= start:
            return 0.0
        if time >= end:
            return amplitude
        return amplitude * (time - start) / (end - start)
    return amplitude * math.cos(2.0 * math.pi * preset["frequency_hz"] * time
                                + math.radians(preset["phase_deg"]))


def _transient_force(entries, count, time):
    force = np.zeros(count)
    for index, preset in entries:
        force[index] += _preset_value(preset, time)
    return force


def _state_vector(mesh, free, mapping):
    position = {dof: index for index, dof in enumerate(free)}
    nodes = {key[1]: index for index, key in enumerate(mesh.node_order)
             if key[0] == "input"}
    values = np.zeros(len(free))
    for name, components in mapping.items():
        index = nodes[name]
        for component, value in components.items():
            values[position[mesh.node_dofs[index][_COMPONENTS_STATE.index(component)]]] = value
    return values


_COMPONENTS_STATE = ("ux", "uy", "rz")


def _peak(values, times):
    flat = int(np.abs(values).argmax())
    row, column = np.unravel_index(flat, values.shape)
    return (float(abs(values[row, column])), float(times[row]), int(column))


def _solve_transient(spec, cho_factor, cho_solve):
    mesh = spec.mesh
    free, _ = free_dofs(spec, mesh)
    if not len(free):
        raise ModelValidationError("/model/supports", "no_free_dofs",
                                   "the model has no free degrees of freedom")
    stiffness_full = assemble_stiffness(spec)
    mass_full = assemble_mass(spec).matrix
    stiffness = stiffness_full[np.ix_(free, free)]
    mass = mass_full[np.ix_(free, free)]
    coefficients = spec.dynamics.damping
    damping = coefficients.alpha * mass + coefficients.beta * stiffness

    grid = spec.dynamics.time
    times = np.asarray(grid.times_s, dtype=float)
    dt = float(grid.dt_s)
    steps = int(grid.steps)
    gamma, beta = 0.5, 0.25
    effective = (stiffness + gamma / (beta * dt) * damping
                 + 1.0 / (beta * dt**2) * mass)
    try:
        factor = cho_factor(effective)
        mass_factor = cho_factor(mass)
    except np.linalg.LinAlgError as error:
        raise NewmarkSolveError(
            "newmark_singular", "the effective or mass matrix could not be factored") from error

    entries = _transient_presets(spec, mesh, free)
    forces = np.array([_transient_force(entries, len(free), time) for time in times])
    displacement = np.zeros((steps + 1, len(free)))
    velocity = np.zeros((steps + 1, len(free)))
    acceleration = np.zeros((steps + 1, len(free)))
    displacement[0] = _state_vector(mesh, free, spec.dynamics.initial_displacement)
    velocity[0] = _state_vector(mesh, free, spec.dynamics.initial_velocity)
    acceleration[0] = cho_solve(mass_factor,
                                forces[0] - damping @ velocity[0] - stiffness @ displacement[0])
    if not np.isfinite(acceleration[0]).all():
        raise NewmarkSolveError("newmark_non_finite",
                                "the initial acceleration is not finite", step=0,
                                time_s=float(times[0]))
    initial_residual = (mass @ acceleration[0] + damping @ velocity[0]
                        + stiffness @ displacement[0] - forces[0])
    initial_scale = (float(np.linalg.norm(mass @ acceleration[0]))
                     + float(np.linalg.norm(damping @ velocity[0]))
                     + float(np.linalg.norm(stiffness @ displacement[0]))
                     + float(np.linalg.norm(forces[0])))
    worst_error = (float(np.linalg.norm(initial_residual)) / initial_scale
                   if initial_scale else 0.0)
    worst_step = 0
    if not math.isfinite(worst_error) or worst_error > _MAX_NEWMARK_ERROR:
        raise NewmarkSolveError("newmark_not_converged",
                                "the initial state exceeds the backward-error gate",
                                step=0, time_s=float(times[0]))
    for step in range(steps):
        rhs = (forces[step + 1]
               + mass @ (displacement[step] / (beta * dt**2)
                         + velocity[step] / (beta * dt)
                         + (1.0 / (2.0 * beta) - 1.0) * acceleration[step])
               + damping @ (gamma / (beta * dt) * displacement[step]
                            + (gamma / beta - 1.0) * velocity[step]
                            + dt * (gamma / (2.0 * beta) - 1.0) * acceleration[step]))
        displacement[step + 1] = cho_solve(factor, rhs)
        velocity[step + 1] = (
            gamma / (beta * dt) * (displacement[step + 1] - displacement[step])
            + (1.0 - gamma / beta) * velocity[step]
            + dt * (1.0 - gamma / (2.0 * beta)) * acceleration[step])
        acceleration[step + 1] = (
            (displacement[step + 1] - displacement[step]) / (beta * dt**2)
            - velocity[step] / (beta * dt)
            - (1.0 / (2.0 * beta) - 1.0) * acceleration[step])
        if not (np.isfinite(displacement[step + 1]).all()
                and np.isfinite(velocity[step + 1]).all()
                and np.isfinite(acceleration[step + 1]).all()):
            raise NewmarkSolveError("newmark_non_finite", "the state is not finite",
                                    step=step + 1, time_s=float(times[step + 1]))
        residual = (mass @ acceleration[step + 1] + damping @ velocity[step + 1]
                    + stiffness @ displacement[step + 1] - forces[step + 1])
        denominator = (float(np.linalg.norm(mass @ acceleration[step + 1]))
                       + float(np.linalg.norm(damping @ velocity[step + 1]))
                       + float(np.linalg.norm(stiffness @ displacement[step + 1]))
                       + float(np.linalg.norm(forces[step + 1])))
        error = float(np.linalg.norm(residual)) / denominator if denominator else 0.0
        if not math.isfinite(error) or error > _MAX_NEWMARK_ERROR:
            raise NewmarkSolveError(
                "newmark_not_converged", "the step exceeds the backward-error gate",
                step=step + 1, time_s=float(times[step + 1]))
        if error > worst_error:
            worst_error, worst_step = error, step + 1

    full_displacement = np.zeros((steps + 1, mesh.dof_count))
    full_velocity = np.zeros((steps + 1, mesh.dof_count))
    full_acceleration = np.zeros((steps + 1, mesh.dof_count))
    full_displacement[:, free] = displacement
    full_velocity[:, free] = velocity
    full_acceleration[:, free] = acceleration
    energy = np.array([
        0.5 * float(full_velocity[index] @ mass_full @ full_velocity[index])
        + 0.5 * float(full_displacement[index] @ stiffness_full @ full_displacement[index])
        for index in range(steps + 1)])
    absolute = np.abs(full_displacement)
    peak_index = absolute.argmax(axis=0)
    diagnostics = ("rayleigh_zero",) if (coefficients.alpha == 0.0
                                         and coefficients.beta == 0.0) else ()
    return TransientResult(
        model=spec,
        times_s=times,
        displacement=full_displacement,
        velocity=full_velocity,
        acceleration=full_acceleration,
        energy=energy,
        dof_labels=dof_labels(mesh),
        start_s=float(grid.start_s),
        dt_s=dt,
        peak_displacement=absolute.max(axis=0),
        peak_time_s=times[peak_index],
        global_peaks={
            "displacement": _peak(full_displacement, times),
            "velocity": _peak(full_velocity, times),
            "acceleration": _peak(full_acceleration, times),
        },
        worst_backward_error=worst_error,
        worst_backward_step=worst_step,
        diagnostics=diagnostics,
        versions=versions(),
        input_hash=canonical_dynamics_input_hash(spec),
    )


def transient_response(model):
    """Integrate a schema 1.5 transient declaration with linear Newmark."""
    spec = validate_dynamics_model(model)
    if spec.analysis_type != "transient":
        raise ModelValidationError("/analysis/type", "unsupported_analysis",
                                   "transient_response requires a transient analysis")
    cho_factor, cho_solve = _linalg_cholesky()
    return _solve_transient(spec, cho_factor, cho_solve)
