"""Preload statics and axial-force geometric stiffness for linear buckling."""

from dataclasses import dataclass
import math

import numpy as np

from .dynamics.system import assemble_stiffness, dof_labels, free_dofs, versions
from .model import (ModelValidationError, canonical_buckling_input_hash,
                    validate_buckling_model)

_MAX_EIGEN_RESIDUAL = 1e-9


class BucklingError(ArithmeticError):
    """A stable machine diagnostic for a failed buckling step."""

    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class PreloadResult:
    """Static preload equilibrium and per-segment axial forces."""

    model: object
    displacement: np.ndarray
    reactions: np.ndarray
    axial_forces: tuple
    relative_residual: float
    dof_labels: tuple
    versions: dict
    input_hash: str

    def __post_init__(self):
        if self.displacement.shape != (len(self.dof_labels),):
            raise ValueError("preload displacement must match the DOF labels")
        if self.reactions.shape != self.displacement.shape:
            raise ValueError("preload reactions must match the displacement")
        if not np.isfinite(self.displacement).all() or not np.isfinite(self.reactions).all():
            raise ValueError("preload result values must be finite")
        if not all(math.isfinite(value) for value in self.axial_forces):
            raise ValueError("axial forces must be finite")
        if not math.isfinite(self.relative_residual) or self.relative_residual < 0:
            raise ValueError("the preload residual must be finite and non-negative")


def _geometric_local(member, length, axial):
    if member["kind"] == "truss2d":
        return axial / length * np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 1.0],
        ])
    return axial / length * np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 6 / 5, length / 10, 0.0, -6 / 5, length / 10],
        [0.0, length / 10, 2 * length**2 / 15, 0.0, -length / 10, -length**2 / 30],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -6 / 5, -length / 10, 0.0, 6 / 5, -length / 10],
        [0.0, length / 10, -length**2 / 30, 0.0, -length / 10, 2 * length**2 / 15],
    ])


def _transformation(member, cosine, sine):
    if member["kind"] == "truss2d":
        rotation = np.array([[cosine, sine], [-sine, cosine]])
        transform = np.zeros((4, 4))
        transform[:2, :2] = transform[2:, 2:] = rotation
        return transform
    rotation = np.array([[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    transform = np.zeros((6, 6))
    transform[:3, :3] = transform[3:, 3:] = rotation
    return transform


def assemble_geometric(spec, displacement):
    """Return the global geometric stiffness and per-segment axial forces."""
    mesh = spec.mesh
    matrix = np.zeros((mesh.dof_count, mesh.dof_count))
    axial_forces = []
    for segment in mesh.segments:
        member = spec.model["members"][segment.member]
        length, cosine, sine = segment.geometry
        width = 3 if member["kind"] == "frame2d" else 2
        dofs = (*mesh.node_dofs[segment.nodes[0]][:width],
                *mesh.node_dofs[segment.nodes[1]][:width])
        direction = np.zeros(2 * width)
        direction[0], direction[1] = -cosine, -sine
        direction[width], direction[width + 1] = cosine, sine
        axial = (member["young_modulus"] * member["area"] / length
                 * float(direction @ displacement[list(dofs)]))
        axial_forces.append(axial)
        transform = _transformation(member, cosine, sine)
        matrix[np.ix_(dofs, dofs)] += (transform.T @ _geometric_local(member, length, axial)
                                       @ transform)
    if not np.isfinite(matrix).all():
        raise BucklingError("non_finite_geometric", "the geometric stiffness is not finite")
    scale = max(float(np.max(np.abs(matrix))), 1.0)
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12 * scale):
        raise BucklingError("asymmetric_geometric", "the geometric stiffness is not symmetric")
    matrix = (matrix + matrix.T) / 2
    matrix.setflags(write=False)
    return matrix, tuple(axial_forces)


def _preload(spec):
    from .structures import StructureSolveError, _solve_structure

    mesh = spec.mesh
    try:
        result = _solve_structure(spec, spec.loads["nodal"])
    except StructureSolveError as error:
        raise BucklingError("unstable_preload_model",
                            f"the preload solve failed: {error.code}") from error
    matrix, axial_forces = assemble_geometric(spec, result.displacement)
    preload = PreloadResult(
        model=spec,
        displacement=result.displacement,
        reactions=result.reactions,
        axial_forces=axial_forces,
        relative_residual=result.relative_residual,
        dof_labels=dof_labels(mesh),
        versions=versions(),
        input_hash=canonical_buckling_input_hash(spec),
    )
    return preload, matrix


def solve_preload(model):
    """Solve the declared nodal preload and return its equilibrium and axial forces."""
    spec = validate_buckling_model(model)
    preload, _ = _preload(spec)
    return preload




@dataclass(frozen=True, slots=True)
class BucklingResult:
    """Positive load factors and K-orthonormal buckling modes."""

    model: object
    factors: np.ndarray
    modes: np.ndarray
    residuals: np.ndarray
    axial_forces: tuple
    preload_residual: float
    dof_labels: tuple
    diagnostics: tuple
    versions: dict
    input_hash: str

    @property
    def critical_factor(self):
        return float(self.factors[0])

    def __post_init__(self):
        count = len(self.factors)
        if count < 1:
            raise ValueError("a buckling result requires at least one factor")
        if self.modes.shape != (len(self.dof_labels), count):
            raise ValueError("mode dimensions do not match the DOF labels")
        if self.residuals.shape != (count,):
            raise ValueError("residuals must match the factor count")
        if not np.isfinite(self.factors).all() or not (self.factors > 0).all():
            raise ValueError("buckling factors must be finite and positive")
        if not np.all(np.diff(self.factors) >= 0):
            raise ValueError("buckling factors must be ascending")
        if not np.isfinite(self.modes).all() or not np.isfinite(self.residuals).all():
            raise ValueError("buckling result values must be finite")
        if not math.isfinite(self.preload_residual) or self.preload_residual < 0:
            raise ValueError("the preload residual must be finite and non-negative")


def _linalg():
    try:
        from scipy.linalg import eigh
    except (ImportError, ModuleNotFoundError) as error:
        raise ModelValidationError(
            "/analysis", "backend_unavailable",
            "buckling requires the ai-native-fem[sparse] extra") from error
    return eigh


def _solve_buckling(spec, modes, eigh):
    mesh = spec.mesh
    preload, geometric = _preload(spec)
    stiffness = assemble_stiffness(spec)
    free, _ = free_dofs(spec, mesh)
    if not len(free):
        raise ModelValidationError("/model/supports", "no_free_dofs",
                                   "the model has no free degrees of freedom")
    stiffness_block = stiffness[np.ix_(free, free)]
    geometric_block = geometric[np.ix_(free, free)]
    try:
        values, vectors = eigh(-geometric_block, stiffness_block)
    except np.linalg.LinAlgError as error:
        raise BucklingError("buckling_mechanism",
                            "the free stiffness is singular or not positive definite") from error
    if not np.isfinite(values).all() or not np.isfinite(vectors).all():
        raise BucklingError("buckling_not_converged", "the eigenpairs are not finite")
    scale = float(np.max(np.abs(values))) if len(values) else 0.0
    tolerance = 1e-12 * scale if scale > 0 else 0.0
    candidates = sorted((1.0 / value, index) for index, value in enumerate(values)
                        if value > tolerance)
    if not candidates:
        raise BucklingError("buckling_no_positive_factor",
                            "the reference preload has no positive critical factor")
    selected = candidates[:modes]
    factors = np.array([factor for factor, _ in selected])
    shapes = np.zeros((mesh.dof_count, len(selected)))
    residuals = np.zeros(len(selected))
    diagnostics = []
    for column, (factor, index) in enumerate(selected):
        phi = np.array(vectors[:, index], dtype=float)
        peak = int(np.argmax(np.abs(phi)))
        if phi[peak] < 0:
            phi = -phi
        shapes[free, column] = phi
        gap = stiffness_block @ phi + factor * geometric_block @ phi
        denominator = (float(np.linalg.norm(stiffness_block @ phi))
                       + abs(factor) * float(np.linalg.norm(geometric_block @ phi)))
        residual = float(np.linalg.norm(gap)) / denominator if denominator else 0.0
        if not math.isfinite(residual) or residual > _MAX_EIGEN_RESIDUAL:
            raise BucklingError("buckling_not_converged",
                                f"mode {column} exceeds the eigenpair residual gate")
        residuals[column] = residual
        if column and abs(factors[column] - factors[column - 1]) <= 1e-8 * factors[column]:
            diagnostics.append(f"repeated_factor:{column}")
    return BucklingResult(
        model=spec,
        factors=factors,
        modes=shapes,
        residuals=residuals,
        axial_forces=preload.axial_forces,
        preload_residual=preload.relative_residual,
        dof_labels=preload.dof_labels,
        diagnostics=tuple(diagnostics),
        versions=versions(),
        input_hash=canonical_buckling_input_hash(spec),
    )


def solve_buckling(model, modes=None):
    """Solve the linearized buckling problem for a schema 1.6 declaration."""
    spec = validate_buckling_model(model)
    if modes is None:
        modes = spec.modes
    if type(modes) is not int or not 1 <= modes <= 20:
        raise ModelValidationError("/modes", "invalid_value",
                                   "expected an integer from 1 to 20")
    eigh = _linalg()
    return _solve_buckling(spec, modes, eigh)


__all__ = ["BucklingError", "BucklingResult", "PreloadResult",
           "assemble_geometric", "solve_buckling", "solve_preload"]