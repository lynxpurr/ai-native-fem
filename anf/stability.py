"""Preload statics and axial-force geometric stiffness for linear buckling."""

from dataclasses import dataclass
import math

import numpy as np

from .dynamics.system import dof_labels, versions
from .model import (ModelValidationError, canonical_buckling_input_hash,
                    validate_buckling_model)


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


__all__ = ["BucklingError", "PreloadResult", "assemble_geometric", "solve_preload"]