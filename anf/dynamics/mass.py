"""Consistent mass matrices for schema 1.4 planar structures."""

from dataclasses import dataclass

import numpy as np

from ..model import validate_dynamics_model
from .system import dof_labels


@dataclass(frozen=True)
class MassMatrix:
    """A dense global mass matrix with its DOF labels and total mass."""

    matrix: np.ndarray
    dof_labels: tuple[tuple, ...]
    total_mass: float


def _truss_local(mass):
    return mass / 6 * np.array([
        [2, 0, 1, 0],
        [0, 2, 0, 1],
        [1, 0, 2, 0],
        [0, 1, 0, 2],
    ], dtype=float)


def _frame_local(mass, length):
    return mass / 420 * np.array([
        [140, 0, 0, 70, 0, 0],
        [0, 156, 22 * length, 0, 54, -13 * length],
        [0, 22 * length, 4 * length * length, 0, 13 * length, -3 * length * length],
        [70, 0, 0, 140, 0, 0],
        [0, 54, 13 * length, 0, 156, -22 * length],
        [0, -13 * length, -3 * length * length, 0, -22 * length, 4 * length * length],
    ], dtype=float)




def assemble_mass(model):
    """Return the global consistent mass matrix for a schema 1.4 model."""
    spec = validate_dynamics_model(model)
    mesh = spec.mesh
    if mesh is None:
        raise TypeError("modal model did not produce a structure mesh")
    matrix = np.zeros((mesh.dof_count, mesh.dof_count))
    total = 0.0
    for segment in mesh.segments:
        member = spec.model["members"][segment.member]
        length, cosine, sine = segment.geometry
        element_mass = member["density"] * member["area"] * length
        total += element_mass
        if member["kind"] == "truss2d":
            rotation = np.array([[cosine, sine], [-sine, cosine]])
            transform = np.zeros((4, 4))
            transform[:2, :2] = transform[2:, 2:] = rotation
            element = _truss_local(element_mass)
            dofs = (mesh.node_dofs[segment.nodes[0]][0], mesh.node_dofs[segment.nodes[0]][1],
                    mesh.node_dofs[segment.nodes[1]][0], mesh.node_dofs[segment.nodes[1]][1])
        else:
            rotation = np.array([[cosine, sine, 0], [-sine, cosine, 0], [0, 0, 1]])
            transform = np.zeros((6, 6))
            transform[:3, :3] = transform[3:, 3:] = rotation
            element = _frame_local(element_mass, length)
            dofs = (*mesh.node_dofs[segment.nodes[0]], *mesh.node_dofs[segment.nodes[1]])
        matrix[np.ix_(dofs, dofs)] += transform.T @ element @ transform
    if not np.isfinite(matrix).all():
        raise ArithmeticError("non-finite mass matrix")
    scale = max(float(np.max(np.abs(matrix))), 1.0)
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12 * scale):
        raise ArithmeticError("mass matrix is not symmetric")
    matrix = (matrix + matrix.T) / 2
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues[0] <= 1e-12 * max(eigenvalues[-1], 1.0):
        raise ArithmeticError("singular mass matrix")
    matrix.setflags(write=False)
    return MassMatrix(matrix, dof_labels(mesh), total)