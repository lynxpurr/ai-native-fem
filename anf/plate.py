"""Fully integrated rectangular Q4 Reissner-Mindlin plate model.

Only transverse bending of a homogeneous isotropic plate is included. Nodal
DOFs are [w, beta_x, beta_y], with beta denoting slopes in the shear equation
grad(w)-beta. This deliberately simple element is subject to shear locking.
"""
from dataclasses import dataclass
import numpy as np
from ._linear import count, eliminate, finite, positive


@dataclass(frozen=True)
class Plate:
    lx: float
    ly: float
    thickness: float
    young_modulus: float
    poisson_ratio: float
    nx: int = 8
    ny: int = 8
    support: str = "clamped"
    shear_factor: float = 5.0/6.0

    def __post_init__(self):
        for name in ("lx", "ly", "thickness", "young_modulus", "shear_factor"):
            positive(getattr(self, name), name)
        nu = finite(self.poisson_ratio, "poisson_ratio")
        if not -1.0 < nu < 0.5:
            raise ValueError("poisson_ratio must be between -1 and 0.5")
        count(self.nx, "nx", 2); count(self.ny, "ny", 2)
        if 3*(self.nx+1)*(self.ny+1) > 1800:
            raise ValueError("this dense research model supports at most 1800 DOFs")
        if self.support not in {"clamped", "hard_simply_supported"}:
            raise ValueError("support must be clamped or hard_simply_supported")


@dataclass(frozen=True)
class PlateResult:
    x: np.ndarray
    y: np.ndarray
    deflection: np.ndarray
    rotations: np.ndarray
    reactions: np.ndarray
    strain_energy: float
    relative_residual: float


def plate_element(model: Plate, width, height):
    """12x12 rectangular element and unit-pressure vector; corners CCW from LL."""
    if not isinstance(model, Plate):
        raise TypeError("model must be a Plate")
    width = positive(width, "width"); height = positive(height, "height")
    E, nu, t = model.young_modulus, model.poisson_ratio, model.thickness
    bending = E*t**3/(12*(1-nu**2))*np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
    shear = model.shear_factor*E*t/(2*(1+nu))
    stiffness = np.zeros((12, 12)); unit_load = np.zeros(12)
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    points, weights = np.polynomial.legendre.leggauss(2)
    for xi, wx in zip(points, weights):
        for eta, wy in zip(points, weights):
            signs_x, signs_y = corners.T
            shape = (1+signs_x*xi)*(1+signs_y*eta)/4
            dx = signs_x*(1+signs_y*eta)/(2*width)
            dy = signs_y*(1+signs_x*xi)/(2*height)
            curvature = np.zeros((3, 12)); transverse = np.zeros((2, 12))
            for node in range(4):
                col = 3*node
                curvature[0, col+1] = dx[node]
                curvature[1, col+2] = dy[node]
                curvature[2, col+1:col+3] = [dy[node], dx[node]]
                transverse[:, col] = [dx[node], dy[node]]
                transverse[0, col+1] = -shape[node]
                transverse[1, col+2] = -shape[node]
            measure = wx*wy*width*height/4
            stiffness += measure*(curvature.T @ bending @ curvature + shear*transverse.T @ transverse)
            unit_load[::3] += measure*shape
    return stiffness, unit_load


def solve_plate(model: Plate, *, pressure):
    """Uniform pressure on a rectangle with zero boundary displacement data."""
    if not isinstance(model, Plate):
        raise TypeError("model must be a Plate")
    pressure = finite(pressure, "pressure")
    nx, ny = model.nx, model.ny
    size = 3*(nx+1)*(ny+1)
    stiffness = np.zeros((size, size)); force = np.zeros(size)
    local, unit_load = plate_element(model, model.lx/nx, model.ly/ny)
    for j in range(ny):
        for i in range(nx):
            nodes = np.array([j*(nx+1)+i, j*(nx+1)+i+1,
                              (j+1)*(nx+1)+i+1, (j+1)*(nx+1)+i])
            dofs = (3*nodes[:, None]+np.arange(3)).ravel()
            stiffness[np.ix_(dofs, dofs)] += local
            force[dofs] += pressure*unit_load
    fixed = set()
    for j in range(ny+1):
        for i in range(nx+1):
            node = j*(nx+1)+i
            on_x, on_y = i in (0, nx), j in (0, ny)
            if not (on_x or on_y):
                continue
            fixed.add(3*node)
            if model.support == "clamped":
                fixed.update([3*node+1, 3*node+2])
            else:
                if on_x: fixed.add(3*node+2)
                if on_y: fixed.add(3*node+1)
    displacement, reactions, residual = eliminate(stiffness, force, sorted(fixed))
    values = displacement.reshape(ny+1, nx+1, 3)
    return PlateResult(np.linspace(0.0, model.lx, nx+1), np.linspace(0.0, model.ly, ny+1),
                       values[:, :, 0], values[:, :, 1:], reactions.reshape(ny+1, nx+1, 3),
                       float(displacement @ stiffness @ displacement / 2), residual)
