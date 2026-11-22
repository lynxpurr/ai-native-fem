"""Rectangular Q4 and MITC4 Reissner-Mindlin plate models.

Only transverse bending of a homogeneous isotropic plate is included. Nodal
DOFs are [w, beta_x, beta_y], with beta denoting slopes in the shear equation
grad(w)-beta. The default fully integrated Q4 is subject to shear locking.
"""
from dataclasses import dataclass
from math import isfinite
import numpy as np
from ._linear import checked_arithmetic, count, eliminate, finite, positive, strain_energy
from .assembly import DenseAssembly


_SIGNS_X = np.array([-1, 1, 1, -1])
_SIGNS_Y = np.array([-1, -1, 1, 1])


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
    formulation: str = "q4_full"

    def __post_init__(self):
        for name in ("lx", "ly", "thickness", "young_modulus", "shear_factor"):
            object.__setattr__(self, name, positive(getattr(self, name), name))
        nu = finite(self.poisson_ratio, "poisson_ratio")
        object.__setattr__(self, "poisson_ratio", nu)
        if not -1.0 < nu < 0.5:
            raise ValueError("poisson_ratio must be between -1 and 0.5")
        object.__setattr__(self, "nx", count(self.nx, "nx", 2))
        object.__setattr__(self, "ny", count(self.ny, "ny", 2))
        if 3*(self.nx+1)*(self.ny+1) > 1800:
            raise ValueError("this dense research model supports at most 1800 DOFs")
        if not isinstance(self.support, str) or self.support not in {"clamped", "hard_simply_supported"}:
            raise ValueError("support must be clamped or hard_simply_supported")
        if (not isinstance(self.formulation, str)
                or self.formulation not in {"q4_full", "mitc4_rectangle"}):
            raise ValueError("formulation must be q4_full or mitc4_rectangle")
        if self.formulation == "mitc4_rectangle" and self.support != "hard_simply_supported":
            raise ValueError("mitc4_rectangle requires hard_simply_supported edges")


@dataclass(frozen=True)
class PlateResult:
    x: np.ndarray
    y: np.ndarray
    deflection: np.ndarray
    rotations: np.ndarray
    reactions: np.ndarray
    strain_energy: float
    relative_residual: float


def _bending_matrix(model):
    nu, t = model.poisson_ratio, model.thickness
    rigidity = model.young_modulus*t**3/(12*(1-nu**2))
    return rigidity*np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])


def _curvature_matrix(dx, dy):
    curvature = np.zeros((3, 12))
    curvature[0, 1::3] = dx
    curvature[1, 2::3] = dy
    curvature[2, 1::3] = dy
    curvature[2, 2::3] = dx
    return curvature


def _mitc4_shear_matrix(xi, eta, width, height):
    """Interpolate edge-midpoint shear on an axis-aligned rectangle."""
    # Bathe and Dvorkin (1985), eqs. (6)-(7). Each edge value is the
    # transverse displacement difference per unit length minus the mean slope.
    horizontal = (1 + _SIGNS_Y * eta) / 4
    vertical = (1 + _SIGNS_X * xi) / 4
    shear = np.zeros((2, 12))
    shear[0, ::3] = 2 * horizontal * _SIGNS_X / width
    shear[1, ::3] = 2 * vertical * _SIGNS_Y / height
    shear[0, 1::3] = -horizontal
    shear[1, 2::3] = -vertical
    return shear


@checked_arithmetic
def plate_element(model: Plate, width, height):
    """12x12 rectangular element and unit-pressure vector; corners CCW from LL."""
    if not isinstance(model, Plate):
        raise TypeError("model must be a Plate")
    width = positive(width, "width"); height = positive(height, "height")
    E, nu, t = model.young_modulus, model.poisson_ratio, model.thickness
    bending = _bending_matrix(model)
    shear = model.shear_factor*E*t/(2*(1+nu))
    mitc4 = model.formulation == "mitc4_rectangle"
    if mitc4 and (bending[0, 0] <= 0 or not 0 < shear < float("inf")
                  or not isfinite(2 * width) or not isfinite(2 * height)):
        raise ArithmeticError("MITC4 element exceeds the supported floating-point range")
    stiffness = np.zeros((12, 12)); unit_load = np.zeros(12)
    signs_x, signs_y = _SIGNS_X, _SIGNS_Y
    points, weights = np.polynomial.legendre.leggauss(2)
    for xi, wx in zip(points, weights):
        for eta, wy in zip(points, weights):
            shape = (1+signs_x*xi)*(1+signs_y*eta)/4
            dx = signs_x*(1+signs_y*eta)/(2*width)
            dy = signs_y*(1+signs_x*xi)/(2*height)
            curvature = _curvature_matrix(dx, dy)
            if mitc4:
                transverse = _mitc4_shear_matrix(xi, eta, width, height)
            else:
                transverse = np.zeros((2, 12))
                transverse[0, ::3] = dx
                transverse[1, ::3] = dy
                transverse[0, 1::3] = -shape
                transverse[1, 2::3] = -shape
            measure = wx*wy*width*height/4
            stiffness += measure*(curvature.T @ bending @ curvature + shear*transverse.T @ transverse)
            unit_load[::3] += measure*shape
    if mitc4 and (np.any(unit_load[::3] <= 0) or np.any(stiffness.diagonal() <= 0)):
        raise ArithmeticError("MITC4 element exceeds the supported floating-point range")
    return stiffness, unit_load


def solve_plate(model: Plate, *, pressure):
    """Uniform pressure on a rectangle with zero boundary displacement data."""
    return _solve_plate(model, pressure=pressure)


@checked_arithmetic
def _solve_plate(model, *, pressure, sparse=None):
    if not isinstance(model, Plate):
        raise TypeError("model must be a Plate")
    pressure = finite(pressure, "pressure")
    nx, ny = model.nx, model.ny
    size = 3*(nx+1)*(ny+1)
    assembly = DenseAssembly(size) if sparse is None else sparse.assembly(size, 144 * nx * ny)
    force = np.zeros(size)
    width, height = model.lx/nx, model.ly/ny
    mitc4 = model.formulation == "mitc4_rectangle"
    if mitc4 and (width == 0 or height == 0):
        raise ArithmeticError("MITC4 element dimensions underflowed")
    local, unit_load = plate_element(model, width, height)
    local_load = pressure*unit_load
    if mitc4 and pressure != 0 and np.any(local_load[::3] == 0):
        raise ArithmeticError("MITC4 nodal pressure load underflowed")
    for j in range(ny):
        for i in range(nx):
            nodes = np.array([j*(nx+1)+i, j*(nx+1)+i+1,
                              (j+1)*(nx+1)+i+1, (j+1)*(nx+1)+i])
            dofs = (3*nodes[:, None]+np.arange(3)).ravel()
            assembly.add(dofs, local)
            force[dofs] += local_load
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
    if sparse is None:
        stiffness = assembly.finish()
        displacement, reactions, residual = eliminate(stiffness, force, sorted(fixed))
    else:
        stiffness = sparse.assembled(assembly)
        displacement, reactions, residual = sparse.eliminate_classical(
            stiffness, force, sorted(fixed), stride=3, length=max(model.lx, model.ly),
        )
    values = displacement.reshape(ny+1, nx+1, 3)
    energy = strain_energy(stiffness, displacement)
    if mitc4 and pressure != 0 and energy <= 0:
        raise ArithmeticError("nonpositive MITC4 strain energy under nonzero pressure")
    return PlateResult(np.linspace(0.0, model.lx, nx+1), np.linspace(0.0, model.ly, ny+1),
                       values[:, :, 0], values[:, :, 1:], reactions.reshape(ny+1, nx+1, 3),
                       energy, residual)
