"""Cubic Hermite Euler-Bernoulli bending, derived from the stated energy.

See docs/methods.md for the equations and mathematical reference. Displacement
and slope at each node are the two DOFs; this module models one bending plane.
"""
from dataclasses import dataclass
import numpy as np
from ._linear import count, eliminate, finite, positive


@dataclass(frozen=True)
class Beam:
    length: float
    young_modulus: float
    second_moment: float
    elements: int = 8

    def __post_init__(self):
        positive(self.length, "length")
        positive(self.young_modulus, "young_modulus")
        positive(self.second_moment, "second_moment")
        if count(self.elements, "elements") > 200:
            raise ValueError("this dense research model supports at most 200 elements")


@dataclass(frozen=True)
class BeamResult:
    x: np.ndarray
    deflection: np.ndarray
    slope: np.ndarray
    root_reaction: np.ndarray
    strain_energy: float
    relative_residual: float


def beam_element(length, bending_rigidity):
    """Return a 4x4 stiffness and unit uniform-load vector.

    Two-point quadrature exactly integrates both polynomial integrands.
    The local order is [w_left, slope_left, w_right, slope_right].
    """
    length = positive(length, "length")
    rigidity = positive(bending_rigidity, "bending_rigidity")
    stiffness = np.zeros((4, 4))
    unit_load = np.zeros(4)
    points, weights = np.polynomial.legendre.leggauss(2)
    for point, weight in zip(points, weights):
        r = (point + 1.0) / 2.0
        shapes = np.array([1-3*r*r+2*r**3, length*(r-2*r*r+r**3),
                           3*r*r-2*r**3, length*(-r*r+r**3)])
        curvature = np.array([(-6+12*r)/length**2, (-4+6*r)/length,
                              (6-12*r)/length**2, (-2+6*r)/length])
        measure = weight * length / 2.0
        stiffness += rigidity * measure * np.outer(curvature, curvature)
        unit_load += measure * shapes
    return stiffness, unit_load


def solve_cantilever(model: Beam, *, tip_force=0.0, uniform_load=0.0):
    """Constant-section cantilever; positive loads produce positive deflection."""
    if not isinstance(model, Beam):
        raise TypeError("model must be a Beam")
    tip_force = finite(tip_force, "tip_force")
    uniform_load = finite(uniform_load, "uniform_load")
    size = 2 * (model.elements + 1)
    stiffness = np.zeros((size, size))
    force = np.zeros(size)
    local, unit_load = beam_element(model.length/model.elements,
                                    model.young_modulus*model.second_moment)
    for element in range(model.elements):
        dofs = np.arange(2*element, 2*element+4)
        stiffness[np.ix_(dofs, dofs)] += local
        force[dofs] += uniform_load * unit_load
    force[-2] += tip_force
    displacement, reactions, residual = eliminate(stiffness, force, [0, 1])
    return BeamResult(np.linspace(0.0, model.length, model.elements+1),
                      displacement[::2], displacement[1::2], reactions[:2],
                      float(displacement @ stiffness @ displacement / 2), residual)
