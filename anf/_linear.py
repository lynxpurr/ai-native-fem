"""Validation and elimination for the small dense research models."""
from functools import wraps
from numbers import Integral, Real
import math
import numpy as np


MAX_RELATIVE_RESIDUAL = 1.0e-8


class UnstableStiffnessError(ArithmeticError):
    """The free stiffness could not be confirmed to be stable."""


def checked_arithmetic(function):
    """Turn arithmetic range failures into a consistent diagnostic."""
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
                return function(*args, **kwargs)
        except (FloatingPointError, OverflowError, ZeroDivisionError) as error:
            raise ArithmeticError("calculation exceeds the supported floating-point range") from error
    return checked


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be representable as a finite float") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if result == 0.0 and value != 0:
        raise ValueError(f"{name} is too small to represent as a nonzero float")
    return result


def positive(value, name):
    result = finite(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def count(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@checked_arithmetic
def strain_energy(stiffness, displacement):
    value = float(displacement @ stiffness @ displacement / 2)
    if not math.isfinite(value):
        raise ArithmeticError("non-finite strain energy")
    return value


def relative_residual(gap, rhs):
    scale = float(np.max(np.abs(rhs), initial=0.0))
    if scale == 0:
        return math.inf if np.any(gap) else 0.0
    return float(np.linalg.norm(gap / scale) / np.linalg.norm(rhs / scale))


@checked_arithmetic
def eliminate(stiffness, force, fixed):
    """Solve zero prescribed DOFs, rejecting non-finite or unbalanced results.

    The free-DOF Euclidean residual is normalized by the actual free load norm,
    using scaling to avoid squaring very large or small physical load values.
    """
    if np.iscomplexobj(stiffness) or np.iscomplexobj(force):
        raise ValueError("the static system must be real")
    stiffness = np.asarray(stiffness, dtype=float)
    force = np.asarray(force, dtype=float)
    if force.ndim != 1 or stiffness.shape != (force.size, force.size):
        raise ValueError("stiffness and force dimensions do not agree")
    if not np.isfinite(stiffness).all() or not np.isfinite(force).all():
        raise ValueError("stiffness and force must be finite")
    fixed = tuple(fixed)
    if any(isinstance(dof, bool) or not isinstance(dof, Integral)
           or not 0 <= dof < force.size for dof in fixed):
        raise ValueError("fixed DOFs must be integer indices within the system")
    fixed = np.asarray(sorted(set(fixed)), dtype=int)
    free = np.setdiff1d(np.arange(force.size), fixed)
    displacement = np.zeros(force.size)
    try:
        displacement[free] = np.linalg.solve(stiffness[np.ix_(free, free)], force[free])
    except np.linalg.LinAlgError as error:
        raise ArithmeticError("the constrained stiffness system could not be solved") from error
    if not np.isfinite(displacement).all():
        raise ArithmeticError("non-finite solution")
    imbalance = stiffness @ displacement - force
    if not np.isfinite(imbalance).all():
        raise ArithmeticError("non-finite force balance")
    residual = relative_residual(imbalance[free], force[free])
    if not math.isfinite(residual) or residual > MAX_RELATIVE_RESIDUAL:
        raise ArithmeticError("free-DOF relative residual exceeds 1e-8")
    reactions = np.zeros(force.size)
    reactions[fixed] = imbalance[fixed]
    return displacement, reactions, residual
