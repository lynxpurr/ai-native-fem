"""Validation and elimination for the small dense research models."""
from numbers import Integral, Real
import math
import numpy as np


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def positive(value, name):
    result = finite(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def count(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def eliminate(stiffness, force, fixed):
    """Solve with zero prescribed DOFs; return displacement, reactions, residual."""
    free = np.setdiff1d(np.arange(len(force)), np.asarray(sorted(set(fixed)), dtype=int))
    displacement = np.zeros(len(force))
    displacement[free] = np.linalg.solve(stiffness[np.ix_(free, free)], force[free])
    if not np.isfinite(displacement).all():
        raise ArithmeticError("non-finite solution")
    imbalance = stiffness @ displacement - force
    residual = float(np.linalg.norm(imbalance[free]) / max(1.0, np.linalg.norm(force[free])))
    reactions = np.zeros(len(force))
    reactions[list(fixed)] = imbalance[list(fixed)]
    return displacement, reactions, residual
