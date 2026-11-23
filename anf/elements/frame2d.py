"""Two-node planar elements in a right-handed local basis."""

from dataclasses import dataclass
import math

import numpy as np

from .._linear import checked_arithmetic, finite, positive


@dataclass(frozen=True)
class Truss2D:
    """Axial member with global DOFs [ux_i, uy_i, ux_j, uy_j]."""

    length: float
    axial_rigidity: float
    cosine: float = 1.0
    sine: float = 0.0

    def __post_init__(self):
        for name, validator in (("length", positive), ("axial_rigidity", positive),
                                ("cosine", finite), ("sine", finite)):
            object.__setattr__(self, name, validator(getattr(self, name), name))
        if not math.isclose(math.hypot(self.cosine, self.sine), 1.0, rel_tol=1e-12):
            raise ValueError("direction cosine and sine must give a unit vector")

    @checked_arithmetic
    def stiffness(self):
        direction = self._direction()
        return self._axial_coefficient() * np.outer(direction, direction)

    def _direction(self):
        return np.array([-self.cosine, -self.sine, self.cosine, self.sine])

    def _axial_coefficient(self):
        return positive(self.axial_rigidity / self.length, "axial stiffness")

    def transformation(self):
        return np.array([[self.cosine, self.sine, 0, 0],
                         [0, 0, self.cosine, self.sine]])

    @checked_arithmetic
    def local_line_load(self, i, j):
        """Integrate a linear axial line load at the two end nodes."""
        a, b = i[0], j[0]
        return self.length * np.array([a / 3 + b / 6, a / 6 + b / 3])

    @checked_arithmetic
    def end_actions(self, displacement):
        tension = self._axial_coefficient() * (self._direction() @ displacement)
        return np.array([[-tension, 0, 0], [tension, 0, 0]])

    def _loaded_end_actions(self, displacement, local_load):
        actions = self.end_actions(displacement)
        if local_load is not None:
            actions[:, 0] -= local_load
        return actions


@dataclass(frozen=True, kw_only=True)
class Frame2D(Truss2D):
    """Euler–Bernoulli member with [ux, uy, rz] at each end."""

    bending_rigidity: float

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "bending_rigidity", positive(self.bending_rigidity, "bending_rigidity"))

    def transformation(self):
        c, s = self.cosine, self.sine
        rotation = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        transform = np.zeros((6, 6))
        transform[:3, :3] = transform[3:, 3:] = rotation
        return transform

    @checked_arithmetic
    def local_stiffness(self):
        length = self.length
        axial = self._axial_coefficient()
        rotation = positive(self.bending_rigidity / length, "bending stiffness")
        coupling = positive(rotation / length, "bending coupling")
        translation = positive(coupling / length, "transverse stiffness")
        a, b, c = 12 * translation, 6 * coupling, 2 * rotation
        stiffness = np.array([
            [axial, 0, 0, -axial, 0, 0],
            [0, a, b, 0, -a, b],
            [0, b, 2*c, 0, -b, c],
            [-axial, 0, 0, axial, 0, 0],
            [0, -a, -b, 0, a, -b],
            [0, b, c, 0, -b, 2*c],
        ])
        if not np.isfinite(stiffness).all():
            raise ArithmeticError("non-finite frame stiffness")
        return stiffness

    @checked_arithmetic
    def stiffness(self):
        transform = self.transformation()
        return transform.T @ self.local_stiffness() @ transform

    @checked_arithmetic
    def local_load(self, transverse):
        """Consistent nodal force for a constant local +y load per unit length."""
        force = transverse * (self.length / 2)
        moment = (force / 6) * self.length
        if not np.isfinite([force, moment]).all():
            raise ArithmeticError("non-finite equivalent nodal load")
        return np.array([0, force, moment, 0, force, -moment])

    @checked_arithmetic
    def local_line_load(self, i, j):
        """Integrate axial and transverse linear loads in the local basis."""
        axial_i, axial_j = super().local_line_load(i, j)
        a, b = i[1], j[1]
        h = self.length
        transverse = h * np.array([0.35 * a + 0.15 * b, 0.15 * a + 0.35 * b])
        moments = h * (h * np.array([a / 20 + b / 30, -a / 30 - b / 20]))
        return np.array([axial_i, transverse[0], moments[0],
                         axial_j, transverse[1], moments[1]])

    @checked_arithmetic
    def end_actions(self, displacement, transverse=0.0):
        return self._loaded_end_actions(displacement, self.local_load(transverse))

    def _loaded_end_actions(self, displacement, local_load):
        local = self.transformation() @ displacement
        actions = self.local_stiffness() @ local
        if local_load is not None:
            actions -= local_load
        return actions.reshape(2, 3)


def hermite_displacement(fraction, length, transverse_dofs):
    """Interpolate local displacement from [v_i, rz_i, v_j, rz_j]."""
    t = fraction
    shape = np.array([
        (1 - t)**2 * (1 + 2*t),
        length * t * (1 - t)**2,
        t**2 * (3 - 2*t),
        length * t**2 * (t - 1),
    ])
    return float(shape @ transverse_dofs)
