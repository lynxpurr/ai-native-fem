"""Member-end releases and rotational springs for planar frame statics."""

import math

import numpy as np

from ._linear import positive
from .model import _choice, _fields, _positive


def _connections(value, path):
    ends = _fields(value, path, (), {"i": None, "j": None})
    result = {}
    for end in value:
        end_path = f"{path}/{end}"
        connection = _fields(ends[end], end_path, ("kind",), {"stiffness": None})
        kind = _choice(connection["kind"], f"{end_path}/kind", ("release", "rotational_spring"),
                       "unsupported_connection")
        fields = ("kind",) if kind == "release" else ("kind", "stiffness")
        connection = _fields(ends[end], end_path, fields)
        if kind == "rotational_spring":
            connection["stiffness"] = _positive(connection["stiffness"], f"{end_path}/stiffness")
        result[end] = connection
    return result


def _spring_stiffness(connection):
    if connection is None:
        return None
    return 0.0 if connection["kind"] == "release" else connection["stiffness"]


def _inverse_diagonal(beam, spring):
    if spring is None:
        return beam
    if spring < beam:
        return spring / (1 + spring / beam)
    return beam / (1 + beam / spring)


class _JointedFrame:
    """Condensed external stiffness, effective load and beam response of a segment."""

    def __init__(self, element, springs, load):
        self.element = element
        self.springs = springs
        self.load = np.zeros(6) if load is None else load
        self.rotation_load = self.load[[2, 5]]
        self.active = np.array([index for index, spring in enumerate(springs) if spring != 0], dtype=int)
        slots = [0, 1, 3, 4]
        slots.extend(3 * index + 2 for index in self.active)
        slots.sort()
        self.slots = slots
        self.transform = element.transformation()[np.ix_(slots, slots)]
        h = element.length
        chord = 1 / h
        slope_difference = np.array([[0, chord, 1, 0, -chord, 0],
                                     [0, chord, 0, 0, -chord, 1]])
        self.bending = slope_difference[np.ix_(self.active, slots)]
        self.axial = np.array([-1, 0, 0, 1, 0, 0], dtype=float)[slots]

        beam = positive(3 * (element.bending_rigidity / h), "end rotational stiffness")
        self.beam_end_stiffness = beam
        self.flexibility = np.array([[1, -0.5], [-0.5, 1]]) / beam
        inverse = np.array([_inverse_diagonal(beam, springs[index]) for index in self.active])
        if len(inverse) == 2:
            cross = 0.5 * math.sqrt(inverse[0] / beam) * math.sqrt(inverse[1] / beam)
            scaling = np.sqrt(inverse)
            self.rotational = ((scaling[:, None] * np.array([[1, cross], [cross, 1]])) * scaling
                               / (1 - cross * cross))
        else:
            self.rotational = np.diag(inverse)
        self.flexible_load = (self.flexibility @ self.rotation_load)[self.active]

    def stiffness(self):
        local = (self.element._axial_coefficient() * np.outer(self.axial, self.axial)
                 + self.bending.T @ self.rotational @ self.bending)
        return self.transform.T @ local @ self.transform

    def force(self):
        moment_gradient = (self.rotation_load[0] + self.rotation_load[1]) / self.element.length
        axial_i, shear_i, _, axial_j, shear_j, _ = self.load
        translation_load = np.array([axial_i, shear_i - moment_gradient, 0,
                                     axial_j, shear_j + moment_gradient, 0])
        local = translation_load[self.slots] + self.bending.T @ (self.rotational @ self.flexible_load)
        return self.transform.T @ local

    def response(self, displacement):
        local = self.transform @ displacement
        full = np.zeros(6)
        full[self.slots] = local
        chord = (full[4] - full[1]) / self.element.length
        moments = np.zeros(2)
        moments[self.active] = self.rotational @ (self.bending @ local - self.flexible_load)
        elastic_moments = moments + self.rotation_load
        bend = self.flexibility @ elastic_moments
        rotations = bend + chord
        spring_energy = 0.0
        for index, spring in enumerate(self.springs):
            if spring is None:
                rotations[index] = full[3 * index + 2]
            elif spring > 0:
                slip = moments[index] / spring
                # Recover soft springs from the beam even if their moments round to zero.
                if spring >= self.beam_end_stiffness:
                    rotations[index] = full[3 * index + 2] - slip
                spring_energy += (moments[index] / 2) * slip

        extension = self.axial @ local
        tension = self.element._axial_coefficient() * extension
        shear = (elastic_moments[0] + elastic_moments[1]) / self.element.length
        p = self.load
        actions = np.array([[-tension - p[0], shear - p[1], moments[0]],
                            [tension - p[3], -shear - p[4], moments[1]]])
        energy = (tension / 2) * extension + (elastic_moments / 2) @ bend + spring_energy
        return actions, rotations, energy
