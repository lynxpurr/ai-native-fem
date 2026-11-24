"""Modal solver checks against analytic planar frequencies and invariants."""

import sys

import numpy as np
import pytest

scipy = pytest.importorskip("scipy.linalg")

from anf.dynamics import ModalSolveError, assemble_mass, solve_modes
from anf.dynamics.modal import _free_dofs
from anf.model import validate_modal_model


UNITS = {"length": "m", "force": "N", "stress": "Pa"}
E = 210e9
RHO = 7850.0
A = 0.01
I = 1e-6
L = 2.0


def frame(n=1, *, supports=None, nodes=None, analysis=None):
    return {
        "schema_version": "1.4", "problem_type": "planar_structure", "units": dict(UNITS),
        "model": {
            "nodes": nodes or {"a": [0, 0], "b": [L, 0]},
            "members": {"beam": {"kind": "frame2d", "i": "a", "j": "b",
                                 "young_modulus": E, "area": A, "second_moment": I,
                                 "density": RHO, "elements": n}},
            "supports": {"a": {"ux": 0, "uy": 0, "rz": 0}} if supports is None else supports,
        },
        "loads": {},
        "analysis": analysis or {"type": "modal", "modes": 6},
    }


def rod(n):
    nodes = {f"n{i}": [i * L / n, 0] for i in range(n + 1)}
    members = {f"m{i}": {"kind": "truss2d", "i": f"n{i}", "j": f"n{i + 1}",
                         "young_modulus": E, "area": A, "density": RHO}
               for i in range(n)}
    supports = {name: {"uy": 0} for name in nodes}
    supports["n0"]["ux"] = 0
    return {
        "schema_version": "1.4", "problem_type": "planar_structure", "units": dict(UNITS),
        "model": {"nodes": nodes, "members": members, "supports": supports},
        "loads": {}, "analysis": {"type": "modal", "modes": 6},
    }


def relative(actual, expected):
    return np.abs(np.asarray(actual) / expected - 1)


def test_cantilever_frequencies_match_euler_bernoulli():
    result = solve_modes(frame(16), count=3)
    beta = np.array([1.875104068711961, 4.694091132974174, 7.854757438237613])
    exact = beta**2 * np.sqrt(E * I / (RHO * A * L**4)) / (2 * np.pi)
    assert np.all(relative(result.frequencies_hz, exact) <= 1e-4)
    assert result.rigid_mode_count == 0
    assert not result.truncated
    assert np.all(result.residuals <= 1e-9)


def test_simply_supported_frequencies_match_euler_bernoulli():
    supports = {"a": {"ux": 0, "uy": 0}, "b": {"uy": 0}}
    result = solve_modes(frame(16, supports=supports), count=3)
    exact = (np.arange(1, 4) * np.pi)**2 * np.sqrt(E * I / (RHO * A * L**4)) / (2 * np.pi)
    assert np.all(relative(result.frequencies_hz, exact) <= 1e-4)
    assert np.all(relative(result.omega_rad_s, exact * 2 * np.pi) <= 1e-4)


def test_axial_rod_frequencies_are_analytic():
    coarse = solve_modes(rod(8), count=3)
    fine = solve_modes(rod(16), count=3)
    exact = np.array([1, 3, 5]) * np.pi / (2 * L) * np.sqrt(E / RHO) / (2 * np.pi)
    assert np.all(relative(fine.frequencies_hz, exact) <= 2e-2)
    assert np.all(relative(fine.frequencies_hz, exact) < relative(coarse.frequencies_hz, exact))


def test_free_free_frame_has_three_rigid_modes():
    result = solve_modes(frame(2, supports={}), count=6)
    assert result.rigid_mode_count == 3
    assert np.all(result.frequencies_hz[:3] < 1e-3)
    assert result.frequencies_hz[3] > 0
    assert not result.truncated
    assert np.all(result.residuals <= 1e-8)


def test_modes_are_mass_normalized():
    doc = frame(8)
    result = solve_modes(doc, count=3)
    spec = validate_modal_model(doc)
    mass = assemble_mass(spec)
    free, _ = _free_dofs(spec, spec.mesh)
    mass_free = mass.matrix[np.ix_(free, free)]
    np.testing.assert_allclose(result.modes.T @ mass_free @ result.modes,
                               np.eye(3), rtol=0, atol=1e-8)


def test_renumbering_does_not_change_frequencies():
    first = solve_modes(frame(8), count=3)
    doc = frame(8, nodes={"p": [0, 0], "q": [L, 0]})
    doc["model"]["members"]["beam"]["i"] = "p"
    doc["model"]["members"]["beam"]["j"] = "q"
    doc["model"]["supports"] = {"p": {"ux": 0, "uy": 0, "rz": 0}}
    second = solve_modes(doc, count=3)
    np.testing.assert_allclose(first.frequencies_hz, second.frequencies_hz, rtol=1e-12, atol=0)
    assert first.dof_labels != second.dof_labels


def test_requested_count_is_truncated_without_zero_padding():
    result = solve_modes(frame(1, supports={}), count=10)
    assert result.truncated
    assert result.diagnostics == ("modal_truncated",)
    assert len(result.frequencies_hz) == 6
    assert result.rigid_mode_count == 3


def test_nonzero_support_is_rejected():
    doc = frame(4, supports={"a": {"ux": 0.002, "uy": 0, "rz": 0}})
    with pytest.raises(ModalSolveError) as error:
        solve_modes(doc)
    assert error.value.code == "unsupported_constraint"


def test_missing_scipy_reports_backend_unavailable(monkeypatch):
    doc = frame(2)
    monkeypatch.setitem(sys.modules, "scipy.linalg", None)
    with pytest.raises(ModalSolveError) as error:
        solve_modes(doc)
    assert error.value.code == "backend_unavailable"