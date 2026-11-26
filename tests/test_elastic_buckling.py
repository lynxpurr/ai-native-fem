"""Schema 1.6 preload equilibrium and geometric stiffness checks."""

import math

import numpy as np
import pytest

from anf.dynamics.system import assemble_stiffness, free_dofs
from anf.model import ModelValidationError, canonical_buckling_input_hash, validate_buckling_model
from anf.stability import (BucklingError, PreloadResult, assemble_geometric,
                           solve_buckling, solve_preload)

UNITS = {"length": "m", "force": "N", "stress": "Pa"}
E, A, I, L = 210e9, 0.01, 1e-6, 2.0


def model(analysis=None, loads=None, *, elements=16, supports=None, members=None, nodes=None):
    return {
        "schema_version": "1.6",
        "problem_type": "planar_structure",
        "units": dict(UNITS),
        "model": {
            "nodes": nodes if nodes is not None else {"base": [0.0, 0.0], "tip": [L, 0.0]},
            "members": members if members is not None else {
                "col": {"kind": "frame2d", "i": "base", "j": "tip", "elements": elements,
                        "young_modulus": E, "area": A, "second_moment": I}},
            "supports": (supports if supports is not None
                         else {"base": {"ux": 0, "uy": 0, "rz": 0}}),
        },
        "loads": {"nodal": {"tip": {"fx": -1000.0}}} if loads is None else loads,
        "analysis": {"type": "buckling", "modes": 4} if analysis is None else analysis,
    }


def code_of(doc):
    with pytest.raises(ModelValidationError) as info:
        validate_buckling_model(doc)
    return info.value.code, info.value.path


def test_preload_reports_axial_force_and_equilibrium():
    result = solve_preload(model())
    assert isinstance(result, PreloadResult)
    assert len(result.axial_forces) == 16
    assert all(value == pytest.approx(-1000.0, rel=1e-12) for value in result.axial_forces)
    assert result.relative_residual <= 1e-8
    assert result.reactions[0] == pytest.approx(1000.0)
    assert result.reactions[1] == pytest.approx(0.0)
    assert len(result.dof_labels) == 51
    assert result.dof_labels[0] == ("input", "base", "ux")


def test_geometric_stiffness_sign_follows_the_axial_force():
    for load, sign in ((-1000.0, -1), (1000.0, 1)):
        doc = model(loads={"nodal": {"tip": {"fx": load}}})
        preload = solve_preload(doc)
        spec = validate_buckling_model(doc)
        matrix, axial = assemble_geometric(spec, preload.displacement)
        assert all(value == pytest.approx(load, rel=1e-12) for value in axial)
        free, _ = free_dofs(spec, spec.mesh)
        block = matrix[np.ix_(free, free)]
        eigenvalues = np.linalg.eigvalsh(block)
        scale = float(np.linalg.norm(block, 1))
        if sign > 0:
            assert eigenvalues.min() >= -1e-10 * scale
        else:
            assert eigenvalues.max() <= 1e-10 * scale


def test_sloped_truss_axial_force_uses_the_direction_cosine():
    nodes = {"a": [0.0, 0.0], "b": [2.0, 0.0], "c": [1.0, 1.0]}
    members = {
        "left": {"kind": "truss2d", "i": "a", "j": "c", "young_modulus": E, "area": A},
        "right": {"kind": "truss2d", "i": "b", "j": "c", "young_modulus": E, "area": A},
    }
    doc = model(loads={"nodal": {"c": {"fy": -1000.0}}}, nodes=nodes, members=members,
                supports={"a": {"ux": 0, "uy": 0}, "b": {"ux": 0, "uy": 0}})
    result = solve_preload(doc)
    expected = -1000.0 / (2 * math.sin(math.pi / 4))
    assert result.axial_forces == pytest.approx((expected, expected), rel=1e-12)


def test_unstable_preload_becomes_a_buckling_error():
    doc = model(loads={}, nodes={"a": [0.0, 0.0], "b": [L, 0.0]},
                members={"rod": {"kind": "truss2d", "i": "a", "j": "b",
                                 "young_modulus": E, "area": A}},
                supports={"a": {"ux": 0, "uy": 0}, "b": {"ux": 0}})
    with pytest.raises(BucklingError) as info:
        solve_preload(doc)
    assert info.value.code == "unstable_preload_model"


def test_declaration_normalizes_and_hashes():
    doc = model()
    spec = validate_buckling_model(doc)
    assert spec.analysis_type == "buckling"
    assert spec.modes == 4
    again = validate_buckling_model(spec.to_dict())
    assert again.to_dict() == spec.to_dict()
    assert canonical_buckling_input_hash(again) == canonical_buckling_input_hash(spec)


@pytest.mark.parametrize("loads, code", [
    ({"uniform_transverse": {"col": 10.0}}, ("unsupported_load", "/loads/uniform_transverse")),
    ({"member_line": {}}, ("unsupported_load", "/loads/member_line")),
    ({"nodal": {"ghost": {"fx": 1.0}}}, ("unknown_node", "/loads/nodal/ghost")),
])
def test_only_nodal_preloads_are_accepted(loads, code):
    assert code_of(model(loads=loads)) == code


def test_nonzero_supports_are_rejected():
    doc = model(supports={"base": {"ux": 0, "uy": 0, "rz": 0},
                          "tip": {"uy": 0.001}})
    code, path = code_of(doc)
    assert (code, path) == ("unsupported_constraint", "/model/supports/tip/uy")


def test_end_connections_are_rejected():
    doc = model(members={"col": {"kind": "frame2d", "i": "base", "j": "tip", "elements": 4,
                                 "young_modulus": E, "area": A, "second_moment": I,
                                 "end_connections": {"i": {"kind": "release"}}}})
    code, path = code_of(doc)
    assert (code, path) == ("unsupported_member", "/model/members/col/end_connections")


@pytest.mark.parametrize("doc, code", [
    (model(analysis={"type": "static"}), "unsupported_analysis"),
    (model(analysis={"type": "buckling", "modes": 0}), "invalid_value"),
    (model(analysis={"type": "buckling", "modes": 21}), "invalid_value"),
])
def test_analysis_contract(doc, code):
    assert code_of(doc)[0] == code


def test_schema_version_and_density_are_strict():
    doc = model()
    doc["schema_version"] = "1.5"
    assert code_of(doc)[0] == "unsupported_schema"
    doc = model()
    doc["model"]["members"]["col"]["density"] = 7850.0
    assert code_of(doc)[0] == "unknown_field"


# --- B1/B2 and the buckling eigensolver ------------------------------------------

CANTILEVER = {"base": {"ux": 0, "uy": 0, "rz": 0}}
PINNED = {"base": {"ux": 0, "uy": 0}, "tip": {"uy": 0}}
GUIDED = {"base": {"ux": 0, "uy": 0, "rz": 0}, "tip": {"uy": 0, "rz": 0}}
P0 = 1000.0


def _scipy():
    return pytest.importorskip("scipy.linalg")


def euler(supports):
    if supports is CANTILEVER:
        return math.pi**2 * E * I / (4 * L**2)
    if supports is PINNED:
        return math.pi**2 * E * I / L**2
    return 4 * math.pi**2 * E * I / L**2


def test_cantilever_euler_factor_and_mesh_convergence():
    _scipy()
    exact = euler(CANTILEVER)
    errors = {}
    for elements in (8, 16, 32):
        result = solve_buckling(model(supports=CANTILEVER, elements=elements))
        errors[elements] = abs(result.critical_factor * P0 - exact) / exact
    assert errors[16] <= 1e-6
    assert errors[32] <= 1e-7
    for coarse, fine in ((8, 16), (16, 32)):
        assert 8 <= errors[coarse] / errors[fine] <= 32


def test_pinned_pinned_euler_factor():
    _scipy()
    result = solve_buckling(model(supports=PINNED))
    exact = euler(PINNED)
    assert abs(result.critical_factor * P0 - exact) / exact <= 1e-5


def test_fixed_fixed_euler_factor():
    _scipy()
    result = solve_buckling(model(supports=GUIDED))
    exact = euler(GUIDED)
    assert abs(result.critical_factor * P0 - exact) / exact <= 2e-4


def test_reference_load_scaling_invariance():
    _scipy()
    physical = []
    factors = []
    for scale in (0.5, 1.0, 2.0):
        result = solve_buckling(model(loads={"nodal": {"tip": {"fx": -P0 * scale}}}))
        factors.append(result.critical_factor)
        physical.append(result.critical_factor * P0 * scale)
    assert factors[0] == pytest.approx(2 * factors[1], rel=1e-12)
    assert factors[2] == pytest.approx(0.5 * factors[1], rel=1e-12)
    for value in physical[1:]:
        assert abs(value - physical[0]) / physical[0] <= 1e-12


def test_eigenpair_residual_and_orthonormality():
    _scipy()
    doc = model()
    result = solve_buckling(doc)
    assert result.residuals.max() <= 1e-9
    assert result.preload_residual <= 1e-8
    spec = validate_buckling_model(doc)
    free, _ = free_dofs(spec, spec.mesh)
    stiffness = assemble_stiffness(spec)[np.ix_(free, free)]
    shapes = result.modes[free]
    error = np.max(np.abs(shapes.T @ stiffness @ shapes - np.identity(shapes.shape[1])))
    assert error <= 1e-8


def test_result_contract_and_sign_convention():
    _scipy()
    doc = model()
    result = solve_buckling(doc, modes=3)
    assert result.critical_factor == result.factors[0]
    assert result.factors.shape == (3,)
    assert np.all(np.diff(result.factors) >= 0)
    assert result.modes.shape == (51, 3)
    assert result.residuals.shape == (3,)
    assert result.dof_labels[0] == ("input", "base", "ux")
    fixed = [index for index, label in enumerate(result.dof_labels) if label[1] == "base"]
    assert np.all(result.modes[fixed] == 0)
    for column in range(3):
        phi = result.modes[:, column]
        assert phi[int(np.argmax(np.abs(phi)))] > 0
    assert result.axial_forces == pytest.approx(tuple([-P0] * 16), rel=1e-12)


def test_tension_and_zero_preload_have_no_factor():
    _scipy()
    for loads in ({"nodal": {"tip": {"fx": P0}}}, {}):
        with pytest.raises(BucklingError) as info:
            solve_buckling(model(loads=loads))
        assert info.value.code == "buckling_no_positive_factor"


def test_singular_stiffness_is_reported_as_a_mechanism(monkeypatch):
    _scipy()
    import scipy.linalg

    def explode(*args, **kwargs):
        raise np.linalg.LinAlgError("forced")

    monkeypatch.setattr(scipy.linalg, "eigh", explode)
    with pytest.raises(BucklingError) as info:
        solve_buckling(model())
    assert info.value.code == "buckling_mechanism"


def test_mode_argument_bounds():
    with pytest.raises(ModelValidationError) as info:
        solve_buckling(model(), modes=0)
    assert info.value.code == "invalid_value"
    with pytest.raises(ModelValidationError):
        solve_buckling(model(), modes=21)
    with pytest.raises(ModelValidationError):
        solve_buckling(model(), modes="3")
