"""Schema 1.6 preload equilibrium and geometric stiffness checks."""

import math

import numpy as np
import pytest

from anf.dynamics.system import assemble_stiffness, free_dofs
from anf.model import ModelValidationError, canonical_buckling_input_hash, validate_buckling_model
from anf.stability import BucklingError, PreloadResult, assemble_geometric, solve_preload

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
