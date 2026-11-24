"""Schema 1.4 mass assembly checks independent of the modal solver."""

import numpy as np
import pytest

from anf.dynamics import assemble_mass
from anf.model import ModelValidationError, validate_modal_model, validate_model


UNITS = {"length": "m", "force": "N", "stress": "Pa"}


def member(kind, *, density=7850.0, elements=1):
    value = {"kind": kind, "i": "a", "j": "b", "young_modulus": 210e9,
             "area": 0.01, "elements": elements}
    if kind == "frame2d":
        value["second_moment"] = 1e-6
    if density is not None:
        value["density"] = density
    return value


def declaration(kind="frame2d", *, nodes=None, density=7850.0, elements=1,
                supports=None, loads=None, analysis=None, version="1.4"):
    return {
        "schema_version": version,
        "problem_type": "planar_structure",
        "units": dict(UNITS),
        "model": {
            "nodes": nodes or {"a": [0, 0], "b": [2, 0]},
            "members": {"beam": member(kind, density=density, elements=elements)},
            "supports": {} if supports is None else supports,
        },
        "loads": {} if loads is None else loads,
        "analysis": analysis or {"type": "modal", "modes": 6},
    }


def mass_properties(matrix):
    eigenvalues = np.linalg.eigvalsh(matrix)
    assert np.isfinite(matrix).all()
    np.testing.assert_allclose(matrix, matrix.T, rtol=0, atol=0)
    assert eigenvalues[0] > 0
    return eigenvalues


def test_truss_mass_matrix_and_rigid_translation():
    result = assemble_mass(declaration("truss2d"))
    total = 7850.0 * 0.01 * 2.0
    expected = total / 6 * np.array([
        [2, 0, 1, 0],
        [0, 2, 0, 1],
        [1, 0, 2, 0],
        [0, 1, 0, 2],
    ])
    np.testing.assert_allclose(result.matrix, expected, rtol=0, atol=0)
    assert result.total_mass == pytest.approx(total)
    assert result.dof_labels == (("input", "a", "ux"), ("input", "a", "uy"),
                                 ("input", "b", "ux"), ("input", "b", "uy"))
    translation_x = result.matrix[0::2, 0::2].copy()
    assert translation_x.sum() == pytest.approx(total)
    assert result.matrix[1::2, 1::2].sum() == pytest.approx(total)
    mass_properties(result.matrix)


def test_inclined_truss_preserves_rigid_translation():
    nodes = {"a": [0, 0], "b": [3, 4]}
    result = assemble_mass(declaration("truss2d", nodes=nodes))
    total = 7850.0 * 0.01 * 5.0
    assert result.total_mass == pytest.approx(total)
    for vector in ([1, 0, 1, 0], [0, 1, 0, 1]):
        vector = np.array(vector, dtype=float)
        assert vector @ result.matrix @ vector == pytest.approx(total)
    mass_properties(result.matrix)


def test_frame_mass_matches_hermite_matrix_and_rotational_inertia():
    result = assemble_mass(declaration("frame2d"))
    total = 7850.0 * 0.01 * 2.0
    length = 2.0
    expected = total / 420 * np.array([
        [140, 0, 0, 70, 0, 0],
        [0, 156, 22 * length, 0, 54, -13 * length],
        [0, 22 * length, 4 * length**2, 0, 13 * length, -3 * length**2],
        [70, 0, 0, 140, 0, 0],
        [0, 54, 13 * length, 0, 156, -22 * length],
        [0, -13 * length, -3 * length**2, 0, -22 * length, 4 * length**2],
    ])
    np.testing.assert_allclose(result.matrix, expected, rtol=1e-15, atol=0)
    assert result.dof_labels == (("input", "a", "ux"), ("input", "a", "uy"),
                                 ("input", "a", "rz"), ("input", "b", "ux"),
                                 ("input", "b", "uy"), ("input", "b", "rz"))
    assert result.total_mass == pytest.approx(total)
    assert result.matrix[0::3, 0::3].sum() == pytest.approx(total)
    assert result.matrix[1::3, 1::3].sum() == pytest.approx(total)
    rotation = np.array([0, 0, 1, 0, length, 1], dtype=float)
    assert rotation @ result.matrix @ rotation == pytest.approx(total * length**2 / 3)
    mass_properties(result.matrix)


def test_mass_matrix_is_read_only():
    result = assemble_mass(declaration("truss2d"))
    with pytest.raises(ValueError):
        result.matrix[0, 0] = 0.0


def test_density_is_required_and_positive():
    doc = declaration("truss2d", density=None)
    with pytest.raises(ModelValidationError) as error:
        validate_modal_model(doc)
    assert (error.value.code, error.value.path) == (
        "missing_field", "/model/members/beam/density")

    for density in (0.0, -1.0):
        doc = declaration("truss2d", density=density)
        with pytest.raises(ModelValidationError) as error:
            validate_modal_model(doc)
        assert (error.value.code, error.value.path) == (
            "invalid_value", "/model/members/beam/density")


def test_modal_requires_empty_loads_and_modal_analysis():
    doc = declaration("truss2d", loads={"nodal": {"a": {"fy": 1}}})
    with pytest.raises(ModelValidationError) as error:
        validate_modal_model(doc)
    assert error.value.code == "unknown_field"
    assert error.value.path == "/loads/nodal"

    doc = declaration("truss2d", analysis={"type": "static"})
    with pytest.raises(ModelValidationError) as error:
        validate_modal_model(doc)
    assert (error.value.code, error.value.path) == (
        "unsupported_analysis", "/analysis/type")


def test_schema_versions_stay_separate():
    modal = declaration("truss2d")
    with pytest.raises(ModelValidationError) as error:
        validate_model(modal)
    assert error.value.code == "unsupported_schema"

    static = {
        "schema_version": "1.0", "problem_type": "cantilever_bending", "units": dict(UNITS),
        "model": {"length": 2, "young_modulus": 210e9, "second_moment": 1e-6},
        "loads": {"tip_force": 1000}, "analysis": {"type": "static"},
    }
    spec = validate_model(static)
    assert (spec.analysis_type, spec.modes) == ("static", None)
    assert spec.to_dict()["analysis"] == {"type": "static"}
    with pytest.raises(ModelValidationError):
        assemble_mass(static)


def test_modal_round_trip_keeps_density_and_modes():
    doc = declaration("frame2d", elements=2)
    spec = validate_modal_model(doc)
    assert (spec.analysis_type, spec.modes) == ("modal", 6)
    normalized = spec.to_dict()
    assert normalized["analysis"] == {"type": "modal", "modes": 6}
    assert normalized["model"]["members"]["beam"]["density"] == 7850.0
    assert validate_modal_model(normalized).to_dict() == normalized