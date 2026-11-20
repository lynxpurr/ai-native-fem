"""Regression tests for input types, numeric range and solver diagnostics."""
from fractions import Fraction
from unittest.mock import patch
import numpy as np
import pytest
from anf import Beam, Plate, solve_cantilever, solve_plate
from anf._linear import eliminate, strain_energy
from anf.beam import beam_element


def test_numpy_integer_overflow_cannot_bypass_plate_size_limit():
    with np.errstate(over="raise"):
        with pytest.raises(ValueError, match="1800 DOFs"):
            Plate(1, 1, .1, 1e6, .3, nx=np.uint64(2**64-1), ny=2)


def test_models_store_builtin_numeric_types():
    beam = Beam(np.float32(2), np.float64(210e9), Fraction(1, 1000000), np.int64(4))
    assert all(type(getattr(beam, name)) is float for name in ("length", "young_modulus", "second_moment"))
    assert type(beam.elements) is int
    plate = Plate(np.float32(1), 1, .2, np.float64(70e9), np.float32(.3), nx=np.uint64(4), ny=np.int64(4))
    assert type(plate.nx) is int and type(plate.ny) is int
    assert type(plate.poisson_ratio) is float and type(plate.young_modulus) is float
    assert np.isfinite(solve_plate(plate, pressure=np.float64(1000)).deflection).all()


@pytest.mark.parametrize("value", [10**1000, Fraction(10**1000, 3), complex(1, 1), True, float("nan")])
def test_unrepresentable_or_nonreal_model_input_has_clear_error(value):
    with pytest.raises(ValueError): Beam(value, 1, 1)


@pytest.mark.parametrize("support", [[], {}, None, 7])
def test_bad_support_is_a_validation_error(support):
    with pytest.raises(ValueError, match="support must be"):
        Plate(1, 1, .1, 1e6, .3, support=support)


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_nonfinite_system_is_rejected_even_with_zero_load(bad):
    with pytest.raises(ValueError, match="finite"):
        eliminate(np.array([[bad]]), np.zeros(1), [])
    with pytest.raises(ValueError, match="finite"):
        eliminate(np.eye(1), np.array([bad]), [])


@pytest.mark.parametrize("scale", [1e-200, 1e-15, 1.0, 1e200])
def test_residual_is_relative_and_bad_solves_do_not_disappear_at_small_loads(scale):
    with patch("numpy.linalg.solve", return_value=np.zeros(1)):
        with pytest.raises(ArithmeticError, match="relative residual"):
            eliminate(np.array([[2*scale]]), np.array([scale]), [])


@pytest.mark.parametrize("scale", [1e-200, 1e-15, 1.0, 1e200])
def test_balanced_solve_and_energy_are_stable_under_unit_scaling(scale):
    matrix, force = np.array([[2*scale]]), np.array([scale])
    displacement, reactions, residual = eliminate(matrix, force, [])
    assert displacement[0] == pytest.approx(.5)
    assert residual < 1e-14
    assert not reactions.any()
    assert strain_energy(matrix, displacement)/scale == pytest.approx(.25)


@pytest.mark.parametrize("load", [1e-100, 1.0, 1e100])
def test_beam_response_is_load_scale_invariant(load):
    result = solve_cantilever(Beam(2, 210e9, 1e-6, elements=4), tip_force=load)
    assert result.deflection[-1]/load == pytest.approx(8/(3*210e9*1e-6), rel=1e-10)
    assert result.relative_residual < 1e-8


def test_fully_constrained_system_keeps_reactions_from_generator_indices():
    displacement, reactions, residual = eliminate(np.eye(2), np.array([3., 4.]), (i for i in (0, 1)))
    np.testing.assert_array_equal(displacement, [0, 0])
    np.testing.assert_array_equal(reactions, [-3, -4])
    assert residual == 0


@pytest.mark.parametrize("fixed", [[-1], [1], [True], [.5], [0, "bad"]])
def test_invalid_fixed_indices_are_rejected(fixed):
    with pytest.raises(ValueError, match="fixed DOFs"):
        eliminate(np.eye(1), np.ones(1), fixed)


def test_singular_system_nonfinite_solution_and_unrepresentable_energy_are_rejected():
    with pytest.raises(ArithmeticError, match="could not be solved"):
        eliminate(np.zeros((1, 1)), np.ones(1), [])
    with patch("numpy.linalg.solve", return_value=np.array([float("nan")])):
        with pytest.raises(ArithmeticError, match="non-finite solution"):
            eliminate(np.eye(1), np.ones(1), [])
    with pytest.raises(ArithmeticError):
        strain_energy(np.eye(1), np.array([1e200]))


def test_element_range_failure_is_controlled_and_numpy_error_state_is_restored():
    before = np.geterr().copy()
    with pytest.raises(ArithmeticError, match="floating-point range"):
        beam_element(1e-200, 1.0)
    assert np.geterr() == before
