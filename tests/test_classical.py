"""Independent closed forms, equilibrium and element invariants."""
import numpy as np
import pytest
from anf import Beam, Plate, solve_cantilever, solve_plate
from anf.beam import beam_element
from anf.plate import plate_element


@pytest.mark.parametrize("elements", [1, 3, 12])
@pytest.mark.parametrize("tip,line", [(100.0, 0.0), (0.0, 45.0), (-100.0, 23.0)])
def test_beam_matches_integrated_differential_equation(elements, tip, line):
    L, E, I = 2.3, 70e9, 2.1e-6
    result = solve_cantilever(Beam(L, E, I, elements), tip_force=tip, uniform_load=line)
    expected = tip*L**3/(3*E*I) + line*L**4/(8*E*I)
    slope = tip*L**2/(2*E*I) + line*L**3/(6*E*I)
    assert result.deflection[-1] == pytest.approx(expected, rel=1e-8)
    assert result.slope[-1] == pytest.approx(slope, rel=1e-8)
    np.testing.assert_allclose(result.root_reaction, [-tip-line*L, -tip*L-line*L**2/2], rtol=1e-8)
    assert result.relative_residual < 1e-8
    assert result.strain_energy > 0


def test_beam_element_rigid_motion_and_closed_matrix():
    L, EI = 1.7, 130.0
    matrix, load = beam_element(L, EI)
    expected = EI/L**3*np.array([[12, 6*L, -12, 6*L],
        [6*L, 4*L*L, -6*L, 2*L*L], [-12, -6*L, 12, -6*L],
        [6*L, 2*L*L, -6*L, 4*L*L]])
    np.testing.assert_allclose(matrix, expected, rtol=1e-14)
    np.testing.assert_allclose(matrix @ [1, 0, 1, 0], 0, atol=1e-12)
    np.testing.assert_allclose(matrix @ [0, 1, L, 1], 0, atol=1e-12)
    np.testing.assert_allclose(load, [L/2, L*L/12, L/2, -L*L/12], rtol=1e-14)


def test_zero_beam_load_is_zero_response():
    result = solve_cantilever(Beam(1, 10, 2))
    assert not result.deflection.any()
    assert not result.root_reaction.any()
    assert result.strain_energy == 0


@pytest.mark.parametrize("field,value", [("length", 0), ("young_modulus", float("inf")),
    ("second_moment", -1), ("elements", True), ("elements", 2.5), ("elements", 201)])
def test_invalid_beam_inputs(field, value):
    data = dict(length=1, young_modulus=10, second_moment=1, elements=4)
    data[field] = value
    with pytest.raises(ValueError): Beam(**data)


def make_plate(n=4, thickness=0.2, support="hard_simply_supported"):
    return Plate(1, 1, thickness, 70e9, 0.3, nx=n, ny=n, support=support)


def navier_center(model, pressure, modes=31):
    """Sine-mode minimization of the continuous bending/shear energy.

    Each load mode q_mn has compliance 1/(D*k^4)+1/(S*k^2).
    This has no FE assembly, interpolation, calibration or stored numeric data.
    """
    E, nu, t = model.young_modulus, model.poisson_ratio, model.thickness
    D = E*t**3/(12*(1-nu**2)); S = model.shear_factor*E*t/(2*(1+nu))
    value = 0.0
    for m in range(1, modes+1, 2):
        for n in range(1, modes+1, 2):
            wave = (m*np.pi/model.lx)**2 + (n*np.pi/model.ly)**2
            sign = (-1)**((m-1)//2+(n-1)//2)
            value += sign*16*pressure/(np.pi**2*m*n)*(1/(D*wave**2)+1/(S*wave))
    return value


def test_plate_element_has_only_three_rigid_modes_and_conserves_pressure():
    model = make_plate()
    a, b = 0.4, 0.7
    matrix, load = plate_element(model, a, b)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-6)
    points = np.array([[0, 0], [a, 0], [a, b], [0, b]])
    for c, sx, sy in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        motion = np.column_stack((c+sx*points[:, 0]+sy*points[:, 1], np.full(4, sx), np.full(4, sy))).ravel()
        assert np.linalg.norm(matrix @ motion)/np.linalg.norm(matrix) < 1e-14
    values = np.linalg.eigvalsh(matrix)
    assert np.count_nonzero(np.abs(values) < values[-1]*1e-10) == 3
    assert values[3] > 0
    assert sum(load[::3]) == pytest.approx(a*b)
    assert not load[1::3].any() and not load[2::3].any()


@pytest.mark.parametrize("support", ["clamped", "hard_simply_supported"])
def test_plate_load_balance_symmetry_energy_and_zero_boundary_data(support):
    model = make_plate(6, support=support)
    result = solve_plate(model, pressure=1000)
    assert np.sum(result.reactions[:, :, 0]) == pytest.approx(-1000, rel=1e-9)
    np.testing.assert_allclose(result.deflection, result.deflection[::-1, :], rtol=1e-10, atol=1e-18)
    np.testing.assert_allclose(result.deflection, result.deflection[:, ::-1], rtol=1e-10, atol=1e-18)
    assert not result.deflection[0, :].any() and not result.deflection[:, 0].any()
    assert not result.deflection[-1, :].any() and not result.deflection[:, -1].any()
    assert result.strain_energy > 0 and result.deflection[3, 3] > 0
    assert result.relative_residual < 1e-9


def test_thick_plate_refinement_approaches_continuous_navier_solution():
    errors = []
    for n in (4, 8, 16):
        model = make_plate(n, thickness=0.2)
        computed = solve_plate(model, pressure=1000).deflection[n//2, n//2]
        reference = navier_center(model, 1000)
        errors.append(abs(computed/reference-1))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 0.08


def test_known_thin_plate_locking_is_not_hidden():
    model = make_plate(8, thickness=0.005)
    computed = solve_plate(model, pressure=1000).deflection[4, 4]
    assert computed/navier_center(model, 1000) < 0.2


@pytest.mark.parametrize("field,value", [("lx", 0), ("thickness", -1), ("young_modulus", True),
    ("poisson_ratio", 0.5), ("poisson_ratio", -1), ("nx", 1), ("ny", 2.5),
    ("nx", 500), ("support", "unsupported"), ("shear_factor", float("nan"))])
def test_invalid_plate_inputs(field, value):
    data = dict(lx=1, ly=1, thickness=0.2, young_modulus=1e6, poisson_ratio=0.3)
    data[field] = value
    with pytest.raises(ValueError): Plate(**data)


def test_nonfinite_applied_load_is_rejected():
    with pytest.raises(ValueError): solve_cantilever(Beam(1, 10, 1), tip_force=float("nan"))
    with pytest.raises(ValueError): solve_plate(make_plate(), pressure=float("inf"))
