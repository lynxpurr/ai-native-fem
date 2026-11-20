"""Rectangular MITC4 patch tests and independent Navier convergence checks."""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from anf import Plate, solve_plate
from anf.plate import _bending_matrix, _curvature_matrix, _mitc4_shear_matrix, plate_element
from anf.verification import VerificationReport, evaluate_checks, input_parameters, verify_plate


Q4 = "q4_full"
MITC4 = "mitc4_rectangle"
PRESSURE = 1000
THICK_ID = "plate.navier-31.v1"
THIN_ID = "plate.thin-q4-navier-31.v1"
SNAPSHOT_ID = "plate.thin-q4-snapshot.v1"


def plate(lx=1, ly=1, thickness=0.005, n=8, formulation=MITC4):
    return Plate(lx, ly, thickness, 70e9, 0.3, nx=n, ny=n,
                 support="hard_simply_supported", formulation=formulation)


def navier_center(model, modes):
    # Independent continuum modes with modal rotational equilibrium eliminated.
    m = np.arange(1, modes + 1, 2)
    sign_over_index = (-1.0)**(m//2) / m
    wave = (np.pi*m[:, None]/model.lx)**2 + (np.pi*m/model.ly)**2
    nu, t, modulus = model.poisson_ratio, model.thickness, model.young_modulus
    rigidity = modulus*t**3 / (12*(1 - nu**2))
    shear = model.shear_factor*modulus*t / (2*(1 + nu))
    center_load = (16*PRESSURE/np.pi**2) * np.outer(sign_over_index, sign_over_index)
    return np.sum(center_load * (1/(rigidity*wave**2) + 1/(shear*wave)))


@pytest.mark.parametrize("width,height", [(0.25, 0.25), (0.4, 0.7)])
@pytest.mark.parametrize("thickness", [0.2, 0.005])
def test_mitc4_element_patch(width, height, thickness, record_property):
    model = plate(thickness=thickness)
    stiffness, load = plate_element(model, width, height)
    eigenvalues = np.linalg.eigvalsh(stiffness)
    zero = 1e-12 * eigenvalues[-1]
    assert np.max(np.abs(eigenvalues[:3])) < zero < eigenvalues[3]
    symmetry = np.linalg.norm(stiffness - stiffness.T) / np.linalg.norm(stiffness)
    load_error = np.max(np.abs(load[::3] / (width*height/4) - 1))
    assert symmetry < 1e-12
    assert load_error < 1e-12
    assert not load.reshape(4, 3)[:, 1:].any()

    x = np.array([0, width, width, 0])
    y = np.array([0, 0, height, height])
    ones, zeros = np.ones(4), np.zeros(4)
    rigid = [np.column_stack(fields).ravel()
             for fields in [(ones, zeros, zeros), (x, ones, zeros), (y, zeros, ones)]]
    rigid_error = max(np.linalg.norm(stiffness @ motion) / (eigenvalues[-1]*np.linalg.norm(motion))
                      for motion in rigid)
    assert rigid_error < 1e-12

    points = np.polynomial.legendre.leggauss(2)[0]
    shear_matrices, curvature_matrices = [], []
    for xi, eta in product(points, repeat=2):
        shear_matrices.append(_mitc4_shear_matrix(xi, eta, width, height))
        dx = np.array([eta - 1, 1 - eta, 1 + eta, -1 - eta]) / (2*width)
        dy = np.array([xi - 1, -1 - xi, 1 + xi, 1 - xi]) / (2*height)
        curvature_matrices.append(_curvature_matrix(dx, dy))
    bending = _bending_matrix(model)
    nu, modulus = model.poisson_ratio, model.young_modulus
    rigidity = modulus*thickness**3 / (12*(1 - nu**2))
    shear_errors, energy_errors = [], []
    for xx, yy, xy in [(1, 0, 0), (0, 1, 0), (1, -0.7, 0), (0, 0, 1)]:
        rotation_x, rotation_y = xx*x + xy*y, yy*y + xy*x
        w = (xx*x**2 + yy*y**2)/2 + xy*x*y
        motion = np.column_stack((w, rotation_x, rotation_y)).ravel()
        slope_scale = np.max(np.abs([rotation_x, rotation_y]))
        shear_errors.append(max(np.max(np.abs(matrix @ motion)) / slope_scale
                                for matrix in shear_matrices))
        expected_energy = rigidity*width*height/2 * (xx**2 + yy**2 + 2*nu*xx*yy + 2*(1 - nu)*xy**2)
        # Integrate bending directly to avoid cancellation of large shear matrix entries.
        curvatures = [matrix @ motion for matrix in curvature_matrices]
        energy = width*height/8 * sum(curvature @ bending @ curvature for curvature in curvatures)
        energy_errors.append(abs(energy / expected_energy - 1))
    assert max(shear_errors) < 1e-12
    assert max(energy_errors) < 1e-12
    for name, value in dict(width=width, height=height, thickness=thickness,
                            nullity=3, smallest_positive_eigenvalue=eigenvalues[3],
                            symmetry=symmetry, load_error=load_error, rigid_error=rigid_error,
                            constant_curvature_shear=max(shear_errors),
                            constant_curvature_energy_error=max(energy_errors)).items():
        record_property(name, value)


@pytest.mark.parametrize("lx", [1, 2])
@pytest.mark.parametrize("thickness", [0.2, 0.05, 0.01, 0.005])
def test_rectangular_convergence_matrix(lx, thickness, record_property):
    model = plate(lx=lx, thickness=thickness)
    reference_127 = navier_center(model, 127)
    reference = navier_center(model, 255)
    delta = abs(reference_127 / reference - 1)
    assert delta < 1e-6
    for name, value in dict(lx=lx, ly=model.ly, thickness=thickness, reference_127=reference_127,
                            reference_255=reference, reference_delta=delta).items():
        record_property(name, value)

    mitc4_errors = []
    for formulation, n in product((Q4, MITC4), (4, 8, 16)):
        mesh = replace(model, nx=n, ny=n, formulation=formulation)
        result = solve_plate(mesh, pressure=PRESSURE)
        w = result.deflection
        center = w[n//2, n//2]
        error = abs(center / reference - 1)
        assert result.relative_residual <= 1e-8
        assert not w[[0, -1], :].any()
        assert not w[:, [0, -1]].any()
        assert not result.rotations[[0, -1], :, 0].any()
        assert not result.rotations[:, [0, -1], 1].any()
        reflection_error = max(np.max(np.abs(w - w[::-1, :])), np.max(np.abs(w - w[:, ::-1])))
        assert reflection_error / np.max(np.abs(w)) < 1e-7
        balance_error = abs(result.reactions[:, :, 0].sum() / (PRESSURE*mesh.lx*mesh.ly) + 1)
        assert balance_error < 1e-7

        # Uniform pressure assigns one fourth of each adjacent cell's force to the node.
        edge_counts = np.full(n + 1, 2)
        edge_counts[[0, -1]] = 1
        nodal_load = PRESSURE*mesh.lx*mesh.ly/(4*n**2) * np.outer(edge_counts, edge_counts)
        work = np.sum(nodal_load * w)
        work_error = abs(2*result.strain_energy / work - 1)
        assert work_error < 1e-7
        for name, value in dict(center=center, error=error, residual=result.relative_residual,
                                balance_error=balance_error, work_error=work_error).items():
            record_property(f"{formulation}.{n}.{name}", value)
        if formulation == MITC4:
            mitc4_errors.append(error)
    coarse, medium, fine = mitc4_errors
    assert fine < medium < coarse
    assert fine < 0.03


@pytest.mark.parametrize("thickness,n,reference_id,legacy_hash", [
    (0.2, 16, THICK_ID, "d17b6de821561f8e1b2bbc919c634ee011a4a35509205b73dcca683e57b5d835"),
    (0.005, 8, THIN_ID, "010ba5738a4616515c65a29570bb36398b50f8ad606fb89080939785835eee4a"),
])
def test_formulation_selects_distinct_verification_scope(thickness, n, reference_id, legacy_hash):
    model = plate(thickness=thickness, n=n)
    old = Plate(1, 1, thickness, 70e9, 0.3, nx=n, ny=n, support="hard_simply_supported")
    explicit = replace(model, formulation=Q4)
    old_inputs = input_parameters(old, pressure=PRESSURE)
    old_report = verify_plate(old, pressure=PRESSURE)
    assert explicit == old
    assert old_inputs == input_parameters(explicit, pressure=PRESSURE)
    assert "formulation" not in old_inputs["model"]
    assert old_report.reference_ids == (reference_id,)
    assert old_report.input_hash == legacy_hash
    assert evaluate_checks(solve_plate(old, pressure=PRESSURE), reference_id, inputs=old_inputs) == old_report

    inputs = input_parameters(model, pressure=PRESSURE)
    result = solve_plate(model, pressure=PRESSURE)
    report = verify_plate(model, pressure=PRESSURE)
    assert inputs["model"]["formulation"] == MITC4
    assert inputs["formulation"] == "reissner_mindlin_mitc4_rectangle"
    assert report.input_hash != legacy_hash
    assert report.verification_status == "solved_unverified"
    assert report.reference_ids == ()
    assert report.warnings == ("No independent reference was evaluated.",)
    assert VerificationReport.from_json(report.to_json()) == report
    assert evaluate_checks(result, inputs=inputs) == report
    for q4_reference in (THICK_ID, THIN_ID, SNAPSHOT_ID):
        explicit_report = verify_plate(model, pressure=PRESSURE, reference=q4_reference)
        assert explicit_report.checks[-1].outcome == "not_applicable"
        assert explicit_report.verification_status == "solved_unverified"
        assert all("locking" not in warning for warning in explicit_report.warnings)


@pytest.mark.parametrize("formulation", ["mitc4", None, 4, []])
def test_unknown_formulation(formulation):
    with pytest.raises(ValueError, match="formulation must be"):
        plate(formulation=formulation)


def test_clamped_and_distorted_geometry_are_rejected():
    model = plate()
    with pytest.raises(ValueError, match="requires hard_simply_supported"):
        replace(model, support="clamped")
    distorted = [[0, 0], [1, 0], [1.2, 0.9], [0, 1]]
    with pytest.raises(ValueError, match="width must be a finite real number"):
        plate_element(model, distorted, 1)


@pytest.mark.parametrize("changes,pressure", [
    ({"thickness": 1e-200}, 1),
    ({"thickness": 1e200}, 1),
    ({"lx": 5e-324}, 1),
    ({}, 5e-324),
    ({}, 1e-200),
    ({"lx": 10, "ly": 10}, 1e308),
])
def test_mitc4_reports_floating_point_range_failures(changes, pressure):
    model = replace(plate(n=2), **changes)
    with pytest.raises(ArithmeticError):
        solve_plate(model, pressure=pressure)
    assert verify_plate(model, pressure=pressure).solve_status == "numerical_failure"


def test_mitc4_accepts_signed_and_zero_pressure():
    model = plate(n=4)
    positive = solve_plate(model, pressure=PRESSURE)
    negative = solve_plate(model, pressure=-PRESSURE)
    unloaded = solve_plate(model, pressure=0)
    np.testing.assert_array_equal(positive.deflection, -negative.deflection)
    assert positive.strain_energy == negative.strain_energy
    assert not unloaded.deflection.any()
    assert not unloaded.reactions.any()
    assert unloaded.strain_energy == unloaded.relative_residual == 0
