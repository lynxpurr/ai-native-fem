import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from anf import Beam, Plate, solve_cantilever, solve_plate
from anf.verification import (
    REFERENCE_REGISTRY, ReferenceKind, VerificationReport, evaluate_checks,
    input_parameters, verify_cantilever, verify_plate,
)


BEAM_ID = "cantilever.closed-form.v1"
THICK_ID = "plate.navier-31.v1"
THIN_ID = "plate.thin-q4-navier-31.v1"
SNAPSHOT_ID = "plate.thin-q4-snapshot.v1"


def square(thickness, n):
    return Plate(1, 1, thickness, 70e9, 0.3, nx=n, ny=n, support="hard_simply_supported")


def test_cantilever_uses_complete_closed_form_and_residual_alone_is_unverified():
    model = Beam(2, 210e9, 1e-6)
    result = solve_cantilever(model, tip_force=1000)
    inputs = input_parameters(model, tip_force=1000)
    report = verify_cantilever(model, tip_force=1000)
    assert report == evaluate_checks(result, REFERENCE_REGISTRY[BEAM_ID], inputs=inputs)
    assert (report.input_status, report.solve_status, report.verification_status) == (
        "valid", "solved", "verification_passed",
    )
    assert report.reference_ids == (BEAM_ID,)
    assert report.checks[0].threshold == 1e-8
    checks = {check.observable: check for check in report.checks[1:]}
    assert set(checks) == {"tip_deflection", "tip_slope", "root_force", "root_moment"}
    assert checks["tip_deflection"].reference == pytest.approx(8e3 / (3 * 210e3))
    assert checks["tip_slope"].reference == pytest.approx(2e3 / 210e3)
    assert checks["root_force"].reference == -1000
    assert checks["root_moment"].reference == -2000
    assert all(check.outcome == "pass" for check in checks.values())
    residual_only = evaluate_checks(result, inputs=inputs)
    assert residual_only.checks == report.checks[:1]
    assert residual_only.verification_status == "solved_unverified"
    assert verify_cantilever(model, tip_force=1000, reference=None) == residual_only


def test_zero_and_cancelling_beam_references_do_not_divide_by_zero():
    model = Beam(2, 10, 1)
    unloaded = verify_cantilever(model)
    assert unloaded.verification_status == "verification_passed"
    assert all((c.metric, c.reference, c.threshold) == ("absolute", 0.0, 0.0)
               for c in unloaded.checks[1:])
    cancelled = verify_cantilever(model, tip_force=-3, uniform_load=4)
    check = next(c for c in cancelled.checks if c.observable == "tip_deflection")
    assert (check.reference, check.metric, check.outcome) == (0.0, "scaled_absolute", "pass")
    assert check.error_scale == pytest.approx(0.8)


def test_finite_beam_error_fails_accuracy_despite_passing_residual():
    model = Beam(2, 10, 1)
    result = solve_cantilever(model, tip_force=1)
    incorrect = replace(result, deflection=result.deflection * 2)
    report = evaluate_checks(incorrect, BEAM_ID, inputs=input_parameters(model, tip_force=1))
    assert report.solve_status == "solved"
    assert report.checks[0].outcome == "pass"
    assert report.checks[1].outcome == "fail"
    assert report.verification_status == "verification_failed"


@pytest.mark.parametrize("thickness,n,evidence,status,expected,bound", [
    (0.2, 16, THICK_ID, "verification_passed", 9.56331888e-8, 2.052e-12),
    (0.005, 8, THIN_ID, "verification_failed", 5.07047234e-3, 1.141e-9),
])
def test_plate_reference_retains_center_observable_and_truncation_bound(
    thickness, n, evidence, status, expected, bound,
):
    model = square(thickness, n)
    report = verify_plate(model, pressure=1000)
    assert report.solve_status == "solved"
    assert report.verification_status == status
    check = report.checks[-1]
    assert (check.evidence_id, check.unit, check.threshold) == (evidence, "m", 0.08)
    assert check.observed == solve_plate(model, pressure=1000).deflection[n//2, n//2]
    assert check.reference == pytest.approx(expected, rel=1e-8, abs=0)
    assert check.reference_error_bound == pytest.approx(bound, rel=1e-3, abs=0)
    assert check.error == pytest.approx(abs(check.observed / check.reference - 1))
    if thickness == 0.005:
        assert check.observed / check.reference < 0.2
        assert any("shear locking" in warning for warning in report.warnings)
    else:
        assert check.error < 0.08
    assert VerificationReport.from_json(report.to_json()) == report


def test_thin_negative_case_cannot_certify_an_accuracy_pass():
    model = square(0.005, 8)
    result = solve_plate(model, pressure=1000)
    report = verify_plate(model, pressure=1000)
    deflection = result.deflection.copy()
    deflection[4, 4] = report.checks[-1].reference
    checked = evaluate_checks(replace(result, deflection=deflection), THIN_ID,
                              inputs=input_parameters(model, pressure=1000))
    assert checked.checks[-1].outcome == "pass"
    assert checked.verification_status == "solved_unverified"
    assert any("negative baseline changed" in warning for warning in checked.warnings)


def test_no_anchor_for_unsupported_plate_scope():
    plate = square(0.2, 8)
    auto = verify_plate(plate, pressure=1000)
    requested = verify_plate(plate, pressure=1000, reference=THICK_ID)
    assert auto.solve_status == requested.solve_status == "solved"
    assert auto.verification_status == requested.verification_status == "solved_unverified"
    assert auto.reference_ids == ()
    assert requested.checks[-1].outcome == "not_applicable"
    assert requested.checks[-1].observed is None
    assert "applicability domain" in requested.warnings[-1]


@pytest.mark.parametrize("change", [
    {"observable_units": (("tip_deflection", ""),)}, {"formula_location": ""},
    {"scope": " "}, {"threshold": None}, {"threshold": float("inf")}, {"threshold": 0},
    {"error_budget": ""}, {"source": ""},
])
def test_reference_record_requires_verifiable_metadata(change):
    with pytest.raises(ValueError):
        replace(REFERENCE_REGISTRY[BEAM_ID], **change)


def test_forged_but_complete_metadata_cannot_become_an_independent_anchor():
    beam_ref = REFERENCE_REGISTRY[BEAM_ID]
    forged = replace(beam_ref, source="Unreviewed source", threshold=1.0)
    with pytest.raises(ValueError, match="REFERENCE_REGISTRY"):
        verify_cantilever(Beam(2, 10, 1), tip_force=1, reference=forged)
    with pytest.raises(ValueError, match="REFERENCE_REGISTRY"):
        verify_cantilever(Beam(2, 10, 1), tip_force=1, reference="unknown")
    with pytest.raises(TypeError):
        REFERENCE_REGISTRY[forged.id] = forged
    assert {kind.value for kind in ReferenceKind} == {
        "analytic", "manufactured", "independent-numerical", "external-comparison", "self-regression",
    }


def test_same_solver_snapshot_passes_only_a_regression_check():
    model = square(0.005, 8)
    report = verify_plate(model, pressure=1000, reference=SNAPSHOT_ID)
    assert report.checks[-1].outcome == "pass"
    assert report.verification_status == "solved_unverified"
    assert report.input_hash == verify_plate(model, pressure=1000).input_hash
    assert VerificationReport.from_json(report.to_json()) == report


def test_fingerprint_is_from_actual_normalized_inputs():
    model = Beam(2, 10, 1, elements=4)
    result = solve_cantilever(model, tip_force=2)
    inputs = input_parameters(model, tip_force=np.int64(2))
    expected = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()
    reordered = dict(reversed(list(inputs.items())))
    assert evaluate_checks(result, inputs=reordered).input_hash == expected
    assert verify_cantilever(model, tip_force=2.0).input_hash == expected
    assert verify_cantilever(model, tip_force=3).input_hash != expected
    assert verify_cantilever(replace(model, elements=5), tip_force=2).input_hash != expected
    assert verify_cantilever(replace(model, young_modulus=11), tip_force=2).input_hash != expected
    invalid = evaluate_checks(result, inputs={**inputs, "units": "unknown"})
    assert (invalid.input_status, invalid.solve_status, invalid.input_hash) == ("invalid", "not_run", None)


@pytest.mark.parametrize("model,loads,expected_input,expected_solve", [
    ({"length": 0.0, "young_modulus": 10, "second_moment": 1}, {}, "invalid", "not_run"),
    (Beam(2, 10, 1), {"tip_force": float("nan")}, "invalid", "not_run"),
    (Beam(1e-200, 10, 1), {"tip_force": 1}, "valid", "numerical_failure"),
])
def test_invalid_input_and_numeric_failure_never_claim_success(model, loads, expected_input, expected_solve):
    report = verify_cantilever(model, reference=BEAM_ID, **loads)
    assert (report.input_status, report.solve_status) == (expected_input, expected_solve)
    assert (report.input_hash is not None) == (expected_input == "valid")
    assert report.verification_status == "not_run"
    assert all(check.outcome == "not_run" and check.observed is None for check in report.checks)
    assert report.warnings
    assert VerificationReport.from_json(report.to_json()) == report


def test_nonfinite_result_and_out_of_budget_residual_are_failed_solves():
    model = Beam(2, 10, 1)
    result = solve_cantilever(model, tip_force=1)
    inputs = input_parameters(model, tip_force=1)
    for bad in (replace(result, strain_energy=float("nan")),
                replace(result, deflection=np.full_like(result.deflection, np.inf)),
                replace(result, relative_residual=2e-8)):
        report = evaluate_checks(bad, BEAM_ID, inputs=inputs)
        assert report.solve_status == "numerical_failure"
        assert report.verification_status == "not_run"
        assert report.checks[0].observed is None


def test_round_trip_preserves_float_precision_and_rejects_invalid_json_measurements():
    report = verify_cantilever(Beam(2.3, 70e9, 2.1e-6), tip_force=-100, uniform_load=23)
    assert VerificationReport.from_json(report.to_json()) == report
    assert json.loads(report.to_json())["schema_version"] == "1.0"
    for bad in (None, float("nan"), float("inf")):
        data = json.loads(report.to_json())
        data["checks"][1]["observed"] = bad
        with pytest.raises(ValueError):
            VerificationReport.from_json(json.dumps(data))
    data = json.loads(report.to_json())
    data["checks"] = data["checks"][:1]
    with pytest.raises(ValueError, match="contradict"):
        VerificationReport.from_json(json.dumps(data))
    data = json.loads(report.to_json())
    data["checks"][1]["threshold"] = 1.0
    with pytest.raises(ValueError, match="registered metric and threshold"):
        VerificationReport.from_json(json.dumps(data))
