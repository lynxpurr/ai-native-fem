"""Numerical and delivery checks for replayable mesh studies."""

from hashlib import sha256
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from anf import studies
from anf.artifacts import ArtifactError, load_result
from anf.model import ModelValidationError, canonical_input_hash
from anf.reporting import write_report
from anf.studies import StudySpec, run_convergence


_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def document(name="thick-q4"):
    return json.loads((_EXAMPLES / "studies" / f"{name}.json").read_bytes())


def beam():
    return json.loads((_EXAMPLES / "models" / "cantilever.json").read_bytes())


def test_thick_report_replays_exact_inputs_and_artifacts(tmp_path):
    study = run_convergence(StudySpec.from_dict(document()))
    report = write_report(study, tmp_path / "结果 report")
    data = json.loads(report.read_bytes())
    steps = data["steps"]
    assert data["status"] == "complete"
    assert steps[0]["observed_m"] < steps[1]["observed_m"] < steps[2]["observed_m"]
    assert steps[1]["successive_mesh_change_m"] > steps[2]["successive_mesh_change_m"] > 0
    assert steps[0]["reference_check"]["outcome"] == "not_applicable"
    assert steps[2]["verification_status"] == "verification_passed"
    assert steps[2]["reference_check"]["error"] < 0.08
    assert max(step["relative_residual"] for step in steps) <= 1e-8
    assert data["estimate"]["observed_order"] > 0
    assert data["estimate"]["finest_discretization_estimate_m"] > 0
    assert canonical_input_hash(study.spec.model) == data["study"]["base_model_hash"]
    replay = run_convergence(StudySpec.from_dict(data["study"]))
    assert replay.to_dict() == study.to_dict()
    assert str(tmp_path) not in report.read_text(encoding="utf-8")
    for row, step in zip(steps, study.steps, strict=True):
        artifact = row["artifact"]
        directory = report.parent / artifact["path"]
        assert sha256((directory / "manifest.json").read_bytes()).hexdigest() == artifact["manifest_sha256"]
        bundle = load_result(directory)
        assert bundle.input_hash == row["input_hash"]
        for name, values in step.result.arrays.items():
            np.testing.assert_array_equal(values, bundle.arrays[name])
    assert (report.parent / "report.md").is_file()
    chart = ET.parse(report.parent / "convergence.svg")
    assert len(chart.findall(".//{http://www.w3.org/2000/svg}circle")) == 3


def test_thin_locking_negative_retained(tmp_path):
    report = write_report(run_convergence(StudySpec.from_dict(document("thin-q4"))), tmp_path / "thin")
    data = json.loads(report.read_bytes())
    failed = data["steps"][1]
    assert failed["mesh"] == {"nx": 8, "ny": 8}
    assert failed["solve_status"] == "solved"
    assert failed["verification_status"] == "verification_failed"
    assert failed["reference_check"]["error"] > 0.98
    assert any("locking" in item["message"] for item in failed["diagnostics"])
    assert load_result(report.parent / failed["artifact"]["path"]).arrays


@pytest.mark.parametrize(("policy", "statuses"), [
    ("stop", ["numerical_failure", "not_run", "not_run"]),
    ("record_and_continue", ["numerical_failure", "numerical_failure", "not_run"]),
])
def test_failure_and_budget_preserve_every_mesh(tmp_path, policy, statuses):
    model = beam()
    model["model"]["length"] = 1e-280
    study = run_convergence(model, [{"elements": n} for n in (1, 2, 4)], "tip_deflection",
                            max_solves=2, failure_policy=policy)
    report = write_report(study, tmp_path / policy)
    data = json.loads(report.read_bytes())
    assert data["status"] == "partial"
    assert [row["solve_status"] for row in data["steps"]] == statuses
    assert data["estimate"]["reason"] == "incomplete_sequence"
    assert data["steps"][-1]["artifact"] is None
    assert data["steps"][0]["reference_check"] is None
    assert data["steps"][0]["diagnostics"][0]["code"] == "numerical_failure"


@pytest.mark.parametrize(("values", "reason"), [
    ((1.0, 3.0, 2.0), "non_monotone_sequence"),
    ((1.0, 2.0, 4.0), "non_decreasing_change"),
    ((0.0, 0.0, 0.0), "near_zero_change"),
])
def test_estimate_guards(monkeypatch, values, reason):
    # Supply synthetic observations to isolate the estimate's decision rules.
    sequence = iter(values)
    monkeypatch.setattr(studies, "_deflection", lambda result: next(sequence))
    steps = run_convergence(beam(), [{"elements": n} for n in (1, 2, 4)], "tip_deflection")
    assert steps.estimate["reason"] == reason
    assert steps.estimate["observed_order"] is None


def test_order_for_known_mesh_sequence(monkeypatch):
    sequence = iter((1.0, 1.75, 1.9375))
    monkeypatch.setattr(studies, "_deflection", lambda result: next(sequence))
    steps = run_convergence(beam(), [{"elements": n} for n in (1, 2, 4)], "tip_deflection")
    assert steps.estimate["observed_order"] == pytest.approx(2.0)
    assert steps.estimate["finest_discretization_estimate_m"] == pytest.approx(0.0625)


def test_one_mesh_and_odd_rectangular_grid():
    doc = document()
    doc.update(meshes=[{"nx": 5, "ny": 3}], reference=None)
    result = run_convergence(StudySpec.from_dict(doc))
    step = result.steps[0]
    assert step.observed_m == pytest.approx(np.mean(step.result.arrays["deflection"][1:3, 2:4]))
    assert result.estimate["reason"] == "insufficient_meshes"


def test_near_zero_reference_and_inconsistent_refinement(monkeypatch):
    model = beam()
    model["loads"]["tip_force"] = 0
    values = iter((1.0, 1.75, 1.9375))
    monkeypatch.setattr(studies, "_deflection", lambda result: next(values))
    result = run_convergence(model, [{"elements": n} for n in (1, 2, 4)], "tip_deflection",
                             reference="cantilever.closed-form.v1")
    assert result.estimate["reason"] == "near_zero_reference"
    monkeypatch.setattr(studies, "_deflection", lambda result: 0.0)
    result = run_convergence(model, [{"elements": n} for n in (1, 2, 5)], "tip_deflection")
    assert result.estimate["reason"] == "inconsistent_refinement"


def test_bad_replay_hash_and_output_budget(tmp_path):
    doc = document()
    with pytest.raises(ModelValidationError, match="base_model_hash: hash_mismatch"):
        StudySpec.from_dict({**doc, "base_model_hash": "0" * 64})
    with pytest.raises(ModelValidationError, match="/study/meshes/0/nx: invalid_value"):
        StudySpec.from_dict({**doc, "meshes": [{"nx": 1, "ny": 2}]})
    spec = StudySpec.from_dict({**doc, "max_output_bytes": 10, "max_solves": 1})
    with pytest.raises(ArtifactError, match="byte budget"):
        write_report(run_convergence(spec), tmp_path / "over")
    assert not list(tmp_path.iterdir())
