from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path

import pytest

from anf import parameter_studies as studies
from anf.artifacts import ArtifactError
from anf.model import ModelValidationError, canonical_input_hash


P = "loads.tip_force"
E = "model.young_modulus"
INERTIA = "model.second_moment"
L = "model.length"
AXES = ((P, "N", (-1500, 1500)), (E, "Pa", (150e9, 240e9)),
        (INERTIA, "m^4", (8e-7, 2e-6)), (L, "m", (1, 2.5)))
BASE = {P: 1000, E: 210e9, INERTIA: 1e-6, L: 2}
STEPS = (0.008, 0.004, 0.002, 0.001)


def beam(parameters=BASE):
    return {
        "schema_version": "1.0", "problem_type": "cantilever_bending",
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "model": {"young_modulus": parameters[E], "second_moment": parameters[INERTIA], "length": parameters[L],
                  "elements": 8},
        "loads": {"tip_force": parameters[P]}, "analysis": {"type": "static"},
    }


def axis(path, unit, bounds, values):
    return {"path": path, "unit": unit, "bounds": bounds, "values": values}


def force_grid(values=(800, 1000, 1200), **options):
    return studies.ParameterGrid(beam(), [axis(P, "N", (-1500, 1500), values)], **options)


def independent(parameters):
    load, modulus, inertia, length = (Fraction(parameters[path]) for path, _, _ in AXES)
    return load * length**3 / (3 * modulus * inertia)


@pytest.mark.parametrize("parameters", [BASE, {P: -600, E: 180e9, INERTIA: 1.4e-6, L: 1.5}])
def test_ps1_four_derivatives_and_step_sequence(tmp_path, parameters):
    grid = studies.ParameterGrid(beam(parameters), [axis(p, u, b, [parameters[p]]) for p, u, b in AXES])
    report = studies.finite_difference_sensitivity(grid, tmp_path / "sensitivity")
    assert report["status"] == "complete"
    assert len(report["samples"]) == len(report["attempts"]) == 33
    assert report["step_ratios"] == list(STEPS)
    assert report["base_model_hash"] == canonical_input_hash(beam(parameters))
    exact = independent(parameters)
    expected = {P: exact / parameters[P], E: -exact / parameters[E],
                INERTIA: -exact / parameters[INERTIA], L: 3 * exact / parameters[L]}
    by_id = {sample["id"]: sample for sample in report["samples"]}
    for path, unit in zip(expected, ("m/N", "m/Pa", "m^-3", "m/m"), strict=True):
        result = report["sensitivities"][path]
        assert result["status"] == "estimated" and result["unit"] == unit
        assert abs(result["derivative"] / float(expected[path]) - 1) <= 5e-6
        for level, ratio in zip(result["levels"], STEPS, strict=True):
            below, above = (by_id[level[f"{side}_id"]] for side in ("minus", "plus"))
            assert level["ratio"] == ratio and level["reason"] is None
            for sample in (below, above):
                assert sample["input_hash"] == canonical_input_hash(beam(sample["parameters"]))
                assert abs(sample["observed_m"] / float(independent(sample["parameters"])) - 1) <= 2e-9
                assert sample["relative_residual"] <= 1e-8
            x0, x1 = (Fraction(row["parameters"][path]) for row in (below, above))
            exact_difference = (independent(above["parameters"]) - independent(below["parameters"])) / (x1 - x0)
            assert abs(level["derivative"] / float(exact_difference) - 1) <= 3e-9 / ratio


def test_ps3_grid_order_limits_diagnostics_and_report(tmp_path):
    grid = studies.ParameterGrid(beam(), [axis(P, "N", (-1200, 1200), [0, 1000, 2000]),
                                          axis(E, "Pa", (1e9, 1.7e308), [1.7e308, 210e9])], max_solves=3)
    # The declared E and I are finite, but their product overflows during assembly.
    doc = grid.to_dict()
    doc["model"]["model"]["second_moment"] = 2
    directory = tmp_path / "grid"
    report = studies.run_parameter_study(json.dumps(doc), directory)
    assert [sample["parameters"] for sample in report["samples"]] == [
        {P: p, E: e} for p in (0, 1000, 2000) for e in (1.7e308, 210e9)
    ]
    assert report["counts"] == {"solved": 1, "numerical_failure": 2, "out_of_domain": 2,
                                 "invalid_input": 0, "not_run": 1}
    assert report["status"] == "partial" and len(report["attempts"]) == 3
    assert report["samples"][3]["diagnostics"][0]["code"] == "solve_budget_exhausted"
    assert report["samples"][1]["observed_m"] == 0
    assert all(sample["input_hash"] for sample in report["samples"])
    assert json.loads((directory / "report.json").read_bytes()) == report
    plot = (directory / "study.svg").read_bytes()
    assert sha256(plot).hexdigest() == report["plot"]["sha256"]
    assert b"6 samples" in plot and b"[-1200, 1200] N" in plot and plot.count(b"<circle ") == 1
    assert studies.run_parameter_study(report["study"], directory) == report


def test_ps3_invalid_physical_model_is_retained(tmp_path):
    grid = studies.ParameterGrid(beam(), [axis(E, "Pa", (150e9, 240e9), [-1, 200e9])])
    results = studies.run_parameter_study(grid, tmp_path / "invalid")
    assert results["counts"]["out_of_domain"] == 1
    assert results["samples"][0]["input_hash"] is None
    assert results["samples"][0]["verification_status"] == "not_run"


@pytest.mark.parametrize(("modify", "code"), [
    (lambda doc: doc["axes"][0].update(unit="kN"), "unsupported_unit"),
    (lambda doc: doc["axes"][0].update(path="model.elements"), "unsupported_parameter"),
    (lambda doc: doc["axes"].extend(doc["axes"]), "duplicate_axis"),
    (lambda doc: doc["axes"][0].update(values=[0, -0.0]), "duplicate_value"),
    (lambda doc: doc["axes"][0].update(values=[True]), "invalid_type"),
    (lambda doc: doc["axes"][0].update(values=[float("inf")]), "invalid_value"),
    (lambda doc: doc["axes"].extend([axis(E, "Pa", (1, 100), list(range(1, 100)))]), "size_limit"),
    (lambda doc: doc["model"]["loads"].update(uniform_load=1), "unsupported_load"),
    (lambda doc: doc.update(max_solves=65), "invalid_value"),
])
def test_ps3_reject_invalid_request_before_creating_output(tmp_path, modify, code):
    doc = force_grid().to_dict()
    modify(doc)
    output = tmp_path / "invalid"
    with pytest.raises(ModelValidationError) as error:
        studies.run_parameter_study(doc, output)
    assert error.value.code == code
    assert not output.exists()


@pytest.mark.parametrize("force, bounds, budget, reason", [
    (0, (-1500, 1500), 64, "target_near_zero"),
    (1000, (-1500, 1000), 64, "out_of_domain"),
    (1000, (-1500, 1500), 2, "solve_budget_exhausted"),
])
def test_ps4_incomplete_stencil_returns_null(tmp_path, force, bounds, budget, reason):
    grid = studies.ParameterGrid(beam({**BASE, P: force}), [axis(P, "N", bounds, [force])], max_solves=budget)
    report = studies.finite_difference_sensitivity(grid, tmp_path / "sensitivity")
    value = report["sensitivities"][P]
    assert (value["status"], value["reason"], value["derivative"]) == ("unavailable", reason, None)
    assert len(report["samples"]) == 9
    if force == 0:
        assert report["samples"][0]["observed_m"] == pytest.approx(0, abs=2e-12)
        assert len(report["attempts"]) == 1 and report["counts"]["not_run"] == 8
    if reason == "out_of_domain":
        assert report["counts"]["out_of_domain"] == 4


def test_ps4_unstable_finite_sequence_is_unavailable(tmp_path, monkeypatch):
    solve = studies._solve

    def noisy(model, reference):
        outcome = solve(model, reference)
        if model.loads["tip_force"] == 1001:
            outcome["observed_m"] *= 1.01
        return outcome

    monkeypatch.setattr(studies, "_solve", noisy)
    report = studies.finite_difference_sensitivity(force_grid(), tmp_path / "noise")
    derivative = report["sensitivities"][P]
    assert report["counts"]["solved"] == 9 and report["status"] == "partial"
    assert (derivative["reason"], derivative["derivative"]) == ("unstable_sequence", None)
    assert all(level["derivative"] is not None for level in derivative["levels"])


@pytest.mark.parametrize("stage", ["before_reservation", "during_solve", "after_commit", "before_publish"])
def test_ps2_interruption_replays_unique_samples(tmp_path, monkeypatch, stage):
    grid = force_grid(max_solves=4)
    directory = tmp_path / "interrupt"
    original_write = studies._write_record
    original_solve = studies._solve
    calls = []

    def solve(model, reference):
        calls.append(model.loads["tip_force"])
        return original_solve(model, reference)

    def write(path, state):
        if path.name == "checkpoint.json" and state["attempts"]:
            if stage == "before_reservation":
                raise KeyboardInterrupt
            if stage == "after_commit" and state["rows"]:
                original_write(path, state)
                raise KeyboardInterrupt
        original_write(path, state)

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(studies, "_solve", solve)
    with monkeypatch.context() as patch:
        if stage == "during_solve":
            patch.setattr(studies, "_solve", interrupt)
        elif stage == "before_publish":
            patch.setattr(studies, "_publish", interrupt)
        else:
            patch.setattr(studies, "_write_record", write)
        with pytest.raises(KeyboardInterrupt):
            studies.run_parameter_study(grid, directory)
    assert not (directory / "report.json").exists()
    resumed = studies.run_parameter_study(grid, directory)
    fresh = studies.run_parameter_study(grid, tmp_path / "fresh")
    assert resumed["samples"] == fresh["samples"]
    assert calls == [800, 1000, 1200] * 2
    assert [item["status"] for item in resumed["attempts"]] == (
        ["interrupted", "finished", "finished", "finished"] if stage == "during_solve" else ["finished"] * 3
    )


def test_ps2_interrupted_attempt_consumes_final_reservation(tmp_path, monkeypatch):
    grid = force_grid(max_solves=1)
    directory = tmp_path / "resumable"

    def interrupt(*args):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(studies, "_solve", interrupt)
        with pytest.raises(KeyboardInterrupt):
            studies.run_parameter_study(grid, directory)
    report = studies.run_parameter_study(grid, directory)
    assert len(report["attempts"]) == 1 and report["attempts"][0]["status"] == "interrupted"
    assert report["counts"]["not_run"] == 3


def test_ps2_cache_identity_and_corruption(tmp_path, monkeypatch):
    original = studies._execution_identity()
    version = {**original, "shared_cache_available": True}
    monkeypatch.setattr(studies, "_execution_identity", lambda: version)
    cache = tmp_path / "cache"
    grid = force_grid()
    first = studies.run_parameter_study(grid, tmp_path / "first", cache_dir=cache)
    second = studies.run_parameter_study(grid, tmp_path / "second", cache_dir=cache)
    assert not second["attempts"]
    assert [item["origin"] for item in second["samples"]] == ["cache"] * 3
    assert [item["input_hash"] for item in first["samples"]] == [item["input_hash"] for item in second["samples"]]
    version = {**version, "anf_build": "different build at same version"}
    changed = studies.run_parameter_study(grid, tmp_path / "changed", cache_dir=cache)
    assert len(changed["attempts"]) == 3
    with pytest.raises(ArtifactError, match="identity differs"):
        studies.run_parameter_study(grid, tmp_path / "first", cache_dir=cache)
    for entry in cache.glob("*.json"):
        data = json.loads(entry.read_bytes())
        data["body"]["outcome"]["observed_m"] = 0
        entry.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ArtifactError, match="digest"):
        studies.run_parameter_study(grid, tmp_path / "corrupt", cache_dir=cache)


def test_ps2_exclusive_writer_and_corrupt_checkpoint(tmp_path):
    grid = force_grid()
    work = tmp_path / "lock"
    work.mkdir()
    with studies._exclusive(work), pytest.raises(ArtifactError, match="already in use"):
        studies.run_parameter_study(grid, work)
    assert not (work / "checkpoint.json").exists()
    studies.run_parameter_study(grid, work)
    with pytest.raises(ArtifactError, match="request, plan"):
        studies.run_parameter_study(force_grid(values=(1,)), work)
    path = work / "checkpoint.json"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ArtifactError, match="invalid study record"):
        studies.run_parameter_study(grid, work)


def test_ps3_output_budget_keeps_checkpoint_and_never_publishes_report(tmp_path):
    work = tmp_path / "small"
    with pytest.raises(ArtifactError, match="report exceeds its byte budget"):
        studies.run_parameter_study(force_grid(max_solves=1, max_output_bytes=100), work)
    assert (work / "checkpoint.json").is_file()
    assert not (work / "report.json").exists() and not (work / "study.svg").exists()


def test_study_rejects_non_cantilever(tmp_path):
    plate = Path(__file__).parent.parent / "examples/models/thick-q4.json"
    with pytest.raises(ModelValidationError) as error:
        studies.run_parameter_study({"model": json.loads(plate.read_bytes()), "axes": force_grid().to_dict()["axes"]},
                                    tmp_path / "plate")
    assert error.value.code == "unsupported_problem"
