"""Product-level load case checks using equilibrium and closed-form anchors."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from anf import artifacts
from anf.application import run_analysis, run_load_cases
from anf.artifacts import ArtifactError, load_load_case_results, save_load_case_results
from anf.model import ModelValidationError, validate_load_case_model
from anf.recovery import member_forces


EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "structures"
COMPONENTS = ("axial", "shear", "moment")


def example(name):
    return json.loads((EXAMPLES / f"{name}.json").read_bytes())


def line(i, j, basis="local"):
    return {"member_line": {"beam": {"basis": basis, "i": i, "j": j}}}


def values(actual, reference, zero_atol):
    actual = np.asarray(actual)
    reference = np.broadcast_to(reference, actual.shape)
    zero = reference == 0
    np.testing.assert_allclose(actual[zero], reference[zero], rtol=0, atol=zero_atol)
    np.testing.assert_allclose(actual[~zero], reference[~zero], rtol=1e-8, atol=0)


@pytest.mark.parametrize("elements", [1, 2, 4, 8])
def test_cantilever_self_weight_and_effective_identity(elements):
    doc = example("load-cases")
    doc["model"]["members"]["beam"]["elements"] = elements
    doc["loads"]["cases"]["explicit"] = line([0, -250], [0, -250])
    batch = run_load_cases(doc, ids=["gravity", "explicit"])
    gravity, explicit = batch.results["gravity"], batch.results["explicit"]
    assert gravity.input_hash == explicit.input_hash
    assert batch.selected_ids == ("explicit", "gravity")
    assert batch.model.to_dict() == validate_load_case_model(json.dumps(batch.model.to_dict())).to_dict()
    assert len(gravity.arrays["node_uy"]) == elements + 1
    assert batch.envelope.node_order == batch.envelope.frame_node_order == ("base", "tip")
    assert batch.envelope.displacement["uy"].minimum_id.tolist() == ["explicit", "explicit"]
    values(gravity.arrays["node_uy"][:2], [0, -0.0025], 1e-10)
    values(gravity.arrays["node_rz"][:2], [0, -1 / 600], 1e-10)
    values(gravity.arrays["reaction_fy"][:2], [500, 0], 1e-6)
    values(gravity.arrays["reaction_mz"][:2], [500, 0], 1e-6)
    forces = member_forces(gravity)
    distance_to_tip = 2 * (1 - np.asarray(forces.stations))
    values(forces.axial, 0, 1e-6)
    values(forces.shear, -250 * distance_to_tip, 1e-6)
    values(forces.moment, -125 * distance_to_tip**2, 1e-6)
    assert not forces.moment.flags.writeable


@pytest.mark.parametrize("elements", [2, 4, 8])
def test_simple_support_uniform_load(elements):
    doc = example("load-cases")
    doc["model"]["nodes"]["tip"] = [4, 0]
    doc["model"]["supports"] = {"base": {"ux": 0, "uy": 0}, "tip": {"uy": 0}}
    doc["model"]["members"]["beam"]["elements"] = elements
    doc["loads"] = {"cases": {"uniform": line([0, -500], [0, -500], "global")}}
    batch = run_load_cases(doc)
    result = batch.results["uniform"]
    middle = batch.ordering["node_order"].index(["generated", "beam", elements // 2])
    values(result.arrays["node_uy"][middle], -1 / 120, 1e-10)
    values(result.arrays["node_rz"][:2], [-1 / 150, 1 / 150], 1e-10)
    values(result.arrays["reaction_fy"][:2], [1000, 1000], 1e-6)
    forces = member_forces(result)
    x = 4 * np.asarray(forces.stations)
    values(forces.shear, -1000 + 500 * x, 1e-6)
    values(forces.moment, 1000 * x - 250 * x**2, 1e-6)


def settlement():
    doc = example("settlement")
    doc["schema_version"] = "1.2"
    doc["loads"] = {
        "cases": {"baseline": {}, "push": {"nodal": {"middle": {"fx": 400}}},
                  "pull": {"nodal": {"middle": {"fx": -200}}}},
        "combinations": {"combo": {"push": 2, "pull": -0.5}, "zero": {"pull": 0}},
    }
    return doc


def test_settlement_combination_and_envelopes():
    model = settlement()
    batch = run_load_cases(model, stations=[1, 0, 0.3])
    combo = batch.results["combo"]
    assert combo.kind == "combination"
    values(combo.arrays["node_ux"], [0, 0.00145, 0.002], 1e-10)
    values(combo.arrays["reaction_fx"], [-1450, 0, 550], 1e-6)
    forces = member_forces(combo)
    values(forces.axial, [[1450] * 3, [550] * 3], 1e-6)
    values(combo.strain_energy, (1450**2 + 550**2) / 2e6, 1e-12)
    baseline, push, pull = (batch.results[name] for name in ("baseline", "push", "pull"))
    for key, actual in combo.arrays.items():
        zero = baseline.arrays[key]
        np.testing.assert_allclose(actual, zero + 2 * (push.arrays[key] - zero)
                                   - 0.5 * (pull.arrays[key] - zero), rtol=0, atol=1e-10)
    assert batch.results["zero"].input_hash == baseline.input_hash
    displacement = batch.envelope.displacement["ux"]
    assert displacement.minimum_id.tolist() == ["baseline", "pull", "baseline"]
    assert displacement.maximum_id.tolist() == ["baseline", "combo", "baseline"]
    assert batch.envelope.sections["axial"].maximum_id.tolist() == [["combo"] * 3, ["pull"] * 3]
    same = run_load_cases(model, ids=list(reversed(batch.selected_ids)), stations=[1, -0.0, 0.3])
    subset = run_load_cases(model, ids=["combo"], stations=[0.3])
    assert same.request_hash == batch.request_hash != subset.request_hash
    assert combo.input_hash == subset.results["combo"].input_hash


@pytest.mark.parametrize("elements", [1, 2, 4])
def test_oblique_trapezoid_reversal_with_prescribed_motion(elements):
    doc = example("load-cases")
    doc["model"]["nodes"]["tip"] = [3, 4]
    doc["model"]["members"]["beam"]["elements"] = elements
    doc["model"]["supports"] = {"base": {"ux": 0.0001, "uy": -0.0005, "rz": 0.0001},
                                "tip": {"ux": 0.0015, "uy": 0.001, "rz": -0.0005}}
    loads = ([0, -500], [250, 0])
    doc["loads"] = {"cases": {"global": line(*loads, "global"),
                             "local": line([-400, -300], [150, -200])}}
    original = run_load_cases(doc)
    a = original.results["global"]
    assert a.input_hash == original.results["local"].input_hash
    doc["loads"] = {"cases": {"global": line(*reversed(loads), "global")}}
    doc["model"]["members"]["beam"].update(i="tip", j="base")
    b = run_load_cases(doc).results["global"]
    assert b.input_hash != a.input_hash
    for name in ("node_ux", "node_uy", "node_rz", "reaction_fx", "reaction_fy", "reaction_mz"):
        values(b.arrays[name][:2], a.arrays[name][:2], 1e-6)
    for component, sign in zip(COMPONENTS, (1, 1, -1), strict=True):
        sections = a.arrays[f"member_section_{component}"]
        zeros = np.abs(sections) < 1e-6
        values(b.arrays[f"member_section_{component}"][:, ::-1],
               np.where(zeros, 0, sign * sections), 1e-6)
    left_fx, right_fx = a.arrays["reaction_fx"][:2]
    left_fy, right_fy = a.arrays["reaction_fy"][:2]
    left_mz, right_mz = a.arrays["reaction_mz"][:2]
    values(left_fx + right_fx, -625, 1e-6)
    values(left_fy + right_fy, 1250, 1e-6)
    # Independent line-load moment about the base from resultant locations L/3 and 2L/3.
    applied_moment = -(1250 * 3 / 3 + 625 * 4 * 2 / 3)
    values(left_mz + right_mz + 3 * right_fy - 4 * right_fx + applied_moment, 0, 1e-6)


def test_truss_accepts_axial_projection_and_combines_weight():
    doc = settlement()
    doc["model"]["nodes"] = {"left": [0, 0], "right": [3, 4]}
    doc["model"]["supports"] = {key: {"ux": 0, "uy": 0} for key in doc["model"]["nodes"]}
    member = doc["model"]["members"].pop("first")
    member["j"] = "right"
    doc["model"]["members"] = {"beam": member}
    doc["loads"] = {"cases": {"base": line([9, 12], [12, 16], "global"),
                             "gravity": {"self_weight": {"acceleration": [3, 4], "density": {"beam": 1e5}}}},
                    "combinations": {"mix": {"base": 2, "gravity": -1}}}
    outcome = run_load_cases(doc).results["mix"]
    end_loads = (25, 35)
    left = -5 * (2 * end_loads[0] + end_loads[1]) / 6
    right = -5 * (end_loads[0] + 2 * end_loads[1]) / 6
    values(outcome.arrays["member_end_axial"], [[left, right]], 1e-6)
    values(outcome.arrays["reaction_fx"], np.array([left, right]) * 0.6, 1e-6)
    values(outcome.arrays["reaction_fy"], np.array([left, right]) * 0.8, 1e-6)
    values(member_forces(outcome).shear, 0, 1e-6)
    values(member_forces(outcome).moment, 0, 1e-6)


def test_generated_segments_follow_original_axis_at_large_coordinate_offset():
    doc = example("load-cases")
    doc["model"]["members"]["beam"]["elements"] = 2
    doc["model"]["nodes"]["tip"] = [6, 8]
    doc["loads"] = {"cases": {"ramp": line([0, -500], [250, 0], "global")}}
    origin = run_load_cases(doc).results["ramp"]
    offset = 10**16
    doc["model"]["nodes"] = {"base": [offset, offset], "tip": [offset + 6, offset + 8]}
    translated = run_load_cases(doc).results["ramp"]
    for name, expected in origin.arrays.items():
        np.testing.assert_array_equal(translated.arrays[name], expected)


@pytest.mark.parametrize(("change", "code"), [
    (lambda d: d["loads"]["combinations"].update(push={"pull": 1}), "duplicate_id"),
    (lambda d: d["loads"]["combinations"].update(outer={"combo": 0}), "unknown_case"),
    (lambda d: d["loads"]["combinations"].update(outer={}), "invalid_value"),
    (lambda d: d["loads"]["combinations"].update(outer={"pull": True}), "invalid_type"),
    (lambda d: d["loads"]["cases"].update(other={"nodal": {"missing": {"fx": 0}}}), "unknown_node"),
    (lambda d: d["loads"]["cases"].update(other={"nodal": {"middle": {"mz": 0}}}), "inactive_dof"),
    (lambda d: d["loads"]["cases"].update(other={"member_line": {
        "first": {"basis": "local", "i": [0, 1], "j": [0, -1]}}}), "unsupported_load"),
    (lambda d: d["loads"]["cases"].update(other={"self_weight": {
        "acceleration": [0, -9.81], "density": {"first": 2500}}}), "unsupported_load"),
    (lambda d: d["loads"]["cases"].update(other={"self_weight": {
        "acceleration": [1, 0], "density": {"first": 0}}}), "invalid_value"),
    (lambda d: d["loads"]["cases"].update(other={"nodal": {"middle": {"fx": float("inf")}}}), "invalid_value"),
    (lambda d: d["model"]["members"]["first"].update(release="rz"), "unknown_field"),
    (lambda d: d["loads"]["cases"].update({f"case{i}": {} for i in range(28)}), "size_limit"),
])
def test_invalid_declarations_are_rejected_before_selection(change, code):
    doc = settlement()
    change(doc)
    with pytest.raises(ModelValidationError) as error:
        run_load_cases(doc, ids=["baseline"])
    assert error.value.code == code


@pytest.mark.parametrize("options", [
    {"ids": []}, {"ids": ["missing"]}, {"ids": ["push", "push"]},
    {"stations": []}, {"stations": [-0.1]}, {"stations": [1.1]},
    {"stations": [True]}, {"stations": [0.0, -0.0]},
])
def test_invalid_selection(options):
    with pytest.raises(ModelValidationError):
        run_load_cases(settlement(), **options)


def partial_batch():
    doc = example("load-cases")
    doc["loads"] = {"cases": {"Bad": {"nodal": {"tip": {"fy": 1e308}}}, "CON": {}, "con": {}}}
    return run_load_cases(doc)


def test_partial_and_failed_artifacts_preserve_diagnostics(tmp_path):
    batch = partial_batch()
    assert batch.envelope_status == "partial"
    failed = batch.results["Bad"]
    assert failed.solve_status == "numerical_failure" and failed.verification_status == "not_run"
    assert failed.diagnostics[0]["code"] == "numerical_range"
    assert batch.envelope.displacement["uy"].minimum_id.tolist() == ["CON", "CON"]
    with pytest.raises(ValueError, match="unavailable"):
        member_forces(failed)
    directory = save_load_case_results(batch, tmp_path, "partial")
    assert sorted(child.name for child in (directory / "cases").iterdir()) == ["0001", "0002"]
    restored = load_load_case_results(directory)
    assert restored.envelope_status == "partial"
    assert restored.request_hash == batch.request_hash
    assert restored.results["Bad"].diagnostics == failed.diagnostics
    assert restored.results["Bad"].input_hash == failed.input_hash
    for case in ("CON", "con"):
        for name, array in batch.results[case].arrays.items():
            assert restored.results[case].arrays[name].tobytes() == array.tobytes()
    with pytest.raises(ArtifactError):
        save_load_case_results(batch, tmp_path, "partial")

    all_fail = run_load_cases(batch.model, ids=["Bad"])
    path = save_load_case_results(all_fail, tmp_path, "unavailable")
    assert not (path / "cases").exists()
    envelope = load_load_case_results(path).envelope
    assert envelope.status == "not_available"
    assert envelope.displacement is None and envelope.sections is None


@pytest.mark.parametrize("damage", [
    "identity", "ordinal", "shape", "rehashed", "ordering", "envelope", "extra", "incomplete",
])
def test_bad_artifact_is_rejected(tmp_path, damage):
    directory = save_load_case_results(partial_batch(), tmp_path, "damaged")
    index = directory / "manifest.json"
    manifest = json.loads(index.read_bytes())
    item = manifest["results"][1]
    path = directory / item["arrays_path"]
    if damage == "identity":
        item["input_hash"] = "0" * 64
    elif damage == "ordinal":
        item["arrays_path"] = manifest["results"][2]["arrays_path"]
    elif damage == "shape":
        item["arrays"]["member_section_moment"]["shape"] = [1, 4]
    elif damage == "rehashed":
        with np.load(path, allow_pickle=False) as archive:
            arrays = dict(archive)
        arrays["member_section_moment"] = arrays["member_section_moment"][:, :-1]
        np.savez_compressed(path, **arrays)
        item["arrays_sha256"] = sha256(path.read_bytes()).hexdigest()
    elif damage == "ordering":
        manifest["ordering"]["node_order"][2][2] = True
    elif damage == "envelope":
        manifest["envelope_status"] = "complete"
    elif damage == "extra":
        (path.parent / "extra").touch()
    index.write_text(json.dumps(manifest), encoding="utf-8")
    if damage == "incomplete":
        index.unlink()
    with pytest.raises(ArtifactError):
        load_load_case_results(directory)


def test_array_expansion_budget_is_shared_across_cases(tmp_path, monkeypatch):
    from zipfile import ZipFile

    directory = save_load_case_results(partial_batch(), tmp_path, "budget")
    with ZipFile(directory / "cases/0001/arrays.npz") as archive:
        size = sum(item.file_size for item in archive.infolist())
    monkeypatch.setattr(artifacts, "_EXPANDED_LIMIT", 2 * size - 1)
    with pytest.raises(ArtifactError, match="expanded byte budget"):
        load_load_case_results(directory)


def test_original_python_and_cli_remain_single_case(tmp_path):
    doc = example("load-cases")
    with pytest.raises(ModelValidationError, match="unsupported_schema"):
        run_analysis(doc)
    with pytest.raises(ModelValidationError, match="unsupported_schema"):
        run_load_cases(example("settlement"))
    model = tmp_path / "model.json"
    model.write_text(json.dumps(doc), encoding="utf-8")
    cli = subprocess.run([sys.executable, "-I", "-m", "anf.cli", "validate", str(model), "--json"],
                         cwd=tmp_path, capture_output=True, check=False)
    assert cli.returncode == 2
    assert json.loads(cli.stdout)["error"]["code"] == "unsupported_schema"
