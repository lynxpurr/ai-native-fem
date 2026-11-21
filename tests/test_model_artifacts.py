"""Declaration, single-solve dispatch and on-disk result contract tests."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import warnings
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import pytest

from anf import Beam, Plate, solve_cantilever, solve_plate
from anf import application
from anf.application import run_analysis
from anf.artifacts import ArtifactError, load_result, save_result
from anf.model import ModelValidationError, canonical_input_hash, validate_model
from anf.verification import verify_cantilever, verify_plate


def declaration(problem, model, loads):
    return {
        "schema_version": "1.0", "problem_type": problem,
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "model": asdict(model), "loads": loads, "analysis": {"type": "static"},
    }


def beam_doc():
    return declaration("cantilever_bending", Beam(2, 210e9, 1e-6), {"tip_force": 1000})


def plate_doc(**kwargs):
    plate = Plate(1, 1, 0.005, 70e9, 0.3, **kwargs)
    return declaration("rectangular_plate", plate, {"pressure": 1000})


def test_model_hash_defaults_and_physical_changes():
    doc = beam_doc()
    model = validate_model(doc)
    canonical = canonical_input_hash(model)
    assert canonical == verify_cantilever(model.model, **model.loads).input_hash

    del doc["model"]["elements"]
    doc["loads"]["uniform_load"] = 0
    assert canonical_input_hash(dict(reversed(list(doc.items())))) == canonical
    doc["model"]["elements"] = 8.0
    doc["model"]["length"] = 2
    assert canonical_input_hash(json.dumps(doc)) == canonical
    assert type(validate_model(doc).model.elements) is int
    doc["model"]["young_modulus"] *= 2
    assert canonical_input_hash(doc) != canonical
    doc = beam_doc()
    doc["loads"]["tip_force"] *= 2
    assert canonical_input_hash(doc) != canonical

    doc = plate_doc(support="hard_simply_supported")
    q4 = canonical_input_hash(doc)
    del doc["model"]["formulation"], doc["model"]["shear_factor"]
    assert canonical_input_hash(doc) == q4
    doc["model"]["support"] = "clamped"
    assert canonical_input_hash(doc) != q4
    doc["model"].update(support="hard_simply_supported", formulation="mitc4_rectangle")
    assert canonical_input_hash(doc) != q4


@pytest.mark.parametrize(("section", "key", "value", "code"), [
    ("", "schema_version", "0.9", "unsupported_schema"),
    ("", "problem_type", "arbitrary_mesh", "unsupported_problem"),
    ("", "include", "model.json", "unknown_field"),
    ("", "ModelBundle", {}, "unknown_field"),
    ("units", "length", "mm", "unsupported_unit"),
    ("analysis", "type", "modal", "unsupported_analysis"),
    ("model", "eval", "solve()", "unknown_field"),
    ("model", "foo/bar~baz", 1, "unknown_field"),
    ("model", "elements", True, "invalid_type"),
    ("model", "elements", 8.25, "invalid_value"),
    ("model", "elements", 201, "invalid_value"),
    ("model", "elements", 0, "invalid_value"),
    ("loads", "import", "some.module", "unknown_field"),
    ("loads", "tip_force", float("nan"), "invalid_value"),
    ("loads", "tip_force", 10**400, "invalid_value"),
])
def test_invalid_declarations_fail_before_fingerprinting_or_solving(monkeypatch, section, key, value, code):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid input reached the physical projection or solver")

    monkeypatch.setattr("anf.model.input_parameters", unexpected)
    monkeypatch.setattr(application, "input_parameters", unexpected)
    monkeypatch.setattr(application, "solve_cantilever", unexpected)
    doc = beam_doc()
    (doc[section] if section else doc)[key] = value
    pointer = f"/{section}" if section else ""
    pointer += "/" + key.replace("~", "~0").replace("/", "~1")
    for entry in (canonical_input_hash, run_analysis):
        with pytest.raises(ModelValidationError) as failure:
            entry(doc)
        assert (failure.value.path, failure.value.code) == (pointer, code)


@pytest.mark.parametrize(("encoded", "path", "code"), [
    (b'"length": 2.0, "length": 3', "/model/length", "duplicate_field"),
    (b'"length": 1e999', "/model/length", "invalid_value"),
    (b'"length": NaN', "/model/length", "invalid_value"),
    (b'"length": 1e-999', "/model/length", "invalid_value"),
    (b'"length": 2.0, "elements": 8.000000000000000001', "/model/elements", "duplicate_field"),
])
def test_json_duplicate_and_nonfinite_values(encoded, path, code):
    payload = json.dumps(beam_doc()).encode().replace(b'"length": 2.0', encoded)
    with pytest.raises(ModelValidationError) as failure:
        validate_model(payload)
    assert (failure.value.path, failure.value.code) == (path, code)


def test_json_count_precision_required_units_and_size_budget():
    payload = json.dumps(beam_doc()).replace('"elements": 8', '"elements": 8.000000000000000001')
    with pytest.raises(ModelValidationError, match="/model/elements: invalid_value"):
        validate_model(payload)
    doc = beam_doc()
    del doc["units"]["force"]
    with pytest.raises(ModelValidationError, match="/units/force: missing_field"):
        validate_model(doc)
    with pytest.raises(ModelValidationError, match="size_limit"):
        validate_model(b" " * (2**20 + 1))


def test_plate_domain():
    doc = plate_doc()
    doc["model"]["formulation"] = "mitc4_rectangle"
    with pytest.raises(ModelValidationError, match="/model/support: unsupported_support"):
        validate_model(doc)
    doc["model"].update(formulation="q4_full", nx=199, ny=3)
    with pytest.raises(ModelValidationError, match="/model: size_limit"):
        validate_model(doc)
    doc["model"]["ny"] = 2.0
    assert validate_model(doc).model.nx == 199


_BEAM = Beam(2, 210e9, 1e-6)
_THICK = Plate(1, 1, 0.2, 70e9, 0.3, nx=16, ny=16, support="hard_simply_supported")
_THIN = Plate(1, 1, 0.005, 70e9, 0.3, support="hard_simply_supported")
_MITC = Plate(1, 1, 0.005, 70e9, 0.3, support="hard_simply_supported", formulation="mitc4_rectangle")
_MITC_RECTANGLE = Plate(2, 1, 0.005, 70e9, 0.3, nx=8, ny=8,
                        support="hard_simply_supported", formulation="mitc4_rectangle")


def physical_arrays(model, result):
    if isinstance(model, Beam):
        return {"x": (result.x, "node", "m"), "deflection": (result.deflection, "node", "m"),
                "slope": (result.slope, "node", "rad"), "root_force": (result.root_reaction[:1], "root_node", "N"),
                "root_moment": (result.root_reaction[1:], "root_node", "N·m")}
    return {"x": (result.x, "x_axis", "m"), "y": (result.y, "y_axis", "m"),
            "deflection": (result.deflection, "node", "m"),
            "beta_x": (result.rotations[:, :, 0], "node", "rad"),
            "beta_y": (result.rotations[:, :, 1], "node", "rad"),
            "reaction_force": (result.reactions[:, :, 0], "node", "N"),
            "reaction_moment_x_dof": (result.reactions[:, :, 1], "node", "N·m"),
            "reaction_moment_y_dof": (result.reactions[:, :, 2], "node", "N·m")}


@pytest.mark.parametrize(("model", "reference", "status"), [
    (_BEAM, "cantilever.closed-form.v1", "verification_passed"),
    (_THICK, "plate.navier-31.v1", "verification_passed"),
    (_THIN, "plate.thin-q4-navier-31.v1", "verification_failed"),
    (_MITC, None, "solved_unverified"),
    (_MITC_RECTANGLE, None, "solved_unverified"),
], ids=("beam", "thick-q4", "thin-q4", "mitc4", "mitc4-rectangle"))
def test_five_solver_cases_and_independent_disk_reader(tmp_path, model, reference, status):
    beam = isinstance(model, Beam)
    loads = {"tip_force": 1000} if beam else {"pressure": 1000}
    problem = "cantilever_bending" if beam else "rectangular_plate"
    original = (solve_cantilever if beam else solve_plate)(model, **loads)
    report = (verify_cantilever if beam else verify_plate)(model, reference=reference, **loads)
    result = run_analysis(declaration(problem, model, loads), {"reference": reference})
    assert result.report == report
    assert result.solve_status == "solved"
    assert result.verification_status == status
    assert result.relative_residual == original.relative_residual <= 1e-8
    assert result.strain_energy == original.strain_energy
    for name, (values, position, unit) in physical_arrays(model, original).items():
        np.testing.assert_array_equal(result.arrays[name], values)
        assert result.array_metadata[name] == {
            "dtype": "<f8", "shape": list(values.shape), "position": position, "unit": unit,
        }
    directory = save_result(result, tmp_path, "case")
    script = """
import json
import sys
from anf.artifacts import load_result
result = load_result(sys.argv[1])
print(json.dumps({
    'report': json.loads(result.report.to_json()), 'model': result.model.to_dict(),
    'versions': dict(result.versions), 'diagnostics': result.diagnostics,
    'energy': result.strain_energy, 'residual': result.relative_residual,
    'metadata': result.array_metadata,
    'arrays': {key: value.tobytes().hex() for key, value in result.arrays.items()},
}))
"""
    # Also support running source tests directly from a clean, uninstalled checkout.
    env = {**os.environ, "PYTHONPATH": str(Path(application.__file__).resolve().parents[1])}
    payload = json.loads(subprocess.check_output([sys.executable, "-c", script, str(directory)],
                                                 cwd=tmp_path, env=env, text=True))
    assert payload == {
        "report": json.loads(result.report.to_json()), "model": result.model.to_dict(),
        "versions": dict(result.versions), "diagnostics": list(result.diagnostics),
        "energy": result.strain_energy, "residual": result.relative_residual,
        "metadata": result.array_metadata,
        "arrays": {key: value.tobytes().hex() for key, value in result.arrays.items()},
    }


def test_reference_options_and_single_solve(monkeypatch):
    calls = []

    def counted(model, **loads):
        calls.append((model, loads))
        return solve_cantilever(model, **loads)

    monkeypatch.setattr(application, "solve_cantilever", counted)
    doc = beam_doc()
    with pytest.raises(ModelValidationError, match="/options/reference: unsupported_reference"):
        run_analysis(doc, {"reference": "auto"})
    with pytest.raises(ModelValidationError, match="/options/output: unknown_field"):
        run_analysis(doc, {"output": "outside"})
    assert calls == []
    default = run_analysis(doc)
    assert len(calls) == 1
    assert default.verification_status == "solved_unverified"
    assert default.report == run_analysis(doc, {"reference": None}).report
    explicit = run_analysis(doc, {"reference": "cantilever.closed-form.v1"})
    assert len(calls) == 3
    assert explicit.verification_status == "verification_passed"
    assert default.input_hash == explicit.input_hash

    mitc = plate_doc(formulation="mitc4_rectangle", support="hard_simply_supported")
    result = run_analysis(mitc, {"reference": "plate.thin-q4-navier-31.v1"})
    assert result.verification_status == "solved_unverified"
    assert result.report.checks[-1].outcome == "not_applicable"


def test_plate_grid_and_standard_stored_npz(tmp_path):
    result = run_analysis(plate_doc(nx=3, ny=5))
    directory = save_result(result, tmp_path, "rectangular-grid")
    arrays = directory / "arrays.npz"
    np.savez(arrays, **result.arrays)
    change_manifest(directory, lambda doc: doc.update(arrays_sha256=sha256(arrays.read_bytes()).hexdigest()))
    loaded = load_result(directory)
    assert loaded.arrays["deflection"].shape == (6, 4)
    assert loaded.arrays["x"].shape == (4,)
    assert loaded.arrays["y"].shape == (6,)
    for name, array in loaded.arrays.items():
        np.testing.assert_array_equal(array, result.arrays[name])


def test_numerical_failure_is_manifest_only(tmp_path):
    doc = beam_doc()
    doc["model"]["length"] = 1e-280
    result = run_analysis(doc, {"reference": "cantilever.closed-form.v1"})
    assert (result.solve_status, result.verification_status) == ("numerical_failure", "not_run")
    assert result.input_hash == canonical_input_hash(doc)
    assert result.diagnostics[0]["code"] == "numerical_failure"
    directory = save_result(result, tmp_path, "failed")
    assert set(path.name for path in directory.iterdir()) == {"manifest.json"}
    loaded = load_result(directory)
    assert not loaded.arrays and loaded.strain_energy is None
    assert loaded.report == result.report


@pytest.fixture
def artifact(tmp_path):
    return save_result(run_analysis(beam_doc()), tmp_path, "case")


def change_manifest(directory, update):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    update(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("change", [
    lambda doc: doc["arrays"]["slope"].update(unit="m"),
    lambda doc: doc["arrays"]["x"].update(shape=[10]),
    lambda doc: doc["arrays"]["x"].update(shape=[True]),
    lambda doc: doc.update(input_hash="0"*64),
    lambda doc: doc.update(verification_status="verification_passed"),
    lambda doc: doc["model"]["model"].update(length=3),
    lambda doc: doc.update(relative_residual=0),
    lambda doc: doc.update(schema_version="1.1"),
    lambda doc: doc.update(diagnostics=[]),
    lambda doc: doc.update(versions={}),
])
def test_contradictory_manifest(artifact, change):
    change_manifest(artifact, change)
    with pytest.raises(ArtifactError):
        load_result(artifact)


def npy(array, version=(1, 0)):
    stream = BytesIO()
    np.lib.format.write_array(stream, array, version=version)
    return stream.getvalue()


def replace_archive(directory, name, content):
    path = directory / "arrays.npz"
    with ZipFile(path) as original:
        members = {key: original.read(key) for key in original.namelist()}
    members.pop("x.npy")
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for key, value in members.items():
            archive.writestr(key, value)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Duplicate name", category=UserWarning)
            archive.writestr(name, content)
    change_manifest(directory, lambda doc: doc.update(arrays_sha256=sha256(path.read_bytes()).hexdigest()))


def unix_link():
    member = ZipInfo("x.npy")
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    return member


@pytest.mark.parametrize(("name", "payload"), [
    ("x.npy", lambda: npy(np.full(9, None, dtype=object))),
    ("x.npy", lambda: npy(np.zeros(9, dtype=">f8"))),
    ("x.npy", lambda: npy(np.zeros(10, dtype="<f8"))),
    ("x.npy", lambda: npy(np.zeros(9, dtype="<f8"), version=(2, 0))),
    ("x.npy", lambda: npy(np.full(9, np.inf, dtype="<f8"))),
    ("x.npy", lambda: npy(np.zeros(9, dtype="<f8")) + b"extra"),
    ("x.npy", lambda: b"\x93NUMPY\x01\x00" + (10241).to_bytes(2, "little")),
    ("deflection.npy", lambda: b"duplicate"),
    ("../x.npy", lambda: b"traversal"),
    ("/x.npy", lambda: b"absolute"),
    ("x.npy/extra", lambda: b"unexpected"),
    (unix_link(), lambda: b"../outside"),
], ids=("object", "endian", "shape", "version", "finite", "trailing", "header-limit", "duplicate",
        "traversal", "absolute", "unexpected", "link"))
def test_unsafe_npz_members(artifact, name, payload):
    replace_archive(artifact, name, payload())
    with pytest.raises(ArtifactError):
        load_result(artifact)


def test_compressed_member_budget(artifact):
    path = artifact / "arrays.npz"
    with ZipFile(path) as archive:
        members = {key: archive.read(key) for key in archive.namelist() if key != "x.npy"}
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for key, value in members.items():
            archive.writestr(key, value)
        with archive.open("x.npy", "w") as member:
            chunk = bytes(1024 * 1024)
            for _ in range(65):
                member.write(chunk)
    change_manifest(artifact, lambda doc: doc.update(arrays_sha256=sha256(path.read_bytes()).hexdigest()))
    with pytest.raises(ArtifactError, match="expanded byte budget"):
        load_result(artifact)


def test_archive_size_member_count_and_corruption(artifact):
    arrays = artifact / "arrays.npz"
    original = arrays.read_bytes()
    with arrays.open("r+b") as stream:
        stream.truncate(64 * 1024 * 1024 + 1)
    with pytest.raises(ArtifactError, match="size limit"):
        load_result(artifact)
    arrays.write_bytes(original[:-1])
    with pytest.raises(ArtifactError, match="digest"):
        load_result(artifact)
    with ZipFile(arrays, "w") as archive:
        for index in range(33):
            archive.writestr(f"{index}.npy", b"")
    change_manifest(artifact, lambda doc: doc.update(arrays_sha256=sha256(arrays.read_bytes()).hexdigest()))
    with pytest.raises(ArtifactError, match="member budget"):
        load_result(artifact)


def test_output_boundaries_and_interruption(tmp_path, artifact, monkeypatch):
    result = run_analysis(beam_doc())
    original = (artifact / "manifest.json").read_bytes()
    for name in ("../outside", r"..\outside", "/outside", "a:b", ".", "..", "NUL.txt",
                 "COM¹.txt", "CONOUT$", "outside.", "outside "):
        with pytest.raises(ArtifactError, match="portable artifact name"):
            save_result(result, artifact, name)
    with pytest.raises(ArtifactError):
        save_result(result, tmp_path, "case")
    assert not (tmp_path / "outside").exists()
    assert (artifact / "manifest.json").read_bytes() == original
    named = save_result(result, tmp_path, "梁 1")
    assert load_result(named).report == result.report

    def interrupt(*args, **kwargs):
        raise OSError("simulated manifest publish interruption")

    monkeypatch.setattr(Path, "replace", interrupt)
    with pytest.raises(ArtifactError, match="interruption"):
        save_result(result, tmp_path, "partial")
    partial = tmp_path / "partial"
    assert (partial / "arrays.npz").is_file()
    assert not (partial / "manifest.json").exists()
    with pytest.raises(ArtifactError):
        load_result(partial)


def test_filesystem_link_rejected(tmp_path, artifact):
    manifest = artifact / "manifest.json"
    elsewhere = tmp_path / "outside.json"
    manifest.replace(elsewhere)
    try:
        os.link(elsewhere, manifest)
    except OSError as error:
        pytest.skip(f"temporary filesystem does not provide hard links: {error}")
    with pytest.raises(ArtifactError):
        load_result(artifact)


def test_concurrent_saves_claim_the_name_once(tmp_path):
    result = run_analysis(beam_doc())
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(save_result, result, tmp_path, "shared")
        second = executor.submit(save_result, result, tmp_path, "shared")
    assert sum(future.exception() is None for future in (first, second)) == 1
    assert all(future.exception() is None or isinstance(future.exception(), ArtifactError)
               for future in (first, second))
    assert load_result(tmp_path / "shared").report == result.report
