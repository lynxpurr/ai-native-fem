"""Round-trip and rejection tests for schema 1.4 modal artifacts."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy.linalg")

from anf.dynamics import (ModalArtifactError, load_modal_result, save_modal_result,
                          solve_modes)


def frame(n=8):
    return {
        "schema_version": "1.4", "problem_type": "planar_structure",
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "model": {
            "nodes": {"a": [0, 0], "b": [2, 0]},
            "members": {"beam": {"kind": "frame2d", "i": "a", "j": "b",
                                 "young_modulus": 210e9, "area": 0.01,
                                 "second_moment": 1e-6, "density": 7850,
                                 "elements": n}},
            "supports": {"a": {"ux": 0, "uy": 0, "rz": 0}},
        },
        "loads": {},
        "analysis": {"type": "modal", "modes": 6},
    }


def saved(tmp_path):
    result = solve_modes(frame(8), count=3)
    directory = save_modal_result(result, tmp_path, "modal-result")
    return result, directory


def test_round_trip_preserves_modal_result(tmp_path):
    result, directory = saved(tmp_path)
    assert {path.name for path in directory.iterdir()} == {"manifest.json", "arrays.npz"}
    loaded = load_modal_result(directory)
    assert loaded.model.to_dict() == result.model.to_dict()
    assert loaded.input_hash == result.input_hash
    assert loaded.dof_labels == result.dof_labels
    assert loaded.rigid_mode_count == result.rigid_mode_count
    assert loaded.truncated == result.truncated
    assert loaded.diagnostics == result.diagnostics
    np.testing.assert_array_equal(loaded.eigenvalues, result.eigenvalues)
    np.testing.assert_array_equal(loaded.omega_rad_s, result.omega_rad_s)
    np.testing.assert_array_equal(loaded.frequencies_hz, result.frequencies_hz)
    np.testing.assert_array_equal(loaded.modes, result.modes)
    np.testing.assert_array_equal(loaded.residuals, result.residuals)


def test_save_refuses_existing_name(tmp_path):
    result, _ = saved(tmp_path)
    with pytest.raises(ModalArtifactError):
        save_modal_result(result, tmp_path, "modal-result")


def test_save_requires_modal_result(tmp_path):
    with pytest.raises(TypeError):
        save_modal_result({"modes": 1}, tmp_path, "bad")


def test_hash_mismatch_is_rejected(tmp_path):
    _, directory = saved(tmp_path)
    with (directory / "arrays.npz").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ModalArtifactError):
        load_modal_result(directory)


def test_unsupported_schema_is_rejected(tmp_path):
    _, directory = saved(tmp_path)
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["schema_version"] = "1.3"
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ModalArtifactError):
        load_modal_result(directory)


def test_extra_file_is_rejected(tmp_path):
    _, directory = saved(tmp_path)
    (directory / "extra.txt").write_text("no")
    with pytest.raises(ModalArtifactError):
        load_modal_result(directory)


def test_non_finite_array_is_rejected(tmp_path):
    _, directory = saved(tmp_path)
    manifest = json.loads((directory / "manifest.json").read_text())
    with np.load(directory / "arrays.npz", allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["eigenvalues"] = arrays["eigenvalues"].copy()
    arrays["eigenvalues"][0] = np.nan
    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    data = stream.getvalue()
    (directory / "arrays.npz").write_bytes(data)
    manifest["arrays_sha256"] = sha256(data).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ModalArtifactError):
        load_modal_result(directory)