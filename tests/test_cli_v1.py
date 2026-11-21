"""Installed CLI and application contract parity."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

import numpy as np
import pytest

from anf import cli
from anf.application import run_analysis
from anf.artifacts import load_result
from anf.model import canonical_input_hash


_MODELS = Path(__file__).resolve().parents[1] / "examples" / "models"


def invoke(directory, *args):
    executable = Path(sysconfig.get_path("scripts")) / ("anf.exe" if os.name == "nt" else "anf")
    if Path(cli.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        assert executable.is_file(), "the installed package must provide the anf command"
    env = {**os.environ, "PYTHONPATH": str(Path(cli.__file__).resolve().parents[1])}
    command = [str(executable)] if executable.is_file() else [sys.executable, "-m", "anf.cli"]
    result = subprocess.run([*command, *map(str, args)], cwd=directory, env=env,
                            capture_output=True, encoding="utf-8", timeout=30)
    assert result.stderr == ""
    assert len(result.stdout.splitlines()) == 1
    return result.returncode, json.loads(result.stdout)


@pytest.mark.parametrize(("name", "reference", "status"), [
    ("cantilever", "cantilever.closed-form.v1", "verification_passed"),
    ("thick-q4", "plate.navier-31.v1", "verification_passed"),
    ("thin-q4", "plate.thin-q4-navier-31.v1", "verification_failed"),
])
def test_console_script_matches_python(tmp_path, name, reference, status):
    source = tmp_path / "模型 input.json"
    shutil.copyfile(_MODELS / f"{name}.json", source)
    expected = run_analysis(source.read_bytes(), {"reference": reference})
    assert invoke(tmp_path, "--json", "validate", source.name) == (0, {
        "input_status": "valid", "input_hash": canonical_input_hash(expected.model),
        "model": expected.model.to_dict(),
    })
    directory = tmp_path / "结果 bundle"
    code, solved = invoke(tmp_path, "solve", source.name, "--output", directory.name,
                          "--reference", reference, "--json")
    assert code == 0
    assert Path(solved["artifact"]) == directory
    assert solved["input_hash"] == expected.input_hash
    assert solved["verification_status"] == status
    assert solved["arrays"] == expected.array_metadata
    assert invoke(tmp_path, "inspect", directory, "--json") == (0, solved)
    loaded = load_result(directory)
    assert loaded.report == expected.report
    for name, array in expected.arrays.items():
        np.testing.assert_array_equal(loaded.arrays[name], array)


@pytest.mark.parametrize(("case", "pointer", "code"), [
    ("unknown", "/extra", "unknown_field"),
    ("missing", "/units/stress", "missing_field"),
    ("support", "/model/support", "unsupported_support"),
    ("units", "/units/length", "unsupported_unit"),
])
def test_original_model_error_paths(tmp_path, case, pointer, code):
    path = tmp_path / "model.json"
    base = "thick-q4" if case == "support" else "invalid-units"
    doc = json.loads((_MODELS / f"{base}.json").read_bytes())
    if case == "unknown":
        doc["extra"] = False
    elif case == "missing":
        doc["units"].pop("stress")
        doc["units"]["length"] = "m"
    elif case == "support":
        doc["model"].update(formulation="mitc4_rectangle", support="clamped")
    path.write_text(json.dumps(doc), encoding="utf-8")
    status, payload = invoke(tmp_path, "solve", path, "--output", "output", "--json")
    assert status == 2
    assert payload["error"]["path"] == pointer
    assert payload["error"]["code"] == code
    assert not (tmp_path / "output").exists()


def test_numerical_failure_and_io_exit_codes(tmp_path):
    doc = json.loads((_MODELS / "cantilever.json").read_bytes())
    doc["model"]["length"] = 1e-280
    model = tmp_path / "failure.json"
    model.write_text(json.dumps(doc), encoding="utf-8")
    status, result = invoke(tmp_path, "solve", model, "--output", "failure", "--json")
    assert status == 3
    assert result["solve_status"] == "numerical_failure"
    assert result["diagnostics"][0]["code"] == "numerical_failure"
    assert invoke(tmp_path, "inspect", "failure", "--json") == (0, result)
    manifest = tmp_path / "failure" / "manifest.json"
    original = manifest.read_bytes()
    status, result = invoke(tmp_path, "solve", model, "--output", "failure", "--json")
    assert status == 4
    assert result["error"]["path"] == "/output"
    assert manifest.read_bytes() == original
    manifest.write_text("{}", encoding="utf-8")
    status, result = invoke(tmp_path, "inspect", "failure", "--json")
    assert status == 4
    assert result["error"]["code"] == "artifact_io"


def test_validate_and_inspect_only_read(tmp_path, capsys, monkeypatch):
    model = _MODELS / "cantilever.json"
    artifact = tmp_path / "result"
    assert cli.main(["solve", str(model), "--output", str(artifact)]) == 0
    capsys.readouterr()

    def unexpected(*args, **kwargs):
        pytest.fail("unexpected solve or write")

    monkeypatch.setattr(cli, "run_analysis", unexpected)
    monkeypatch.setattr(cli, "save_result", unexpected)
    assert cli.main(["validate", str(model)]) == 0
    assert cli.main(["inspect", str(artifact)]) == 0
    assert {item.name for item in tmp_path.iterdir()} == {"result"}


def test_usage_errors_are_structured_in_json_mode(tmp_path):
    status, payload = invoke(tmp_path, "--json", "solve", "model.json")
    assert status == 2
    assert payload["error"]["code"] == "invalid_command"
    assert "--output" in payload["error"]["message"]
