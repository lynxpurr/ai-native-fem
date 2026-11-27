"""Static result viewer acceptance through a saved artifact and the CLI."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from anf.application import run_analysis
from anf.artifacts import save_result
from anf.viewer import open_result_artifact


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def artifact(tmp_path, example, change=None):
    model = json.loads((EXAMPLES / example).read_bytes())
    if change:
        change(model)
    return save_result(run_analysis(model), tmp_path, "saved result")


def cli(tmp_path, *args):
    run = subprocess.run(
        [sys.executable, "-I", "-m", "anf.cli", *map(str, args), "--json"],
        cwd=tmp_path, capture_output=True, check=False,
    )
    assert run.stdout, run.stderr.decode(errors="replace")
    return run.returncode, json.loads(run.stdout)


def scene_from(path):
    page = path.read_text(encoding="utf-8")
    return json.loads(page.split('id="result-data">', 1)[1].split("</script>", 1)[0])


def test_view_matches_inspect_and_preserves_planar_associations(tmp_path):
    def mixed_model(doc):
        model = doc["model"]
        model["nodes"]["ground"] = [6, 3]
        model["members"]["brace"] = {
            "kind": "truss2d", "i": "rightTop", "j": "ground",
            "young_modulus": 200e9, "area": 0.01,
        }
        model["members"]["girder"]["elements"] = 2
        model["supports"]["ground"] = {"ux": 0, "uy": 0}
        doc["loads"]["uniform_transverse"] = {"girder": -5}

    directory = artifact(tmp_path, "structures/portal.json", mixed_model)
    result = open_result_artifact(directory)
    assert result.solve_status == "solved"
    report = tmp_path / "saved report.html"
    status, summary = cli(tmp_path, "view", directory, "--output", report)
    assert status == 0
    assert summary.pop("output") == str(report)
    assert cli(tmp_path, "inspect", directory) == (0, summary)
    scene = scene_from(report)
    for key in ("input_hash", "solve_status", "verification_status", "diagnostics",
                "relative_residual", "strain_energy"):
        assert scene[key] == summary[key]
    midpoint = scene["labels"].index("girder / subdivision 1")
    left = scene["labels"].index("leftTop")
    right = scene["labels"].index("rightTop")
    assert scene["points"][midpoint] == [2, 3, 0]
    assert [left, midpoint] in scene["cells"]
    assert [midpoint, right] in scene["cells"]
    assert scene["labels"].index("ground") not in scene["frame_nodes"]
    assert ["girder", left, right] in scene["member_ends"]
    assert [[left, right], "girder", "uniform transverse (local +y)", "N/m", -5] in scene["loads"]
    fields = {field["name"]: field for field in scene["fields"]}
    for name, field in fields.items():
        assert field["unit"] == summary["arrays"][name]["unit"]
        assert field["position"] == summary["arrays"][name]["position"]
        assert field["values"] == result.arrays[name].ravel().tolist()
    assert fields["node_rz"]["position"] == "frame_node"
    assert fields["member_end_moment"]["coordinates"] == "Local member axes (i → j)"
    assert scene["displacement"][midpoint] == [
        fields["node_ux"]["values"][midpoint], fields["node_uy"]["values"][midpoint], 0,
    ]


@pytest.mark.parametrize("example,axis", [
    ("models/cantilever.json", 1),
    ("models/thick-q4.json", 2),
])
def test_view_uses_only_translations_for_deformation(tmp_path, example, axis):
    directory = artifact(tmp_path, example)
    report = tmp_path / "report.html"
    assert cli(tmp_path, "view", directory, "--output", report)[0] == 0
    scene = scene_from(report)
    result = open_result_artifact(directory)
    assert scene["fields"]
    for index, value in enumerate(result.arrays["deflection"].ravel()):
        vector = [0, 0, 0]
        vector[axis] = value
        assert scene["displacement"][index] == vector
    if axis == 1:
        assert scene["cells"][0] == [0, 1]
        assert scene["supports"] == [[0, "w", "m", 0], [0, "slope", "rad", 0]]
    else:
        stride = len(result.arrays["x"])
        assert scene["cells"][0] == [0, 1, stride + 1, stride]
        assert [0, "w", "m", 0] in scene["supports"]
        assert stride + 1 not in {row[0] for row in scene["supports"]}


def test_failure_artifact_has_diagnostics_without_result_fields(tmp_path):
    directory = artifact(
        tmp_path, "models/cantilever.json", lambda doc: doc["model"].update(length=1e-280),
    )
    report = tmp_path / "failed.html"
    status, summary = cli(tmp_path, "view", directory, "--output", report)
    assert status == 0
    scene = scene_from(report)
    assert scene["solve_status"] == summary["solve_status"] == "numerical_failure"
    assert scene["verification_status"] == "not_run"
    assert scene["diagnostics"] == summary["diagnostics"]
    assert scene["fields"] == scene["displacement"] == []
    assert scene["points"] and scene["supports"]


@pytest.mark.parametrize("damage", ["digest", "over_limit", "incomplete", "schema"])
def test_view_rejects_invalid_artifact_before_writing(tmp_path, damage):
    directory = artifact(tmp_path, "models/cantilever.json")
    arrays = directory / "arrays.npz"
    manifest = directory / "manifest.json"
    if damage == "digest":
        arrays.write_bytes(b"invalid array digest")
    elif damage == "over_limit":
        manifest.write_bytes(b" " * (1024 * 1024 + 1))
    elif damage == "incomplete":
        manifest.unlink()
    else:
        data = json.loads(manifest.read_bytes())
        data["schema_version"] = "1.4"
        manifest.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "report.html"
    status, summary = cli(tmp_path, "view", directory, "--output", output)
    assert status == 4
    assert summary["error"]["path"] == "/artifact"
    assert not output.exists()


def test_view_requires_new_html_output(tmp_path):
    directory = artifact(tmp_path, "models/cantilever.json")
    output = tmp_path / "existing.html"
    output.write_bytes(b"keep existing data")
    for path in (output, output.with_suffix(".txt"), tmp_path / "missing" / output.name):
        status, summary = cli(tmp_path, "view", directory, "--output", path)
        assert status == 4
        assert summary["error"]["path"] == "/output"
    assert output.read_bytes() == b"keep existing data"
