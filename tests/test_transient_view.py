"""Saved transient series and peak acceptance through the viewer CLI."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from anf.dynamics import TransientResult, save_transient_result
from anf.dynamics.system import dof_labels
from anf.model import canonical_dynamics_input_hash, validate_dynamics_model


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/dynamics/transient-cantilever.json"


@pytest.fixture
def transient():
    doc = json.loads(EXAMPLE.read_bytes())
    doc["model"]["members"]["beam"]["elements"] = 1
    doc["analysis"]["time"] = {"start_s": 0.5, "end_s": 1, "steps": 2}
    doc["loads"] = {}
    model = validate_dynamics_model(doc)
    times = np.array([0.5, 0.75, 1.0])
    displacement = np.zeros((3, 6))
    displacement[:, 4] = [1, 2, 3]
    displacement[:, 5] = [0, -2, 2]
    return TransientResult(
        model=model, times_s=times, displacement=displacement, velocity=np.zeros((3, 6)),
        acceleration=np.zeros((3, 6)), energy=np.zeros(3), dof_labels=dof_labels(model.mesh),
        start_s=0.5, dt_s=0.25, peak_displacement=np.array([0, 0, 0, 0, 3.0, 2.0]),
        peak_time_s=np.array([0.5, 0.5, 0.5, 0.5, 1, 0.75]),
        global_peaks={"displacement": (3.0, 1.0, 4), "velocity": (0.0, 0.5, 0),
                      "acceleration": (0.0, 0.5, 0)},
        worst_backward_error=0.0, worst_backward_step=0,
        diagnostics=(), versions={"anf": "0.1.0a7"}, input_hash=canonical_dynamics_input_hash(model),
    )


def view(tmp_path, result):
    directory = save_transient_result(result, tmp_path, "saved")
    output = tmp_path / "view.html"
    run = subprocess.run(
        [sys.executable, "-I", "-m", "anf.cli", "view", str(directory),
         "--output", str(output), "--json"],
        cwd=tmp_path, capture_output=True, check=False,
    )
    assert run.stdout, run.stderr.decode(errors="replace")
    return run.returncode, json.loads(run.stdout), output


def test_transient_view_preserves_signed_samples_and_first_peak(tmp_path, transient):
    code, summary, output = view(tmp_path, transient)
    assert code == 0
    assert summary["kind"] == "transient_result"
    assert summary["point_count"] == 3
    assert summary["start_s"] == 0.5 and summary["end_s"] == 1
    page = output.read_text(encoding="utf-8")
    data = json.loads(page.split('id="result-data">', 1)[1].split("</script>", 1)[0])
    assert data["times"] == [0.5, 0.75, 1.0]
    assert [item["label"] for item in data["nodes"]] == ["base", "tip"]
    translation, rotation = data["nodes"][1]["components"][1:]
    assert translation == {"name": "uy", "unit": "m", "values": [1, 2, 3], "peak": 3, "peak_time": 1}
    assert rotation == {"name": "rz", "unit": "rad", "values": [0, -2, 2], "peak": 2, "peak_time": 0.75}
    assert data["nodes"][0]["components"][0]["peak_time"] == 0.5


@pytest.mark.parametrize("invalid,diagnostic", [
    ({"times_s": np.array([0.5, 0.5, 1.0])}, "time axis"),
    ({"peak_time_s": np.array([0.5, 0.5, 0.5, 0.5, 1, 1.0])}, "displacement peaks"),
    ({"dof_labels": (("input", "base", "uy"), ("input", "base", "ux"), ("input", "base", "rz"),
                     ("input", "tip", "ux"), ("input", "tip", "uy"), ("input", "tip", "rz"))}, "DOF label"),
])
def test_transient_view_rejects_inconsistent_saved_history(tmp_path, transient, invalid, diagnostic):
    code, summary, output = view(tmp_path, replace(transient, **invalid))
    assert code == 4
    assert summary["error"]["path"] == "/artifact"
    assert diagnostic in summary["error"]["message"]
    assert not output.exists()


@pytest.mark.parametrize("time,expected_times,repeated", [
    ({"start_s": 0.001, "end_s": 0.015, "steps": 7},
     [0.001 + index * ((0.015 - 0.001) / 7) for index in range(8)], False),
    ({"start_s": 2**52, "end_s": 2**52 + 1, "steps": 4},
     [2**52, 2**52, 2**52, 2**52 + 1, 2**52 + 1], True),
    ({"times_s": [0.5, 0.7500000001, 1.0]}, [0.5, 0.7500000001, 1.0], False),
])
def test_transient_view_preserves_published_time_grid_rounding(
    tmp_path, transient, time, expected_times, repeated,
):
    doc = transient.model.to_dict()
    doc["analysis"]["time"] = time
    model = validate_dynamics_model(doc)
    grid = model.dynamics.time
    times = np.asarray(grid.times_s)
    count = len(times)
    assert times.tolist() == expected_times
    displacement = np.zeros((count, len(transient.dof_labels)))
    displacement[1, 4] = -2
    peak_displacement = np.zeros(len(transient.dof_labels))
    peak_displacement[4] = 2
    peak_time_s = np.full(len(transient.dof_labels), times[0])
    peak_time_s[4] = times[1]
    result = replace(
        transient, model=model, times_s=times, dt_s=grid.dt_s, start_s=grid.start_s,
        displacement=displacement, velocity=np.zeros_like(displacement),
        acceleration=np.zeros_like(displacement), energy=np.zeros(count),
        peak_displacement=peak_displacement, peak_time_s=peak_time_s,
        input_hash=canonical_dynamics_input_hash(model),
    )
    status, summary, report = view(tmp_path, result)
    assert status == 0
    assert summary["end_s"] == expected_times[-1]
    html = report.read_text(encoding="utf-8")
    scene = json.loads(html.split('id="result-data">', 1)[1].split("</script>", 1)[0])
    assert scene["times"] == expected_times
    assert scene["repeated_times"] is repeated
