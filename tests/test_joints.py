"""Check rotational joints through the planar load case product."""

from copy import deepcopy
import json

import numpy as np
import pytest

from anf.application import run_load_cases
from anf.artifacts import ArtifactError, load_load_case_results, save_load_case_results
from anf.model import ModelValidationError
from anf.recovery import member_forces


def frame(start, end, elements=1, **connections):
    properties = {"kind": "frame2d", "i": start, "j": end, "young_modulus": 2e11,
                  "area": 0.01, "second_moment": 1e-6, "elements": elements}
    if connections:
        properties["end_connections"] = connections
    return properties


def spring(stiffness):
    return {"kind": "rotational_spring", "stiffness": stiffness}


def release():
    return {"kind": "release"}


def beam_model(elements=1, i=None, j=None, q=(-500, -500), rotation=(0, 0)):
    supports = {"left": {"ux": 0, "uy": 0}, "right": {"uy": 0}}
    for node, connection, angle in zip(supports, (i, j), rotation, strict=True):
        if connection is None or connection["kind"] != "release":
            supports[node]["rz"] = angle
    connections = {end: item for end, item in zip(("i", "j"), (i, j), strict=True) if item is not None}
    line = {"basis": "global", "i": [0, q[0]], "j": [0, q[1]]}
    return {
        "schema_version": "1.3", "problem_type": "planar_structure",
        "analysis": {"type": "static"}, "units": {"length": "m", "force": "N", "stress": "Pa"},
        "model": {"nodes": {"left": [0, 0], "right": [4, 0]},
                  "members": {"beam": frame("left", "right", elements, **connections)}, "supports": supports},
        "loads": {"cases": {"line": {"member_line": {"beam": line}}, "empty": {}}},
    }


def compare(value, target, atol=1e-6):
    value = np.asarray(value)
    target = np.broadcast_to(target, value.shape)
    active = target != 0
    np.testing.assert_allclose(value[active], target[active], rtol=1e-8, atol=0)
    np.testing.assert_allclose(value[~active], target[~active], rtol=0, atol=atol)


@pytest.mark.parametrize("elements", [1, 2, 4, 8])
@pytest.mark.parametrize("ratio", [0, 0.001, 1, 1000, None])
def test_spring_limits_with_uniform_load(elements, ratio):
    k = None if ratio is None else ratio * 150_000
    connection = release() if k == 0 else spring(k) if k is not None else None
    batch = run_load_cases(beam_model(elements, connection, release()))
    outcome = batch.results["line"]
    assert outcome.solve_status == "solved"
    theta_i = 0 if ratio is None else -1 / (150 * (1 + ratio))
    moment_i = 1000 if ratio is None else -k * theta_i
    left = 1000 + moment_i / 4
    theta_j = (1 / 150) - moment_i * 4 / (6 * 200_000)
    compare(outcome.arrays["reaction_fy"][:2], [left, 2000 - left])
    compare(outcome.arrays["member_end_moment"], [[moment_i, 0]])
    compare(outcome.arrays["frame_member_end_rotation"], [[theta_i, theta_j]], atol=1e-10)
    forces = member_forces(outcome)
    x = 4 * np.asarray(forces.stations)
    compare(forces.shear, -left + 500 * x)
    compare(forces.moment, np.where(x == 4, 0, -moment_i + left * x - 250 * x*x))
    assert forces.moment[0, -1] == 0
    assert batch.envelope.frame_node_order == (() if k == 0 else ("left",))
    assert not any(key == ["input", "right"] for key in batch.ordering["frame_node_order"])
    assert outcome.relative_residual <= 1e-8


def test_fully_released_beam_keeps_internal_energy():
    batch = run_load_cases(beam_model(1, release(), release()))
    outcome = batch.results["line"]
    assert "node_rz" not in outcome.arrays
    # In the one-element discretization opposite slopes produce constant curvature.
    strain_energy = 2 * 200_000 * (1 / 150)**2 / 4
    compare(outcome.strain_energy, strain_energy, atol=1e-10)
    assert batch.results["empty"].strain_energy == 0


@pytest.mark.parametrize("elements", [1, 4])
def test_prescribed_joint_rotation_drives_the_beam_once(elements):
    k = 125_000
    g = 0.02
    doc = beam_model(elements, spring(k), release(), rotation=(g, 0), q=(0, 0))
    doc["loads"]["combinations"] = {"combined": {"line": 2, "empty": -0.5}}
    result = run_load_cases(doc).results["combined"]
    moment = g / (1 / k + 4 / (3 * 200_000))
    rotation = g - moment / k
    compare(result.arrays["member_end_moment"], [[moment, 0]])
    compare(result.arrays["reaction_fy"][:2], [moment / 4, -moment / 4])
    compare(result.arrays["frame_member_end_rotation"], [[rotation, -rotation / 2]], atol=1e-10)
    compare(result.strain_energy, 0.5 * moment * g, atol=1e-10)


@pytest.mark.parametrize("left_connection", [spring(100_000), release()])
def test_shared_node_rotation_balances_moment(left_connection):
    model = beam_model()
    body = model["model"]
    body["nodes"]["end"] = [8, 0]
    body["supports"]["right"] = {"ux": 0, "uy": 0}
    body["supports"]["end"] = {"ux": 0, "uy": 0, "rz": 0}
    body["members"] = {"left": frame("left", "right", j=left_connection), "right": frame("right", "end")}
    model["loads"]["cases"] = {"moment": {"nodal": {"right": {"mz": 120}}}}
    batch = run_load_cases(model)
    result = batch.results["moment"]
    left, right = (0, 120) if left_connection["kind"] == "release" else (30, 90)
    compare(result.arrays["member_end_moment"][[0, 1], [1, 0]], [left, right])
    index = batch.ordering["frame_node_order"].index(["input", "right"])
    compare(result.arrays["node_rz"][index], right / 200_000, atol=1e-10)
    compare(result.arrays["frame_member_end_rotation"][0, 1], left / 200_000, atol=1e-10)


@pytest.mark.parametrize("elements", [1, 4])
def test_trapezoid_and_two_springs_reverse_at_prescribed_rotations(elements):
    original = beam_model(elements, spring(100_000), spring(300_000), (-350, 120), (0.005, -0.012))
    reversed_model = deepcopy(original)
    member = reversed_model["model"]["members"]["beam"]
    member["i"], member["j"] = member["j"], member["i"]
    ends = member["end_connections"]
    ends["i"], ends["j"] = ends["j"], ends["i"]
    line = reversed_model["loads"]["cases"]["line"]["member_line"]["beam"]
    line["i"], line["j"] = line["j"], line["i"]
    a = run_load_cases(original).results["line"]
    b = run_load_cases(reversed_model).results["line"]
    compare(b.arrays["reaction_fy"], a.arrays["reaction_fy"])
    compare(b.arrays["reaction_mz"], a.arrays["reaction_mz"])
    compare(b.arrays["member_end_moment"], a.arrays["member_end_moment"][:, ::-1])
    compare(b.arrays["frame_member_end_rotation"], a.arrays["frame_member_end_rotation"][:, ::-1], atol=1e-10)
    compare(member_forces(b).shear, member_forces(a).shear[:, ::-1])
    compare(member_forces(b).moment, -member_forces(a).moment[:, ::-1])
    actions = a.arrays["member_end_moment"][0]
    theta = a.arrays["frame_member_end_rotation"][0]
    compare(actions, (np.array([0.005, -0.012]) - theta) * [100_000, 300_000])
    left, right = a.arrays["reaction_fy"][:2]
    compare(left + right, -4 * (-350 + 120) / 2)
    compare(sum(actions) + 4 * right + 4**2 * (-350 + 2 * 120) / 6, 0)


def test_oblique_connection_rotates_global_reactions():
    doc = beam_model(4, spring(100_000), release(), (-350, 120), (0.005, 0))
    doc["model"]["supports"]["right"]["ux"] = 0
    horizontal = run_load_cases(doc).results["line"]
    doc["model"]["nodes"]["right"] = [2.4, 3.2]
    line = doc["loads"]["cases"]["line"]["member_line"]["beam"]
    for end in ("i", "j"):
        _, load = line[end]
        line[end] = [-0.8 * load, 0.6 * load]
    oblique = run_load_cases(doc).results["line"]
    for key in ("member_end_moment", "frame_member_end_rotation", "member_section_moment"):
        compare(oblique.arrays[key], horizontal.arrays[key], atol=1e-10)
    compare(oblique.arrays["reaction_fx"], -0.8 * horizontal.arrays["reaction_fy"])
    compare(oblique.arrays["reaction_fy"], 0.6 * horizontal.arrays["reaction_fy"])


@pytest.mark.parametrize("soft_stiffness", [1e-300, 5e-324])
def test_extreme_finite_springs_have_bounded_limits(soft_stiffness):
    rigid = run_load_cases(beam_model(j=release())).results["line"]
    stiff = run_load_cases(beam_model(i=spring(1e300), j=release())).results["line"]
    free = run_load_cases(beam_model(i=release(), j=release())).results["line"]
    soft = run_load_cases(beam_model(i=spring(soft_stiffness), j=release())).results["line"]
    for key in ("member_end_moment", "frame_member_end_rotation", "reaction_fy"):
        compare(stiff.arrays[key], rigid.arrays[key], atol=1e-10)
        compare(soft.arrays[key], free.arrays[key], atol=1e-10)


@pytest.mark.parametrize("elements", [1, 4])
def test_double_release_mechanism_is_reported(elements):
    doc = beam_model(elements, release(), release())
    del doc["model"]["supports"]["right"]
    batch = run_load_cases(doc)
    assert batch.envelope_status == "not_available"
    assert batch.results["line"].diagnostics[0]["code"] == "unstable_or_ill_conditioned"


@pytest.mark.parametrize(("change", "code", "path"), [
    (lambda d: d["model"]["members"]["beam"]["end_connections"].update(k=release()), "unknown_field",
     "/model/members/beam/end_connections/k"),
    (lambda d: d["model"]["members"]["beam"]["end_connections"].update(i=spring(0)), "invalid_value",
     "/model/members/beam/end_connections/i/stiffness"),
    (lambda d: d["model"]["members"]["beam"]["end_connections"].update(i=spring(True)), "invalid_type",
     "/model/members/beam/end_connections/i/stiffness"),
    (lambda d: d["model"]["members"]["beam"]["end_connections"].update(i=spring(float("inf"))), "invalid_value",
     "/model/members/beam/end_connections/i/stiffness"),
    (lambda d: d["model"]["members"]["beam"]["end_connections"].update(i={"kind": "hinge"}), "unsupported_connection",
     "/model/members/beam/end_connections/i/kind"),
    (lambda d: d["model"]["members"]["beam"]["end_connections"].update(i={"kind": "release", "stiffness": 0}),
     "unknown_field", "/model/members/beam/end_connections/i/stiffness"),
    (lambda d: d["model"]["supports"]["right"].update(rz=0), "inactive_dof", "/model/supports/right/rz"),
    (lambda d: d["loads"]["cases"]["empty"].update(nodal={"right": {"mz": 0}}), "inactive_dof",
     "/loads/cases/empty/nodal/right/mz"),
    (lambda d: d.update(schema_version="1.2"), "unknown_field", "/model/members/beam/end_connections"),
])
def test_invalid_connection_diagnostics(change, code, path):
    doc = beam_model(j=release())
    change(doc)
    with pytest.raises(ModelValidationError) as caught:
        run_load_cases(doc)
    assert (caught.value.code, caught.value.path) == (code, path)


def test_truss_connection_is_rejected():
    model = beam_model(j=release())
    frame = model["model"]["members"]["beam"]
    frame["kind"] = "truss2d"
    del frame["second_moment"]
    with pytest.raises(ModelValidationError, match="/model/members/beam/end_connections"):
        run_load_cases(model)


def test_truss_only_13_artifact_has_no_rotations(tmp_path):
    doc = beam_model(i=release(), j=release())
    bar = {"kind": "truss2d", "i": "left", "j": "right", "young_modulus": 2e11, "area": 0.01}
    doc["model"]["members"] = {"bar": bar}
    doc["loads"]["cases"] = {"tip": {"nodal": {"right": {"fx": 100}}}}
    batch = run_load_cases(doc)
    artifact = load_load_case_results(save_load_case_results(batch, tmp_path, "truss"))
    result = artifact.results["tip"]
    assert result.solve_status == "solved"
    assert not {"node_rz", "frame_member_end_rotation"} & result.arrays.keys()
    compare(result.arrays["node_ux"], [0, 100 * 4 / (2e11 * 0.01)], atol=1e-10)


def test_versioned_round_trip_and_rigid_identity(tmp_path):
    jointed = run_load_cases(beam_model(2, spring(125_000), release()))
    directory = save_load_case_results(jointed, tmp_path, "joints")
    restored = load_load_case_results(directory)
    key = "frame_member_end_rotation"
    assert restored.schema_version == "1.3" and restored.request_hash == jointed.request_hash
    assert restored.results["line"].arrays[key].tobytes() == jointed.results["line"].arrays[key].tobytes()
    index = directory / "manifest.json"
    manifest = json.loads(index.read_bytes())
    manifest["results"][0]["arrays"][key]["shape"] = [2, 1]
    index.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactError, match="metadata"):
        load_load_case_results(directory)

    no_joints = beam_model()
    older = deepcopy(no_joints)
    older["schema_version"] = "1.2"
    older_batch = run_load_cases(older)
    no_joints["model"]["members"]["beam"]["end_connections"] = {}
    newer_batch = run_load_cases(no_joints)
    assert older_batch.request_hash != newer_batch.request_hash
    assert older_batch.results["line"].input_hash == newer_batch.results["line"].input_hash
    assert jointed.results["line"].input_hash != newer_batch.results["line"].input_hash
    for name, values in older_batch.results["line"].arrays.items():
        np.testing.assert_array_equal(values, newer_batch.results["line"].arrays[name])
    assert key not in load_load_case_results(save_load_case_results(older_batch, tmp_path, "old")).results["line"].arrays
