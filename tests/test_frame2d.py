"""Independent analytic checks and installed-entry acceptance for planar statics."""

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from anf import application
from anf.artifacts import ArtifactError, load_result, save_result
from anf.elements import Frame2D, Truss2D
from anf.elements.frame2d import hermite_displacement
from anf.model import ModelValidationError, canonical_input_hash, validate_model
from anf.studies import StudySpec
from anf.verification import VerificationReport


FIXED = {"ux": 0, "uy": 0, "rz": 0}
REFERENCE = "structure.cantilever-tip.v1"
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def declaration(nodes, members, supports, *, nodal=None, uniform=None):
    return {
        "schema_version": "1.1", "problem_type": "planar_structure",
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "model": {"nodes": nodes, "members": members, "supports": supports},
        "loads": {"nodal": nodal or {}, "uniform_transverse": uniform or {}},
        "analysis": {"type": "static"},
    }


def bar(i, j):
    return {"kind": "truss2d", "i": i, "j": j, "young_modulus": 200e9, "area": 1e-5}


def frame(i, j, elements=1, *, young=210e9, inertia=1e-6):
    return {"kind": "frame2d", "i": i, "j": j, "young_modulus": young,
            "area": 0.01, "second_moment": inertia, "elements": elements}


def cantilever(elements=1):
    return declaration({"a": [0, 0], "b": [2, 0]}, {"beam": frame("a", "b", elements)},
                       {"a": dict(FIXED)}, nodal={"b": {"fy": -1000}})


def simply_supported(elements, *, reversed=False):
    i, j = ("b", "a") if reversed else ("a", "b")
    return declaration({"a": [0, 0], "b": [4, 0]},
                       {"beam": frame(i, j, elements, young=200e9)},
                       {"a": {"ux": 0, "uy": 0}, "b": {"uy": 0}},
                       uniform={"beam": 500 if reversed else -500})


def check(actual, expected, zero):
    actual, expected = np.asarray(actual), np.asarray(expected)
    tolerance = np.where(expected == 0, zero, 1e-8 * np.abs(expected))
    assert actual.shape == expected.shape
    assert np.all(np.abs(actual - expected) <= tolerance), (actual, expected, tolerance)


def node_value(result, node, array):
    if isinstance(node, str):
        node = ["input", node]
    order = "frame_node_order" if array in {"node_rz", "reaction_mz"} else "node_order"
    return result.arrays[array][result.ordering[order].index(node)]


def ends(result, member, axial, shear, moment):
    row = result.ordering["member_order"].index(member)
    for name, expected in zip(("axial", "shear", "moment"), (axial, shear, moment), strict=True):
        check(result.arrays[f"member_end_{name}"][row], expected, 1e-6)


def solved(doc, reference=None):
    result = application.run_analysis(doc, {"reference": reference})
    assert result.solve_status == "solved", result.diagnostics
    assert 0 <= result.relative_residual <= 1e-8
    return result


def equilibrium(result):
    """Check member and global force/moment equilibrium and work, including settlements."""
    model, mesh = result.model, result.model.mesh
    force_x, force_y, moment, work = [], [], [], []
    for index, (node, (x, y)) in enumerate(zip(mesh.node_order, mesh.coordinates, strict=True)):
        nodal = model.loads["nodal"].get(node[1], {}) if node[0] == "input" else {}
        fx = node_value(result, list(node), "reaction_fx")
        fy = node_value(result, list(node), "reaction_fy")
        ux = node_value(result, list(node), "node_ux")
        uy = node_value(result, list(node), "node_uy")
        fx += nodal.get("fx", 0)
        fy += nodal.get("fy", 0)
        mz = 0
        if len(mesh.node_dofs[index]) == 3:
            mz = node_value(result, list(node), "reaction_mz") + nodal.get("mz", 0)
            work.append(mz * node_value(result, list(node), "node_rz"))
        force_x.append(fx)
        force_y.append(fy)
        moment.extend((mz, x * fy, -y * fx))
        work.extend((fx * ux, fy * uy))

    def balanced(terms, zero=1e-6):
        assert abs(math.fsum(terms)) <= max(zero, 1e-8 * max(map(abs, terms)))

    for row, (name, member) in enumerate(model.model["members"].items()):
        i, j = [np.array(model.model["nodes"][member[end]]) for end in ("i", "j")]
        length = math.dist(i, j)
        q = model.loads["uniform_transverse"].get(name, 0)
        n, v, m = [result.arrays[f"member_end_{part}"][row] for part in ("axial", "shear", "moment")]
        balanced(n)
        balanced((*v, q * length))
        balanced((*m, length * v[1], q * length**2 / 2))
        fx, fy = q * np.array([i[1] - j[1], j[0] - i[0]])
        force_x.append(fx)
        force_y.append(fy)
        x, y = (i + j) / 2
        moment.extend((x * fy, -y * fx))

    for segment in mesh.segments:
        q = model.loads["uniform_transverse"].get(segment.member, 0)
        if not q:
            continue
        length, c, s = segment.geometry
        transverse = []
        for node in segment.nodes:
            key = list(mesh.node_order[node])
            transverse.extend((-s * node_value(result, key, "node_ux") + c * node_value(result, key, "node_uy"),
                               node_value(result, key, "node_rz")))
        # Simpson integration of a cubic is exact.
        midpoint = hermite_displacement(0.5, length, transverse)
        work.append(q * length * (transverse[0] + 4 * midpoint + transverse[2]) / 6)

    for terms in (force_x, force_y, moment):
        balanced(terms)
    balanced([*work, -2 * result.strain_energy], 1e-10)


@pytest.mark.parametrize("count", [1, 2, 4, 8])
def test_f1_axial_chain(count):
    nodes = {f"n{i}": [4*i/count, 0] for i in range(count + 1)}
    members = {f"bar{i}": bar(f"n{i}", f"n{i+1}") for i in range(count)}
    supports = {name: {"uy": 0} for name in nodes}
    supports["n0"]["ux"] = 0
    result = solved(declaration(nodes, members, supports, nodal={f"n{count}": {"fx": 1000}}))
    assert "node_rz" not in result.arrays and "reaction_mz" not in result.arrays
    for name, (x, _) in nodes.items():
        check(node_value(result, name, "node_ux"), 1000 * x / 2e6, 1e-10)
        check(node_value(result, name, "node_uy"), 0, 1e-10)
    check(node_value(result, "n0", "reaction_fx"), -1000, 1e-6)
    for name in members:
        ends(result, name, [-1000, 1000], [0, 0], [0, 0])
    check(result.strain_energy, 1, 1e-10)
    equilibrium(result)


@pytest.mark.parametrize("count", [1, 2, 4, 8])
def test_f1_cantilever(count):
    result = solved(cantilever(count), REFERENCE)
    assert result.verification_status == "verification_passed"
    assert len(result.ordering["node_order"]) == count + 1
    rigidity, length, force = 210e3, 2, -1000
    check(node_value(result, "b", "node_uy"), force * length**3 / (3*rigidity), 1e-10)
    check(node_value(result, "b", "node_rz"), force * length**2 / (2*rigidity), 1e-10)
    check(node_value(result, "a", "reaction_fy"), -force, 1e-6)
    check(node_value(result, "a", "reaction_mz"), -force * length, 1e-6)
    ends(result, "beam", [0, 0], [-force, force], [-force*length, 0])
    check(result.strain_energy, force**2 * length**3 / (6*rigidity), 1e-10)
    equilibrium(result)


@pytest.mark.parametrize(("count", "reverse"), [(1, False), (2, False), (4, False), (8, False), (2, True)])
def test_f1_f3_uniform_load(count, reverse):
    result = solved(simply_supported(count, reversed=reverse))
    q, length, rigidity = -500, 4, 200e3
    angle = q * length**3 / (24*rigidity)
    check(node_value(result, "a", "node_rz"), angle, 1e-10)
    check(node_value(result, "b", "node_rz"), -angle, 1e-10)
    force = -q * length / 2
    for name in ("a", "b"):
        check(node_value(result, name, "reaction_fy"), force, 1e-6)
    v = -force if reverse else force
    ends(result, "beam", [0, 0], [v, v], [0, 0])
    continuum_midspan = 5*q*length**4 / (384*rigidity)
    if count == 1:
        transverse = [node_value(result, node, dof) for node in ("a", "b")
                      for dof in ("node_uy", "node_rz")]
        midspan = hermite_displacement(0.5, length, transverse)
        check(midspan, 0.8 * continuum_midspan, 1e-10)
    else:
        midspan = node_value(result, ["generated", "beam", count//2], "node_uy")
        check(midspan, continuum_midspan, 1e-10)
    continuum_energy = q**2 * length**5 / (240*rigidity)
    missing = q**2 * length**5 / (1440*rigidity*count**4)
    check(result.strain_energy, continuum_energy - missing, 1e-10)
    check(continuum_energy - result.strain_energy, missing, 1e-10)
    equilibrium(result)


def test_f2_portal_force_method():
    h, length, force, axial, bending = 3, 4, 1000, 2e9, 2e6
    nodes = {"a": [0, 0], "b": [0, h], "c": [length, h], "d": [length, 0]}
    members = {i+j: frame(i, j, young=200e9, inertia=1e-5) for i, j in ("ab", "bc", "cd")}
    result = solved(declaration(nodes, members, {"a": FIXED, "d": {"uy": 0}},
                                nodal={"c": {"fx": force}}))
    # Integrate the unit-load force diagrams on the frame with its right roller removed.
    fxx = h**3 / (3*bending) + length/axial
    fxy = -length*h**2 / (2*bending)
    fyy = (length**2*h + length**3/3)/bending + 2*h/axial
    roller = -fxy * force / fyy
    dx, dy = fxx*force + fxy*roller, -roller*h/axial
    moment = force*h - roller*length
    check(node_value(result, "c", "node_ux"), dx, 1e-10)
    check(node_value(result, "c", "node_uy"), dy, 1e-10)
    check(node_value(result, "d", "reaction_fy"), roller, 1e-6)
    check([node_value(result, "a", key) for key in ("reaction_fx", "reaction_fy", "reaction_mz")],
          [-force, -roller, moment], 1e-6)
    ends(result, "ab", [-roller, roller], [force, -force], [moment, length*roller])
    ends(result, "bc", [-force, force], [-roller, roller], [-length*roller, 0])
    ends(result, "cd", [roller, -roller], [0, 0], [0, 0])
    check(result.strain_energy, force*dx/2, 1e-10)
    equilibrium(result)


def test_f2_shared_translation():
    doc = cantilever()
    doc["model"]["nodes"]["c"] = [2, 2]
    doc["model"]["members"]["tie"] = bar("b", "c")
    doc["model"]["supports"]["c"] = {"ux": 0, "uy": 0}
    result = solved(doc, REFERENCE)
    assert result.verification_status == "solved_unverified"
    assert all(check.outcome == "not_applicable" for check in result.report.checks[1:])
    assert result.ordering["frame_node_order"] == [["input", "a"], ["input", "b"]]
    flexural, tie = 3*210e3/2**3, 2e6/2
    displacement = -1000 / (flexural + tie)
    bending_force, tension = -flexural * displacement, -tie * displacement
    check(node_value(result, "b", "node_uy"), displacement, 1e-10)
    check(node_value(result, "b", "node_rz"), 3*displacement/4, 1e-10)
    check(node_value(result, "c", "reaction_fy"), tension, 1e-6)
    check([node_value(result, "a", key) for key in ("reaction_fx", "reaction_fy", "reaction_mz")],
          [0, bending_force, 2*bending_force], 1e-6)
    ends(result, "beam", [0, 0], [bending_force, -bending_force], [2*bending_force, 0])
    ends(result, "tie", [-tension, tension], [0, 0], [0, 0])
    check(result.strain_energy, -1000*displacement/2, 1e-10)
    equilibrium(result)


@pytest.mark.parametrize("transform", ["rotate_translate_rename", "reverse"])
def test_f3_cantilever_covariance(transform):
    doc = cantilever(2)
    baseline = solved(doc)
    model = doc["model"]
    if transform == "reverse":
        model["members"]["beam"].update(i="b", j="a")
        root, tip = "a", "b"
        basis = np.eye(2)
    else:
        basis = np.array([[0.6, -0.8], [0.8, 0.6]])
        root, tip = "zroot", "atip"
        offset = np.array([10, -7])
        model["nodes"] = {root: list(offset), tip: list(offset + basis @ [2, 0])}
        model["nodes"] = {name: list(map(float, point)) for name, point in model["nodes"].items()}
        model["supports"] = {root: FIXED}
        model["members"]["beam"].update(i=root, j=tip)
        doc["loads"]["nodal"] = {tip: dict(zip(("fx", "fy"), map(float, basis @ [0, -1000])))}
    result = solved(doc)
    assert result.input_hash != baseline.input_hash
    for array, node, original, tol in (
        (("node_ux", "node_uy"), tip, "b", 1e-10),
        (("reaction_fx", "reaction_fy"), root, "a", 1e-6),
    ):
        check(basis.T @ [node_value(result, node, key) for key in array],
              [node_value(baseline, original, key) for key in array], tol)
    check(node_value(result, tip, "node_rz"), node_value(baseline, "b", "node_rz"), 1e-10)
    check(node_value(result, root, "reaction_mz"), 2000, 1e-6)
    for component, sign, actions in (
        ("axial", -1, [0, 0]), ("shear", -1, [1000, -1000]), ("moment", 1, [2000, 0]),
    ):
        field = f"member_end_{component}"
        expected = np.array([actions])
        check(baseline.arrays[field], expected, 1e-6)
        if transform == "reverse":
            expected = sign * expected[:, ::-1]
        check(result.arrays[field], expected, 1e-6)
    check(result.strain_energy, baseline.strain_energy, 1e-10)
    equilibrium(result)


@pytest.mark.parametrize("case", ["free", "fixed", "zero"])
def test_f4_prescribed_displacement(case, monkeypatch):
    nodes = {"a": [0, 0], "c": [4, 0]}
    end = {"uy": 0}
    if case != "zero":
        end["ux"] = .002
    members = {"ac": bar("a", "c")}
    supports = {"a": {"ux": 0, "uy": 0}, "c": end}
    if case == "free":
        nodes["b"], supports["b"] = [2, 0], {"uy": 0}
        members = {"ab": bar("a", "b"), "bc": bar("b", "c")}
        from anf import structures

        equilibrate = structures._equilibrate

        def check_rhs(matrix, rhs):
            check(rhs, [2000], 1e-6)
            return equilibrate(matrix, rhs)

        monkeypatch.setattr(structures, "_equilibrate", check_rhs)
    result = solved(declaration(nodes, members, supports))
    if case == "free":
        check(node_value(result, "b", "node_ux"), 0.001, 1e-10)
    else:
        assert result.relative_residual == 0
    tension = 0 if case == "zero" else 1000
    check(node_value(result, "c", "node_ux"), end.get("ux", 0), 1e-10)
    check(node_value(result, "a", "reaction_fx"), -tension, 1e-6)
    check(node_value(result, "c", "reaction_fx"), tension, 1e-6)
    for name in members:
        ends(result, name, [-tension, tension], [0, 0], [0, 0])
    check(result.strain_energy, tension * end.get("ux", 0)/2, 1e-10)
    equilibrium(result)


def mechanism(point):
    return declaration({"a": [0, 0], "b": point}, {"bar": bar("a", "b")},
                       {"a": {"ux": 0, "uy": 0}})


@pytest.mark.parametrize(("point", "reason"), [([4, 0], "diagonal"), ([3, 4], "eigenvalue")])
def test_f4_mechanism_writes_manifest_only(point, reason, tmp_path):
    result = application.run_analysis(mechanism(point))
    assert (result.solve_status, result.verification_status) == ("numerical_failure", "not_run")
    assert result.diagnostics[0]["code"] == "unstable_or_ill_conditioned"
    assert reason in result.diagnostics[0]["message"]
    artifact = save_result(result, tmp_path, "mechanism")
    assert {file.name for file in artifact.iterdir()} == {"manifest.json"}
    assert load_result(artifact).diagnostics == result.diagnostics


@pytest.mark.parametrize("kind", ["truss", "frame"])
def test_element_rigid_modes_rank_and_energy(kind):
    length, cosine, sine = 5, 0.6, 0.8
    if kind == "frame":
        element = Frame2D(length, 2e6, cosine, sine, bending_rigidity=200e3)
        scale = np.array([1, 1, 1/length]*2)
        modes = ([1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 1, -4, 3, 1])
        rank = 3
        load = element.local_load(-10)
        check(load[1] + load[4], -50, 1e-6)
        check(load[2] + length*load[4] + load[5], -125, 1e-6)
    else:
        element = Truss2D(length, 2e6, cosine, sine)
        scale = np.ones(4)
        modes = ([1, 0, 1, 0], [0, 1, 0, 1], [0, 0, -4, 3])
        rank = 1
    stiffness = scale[:, None] * element.stiffness() * scale
    norm = np.linalg.norm(stiffness)
    assert np.linalg.norm(stiffness - stiffness.T) <= 1e-12 * norm
    for mode in modes:
        unit_mode = np.asarray(mode) / scale
        assert np.linalg.norm(stiffness @ unit_mode) <= 1e-12 * norm * np.linalg.norm(unit_mode)
    spectral = np.linalg.eigvalsh(stiffness)
    assert np.count_nonzero(spectral > 1e-12 * norm) == rank
    assert np.min(spectral) >= -1e-12 * norm
    axial_mode = np.array([0, 0, 0, cosine, sine, 0] if kind == "frame" else [0, 0, cosine, sine])
    check(axial_mode @ element.stiffness() @ axial_mode / 2, 2e6/(2*length), 1e-10)


@pytest.mark.parametrize(("code", "path", "change"), [
    ("duplicate_node", "/model/nodes/b", lambda d: d["model"]["nodes"].update(b=[-0., 0])),
    ("zero_length", "/model/members/bar/j", lambda d: d["model"]["members"]["bar"].update(j="a")),
    ("duplicate_member", "/model/members/z", lambda d: d["model"]["members"].update(z=bar("b", "a"))),
    ("unknown_node", "/model/members/bar/j", lambda d: d["model"]["members"]["bar"].update(j="c")),
    ("isolated_node", "/model/nodes/c", lambda d: d["model"]["nodes"].update(c=[1, 2])),
    ("inactive_dof", "/model/supports/a/rz", lambda d: d["model"]["supports"]["a"].update(rz=0)),
    ("inactive_dof", "/loads/nodal/b/mz", lambda d: d["loads"]["nodal"].update(b={"mz": 0})),
    ("unknown_node", "/loads/nodal/z", lambda d: d["loads"]["nodal"].update(z={"fx": 1})),
    ("unknown_member", "/loads/uniform_transverse/z", lambda d: d["loads"]["uniform_transverse"].update(z=1)),
    ("unsupported_load", "/loads/uniform_transverse/bar",
     lambda d: d["loads"]["uniform_transverse"].update(bar=0)),
    ("invalid_value", "/model/members/bar/elements", lambda d: d["model"]["members"]["bar"].update(elements=2)),
    ("invalid_id", "/model/nodes/bad~1id", lambda d: d["model"]["nodes"].update({"bad/id": [1, 2]})),
    ("unsupported_unit", "/units/length", lambda d: d["units"].update(length="mm")),
    ("unsupported_schema", "/schema_version", lambda d: d.update(schema_version="1.2")),
    ("unsupported_problem", "/problem_type", lambda d: d.update(schema_version="1.0")),
    ("unrepresentable_geometry", "/model/members/bar",
     lambda d: d["model"]["nodes"].update(a=[-1e308, 0], b=[1e308, 0])),
])
def test_invalid_structure(code, path, change):
    doc = mechanism([4, 0])
    change(doc)
    with pytest.raises(ModelValidationError) as caught:
        canonical_input_hash(doc)
    assert (caught.value.code, caught.value.path) == (code, path)


def test_size_budget_and_unrepresentable_mesh():
    nodes = {f"n{i}": [i, 0] for i in range(8)}
    members = {f"m{i}": frame(f"n{i}", f"n{i+1}", 32) for i in range(7)}
    with pytest.raises(ModelValidationError, match="size_limit"):
        validate_model(declaration(nodes, members, {}))
    doc = cantilever(2)
    doc["model"]["nodes"] = {"a": [1e16, 0], "b": [1e16 + 2, 0]}
    with pytest.raises(ModelValidationError, match="unrepresentable_geometry"):
        validate_model(doc)
    doc = json.dumps(cantilever()).replace('"nodes": {', '"nodes": {}, "nodes": {')
    with pytest.raises(ModelValidationError, match="duplicate_field"):
        validate_model(doc)


def test_hash_defaults_portable_and_solves_once(monkeypatch):
    doc = cantilever()
    other = deepcopy(doc)
    del other["model"]["members"]["beam"]["elements"]
    other["model"]["nodes"]["a"][0] = -0.0
    other["model"]["supports"]["a"]["ux"] = -0.0
    other["loads"]["uniform_transverse"]["beam"] = -0.0
    other["loads"]["nodal"]["b"]["mz"] = 0
    expected_hash = canonical_input_hash(doc)
    assert canonical_input_hash(json.dumps(other, sort_keys=True)) == expected_hash
    real = application.solve_structure
    calls = []

    def wrapped(model):
        calls.append(model)
        return real(model)

    monkeypatch.setattr(application, "solve_structure", wrapped)
    result = solved(other, REFERENCE)
    assert len(calls) == 1
    assert result.input_hash == expected_hash
    assert result.schema_version == "1.1"
    assert result.report.checks[0].evidence_id == "planar_structure.relative-residual.v1"
    assert VerificationReport.from_json(result.report.to_json()) == result.report
    with pytest.raises(ModelValidationError, match="unsupported_reference"):
        application.run_analysis(doc, {"reference": "cantilever.closed-form.v1"})
    with pytest.raises(ModelValidationError, match="unsupported_problem"):
        StudySpec(doc, [{"elements": 1}], "tip_deflection")


@pytest.mark.parametrize("corruption", ["ordering", "unit", "schema", "array_digest", "npz"])
def test_structural_artifact_policy(tmp_path, corruption):
    result = solved(cantilever(2), REFERENCE)
    artifact = save_result(result, tmp_path, "result")
    loaded = load_result(artifact)
    assert loaded.ordering == result.ordering and loaded.report == result.report
    for name, array in loaded.arrays.items():
        np.testing.assert_array_equal(array, result.arrays[name])
    manifest = artifact / "manifest.json"
    data = json.loads(manifest.read_bytes())
    if corruption == "ordering":
        data["ordering"]["node_order"].reverse()
    elif corruption == "unit":
        data["arrays"]["member_end_moment"]["unit"] = "N"
    elif corruption == "schema":
        data["report"]["schema_version"] = "1.0"
    elif corruption == "array_digest":
        data["arrays_sha256"] = "0" * 64
    else:
        payload = b"not a zip archive"
        (artifact / "arrays.npz").write_bytes(payload)
        data["arrays_sha256"] = sha256(payload).hexdigest()
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ArtifactError):
        load_result(artifact)


@pytest.mark.parametrize(("name", "formulation", "width", "fingerprint", "reference", "status"), [
    ("cantilever", None, None, "50d38b0a536b56efe8e5ab185b595cb7d70ddb54e014d1eb39f128cc5de3e54e",
     "cantilever.closed-form.v1", "verification_passed"),
    ("thick-q4", None, None, "d17b6de821561f8e1b2bbc919c634ee011a4a35509205b73dcca683e57b5d835",
     "plate.navier-31.v1", "verification_passed"),
    ("thin-q4", None, None, "010ba5738a4616515c65a29570bb36398b50f8ad606fb89080939785835eee4a",
     "plate.thin-q4-navier-31.v1", "verification_failed"),
    ("thin-q4", "mitc4_rectangle", 1, "8eb4ac137f9aa4cbad4fc96fbb06318b70201cd5575e0a8ba0e6eccab0481247",
     None, "solved_unverified"),
    ("thin-q4", "mitc4_rectangle", 2, "1fb61e17fd51da9d83f399c858e8b983f607c84d7cc1288638aa43ed149c53ef",
     None, "solved_unverified"),
])
def test_r1_published_fingerprints(name, formulation, width, fingerprint, reference, status, tmp_path):
    doc = json.loads((EXAMPLES / "models" / f"{name}.json").read_bytes())
    if formulation:
        doc["model"].update(formulation=formulation, lx=width)
    result = solved(doc, reference)
    assert result.schema_version == "1.0"
    assert result.input_hash == canonical_input_hash(doc) == fingerprint
    assert result.verification_status == status
    path = save_result(result, tmp_path, "legacy")
    checked = cli(tmp_path, "inspect", path)
    assert checked["report"] == json.loads(result.report.to_json())


def cli(cwd, *args, code=0):
    command = [sys.executable, "-I", "-m", "anf.cli", *map(str, args), "--json"]
    result = subprocess.run(command, cwd=cwd, capture_output=True)
    assert result.returncode == code, result.stderr
    return json.loads(result.stdout)


def test_real_stdio_and_cli_match_python(tmp_path):
    pytest.importorskip("mcp")
    from test_mcp_local import call, connection

    anyio = pytest.importorskip("anyio")
    doc_path = EXAMPLES / "structures" / "cantilever.json"
    doc = json.loads(doc_path.read_bytes())
    expected = solved(doc, REFERENCE)
    validate = cli(tmp_path, "validate", doc_path)
    actual = cli(tmp_path, "solve", doc_path, "--output", "cli", "--reference", REFERENCE)
    assert cli(tmp_path, "inspect", "cli") == actual
    assert validate["input_hash"] == expected.input_hash
    assert actual["report"] == json.loads(expected.report.to_json())
    assert actual["ordering"] == expected.ordering
    bad = mechanism([3, 4])
    bad_path = tmp_path / "mechanism.json"
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    failure = cli(tmp_path, "solve", bad_path, "--output", "failure", code=3)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def run():
        async with connection(workspace) as session:
            caps = await call(session, "capabilities", {})
            assert caps["model_schemas"] == caps["result_schemas"] == ["1.0", "1.1"]
            assert caps["problems"]["planar_structure"]["max_dofs"] == 600
            assert "planar_structure" not in caps["observables"]
            tools = {tool.name: tool.inputSchema for tool in (await session.list_tools()).tools}
            choices = tools["validate_model"]["properties"]["model"]["oneOf"]
            assert choices[1]["properties"]["schema_version"] == {"const": "1.1"}
            assert tools["convergence_study"]["properties"]["study"]["properties"]["model"] == choices[0]
            validated = await call(session, "validate_model", {"model": doc})
            response = await call(session, "solve_model", {"model": doc, "reference": REFERENCE})
            assert validated["input_hash"] == response["input_hash"] == expected.input_hash
            for key in ("input_hash", "solve_status", "verification_status", "diagnostics"):
                assert response[key] == actual[key]
            identity = response["artifact_id"]
            assert await call(session, "inspect_artifact", {"artifact_id": identity}) == response
            mcp = load_result(workspace / identity)
            assert mcp.ordering == expected.ordering and mcp.report == expected.report
            for name, values in expected.arrays.items():
                np.testing.assert_allclose(mcp.arrays[name], values, atol=1e-12, rtol=1e-10)
            failed = await call(session, "solve_model", {"model": bad}, error=True)
            assert failed["error"]["code"] == "unstable_or_ill_conditioned"
            for key in ("input_hash", "solve_status", "verification_status", "diagnostics"):
                assert failed[key] == failure[key]
            declined = await call(session, "convergence_study", {
                "study": {"model": doc, "meshes": [{"elements": 2}], "observable": "tip_deflection"},
            }, error=True)
            assert declined["error"]["code"] == "unsupported_problem"

    anyio.run(run)


@pytest.mark.parametrize(("name", "node", "field", "expected"), [
    ("portal", "rightBase", "reaction_fy", 259.5929198434751),
    ("settlement", "middle", "node_ux", 0.001),
])
def test_packaged_examples(name, node, field, expected):
    result = solved((EXAMPLES / "structures" / f"{name}.json").read_bytes())
    check(node_value(result, node, field), expected, 0)
    equilibrium(result)
