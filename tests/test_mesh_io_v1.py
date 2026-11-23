"""Native-entry tests with independent rod, beam and plate references."""

from decimal import Decimal, localcontext
import math
from pathlib import Path

import numpy as np
import pytest

from anf.application import run_analysis
from anf.artifacts import ArtifactError, load_result, save_result
from anf.mesh_io import MeshDocument, import_mesh, validate_topology
from anf.meshing import generate_supported_mesh
from anf.model import ModelValidationError, canonical_input_hash, validate_model


def line(kind="frame2d", *, unit="m", moved=False):
    factor = 1000 if unit == "mm" else 1
    start, end = ([7, -1], [7, 1]) if moved else ([0, 0], [2, 0])
    return generate_supported_mesh({"generator": "line", "length_unit": unit, "property": "steel",
                                    "start": [x * factor for x in start], "end": [x * factor for x in end],
                                    "segments": 1, "kind": kind})


def binding(kind="frame2d", *, moved=False):
    material = {"young_modulus": 210e9, "area": .01}
    fixed = {"ux": 0, "uy": 0}
    if kind == "frame2d":
        material["second_moment"] = 1e-6
        fixed["rz"] = 0
    supports = [{"set": "root", "values": fixed}]
    if kind == "truss2d":
        supports.append({"set": "tip", "values": {"ux" if moved else "uy": 0}})
    force = {"fx": -1000} if moved else {"fy": 1000}
    if kind == "truss2d":
        force = {"fy": 1000} if moved else {"fx": 1000}
    return {"properties": {"steel": material}, "supports": supports,
            "loads": [{"set": "tip", "values": force}]}


def plate(*, moved=False, nx=8, ny=8):
    return generate_supported_mesh({"generator": "rectangle", "length_unit": "m", "property": "plate",
                                    "origin": [7, -1] if moved else [0, 0],
                                    "u": [0, 2] if moved else [2, 0], "v": [-1, 0] if moved else [0, 1],
                                    "nx": nx, "ny": ny})


def plate_binding(formulation="mitc4_rectangle"):
    return {"properties": {"plate": {"young_modulus": 210e9, "poisson_ratio": .3, "thickness": .01}},
            "pressure": 1000, "support": "hard_simply_supported", "formulation": formulation}


def renumber(mesh, *, reverse=False):
    doc = mesh.to_dict()
    doc["provenance"] = None
    remap = {node["id"]: 9007199254740991 - 17 * i for i, node in enumerate(doc["nodes"])}
    for node in doc["nodes"]:
        node["id"] = remap[node["id"]]
    for cell in doc["cells"]:
        cell["id"] = 1000001 + 31 * int(cell["id"])
        cell["nodes"] = [remap[key] for key in cell["nodes"]]
        if reverse:
            cell["nodes"].reverse()
    for group in doc["sets"]:
        group["ids"] = [remap[key] if group["kind"] == "node" else 1000001 + 31 * int(key) for key in group["ids"]]
    if doc["grid"] is not None:
        for key in ("origin", "u_corner", "v_corner"):
            doc["grid"][key] = remap[doc["grid"][key]]
    doc["nodes"].reverse()
    doc["cells"].reverse()
    doc["sets"].reverse()
    return import_mesh(doc)


def solved(mesh, bind, backend="dense"):
    result = run_analysis(mesh.to_model(bind), {"backend": backend})
    assert result.solve_status == "solved", result.diagnostics
    assert result.relative_residual <= 1e-8
    return result


def near(actual, expected, absolute):
    assert abs(actual - expected) <= absolute + 1e-8 * abs(expected), (actual, expected)


@pytest.mark.parametrize("kind", ["frame2d", "truss2d"])
@pytest.mark.parametrize("unit,moved,reverse", [("m", False, False), ("mm", False, True), ("m", True, False)])
def test_me1_member_physics_roundtrip_and_source_mapping(tmp_path, kind, unit, moved, reverse):
    original = line(kind, unit=unit, moved=moved)
    mesh = renumber(original, reverse=reverse)
    path = tmp_path / "mesh.json"
    path.write_bytes(mesh.to_json())
    restored = import_mesh(path)
    assert restored == mesh
    bind = binding(kind, moved=moved)
    result = solved(restored, bind)
    groups = {group["name"]: group["ids"] for group in mesh.to_dict()["sets"]}
    by_source = {row["source_id"]: row for row in mesh.mapping["nodes"]}
    tip, root = (by_source[groups[name][0]] for name in ("tip", "root"))
    expected = 4 / 315 if kind == "frame2d" else 2 / 2100000
    expected_u = (-expected, 0) if moved and kind == "frame2d" else ((0, expected) if moved else
                 ((0, expected) if kind == "frame2d" else (expected, 0)))
    for array, value in zip(("node_ux", "node_uy"), expected_u):
        near(result.arrays[array][tip["result_index"]], value, 1e-10)
    force = bind["loads"][0]["values"]
    for array, key in (("reaction_fx", "fx"), ("reaction_fy", "fy")):
        near(result.arrays[array][root["result_index"]], -force.get(key, 0), 1e-6)
    near(result.strain_energy, 400 / 63 if kind == "frame2d" else 1000 / 2100000, 1e-10)
    cell = mesh.mapping["cells"][0]
    assert cell["source_connection"] == mesh.to_dict()["cells"][0]["nodes"]
    assert result.ordering["member_order"][cell["result_index"]] == cell["model_id"]
    if kind == "frame2d":
        near(result.arrays["node_rz"][tip["frame_result_index"]], 1 / 105, 1e-10)
        near(result.arrays["reaction_mz"][root["frame_result_index"]], -2000, 1e-6)
        # End moments reverse their endpoint positions under source connectivity reversal.
        near(result.arrays["member_end_moment"][0][1 if reverse else 0], -2000, 1e-6)
    save_result(result, tmp_path, "result")
    reread = load_result(tmp_path / "result")
    assert reread.input_hash == result.input_hash
    for key in result.arrays:
        np.testing.assert_array_equal(reread.arrays[key], result.arrays[key])


def test_me1_order_units_decimal_and_fingerprint():
    mesh = renumber(line())
    doc = mesh.to_dict()
    doc["nodes"].reverse()
    doc["cells"][0]["id"] = Decimal(str(doc["cells"][0]["id"]))
    doc["sets"].reverse()
    reordered = validate_topology(doc)
    assert reordered == mesh
    assert canonical_input_hash(mesh.to_model(binding())) == canonical_input_hash(reordered.to_model(binding()))
    assert canonical_input_hash(line(unit="mm").to_model(binding())) == canonical_input_hash(line().to_model(binding()))
    raw = line().to_json().replace(b'"xy":[0,0]', b'"xy":[0.00000000000000000000000000000000001,0]')
    assert import_mesh(import_mesh(raw).to_json()) == import_mesh(raw)


def navier(modes):
    bending = 210e9 * .01**3 / (12 * (1 - .3**2))
    shear = (5 / 6) * 210e9 * .01 / (2 * (1 + .3))
    terms = []
    for m in range(1, modes + 1, 2):
        for n in range(1, modes + 1, 2):
            wave = (math.pi * m / 2)**2 + (math.pi * n)**2
            terms.append(16000 * (-1)**((m + n) // 2 - 1) / (math.pi**2 * m * n)
                         * (1 / (bending * wave**2) + 1 / (shear * wave)))
    return math.fsum(terms)


@pytest.mark.parametrize("formulation", ["mitc4_rectangle", "q4_full"])
def test_me1_me3_plate_projection_and_independent_reference(formulation):
    mesh = renumber(plate(moved=True))
    bind = plate_binding(formulation)
    result = solved(mesh, bind)
    reference = solved(plate(), bind)
    assert mesh.mapping["projection"]["jacobian_m2"] == 1 / 128
    assert canonical_input_hash(result.model) == canonical_input_hash(reference.model)
    for key in result.arrays:
        np.testing.assert_allclose(result.arrays[key], reference.arrays[key], rtol=1e-8, atol=1e-10)
    # Direct existing declaration is a separate mapping baseline.
    direct = validate_model({"schema_version": "1.0", "problem_type": "rectangular_plate",
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "model": {"lx": 2, "ly": 1, "nx": 8, "ny": 8, "young_modulus": 210e9,
                  "poisson_ratio": .3, "thickness": .01, "support": bind["support"], "formulation": formulation},
        "loads": {"pressure": 1000}, "analysis": {"type": "static"}})
    assert canonical_input_hash(direct) == result.input_hash
    if formulation == "mitc4_rectangle":
        coarse, fine = navier(127), navier(255)
        assert abs(coarse / fine - 1) <= 2e-7
        assert abs(result.arrays["deflection"][4, 4] / fine - 1) <= .01
    assert abs(result.arrays["reaction_force"].sum() / 2000 + 1) <= 1e-7
    weights = np.outer([1, *[2] * 7, 1], [1, *[2] * 7, 1]) * (2000 / (4 * 64))
    assert abs(2 * result.strain_energy / np.sum(weights * result.arrays["deflection"]) - 1) <= 1e-7
    for row in mesh.mapping["cells"]:
        iy, ix = divmod(row["result_index"], 8)
        assert row["model_connection"] == [[iy, ix], [iy, ix+1], [iy+1, ix+1], [iy+1, ix]]


@pytest.mark.parametrize("modify,code", [
    (lambda d: d.update(node_count=3), "count_mismatch"),
    (lambda d: d["nodes"][1].update(id=d["nodes"][0]["id"]), "duplicate_id"),
    (lambda d: d["nodes"][0].update(id=True), "invalid_type"),
    (lambda d: d["nodes"][0].update(id="1"), "invalid_type"),
    (lambda d: d["nodes"][0].update(id=2**53), "invalid_value"),
    (lambda d: d["nodes"][1].update(xy=[0, 0]), "duplicate_node"),
    (lambda d: d["cells"][0].update(kind="hex8"), "unsupported_element"),
    (lambda d: d["cells"][0].update(nodes=[1, 99]), "unknown_node"),
    (lambda d: d["cells"][0].update(nodes=[1, 1]), "degenerate_cell"),
    (lambda d: d["sets"][0].update(ids=[1, 1]), "duplicate_id"),
    (lambda d: d["sets"][0].update(ids=[99]), "unknown_entity"),
    (lambda d: d["sets"][0].update(name="../root"), "invalid_name"),
    (lambda d: d.update(length_unit="inch"), "unsupported_unit"),
    (lambda d: d.update(mesh_version="2.0"), "unsupported_mesh_version"),
    (lambda d: d.update(unused=3), "unknown_field"),
    (lambda d: d.update(provenance={"nested": [[[[[[[[[0]]]]]]]]]}), "size_limit"),
])
def test_me2_rejections(modify, code):
    doc = line().to_dict()
    modify(doc)
    with pytest.raises(ModelValidationError) as error:
        import_mesh(doc)
    assert error.value.code == code


def test_me2_duplicates_isolated_sets_and_binding_conflicts():
    doc = line().to_dict()
    doc["cells"].append({**doc["cells"][0], "id": 44, "nodes": [2, 1]})
    doc["cell_count"] = 2
    with pytest.raises(ModelValidationError, match="duplicate_cell"):
        import_mesh(doc)
    doc = line().to_dict()
    doc["node_count"] = 3
    doc["nodes"].append({"id": 7, "xy": [3, 0]})
    with pytest.raises(ModelValidationError, match="isolated_node"):
        import_mesh(doc)
    for key, code in (("properties", "unknown_property"), ("loads", "unknown_set")):
        bind = binding()
        if key == "properties":
            bind[key] = {}
        else:
            bind[key][0]["set"] = "missing"
        with pytest.raises(ModelValidationError, match=code):
            line().to_model(bind)
    bind = binding()
    bind["supports"].append({"set": "all_nodes", "values": {"ux": .01}})
    with pytest.raises(ModelValidationError, match="conflicting_assignment"):
        line().to_model(bind)


@pytest.mark.parametrize("mode", ["clockwise", "crossed", "sheared", "trapezoid", "nonuniform", "missing", "collapsed"])
def test_me3_reject_distorted_positive_jacobian_and_other_invalid_grids(mode):
    doc = plate(nx=2, ny=2).to_dict()
    if mode in ("clockwise", "crossed"):
        cell = doc["cells"][0]
        cell["nodes"] = [cell["nodes"][i] for i in ((0, 3, 2, 1) if mode == "clockwise" else (0, 2, 1, 3))]
    elif mode == "sheared":
        for node in doc["nodes"]:
            node["xy"][0] += node["xy"][1]
    elif mode == "trapezoid":
        for node in doc["nodes"]:
            node["xy"][0] *= 1 + node["xy"][1] / 10
    elif mode == "nonuniform":
        doc["nodes"][1]["xy"][0] += Decimal('.01')
    elif mode == "missing":
        doc["cells"].pop()
        doc["cell_count"] -= 1
        doc["sets"] = []
    else:
        doc["nodes"][1]["xy"] = doc["nodes"][0]["xy"]
    with pytest.raises(ModelValidationError):
        import_mesh(doc)


def test_me3_cycle_equivalence_and_geometry_budget():
    mesh = plate(nx=2, ny=2)
    doc = mesh.to_dict()
    for cell in doc["cells"]:
        cell["nodes"] = cell["nodes"][1:] + cell["nodes"][:1]
    near_model = import_mesh(doc).to_model(plate_binding())
    assert canonical_input_hash(near_model) == canonical_input_hash(mesh.to_model(plate_binding()))
    doc["nodes"][4]["xy"][0] += Decimal('1e-12')
    assert import_mesh(doc).mapping["projection"]["max_correction_m"] <= 5e-11
    doc["nodes"][4]["xy"][0] += Decimal('1e-8')
    with pytest.raises(ModelValidationError, match="invalid_rectangle"):
        import_mesh(doc)
    doc = mesh.to_dict()
    doc["grid"]["v_corner"] = 8  # Top middle is not the declared perpendicular corner.
    with pytest.raises(ModelValidationError, match="invalid_rectangle"):
        import_mesh(doc)


def test_me4_bounded_input_and_generation(tmp_path):
    raw = line().to_json()
    with pytest.raises(ModelValidationError, match="duplicate_field"):
        import_mesh(raw.replace(b'"mesh_version":"1.0"', b'"mesh_version":"1.0","mesh_version":"1.0"'))
    with pytest.raises(ModelValidationError, match="invalid_json"):
        import_mesh(b'\xff')
    for blob in (b'PK\x03\x04', b'\x1f\x8b', b'BZh', b'\xfd7zXZ'):
        with pytest.raises(ModelValidationError, match="unsupported_format"):
            import_mesh(blob)
    with pytest.raises(ModelValidationError, match="unsupported_format"):
        import_mesh(Path("unknown.msh"))
    with pytest.raises(ModelValidationError, match="size_limit"):
        import_mesh(raw + b' ' * (1024 * 1024))
    path = tmp_path / "large.json"
    path.write_bytes(raw + b' ' * (1024 * 1024))
    with pytest.raises(ArtifactError):
        import_mesh(path)
    for parameters in ({"generator": "gmsh"}, {**plate().to_dict()["provenance"]["parameters"], "nx": 199, "ny": 199}):
        with pytest.raises(ModelValidationError):
            generate_supported_mesh(parameters)
    for mesh in (line(), plate()):
        provenance = mesh.to_dict()["provenance"]
        assert generate_supported_mesh(provenance["parameters"], source_sha256=provenance["source_sha256"]) == mesh
    assert len(plate(nx=2, ny=199).mapping["nodes"]) == 600


def test_coordinate_underflow_overflow_and_decimal_context():
    doc = line().to_dict()
    for value in (Decimal('1e-400'), Decimal('1e309'), float('nan')):
        doc["nodes"][1]["xy"][0] = value
        with pytest.raises(ModelValidationError):
            import_mesh(doc)
    parameters = line().to_dict()["provenance"]["parameters"]
    with localcontext() as context:
        context.prec = 6
        assert generate_supported_mesh(parameters) == line()


def test_me1_arbitrary_rotation_and_mixed_rotation_mapping():
    angle = .37
    c, s = math.cos(angle), math.sin(angle)
    mesh = generate_supported_mesh({"generator": "line", "length_unit": "m", "property": "steel",
        "start": [3, -2], "end": [3 + 2*c, -2 + 2*s], "segments": 1, "kind": "frame2d"})
    bind = binding()
    bind["loads"][0]["values"] = {"fx": -1000*s, "fy": 1000*c}
    result = solved(mesh, bind)
    near(result.arrays["node_ux"][1], -s * 4/315, 1e-10)
    near(result.arrays["node_uy"][1], c * 4/315, 1e-10)
    near(result.strain_energy, 400/63, 1e-10)
    doc = line().to_dict()
    doc["nodes"].append({"id": 3, "xy": [-1, 0]})
    doc["cells"].append({"id": 2, "kind": "truss2d", "nodes": [3, 1], "property": "rod"})
    doc.update(node_count=3, cell_count=2, provenance=None)
    mixed = import_mesh(doc)
    rows = mixed.mapping["nodes"]
    assert rows[2]["frame_result_index"] is None and rows[2]["result_index"] == 2


def test_me3_q4_thin_locking_failure_survives_mesh_projection():
    parameters = plate().to_dict()["provenance"]["parameters"]
    parameters.update(u=[1, 0], v=[0, 1])
    mesh = generate_supported_mesh(parameters)
    bind = plate_binding("q4_full")
    bind["properties"]["plate"].update(thickness=.005, young_modulus=70e9)
    result = run_analysis(mesh.to_model(bind), {"reference": "plate.thin-q4-navier-31.v1"})
    assert result.solve_status == "solved"
    assert result.verification_status == "verification_failed"


def test_me2_set_and_entity_limits_and_generator_precision():
    doc = line().to_dict()
    doc["sets"] = [{"name": f"set{i}", "kind": "node", "ids": [1]} for i in range(65)]
    with pytest.raises(ModelValidationError, match="size_limit"):
        import_mesh(doc)
    mesh = generate_supported_mesh({"generator": "line", "length_unit": "m", "property": "rod",
        "start": [0, 0], "end": [2, 0], "segments": 199, "kind": "truss2d"})
    doc = mesh.to_dict()
    doc["sets"] = [{"name": f"set{i}", "kind": "node", "ids": list(range(1, 201))} for i in range(21)]
    with pytest.raises(ModelValidationError, match="size_limit"):
        import_mesh(doc)
    parameters = line().to_dict()["provenance"]["parameters"]
    parameters.update(start=[Decimal('123456.123456789'), 0], end=[Decimal('123458.123456789'), 0], segments=3)
    baseline = generate_supported_mesh(parameters)
    with localcontext() as context:
        context.prec = 6
        assert generate_supported_mesh(parameters) == baseline


def test_me1_optional_sparse_uses_same_native_projection():
    pytest.importorskip("scipy")
    for mesh, bind in ((renumber(line()), binding()), (plate(), plate_binding())):
        dense, sparse = (solved(mesh, bind, backend) for backend in ("dense", "scipy"))
        assert dense.input_hash == sparse.input_hash
        for key in dense.arrays:
            np.testing.assert_allclose(dense.arrays[key], sparse.arrays[key], rtol=1e-8, atol=1e-10)
