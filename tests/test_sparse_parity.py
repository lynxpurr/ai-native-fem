"""Installed-package checks for the optional static sparse backend."""

import json
import sys

import numpy as np
import pytest

import test_frame2d as frames
from anf import application, structures
from anf.artifacts import load_result, save_result
from anf.assembly import DenseAssembly
from anf.model import ModelValidationError


run_analysis = application.run_analysis


@pytest.fixture
def sparse(monkeypatch):
    scipy = pytest.importorskip("scipy")
    from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

    def no_dense(*args, **kwargs):
        pytest.fail("the sparse path attempted to build a dense global matrix")

    def solve(doc, options=None):
        with monkeypatch.context() as guard:
            guard.setattr(DenseAssembly, "__init__", no_dense)
            for matrix in (coo_matrix, csc_matrix, csr_matrix):
                guard.setattr(matrix, "toarray", no_dense)
                guard.setattr(matrix, "todense", no_dense)
            result = run_analysis(doc, {**(options or {}), "backend": "scipy"})
        assert result.execution["backend"] == "scipy"
        assert result.execution["scipy"] == scipy.__version__
        assert result.execution["stiffness_nnz"] <= result.execution["dofs"]**2
        if result.solve_status == "solved":
            assert result.relative_residual <= 1e-8
            assert all(np.isfinite(result.execution[key]) and result.execution[key] >= 0
                       for key in ("absolute_residual_n", "backward_error", "assembly_seconds", "solve_seconds"))
        return result

    return solve


def assert_parity(dense, sparse_result):
    assert sparse_result.input_hash == dense.input_hash
    assert sparse_result.ordering == dense.ordering
    assert sparse_result.verification_status == dense.verification_status
    assert sparse_result.arrays.keys() == dense.arrays.keys()
    for key, values in dense.arrays.items():
        force = key.startswith(("reaction", "root_force", "root_moment", "member_end"))
        np.testing.assert_allclose(sparse_result.arrays[key], values, rtol=1e-8,
                                   atol=1e-6 if force else 1e-10)
    np.testing.assert_allclose(sparse_result.strain_energy, dense.strain_energy, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("name,formulation,width,reference,status", [
    ("cantilever", None, None, "cantilever.closed-form.v1", "verification_passed"),
    ("thick-q4", None, None, "plate.navier-31.v1", "verification_passed"),
    ("thin-q4", None, None, "plate.thin-q4-navier-31.v1", "verification_failed"),
    ("thin-q4", "mitc4_rectangle", 1, None, "solved_unverified"),
    ("thin-q4", "mitc4_rectangle", 2, None, "solved_unverified"),
])
def test_classical_references(name, formulation, width, reference, status, sparse, tmp_path):
    doc = json.loads((frames.EXAMPLES / "models" / f"{name}.json").read_bytes())
    if formulation:
        doc["model"].update(formulation=formulation, lx=width)
    options = {"reference": reference}
    result = sparse(doc, options)
    assert_parity(run_analysis(doc, options), result)
    assert result.verification_status == status
    readback = load_result(save_result(result, tmp_path, "sparse"))
    assert readback.report == result.report
    assert readback.execution is None
    for key, values in result.arrays.items():
        np.testing.assert_array_equal(readback.arrays[key], values)


@pytest.mark.parametrize("case,args", [
    *((frames.test_f1_axial_chain, (n,)) for n in (1, 2, 4, 8)),
    *((frames.test_f1_cantilever, (n,)) for n in (1, 2, 4, 8)),
    *((frames.test_f1_f3_uniform_load, (n, False)) for n in (1, 2, 4, 8)),
    (frames.test_f1_f3_uniform_load, (2, True)),
    (frames.test_f2_portal_force_method, ()),
    (frames.test_f2_shared_translation, ()),
    (frames.test_f3_cantilever_covariance, ("rotate_translate_rename",)),
    (frames.test_f3_cantilever_covariance, ("reverse",)),
], ids=lambda item: item.__name__ if callable(item) else None)
def test_f1_to_f3(case, args, sparse, monkeypatch):
    def compare(doc, options):
        result = sparse(doc, options)
        assert_parity(run_analysis(doc, options), result)
        return result

    monkeypatch.setattr(frames.application, "run_analysis", compare)
    case(*args)


@pytest.mark.parametrize("mode", ["free", "fixed", "zero"])
def test_prescribed_displacement(mode, sparse, monkeypatch):
    from anf.backends import scipy

    solve_linear = scipy.solve_linear

    def check_effective_load(matrix, rhs, *, structural):
        assert structural
        np.testing.assert_allclose(rhs, [2000 if mode == "free" else 0], rtol=1e-8, atol=1e-6)
        return solve_linear(matrix, rhs, structural=structural)

    def compare(doc, options):
        result = sparse(doc, options)
        assert_parity(run_analysis(doc, options), result)
        return result

    monkeypatch.setattr(scipy, "solve_linear", check_effective_load)
    monkeypatch.setattr(frames.application, "run_analysis", compare)
    frames.test_f4_prescribed_displacement(mode, monkeypatch)


@pytest.mark.parametrize("point,reason", [([4, 0], "diagonal"), ([3, 4], "shifted stiffness")])
def test_mechanisms_and_failure_artifacts(point, reason, sparse, tmp_path):
    result = sparse(frames.mechanism(point))
    assert (result.solve_status, result.verification_status) == ("numerical_failure", "not_run")
    assert result.diagnostics[0]["code"] == "unstable_or_ill_conditioned"
    assert reason in result.diagnostics[0]["message"]
    path = save_result(result, tmp_path, "mechanism")
    assert {file.name for file in path.iterdir()} == {"manifest.json"}
    assert load_result(path).diagnostics == result.diagnostics


def test_nearly_parallel_members(sparse):
    nodes = {"a": [1, 1], "b": [2, 2.0000004], "c": [0, 0]}
    members = {node: frames.bar(node, "c") for node in ("a", "b")}
    supports = {node: {"ux": 0, "uy": 0} for node in ("a", "b")}
    doc = frames.declaration(nodes, members, supports, nodal={"c": {"fx": 1, "fy": 1}})
    result = sparse(doc)
    assert result.diagnostics[0]["code"] == "unstable_or_ill_conditioned"
    assert "shifted stiffness" in result.diagnostics[0]["message"]


def test_work_gate_rejects_disagreement():
    with pytest.raises(ArithmeticError, match="work and strain energy disagree"):
        structures._check_sparse_work(np.array([1.0]), np.array([1.0]), np.array([1.01]),
                                      np.array([0.0]), [])


def test_duplicate_triplets_cancel_in_canonical_csr(sparse):
    from anf.backends.scipy import CooAssembly

    assembly = CooAssembly(5, 13)
    assembly.add(np.array([4, 1, 0]), np.ones((3, 3)))
    assembly.add(np.array([1, 4]), -np.ones((2, 2)))
    matrix = assembly.finish()
    assert matrix.has_canonical_format
    assert matrix.nnz == 5 and matrix[1, 4] == 0
    assert matrix.shape == (5, 5)
    np.testing.assert_array_equal(matrix @ np.arange(5), [5, 0, 0, 0, 0])


def test_backend_selection_and_missing_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "anf.backends.scipy", None)
    doc = frames.cantilever()
    assert run_analysis(doc).solve_status == "solved"
    assert run_analysis(doc, {"backend": "dense"}).execution is None
    with pytest.raises(ModelValidationError, match="/options/backend: backend_unavailable"):
        run_analysis(doc, {"backend": "scipy"})
    with pytest.raises(ModelValidationError, match="/options/backend: unsupported_backend"):
        run_analysis(doc, {"backend": "other"})


@pytest.mark.parametrize("name", ["load-cases", "frame-joints"])
def test_batch_models_remain_on_the_batch_entry_point(name):
    doc = (frames.EXAMPLES / "structures" / f"{name}.json").read_bytes()
    with pytest.raises(ModelValidationError, match="/schema_version: unsupported_schema"):
        run_analysis(doc, {"backend": "scipy"})
    assert application.run_load_cases(doc).results
