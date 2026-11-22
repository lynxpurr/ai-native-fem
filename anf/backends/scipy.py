"""COO assembly and sparse direct static solves using SciPy and SuperLU."""

from time import perf_counter

import numpy as np
import scipy
from scipy.sparse import coo_matrix, diags, eye
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import splu

from .._linear import MAX_RELATIVE_RESIDUAL, UnstableStiffnessError, relative_residual


class CooAssembly:
    """Collect bounded element triplets in the model's canonical order."""

    def __init__(self, size, entry_limit):
        self.size = size
        self.rows = np.empty(entry_limit, dtype=np.int32)
        self.columns = np.empty(entry_limit, dtype=np.int32)
        self.values = np.empty(entry_limit)
        self.used = 0

    def add(self, dofs, stiffness):
        width = len(dofs)
        stop = self.used + width**2
        if stop > len(self.values):
            raise ArithmeticError("COO assembly exceeds its element contribution budget")
        span = slice(self.used, stop)
        self.rows[span] = np.repeat(dofs, width)
        self.columns[span] = np.tile(dofs, width)
        self.values[span] = stiffness.ravel()
        self.used = stop

    def finish(self):
        span = slice(self.used)
        stiffness = coo_matrix(
            (self.values[span], (self.rows[span], self.columns[span])),
            shape=(self.size, self.size),
        )
        stiffness.sum_duplicates()
        stiffness = stiffness.tocsr()
        stiffness.eliminate_zeros()
        stiffness.sort_indices()
        if not np.isfinite(stiffness.data).all():
            raise ArithmeticError("non-finite assembled stiffness")
        return stiffness


def _equilibrated(matrix):
    diagonal = matrix.diagonal()
    if np.any(diagonal <= 0):
        raise UnstableStiffnessError("non-positive free stiffness diagonal")
    scales = 1 / np.sqrt(diagonal)
    scaling = diags(scales, format="csr")
    balanced = (scaling @ matrix) @ scaling
    if not np.isfinite(balanced.data).all():
        raise ArithmeticError("non-finite equilibrated free stiffness")
    return balanced, scales


def _check_stability(matrix):
    # The infinity norm bounds the largest eigenvalue of the symmetric part.
    # Check the roundoff-level skew before using that part for the diagnostic.
    row_bound = float(np.asarray(abs(matrix).sum(axis=1)).max())
    if not np.isfinite(row_bound):
        raise ArithmeticError("non-finite stiffness spectral bound")
    epsilon = np.finfo(float).eps
    skew = matrix - matrix.T
    if np.max(np.abs(skew.data), initial=0) > 8 * epsilon * row_bound:
        raise ArithmeticError("equilibrated free stiffness lost symmetry")
    symmetric = matrix * 0.5 + matrix.T * 0.5
    bound = float(np.asarray(abs(symmetric).sum(axis=1)).max())
    pivot_limit = 8 * epsilon * bound
    order = reverse_cuthill_mckee(symmetric, symmetric_mode=True)
    size = matrix.shape[0]
    shift = 64 * epsilon * size * bound
    shifted = symmetric[order, :][:, order].tocsc() - shift * eye(size, format="csc")
    try:
        factor = splu(shifted, permc_spec="NATURAL", diag_pivot_thresh=0,
                      options={"Equil": False, "SymmetricMode": True})
    except RuntimeError as error:
        raise UnstableStiffnessError("shifted stiffness factorization failed") from error
    indices = np.arange(size)
    pivots = factor.U.diagonal()
    if (not np.array_equal(factor.perm_r, indices) or not np.array_equal(factor.perm_c, indices)
            or not np.isfinite(pivots).all() or np.any(pivots <= pivot_limit)):
        raise UnstableStiffnessError("shifted stiffness lacks sufficiently positive diagonal pivots")


def solve_linear(matrix, rhs, *, structural):
    """Solve a free subproblem, applying the structural stability gate when requested."""
    if not np.isfinite(matrix.data).all() or not np.isfinite(rhs).all():
        raise ArithmeticError("non-finite free stiffness or effective load")
    if structural:
        matrix, scales = _equilibrated(matrix)
        _check_stability(matrix)
        rhs = scales * rhs
    try:
        displacement = splu(matrix.tocsc()).solve(rhs)
    except RuntimeError as error:
        raise UnstableStiffnessError("sparse free stiffness factorization failed") from error
    if structural:
        displacement = scales * displacement
    if not np.isfinite(displacement).all():
        raise ArithmeticError("non-finite sparse displacement")
    return displacement


class SparseSolve:
    """One sparse execution and its in-memory numerical measurements."""

    def __init__(self):
        self.metrics = {"backend": "scipy", "scipy": scipy.__version__}

    def assembly(self, size, entry_limit):
        self._assembly_start = perf_counter()
        return CooAssembly(size, entry_limit)

    def assembled(self, assembly):
        matrix = assembly.finish()
        self.metrics.update(dofs=matrix.shape[0], stiffness_nnz=matrix.nnz,
                            assembly_seconds=perf_counter() - self._assembly_start)
        return matrix

    def solve(self, matrix, rhs, *, structural):
        start = perf_counter()
        try:
            return solve_linear(matrix, rhs, structural=structural)
        finally:
            self.metrics["solve_seconds"] = perf_counter() - start

    def measure_balance(self, matrix, gap, rhs, displacement, rotational, length):
        if not np.any(gap):
            self.metrics.update(absolute_residual_n=0.0, backward_error=0.0)
            return
        weights = np.ones(rhs.size)
        column_scale = np.ones(rhs.size)
        weights[rotational] = 1 / length
        column_scale[rotational] = length
        scaled = (diags(weights) @ matrix) @ diags(1 / column_scale)
        row_norm = float(np.asarray(abs(scaled).sum(axis=1)).max())
        absolute = float(np.max(np.abs(weights * gap)))
        denominator = (row_norm * np.max(np.abs(column_scale * displacement))
                       + np.max(np.abs(weights * rhs)))
        if not np.isfinite([absolute, denominator]).all() or denominator <= 0:
            raise ArithmeticError("non-finite scaled sparse balance measurements")
        self.metrics.update(absolute_residual_n=absolute, backward_error=float(absolute / denominator))

    def zero_balance(self):
        self.metrics.update(solve_seconds=0.0, absolute_residual_n=0.0, backward_error=0.0)

    def eliminate_classical(self, stiffness, force, fixed, *, stride, length):
        fixed = np.asarray(fixed, dtype=int)
        free = np.setdiff1d(np.arange(force.size), fixed)
        displacement = np.zeros(force.size)
        matrix = stiffness[free, :][:, free]
        rhs = force[free]
        displacement[free] = self.solve(matrix, rhs, structural=False)
        imbalance = stiffness @ displacement - force
        if not np.isfinite(imbalance).all():
            raise ArithmeticError("non-finite sparse force balance")
        gap = imbalance[free]
        residual = relative_residual(gap, rhs)
        if not np.isfinite(residual) or residual > MAX_RELATIVE_RESIDUAL:
            raise ArithmeticError("free-DOF relative residual exceeds 1e-8")
        self.measure_balance(matrix, gap, rhs, displacement[free], free % stride != 0, length)
        reactions = np.zeros_like(force)
        reactions[fixed] = imbalance[fixed]
        return displacement, reactions, residual
