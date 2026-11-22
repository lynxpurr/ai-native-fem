"""Measure the installed static solvers in fresh, single-threaded processes."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch


THREADS = dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"), "1")
SCALES = {"beam": (8, 32, 64, 128, 200), "plate": (4, 8, 16, 23), "frame": (12, 60, 120, 199)}


def declaration(kind, size):
    doc = {"schema_version": "1.0", "units": {"length": "m", "force": "N", "stress": "Pa"},
           "analysis": {"type": "static"}}
    if kind == "beam":
        doc.update(problem_type="cantilever_bending", loads={"tip_force": 1000},
                   model={"length": 2, "young_modulus": 210e9, "second_moment": 1e-6, "elements": size})
    elif kind == "plate":
        doc.update(problem_type="rectangular_plate", loads={"pressure": 800},
                   model={"lx": 1, "ly": 1, "thickness": 0.1, "young_modulus": 70e9,
                          "poisson_ratio": 0.3, "nx": size, "ny": size, "support": "hard_simply_supported"})
    else:
        names = [f"n{i:03d}" for i in range(size + 1)]
        rotations = [0.0, *(0.001 * (-1)**i for i in range(1, size)), 0.0]
        nodes = {name: [2 * i, 0] for i, name in enumerate(names)}
        supports = {name: {"ux": 0, "uy": 0} for name in names}
        supports[names[0]]["rz"] = supports[names[-1]]["rz"] = 0
        members = {f"m{i:03d}": {"kind": "frame2d", "i": left, "j": right,
                                "young_modulus": 200e9, "area": 0.01, "second_moment": 1e-6}
                   for i, (left, right) in enumerate(zip(names, names[1:]))}
        # For fixed translations, neighboring members supply 4EI/L and 2EI/L
        # moment coefficients to the manufactured joint rotations.
        nodal = {names[i]: {"mz": 1e5 * (2*rotations[i-1] + 8*rotations[i] + 2*rotations[i+1])}
                 for i in range(1, size)}
        doc.update(schema_version="1.1", problem_type="planar_structure",
                   model={"nodes": nodes, "members": members, "supports": supports}, loads={"nodal": nodal})
    return doc


def peak_bytes():
    if sys.platform != "win32":
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak if sys.platform == "darwin" else peak * 1024
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("faults", wintypes.DWORD),
                    ("peak", ctypes.c_size_t), ("remaining", ctypes.c_size_t * 7)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    get_memory = kernel.K32GetProcessMemoryInfo
    get_memory.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
    get_memory.restype = wintypes.BOOL
    counters = Counters()
    counters.size = ctypes.sizeof(counters)
    if not get_memory(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.size):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.peak


def sample(kind, size, backend):
    import numpy as np
    from anf import structures
    from anf.application import run_analysis
    from anf.assembly import DenseAssembly

    doc = declaration(kind, size)
    timings = {"solve_seconds": 0.0}
    start_assembly, end_assembly = DenseAssembly.__init__, DenseAssembly.finish

    def begin(assembly, *args):
        timings["assembly_start"] = perf_counter()
        start_assembly(assembly, *args)

    def finish(assembly):
        matrix = end_assembly(assembly)
        timings["assembly_seconds"] = perf_counter() - timings.pop("assembly_start")
        timings.update(dofs=matrix.shape[0], stiffness_nnz=int(np.count_nonzero(matrix)))
        return matrix

    def timed_solve(solve):
        def measured(*args):
            start = perf_counter()
            try:
                return solve(*args)
            finally:
                timings["solve_seconds"] = perf_counter() - start
        return measured

    with ExitStack() as stack:
        if backend == "dense":
            stack.enter_context(patch.object(DenseAssembly, "__init__", begin))
            stack.enter_context(patch.object(DenseAssembly, "finish", finish))
            solver, name = (structures, "_equilibrate") if kind == "frame" else (np.linalg, "solve")
            stack.enter_context(patch.object(solver, name, timed_solve(getattr(solver, name))))
        start = perf_counter()
        result = run_analysis(doc, {"backend": backend})
        elapsed = perf_counter() - start
    peak = peak_bytes()
    metrics = dict(result.execution) if result.execution is not None else timings
    reference = {}
    if result.solve_status == "solved":
        arrays = result.arrays
        if kind == "frame":
            expected = [0.0, *(0.001 * (-1)**i for i in range(1, size)), 0.0]
            reference["node_rz"] = error_ratio(arrays["node_rz"], expected, 1e-10)
        elif kind == "plate":
            reference["vertical_reaction"] = error_ratio(arrays["reaction_force"].sum(), -800, 1e-6)
        else:
            reference["tip_deflection"] = error_ratio(arrays["deflection"][-1], 8000 / (3 * 210000), 1e-10)
            reference["strain_energy"] = error_ratio(result.strain_energy, 4e6 / (3 * 210000), 1e-10)
    return dict(metrics, backend=backend, kind=kind, size=size, peak_bytes=peak, total_seconds=elapsed,
                numpy=np.__version__, status=result.solve_status, diagnostics=result.diagnostics,
                residual=result.relative_residual, energy=result.strain_energy,
                reference_error_over_tolerance=reference,
                arrays={name: array.tolist() for name, array in result.arrays.items()})


def error_ratio(actual, expected, absolute, *, reference=True):
    import numpy as np

    expected = np.asarray(expected)
    relative = 1e-8 * np.abs(expected)
    tolerance = np.where(expected == 0, absolute, relative) if reference else absolute + relative
    return float(np.max(np.abs(np.asarray(actual) - expected) / tolerance, initial=0))


def compare(dense, sparse):
    if dense["status"] != "solved" or sparse["status"] != "solved":
        return None
    assert dense["arrays"].keys() == sparse["arrays"].keys()
    errors = {}
    for key, array in dense["arrays"].items():
        force = key.startswith(("reaction", "root_force", "root_moment", "member_end"))
        errors[key] = error_ratio(sparse["arrays"][key], array, 1e-6 if force else 1e-10,
                                  reference=False)
    errors["energy"] = error_ratio(sparse["energy"], dense["energy"], 1e-10, reference=False)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", nargs=3, metavar=("KIND", "SIZE", "BACKEND"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    os.environ.update(THREADS)
    if args.worker:
        kind, size, backend = args.worker
        print(json.dumps(sample(kind, int(size), backend), allow_nan=False))
        return
    samples = []
    for kind, values in SCALES.items():
        for size in values:
            pair = {"kind": kind, "size": size, "passed_comparisons": 0,
                    "parity_error_over_tolerance": [], "samples": []}
            for _ in range(args.repeats):
                rows = [
                    json.loads(subprocess.check_output(
                        [sys.executable, str(Path(__file__).resolve()), "--worker", kind, str(size), backend],
                        text=True, timeout=60,
                    ))
                    for backend in ("dense", "scipy")
                ]
                parity = compare(*rows)
                pair["parity_error_over_tolerance"].append(parity)
                if parity is not None:
                    measurements = [*parity.values()]
                    for row in rows:
                        measurements.extend(row["reference_error_over_tolerance"].values())
                    pair["passed_comparisons"] += all(value <= 1 for value in measurements)
                for row in rows:
                    row.pop("arrays")
                    pair["samples"].append(row)
            for backend in ("dense", "scipy"):
                rows = [row for row in pair["samples"] if row["backend"] == backend]
                pair[backend] = {field: median(row[field] for row in rows)
                                 for field in ("assembly_seconds", "solve_seconds", "total_seconds")}
                pair[backend]["peak_bytes"] = max(row["peak_bytes"] for row in rows)
            samples.append(pair)
    print(json.dumps({"platform": platform.platform(), "processor": platform.processor(),
                      "python": platform.python_version(), "threads": THREADS, "runs": samples},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
