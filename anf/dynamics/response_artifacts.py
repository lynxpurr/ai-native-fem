"""Versioned harmonic response artifacts for schema 1.5."""

from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import struct
from zipfile import BadZipFile
import zlib

import numpy as np

from ..artifacts import _HEADER_LIMIT, _read_arrays, _read_regular, _require_directory
from ..model import _STATE_LIMIT, canonical_dynamics_input_hash, validate_dynamics_model
from .response import HarmonicResult, TransientResult
from .system import dof_labels

_JSON_LIMIT = 1024 * 1024
_ARRAY_LIMIT = 64 * 1024 * 1024
_MANIFEST = "manifest.json"
_ARRAYS = "arrays.npz"
_FIELDS = ("frequencies_hz", "omega_rad_s", "displacements", "magnitudes",
           "phases_deg", "backward_errors", "conditions")


class HarmonicArtifactError(ValueError):
    """Invalid or incomplete harmonic artifact."""


def _regular(path, limit):
    if path.is_symlink() or not path.is_file():
        raise HarmonicArtifactError("expected a regular file")
    size = path.stat().st_size
    if size > limit:
        raise HarmonicArtifactError(f"file exceeds {limit} bytes")
    return path.read_bytes()


def _safe_name(name):
    if (not isinstance(name, str) or not name or name in (".", "..")
            or "/" in name or "\\" in name):
        raise HarmonicArtifactError("invalid artifact name")
    return name


def _payload(result):
    arrays = {
        "frequencies_hz": np.asarray(result.frequencies_hz, dtype=float),
        "omega_rad_s": np.asarray(result.omega_rad_s, dtype=float),
        "displacements": np.asarray(result.displacements, dtype=complex),
        "magnitudes": np.asarray(result.magnitudes, dtype=float),
        "phases_deg": np.asarray(result.phases_deg, dtype=float),
        "backward_errors": np.asarray(result.backward_errors, dtype=float),
        "conditions": np.asarray(result.conditions, dtype=float),
    }
    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    data = stream.getvalue()
    manifest = {
        "schema_version": "1.5",
        "kind": "harmonic_result",
        "model": result.model.to_dict(),
        "input_hash": result.input_hash,
        "frequency_count": len(result.frequencies_hz),
        "diagnostics": list(result.diagnostics),
        "versions": dict(result.versions),
        "dof_labels": [list(label) for label in result.dof_labels],
        "arrays": {
            name: {"shape": list(array.shape), "dtype": str(array.dtype)}
            for name, array in arrays.items()
        },
        "arrays_sha256": sha256(data).hexdigest(),
    }
    return manifest, data


def save_harmonic_result(result, root, name):
    """Create root/name and publish the harmonic manifest last."""
    if not isinstance(result, HarmonicResult):
        raise TypeError("expected a HarmonicResult")
    _safe_name(name)
    try:
        root = Path(root).resolve(strict=True)
        if not root.is_dir():
            raise HarmonicArtifactError("artifact root must be a directory")
        directory = root / name
        directory.mkdir(mode=0o700)
        manifest, data = _payload(result)
        (directory / _ARRAYS).write_bytes(data)
        text = (json.dumps(manifest, allow_nan=False, indent=2) + "\n").encode("utf-8")
        if len(text) > _JSON_LIMIT:
            raise HarmonicArtifactError("harmonic manifest exceeds 1 MiB")
        (directory / _MANIFEST).write_bytes(text)
        return directory
    except HarmonicArtifactError:
        raise
    except (OSError, ValueError) as error:
        raise HarmonicArtifactError(f"could not save harmonic result: {error}") from error


def _manifest(directory):
    data = _regular(directory / _MANIFEST, _JSON_LIMIT)
    try:
        manifest = json.loads(data.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite constant {value}")))
    except (UnicodeError, ValueError) as error:
        raise HarmonicArtifactError("invalid harmonic manifest") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "1.5":
        raise HarmonicArtifactError("unsupported harmonic result schema")
    required = ("schema_version", "kind", "model", "input_hash", "frequency_count",
                "diagnostics", "versions", "dof_labels", "arrays", "arrays_sha256")
    if set(manifest) != set(required) or manifest.get("kind") != "harmonic_result":
        raise HarmonicArtifactError("incomplete harmonic manifest")
    return manifest


def _arrays(directory, manifest, labels):
    data = _regular(directory / _ARRAYS, _ARRAY_LIMIT)
    if sha256(data).hexdigest() != manifest["arrays_sha256"]:
        raise HarmonicArtifactError("array digest does not match the manifest")
    try:
        with np.load(BytesIO(data), allow_pickle=False) as archive:
            if set(archive.files) != set(_FIELDS):
                raise HarmonicArtifactError("unexpected harmonic array set")
            arrays = {name: np.asarray(archive[name]) for name in _FIELDS}
    except (OSError, ValueError, EOFError) as error:
        raise HarmonicArtifactError(f"invalid harmonic arrays: {error}") from error
    count = manifest["frequency_count"]
    if not isinstance(count, int) or not 1 <= count <= 200:
        raise HarmonicArtifactError("invalid frequency count")
    expected = (len(labels), count)
    for name in ("displacements", "magnitudes", "phases_deg"):
        if arrays[name].shape != expected:
            raise HarmonicArtifactError("response dimensions do not match the DOF labels")
    for name in ("frequencies_hz", "omega_rad_s", "backward_errors", "conditions"):
        if arrays[name].shape != (count,):
            raise HarmonicArtifactError("per-frequency arrays must share the frequency count")
    for name, array in arrays.items():
        metadata = manifest["arrays"].get(name)
        if not isinstance(metadata, dict) or metadata.get("dtype") != str(array.dtype):
            raise HarmonicArtifactError("missing harmonic array metadata")
        if list(array.shape) != metadata.get("shape"):
            raise HarmonicArtifactError("harmonic array shape does not match the manifest")
        if not np.isfinite(array).all():
            raise HarmonicArtifactError("harmonic arrays must be finite")
    return arrays


def load_harmonic_result(directory):
    """Load and validate a complete harmonic artifact."""
    directory = Path(directory)
    try:
        if not directory.is_dir():
            raise HarmonicArtifactError("expected an artifact directory")
        if {entry.name for entry in directory.iterdir()} != {_MANIFEST, _ARRAYS}:
            raise HarmonicArtifactError("unexpected harmonic artifact entries")
        manifest = _manifest(directory)
        model = validate_dynamics_model(manifest["model"])
        if model.analysis_type != "harmonic":
            raise HarmonicArtifactError("artifact model is not a harmonic declaration")
        if manifest["input_hash"] != canonical_dynamics_input_hash(model):
            raise HarmonicArtifactError("input hash does not match the model")
        labels = manifest["dof_labels"]
        if (not isinstance(labels, list) or not all(
                isinstance(label, list) and 2 <= len(label) <= 4
                and isinstance(label[0], str)
                and all(isinstance(item, (str, int)) and not isinstance(item, bool)
                        for item in label[1:]) for label in labels)):
            raise HarmonicArtifactError("invalid DOF labels")
        arrays = _arrays(directory, manifest, labels)
        versions = manifest["versions"]
        if (not isinstance(versions, dict) or not versions
                or not all(isinstance(key, str) and isinstance(value, str)
                           for key, value in versions.items())):
            raise HarmonicArtifactError("invalid version record")
        diagnostics = manifest["diagnostics"]
        if not isinstance(diagnostics, list) or not all(
                isinstance(item, str) for item in diagnostics):
            raise HarmonicArtifactError("invalid diagnostics")
        return HarmonicResult(
            model=model,
            frequencies_hz=arrays["frequencies_hz"],
            omega_rad_s=arrays["omega_rad_s"],
            displacements=arrays["displacements"],
            magnitudes=arrays["magnitudes"],
            phases_deg=arrays["phases_deg"],
            dof_labels=tuple(tuple(label) for label in labels),
            backward_errors=arrays["backward_errors"],
            conditions=arrays["conditions"],
            diagnostics=tuple(diagnostics),
            versions=dict(versions),
            input_hash=manifest["input_hash"],
        )
    except HarmonicArtifactError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise HarmonicArtifactError(f"invalid harmonic result artifact: {error}") from error


__all__ = ["HarmonicArtifactError", "TransientArtifactError",
           "load_harmonic_result", "load_transient_result",
           "save_harmonic_result", "save_transient_result"]

_TRANSIENT_FIELDS = ("times_s", "displacement", "velocity", "acceleration", "energy",
                     "peak_displacement", "peak_time_s")


class TransientArtifactError(ValueError):
    """Invalid or incomplete transient artifact."""


def _transient_payload(result):
    arrays = {
        "times_s": np.asarray(result.times_s, dtype=float),
        "displacement": np.asarray(result.displacement, dtype=float),
        "velocity": np.asarray(result.velocity, dtype=float),
        "acceleration": np.asarray(result.acceleration, dtype=float),
        "energy": np.asarray(result.energy, dtype=float),
        "peak_displacement": np.asarray(result.peak_displacement, dtype=float),
        "peak_time_s": np.asarray(result.peak_time_s, dtype=float),
    }
    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    data = stream.getvalue()
    peaks = {name: {"value": value, "time_s": time, "dof": dof}
             for name, (value, time, dof) in result.global_peaks.items()}
    manifest = {
        "schema_version": "1.5",
        "kind": "transient_result",
        "model": result.model.to_dict(),
        "input_hash": result.input_hash,
        "point_count": len(result.times_s),
        "steps": len(result.times_s) - 1,
        "start_s": result.start_s,
        "dt_s": result.dt_s,
        "global_peaks": peaks,
        "worst_backward_error": result.worst_backward_error,
        "worst_backward_step": result.worst_backward_step,
        "diagnostics": list(result.diagnostics),
        "versions": dict(result.versions),
        "dof_labels": [list(label) for label in result.dof_labels],
        "arrays": {
            name: {"shape": list(array.shape), "dtype": str(array.dtype)}
            for name, array in arrays.items()
        },
        "arrays_sha256": sha256(data).hexdigest(),
    }
    return manifest, data


def save_transient_result(result, root, name):
    """Create root/name and publish the transient manifest last."""
    if not isinstance(result, TransientResult):
        raise TypeError("expected a TransientResult")
    _safe_name(name)
    try:
        root = Path(root).resolve(strict=True)
        if not root.is_dir():
            raise TransientArtifactError("artifact root must be a directory")
        directory = root / name
        directory.mkdir(mode=0o700)
        manifest, data = _transient_payload(result)
        (directory / _ARRAYS).write_bytes(data)
        text = (json.dumps(manifest, allow_nan=False, indent=2) + "\n").encode("utf-8")
        if len(text) > _JSON_LIMIT:
            raise TransientArtifactError("transient manifest exceeds 1 MiB")
        (directory / _MANIFEST).write_bytes(text)
        return directory
    except TransientArtifactError:
        raise
    except (OSError, ValueError) as error:
        raise TransientArtifactError(f"could not save transient result: {error}") from error


def _transient_manifest(directory, *, for_view):
    read = _read_regular if for_view else _regular
    data = read(directory / _MANIFEST, _JSON_LIMIT)
    try:
        manifest = json.loads(data.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite constant {value}")))
    except (UnicodeError, ValueError) as error:
        raise TransientArtifactError("invalid transient manifest") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "1.5":
        raise TransientArtifactError("unsupported transient result schema")
    required = ("schema_version", "kind", "model", "input_hash", "point_count",
                "steps", "start_s", "dt_s", "global_peaks", "worst_backward_error",
                "worst_backward_step", "diagnostics", "versions", "dof_labels",
                "arrays", "arrays_sha256")
    if set(manifest) != set(required) or manifest.get("kind") != "transient_result":
        raise TransientArtifactError("incomplete transient manifest")
    return manifest


def _transient_arrays(directory, manifest, labels):
    data = _regular(directory / _ARRAYS, _ARRAY_LIMIT)
    if sha256(data).hexdigest() != manifest["arrays_sha256"]:
        raise TransientArtifactError("array digest does not match the manifest")
    try:
        with np.load(BytesIO(data), allow_pickle=False) as archive:
            if set(archive.files) != set(_TRANSIENT_FIELDS):
                raise TransientArtifactError("unexpected transient array set")
            arrays = {name: np.asarray(archive[name]) for name in _TRANSIENT_FIELDS}
    except (OSError, ValueError, EOFError) as error:
        raise TransientArtifactError(f"invalid transient arrays: {error}") from error
    points = manifest["point_count"]
    if not isinstance(points, int) or not 2 <= points <= 5001:
        raise TransientArtifactError("invalid time-point count")
    if manifest["steps"] != points - 1:
        raise TransientArtifactError("step count does not match the time points")
    expected = (points, len(labels))
    for name in ("displacement", "velocity", "acceleration"):
        if arrays[name].shape != expected:
            raise TransientArtifactError("history dimensions do not match the DOF labels")
    for name in ("times_s", "energy"):
        if arrays[name].shape != (points,):
            raise TransientArtifactError("time arrays must match the point count")
    for name in ("peak_displacement", "peak_time_s"):
        if arrays[name].shape != (len(labels),):
            raise TransientArtifactError("peak arrays must match the DOF labels")
    for name, array in arrays.items():
        metadata = manifest["arrays"].get(name)
        if not isinstance(metadata, dict) or metadata.get("dtype") != str(array.dtype):
            raise TransientArtifactError("missing transient array metadata")
        if list(array.shape) != metadata.get("shape"):
            raise TransientArtifactError("transient array shape does not match the manifest")
        if not np.isfinite(array).all():
            raise TransientArtifactError("transient arrays must be finite")
    return arrays


def _transient_view_arrays(directory, manifest, model, labels):
    grid = model.dynamics.time
    points = grid.steps + 1
    if (type(manifest["point_count"]) is not int or type(manifest["steps"]) is not int
            or manifest["point_count"] != points or manifest["steps"] != grid.steps):
        raise TransientArtifactError("time-point count does not match the model")
    if labels != dof_labels(model.mesh):
        raise TransientArtifactError("DOF label order does not match the model")

    shapes = {
        "times_s": (points,), "energy": (points,),
        "displacement": (points, len(labels)), "velocity": (points, len(labels)),
        "acceleration": (points, len(labels)),
        "peak_displacement": (len(labels),), "peak_time_s": (len(labels),),
    }
    expected = {name: {"shape": list(shape), "dtype": "float64"} for name, shape in shapes.items()}
    if manifest["arrays"] != expected:
        raise TransientArtifactError("transient array metadata does not match the model")
    data = _read_regular(directory / _ARRAYS, _ARRAY_LIMIT)
    if sha256(data).hexdigest() != manifest["arrays_sha256"]:
        raise TransientArtifactError("array digest does not match the manifest")
    layout = {name: (shape, "", "") for name, shape in shapes.items()}
    budget = sum(8 * math.prod(shape) + 10 + _HEADER_LIMIT for shape in shapes.values())
    arrays, _ = _read_arrays(data, layout, budget, allow_fortran=True, allow_big_endian=True,
                             element_limit=_STATE_LIMIT)
    return arrays


def _validate_transient_view(result):
    grid = result.model.dynamics.time
    times = result.times_s
    intervals = np.diff(times)
    ulp = max(math.ulp(grid.start_s), math.ulp(grid.end_s))
    tolerance = 1e-9 * result.dt_s + 2 * ulp
    if (result.start_s != grid.start_s or times[0] != grid.start_s
            or abs(times[-1] - grid.end_s) > math.ulp(grid.end_s)
            or np.any(intervals <= 0) or np.any(np.abs(intervals - result.dt_s) > tolerance)
            or abs(result.dt_s - grid.dt_s) > tolerance):
        raise TransientArtifactError("saved time axis is inconsistent with the declared grid")
    absolute = np.abs(result.displacement)
    indices = absolute.argmax(axis=0)
    if (not np.array_equal(result.peak_displacement, absolute[indices, np.arange(len(indices))])
            or not np.array_equal(result.peak_time_s, times[indices])):
        raise TransientArtifactError("saved displacement peaks do not match the history")
    if (not math.isfinite(result.worst_backward_error) or result.worst_backward_error < 0
            or not 0 <= result.worst_backward_step < len(times)):
        raise TransientArtifactError("invalid backward-error record")


def _transient_peaks(value):
    if not isinstance(value, dict) or set(value) != {"displacement", "velocity", "acceleration"}:
        raise TransientArtifactError("invalid global peaks")
    peaks = {}
    for name, item in value.items():
        if (not isinstance(item, dict)
                or set(item) != {"value", "time_s", "dof"}
                or not isinstance(item["dof"], int)):
            raise TransientArtifactError("invalid global peak entry")
        for key in ("value", "time_s"):
            if not isinstance(item[key], (int, float)) or isinstance(item[key], bool):
                raise TransientArtifactError("invalid global peak value")
        peaks[name] = (float(item["value"]), float(item["time_s"]), item["dof"])
    return peaks


def load_transient_result(directory):
    """Load and validate a complete transient artifact."""
    return _load_transient_result(directory, for_view=False)


def _load_transient_result(directory, *, for_view):
    directory = Path(directory)
    try:
        if for_view:
            _require_directory(directory)
        if not directory.is_dir():
            raise TransientArtifactError("expected an artifact directory")
        if any(entry.name not in {_MANIFEST, _ARRAYS} for entry in directory.iterdir()):
            raise TransientArtifactError("unexpected transient artifact entries")
        manifest = _transient_manifest(directory, for_view=for_view)
        model = validate_dynamics_model(manifest["model"])
        if model.analysis_type != "transient":
            raise TransientArtifactError("artifact model is not a transient declaration")
        if manifest["input_hash"] != canonical_dynamics_input_hash(model):
            raise TransientArtifactError("input hash does not match the model")
        labels = manifest["dof_labels"]
        if (not isinstance(labels, list) or not all(
                isinstance(label, list) and 2 <= len(label) <= 4
                and isinstance(label[0], str)
                and all(isinstance(item, (str, int)) and not isinstance(item, bool)
                        for item in label[1:]) for label in labels)):
            raise TransientArtifactError("invalid DOF labels")
        labels = tuple(tuple(label) for label in labels)
        arrays = (_transient_view_arrays(directory, manifest, model, labels) if for_view
                  else _transient_arrays(directory, manifest, labels))
        versions = manifest["versions"]
        if (not isinstance(versions, dict) or not versions
                or not all(isinstance(key, str) and isinstance(value, str)
                           for key, value in versions.items())):
            raise TransientArtifactError("invalid version record")
        diagnostics = manifest["diagnostics"]
        if not isinstance(diagnostics, list) or not all(
                isinstance(item, str) for item in diagnostics):
            raise TransientArtifactError("invalid diagnostics")
        if (not isinstance(manifest["start_s"], (int, float))
                or isinstance(manifest["start_s"], bool)
                or not isinstance(manifest["dt_s"], (int, float))
                or isinstance(manifest["dt_s"], bool)
                or manifest["dt_s"] <= 0):
            raise TransientArtifactError("invalid time grid scalars")
        for name in ("worst_backward_error",):
            if (not isinstance(manifest[name], (int, float))
                    or isinstance(manifest[name], bool)):
                raise TransientArtifactError("invalid backward-error record")
        if not isinstance(manifest["worst_backward_step"], int):
            raise TransientArtifactError("invalid backward-error step")
        result = TransientResult(
            model=model,
            times_s=arrays["times_s"],
            displacement=arrays["displacement"],
            velocity=arrays["velocity"],
            acceleration=arrays["acceleration"],
            energy=arrays["energy"],
            dof_labels=labels,
            start_s=float(manifest["start_s"]),
            dt_s=float(manifest["dt_s"]),
            peak_displacement=arrays["peak_displacement"],
            peak_time_s=arrays["peak_time_s"],
            global_peaks=_transient_peaks(manifest["global_peaks"]),
            worst_backward_error=float(manifest["worst_backward_error"]),
            worst_backward_step=manifest["worst_backward_step"],
            diagnostics=tuple(diagnostics),
            versions=dict(versions),
            input_hash=manifest["input_hash"],
        )
        if for_view:
            _validate_transient_view(result)
        return result
    except TransientArtifactError:
        raise
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError, SyntaxError,
            BadZipFile, struct.error, zlib.error) as error:
        raise TransientArtifactError(f"invalid transient result artifact: {error}") from error
