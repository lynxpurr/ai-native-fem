"""Versioned harmonic response artifacts for schema 1.5."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from ..model import canonical_dynamics_input_hash, validate_dynamics_model
from .response import HarmonicResult

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


__all__ = ["HarmonicArtifactError", "load_harmonic_result", "save_harmonic_result"]