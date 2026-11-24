"""Versioned modal result artifacts for schema 1.4."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from ..model import canonical_modal_input_hash, validate_modal_model
from .modal import ModalResult


_JSON_LIMIT = 1024 * 1024
_ARRAY_LIMIT = 64 * 1024 * 1024
_MANIFEST = "manifest.json"
_ARRAYS = "arrays.npz"
_FIELDS = ("eigenvalues", "omega_rad_s", "frequencies_hz", "modes", "residuals")


class ModalArtifactError(ValueError):
    """Invalid or incomplete modal artifact."""


def _regular(path, limit):
    if path.is_symlink() or not path.is_file():
        raise ModalArtifactError("expected a regular file")
    size = path.stat().st_size
    if size > limit:
        raise ModalArtifactError(f"file exceeds {limit} bytes")
    return path.read_bytes()


def _safe_name(name):
    if (not isinstance(name, str) or not name or name in (".", "..")
            or "/" in name or "\\" in name):
        raise ModalArtifactError("invalid artifact name")
    return name


def _payload(result):
    arrays = {
        "eigenvalues": np.asarray(result.eigenvalues, dtype=float),
        "omega_rad_s": np.asarray(result.omega_rad_s, dtype=float),
        "frequencies_hz": np.asarray(result.frequencies_hz, dtype=float),
        "modes": np.asarray(result.modes, dtype=float),
        "residuals": np.asarray(result.residuals, dtype=float),
    }
    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    data = stream.getvalue()
    manifest = {
        "schema_version": "1.4",
        "kind": "modal_result",
        "model": result.model.to_dict(),
        "input_hash": result.input_hash,
        "mode_count": len(result.eigenvalues),
        "rigid_mode_count": result.rigid_mode_count,
        "truncated": result.truncated,
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


def save_modal_result(result, root, name):
    """Create root/name and publish the modal manifest last."""
    if not isinstance(result, ModalResult):
        raise TypeError("expected a ModalResult")
    _safe_name(name)
    try:
        root = Path(root).resolve(strict=True)
        if not root.is_dir():
            raise ModalArtifactError("artifact root must be a directory")
        directory = root / name
        directory.mkdir(mode=0o700)
        manifest, data = _payload(result)
        (directory / _ARRAYS).write_bytes(data)
        text = (json.dumps(manifest, allow_nan=False, indent=2) + "\n").encode("utf-8")
        if len(text) > _JSON_LIMIT:
            raise ModalArtifactError("modal manifest exceeds 1 MiB")
        (directory / _MANIFEST).write_bytes(text)
        return directory
    except ModalArtifactError:
        raise
    except (OSError, ValueError) as error:
        raise ModalArtifactError(f"could not save modal result: {error}") from error


def _manifest(directory):
    data = _regular(directory / _MANIFEST, _JSON_LIMIT)
    try:
        manifest = json.loads(data.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite constant {value}")))
    except (UnicodeError, ValueError) as error:
        raise ModalArtifactError("invalid modal manifest") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "1.4":
        raise ModalArtifactError("unsupported modal result schema")
    required = ("schema_version", "kind", "model", "input_hash", "mode_count",
                "rigid_mode_count", "truncated", "diagnostics", "versions",
                "dof_labels", "arrays", "arrays_sha256")
    if set(manifest) != set(required) or manifest.get("kind") != "modal_result":
        raise ModalArtifactError("incomplete modal manifest")
    return manifest


def _arrays(directory, manifest, labels):
    data = _regular(directory / _ARRAYS, _ARRAY_LIMIT)
    if sha256(data).hexdigest() != manifest["arrays_sha256"]:
        raise ModalArtifactError("array digest does not match the manifest")
    try:
        with np.load(BytesIO(data), allow_pickle=False) as archive:
            if set(archive.files) != set(_FIELDS):
                raise ModalArtifactError("unexpected modal array set")
            arrays = {name: np.asarray(archive[name], dtype=float) for name in _FIELDS}
    except (OSError, ValueError, EOFError) as error:
        raise ModalArtifactError(f"invalid modal arrays: {error}") from error
    count = manifest["mode_count"]
    if not isinstance(count, int) or not 0 <= count <= 20:
        raise ModalArtifactError("invalid mode count")
    if arrays["modes"].shape != (len(labels), count):
        raise ModalArtifactError("mode shape dimensions do not match the DOF labels")
    for name, array in arrays.items():
        metadata = manifest["arrays"].get(name)
        if not isinstance(metadata, dict) or metadata.get("dtype") != "float64":
            raise ModalArtifactError("missing modal array metadata")
        if list(array.shape) != metadata.get("shape"):
            raise ModalArtifactError("modal array shape does not match the manifest")
        if not np.isfinite(array).all():
            raise ModalArtifactError("modal arrays must be finite")
    if any(len(arrays[name]) != count for name in
           ("eigenvalues", "omega_rad_s", "frequencies_hz", "residuals")):
        raise ModalArtifactError("modal arrays must share the mode count")
    return arrays


def load_modal_result(directory):
    """Load and validate a complete modal artifact."""
    directory = Path(directory)
    try:
        if not directory.is_dir():
            raise ModalArtifactError("expected an artifact directory")
        if {entry.name for entry in directory.iterdir()} != {_MANIFEST, _ARRAYS}:
            raise ModalArtifactError("unexpected modal artifact entries")
        manifest = _manifest(directory)
        model = validate_modal_model(manifest["model"])
        if manifest["input_hash"] != canonical_modal_input_hash(model):
            raise ModalArtifactError("input hash does not match the model")
        labels = manifest["dof_labels"]
        if (not isinstance(labels, list) or not all(
                isinstance(label, list) and 2 <= len(label) <= 4
                and isinstance(label[0], str)
                and all(isinstance(item, (str, int)) and not isinstance(item, bool)
                        for item in label[1:]) for label in labels)):
            raise ModalArtifactError("invalid DOF labels")
        arrays = _arrays(directory, manifest, labels)
        versions = manifest["versions"]
        if (not isinstance(versions, dict) or not versions
                or not all(isinstance(key, str) and isinstance(value, str)
                           for key, value in versions.items())):
            raise ModalArtifactError("invalid version record")
        diagnostics = manifest["diagnostics"]
        if not isinstance(diagnostics, list) or not all(
                isinstance(item, str) for item in diagnostics):
            raise ModalArtifactError("invalid diagnostics")
        if not isinstance(manifest["truncated"], bool):
            raise ModalArtifactError("invalid truncation flag")
        if not isinstance(manifest["rigid_mode_count"], int):
            raise ModalArtifactError("invalid rigid-mode count")
        return ModalResult(
            model=model,
            eigenvalues=arrays["eigenvalues"],
            omega_rad_s=arrays["omega_rad_s"],
            frequencies_hz=arrays["frequencies_hz"],
            modes=arrays["modes"],
            dof_labels=tuple(tuple(label) for label in labels),
            residuals=arrays["residuals"],
            rigid_mode_count=manifest["rigid_mode_count"],
            truncated=manifest["truncated"],
            diagnostics=tuple(diagnostics),
            versions=dict(versions),
            input_hash=manifest["input_hash"],
        )
    except ModalArtifactError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ModalArtifactError(f"invalid modal result artifact: {error}") from error


__all__ = ["ModalArtifactError", "load_modal_result", "save_modal_result"]