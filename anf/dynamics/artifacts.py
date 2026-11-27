"""Versioned modal result artifacts for schema 1.4."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import struct
from zipfile import BadZipFile
import zlib

import numpy as np

from ..artifacts import _HEADER_LIMIT, _read_arrays, _read_regular, _require_directory
from ..model import canonical_modal_input_hash, validate_modal_model
from .modal import ModalResult
from .system import dof_labels, free_dofs


_JSON_LIMIT = 1024 * 1024
_ARRAY_LIMIT = 1024 * 1024
_MANIFEST = "manifest.json"
_ARRAYS = "arrays.npz"
_FIELDS = ("eigenvalues", "omega_rad_s", "frequencies_hz", "modes", "residuals")


class ModalArtifactError(ValueError):
    """Invalid or incomplete modal artifact."""


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
    data = _read_regular(directory / _MANIFEST, _JSON_LIMIT)
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
    count = manifest["mode_count"]
    if type(count) is not int or not 0 <= count <= min(len(labels), 20):
        raise ModalArtifactError("invalid mode count")
    layout = {name: ((count,), "mode", "") for name in _FIELDS}
    layout["modes"] = ((len(labels), count), "dof", "")
    expected = {name: {"shape": list(shape), "dtype": "float64"}
                for name, (shape, _, _) in layout.items()}
    if manifest["arrays"] != expected:
        raise ModalArtifactError("modal metadata does not match model-derived array shapes")

    data = _read_regular(directory / _ARRAYS, _ARRAY_LIMIT)
    if sha256(data).hexdigest() != manifest["arrays_sha256"]:
        raise ModalArtifactError("array digest does not match the manifest")
    budget = (len(labels) + 4) * count * 8 + len(layout) * (10 + _HEADER_LIMIT)
    # The published writer used native byte order and preserved Fortran order.
    arrays, _ = _read_arrays(data, layout, budget, allow_fortran=True, allow_big_endian=True)
    if any(np.any(arrays[name] < 0) for name in ("frequencies_hz", "omega_rad_s", "residuals")):
        raise ModalArtifactError("modal frequencies and residuals must be nonnegative")
    return {name: array.astype(float, copy=False) for name, array in arrays.items()}


def load_modal_result(directory):
    """Load and validate a complete modal artifact."""
    directory = Path(directory)
    try:
        _require_directory(directory)
        if any(entry.name not in {_MANIFEST, _ARRAYS} for entry in directory.iterdir()):
            raise ModalArtifactError("unexpected modal artifact entries")
        manifest = _manifest(directory)
        model = validate_modal_model(manifest["model"])
        if manifest["input_hash"] != canonical_modal_input_hash(model):
            raise ModalArtifactError("input hash does not match the model")
        free, supported = free_dofs(model, model.mesh)
        full_labels = dof_labels(model.mesh)
        labels = tuple(full_labels[index] for index in free)
        if manifest["dof_labels"] != [list(label) for label in labels]:
            raise ModalArtifactError("DOF label order does not match the free model DOFs")
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
        if (type(manifest["rigid_mode_count"]) is not int
                or manifest["rigid_mode_count"] != (0 if supported else 3)):
            raise ModalArtifactError("invalid rigid-mode count")
        return ModalResult(
            model=model,
            eigenvalues=arrays["eigenvalues"],
            omega_rad_s=arrays["omega_rad_s"],
            frequencies_hz=arrays["frequencies_hz"],
            modes=arrays["modes"],
            dof_labels=labels,
            residuals=arrays["residuals"],
            rigid_mode_count=manifest["rigid_mode_count"],
            truncated=manifest["truncated"],
            diagnostics=tuple(diagnostics),
            versions=dict(versions),
            input_hash=manifest["input_hash"],
        )
    except ModalArtifactError:
        raise
    except (OSError, ValueError, SyntaxError, TypeError, KeyError, RuntimeError, EOFError,
            BadZipFile, struct.error, zlib.error) as error:
        raise ModalArtifactError(f"invalid modal result artifact: {error}") from error


__all__ = ["ModalArtifactError", "load_modal_result", "save_modal_result"]