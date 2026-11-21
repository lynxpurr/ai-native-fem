"""Bounded, data-only JSON/NPZ storage for declarative analysis results."""

from ast import literal_eval
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
from types import MappingProxyType
from zipfile import BadZipFile, ZIP_DEFLATED, ZIP_STORED, ZipFile
import zlib

import numpy as np

from .beam import Beam, BeamResult
from .model import ModelSpec, _JSON_LIMIT, _decode_json, _fields, _number, validate_model
from .verification import VerificationReport, _describe_case, input_parameters


_ZIP_LIMIT = 64 * 1024 * 1024
_MEMBER_LIMIT = 32
_EXPANDED_LIMIT = 64 * 1024 * 1024
_ELEMENT_LIMIT = 1_000_000
_HEADER_LIMIT = 10 * 1024
_DTYPE = np.dtype("<f8")
_MANIFEST = "manifest.json"
_ARRAYS = "arrays.npz"
_ROOT_FIELDS = ("schema_version", "model", "input_hash", "solve_status", "verification_status",
                "versions", "reference", "report", "diagnostics", "strain_energy",
                "relative_residual", "arrays", "arrays_sha256")
_EOCD = struct.Struct("<4s4H2IH")
_CENTRAL = struct.Struct("<4s6H3I5H2I")
_LOCAL = struct.Struct("<4s5H3I2H")


class ArtifactError(ValueError):
    """An unsafe, incomplete or inconsistent result artifact."""


def _array_layout(model):
    if type(model.model) is Beam:
        points = (model.model.elements + 1,)
        return {
            "x": (points, "node", "m"),
            "deflection": (points, "node", "m"),
            "slope": (points, "node", "rad"),
            "root_force": ((1,), "root_node", "N"),
            "root_moment": ((1,), "root_node", "N·m"),
        }
    nx, ny = model.model.nx + 1, model.model.ny + 1
    grid = (ny, nx)
    return {
        "x": ((nx,), "x_axis", "m"),
        "y": ((ny,), "y_axis", "m"),
        "deflection": (grid, "node", "m"),
        "beta_x": (grid, "node", "rad"),
        "beta_y": (grid, "node", "rad"),
        "reaction_force": (grid, "node", "N"),
        "reaction_moment_x_dof": (grid, "node", "N·m"),
        "reaction_moment_y_dof": (grid, "node", "N·m"),
    }


def _result_arrays(result):
    arrays = {"x": result.x, "deflection": result.deflection}
    if isinstance(result, BeamResult):
        arrays.update(slope=result.slope, root_force=result.root_reaction[:1],
                      root_moment=result.root_reaction[1:])
    else:
        arrays.update(y=result.y, beta_x=result.rotations[..., 0], beta_y=result.rotations[..., 1],
                      reaction_force=result.reactions[..., 0],
                      reaction_moment_x_dof=result.reactions[..., 1],
                      reaction_moment_y_dof=result.reactions[..., 2])
    return {name: np.asarray(array, dtype=_DTYPE) for name, array in arrays.items()}


def _array_metadata(layout):
    return {name: {"shape": list(shape), "position": position, "unit": unit, "dtype": _DTYPE.str}
            for name, (shape, position, unit) in layout.items()}


def _diagnostics(report):
    code = "numerical_failure" if report.solve_status == "numerical_failure" else "verification_warning"
    return tuple({"code": code, "message": warning} for warning in report.warnings)


def _validate_context(model, report, reference, versions, energy, residual):
    inputs = input_parameters(model.model, **model.loads)
    fingerprint, scope = _describe_case(inputs)
    ids = () if reference is None else (reference,)
    if (not isinstance(report, VerificationReport)
            or (report.input_hash, report.scope, report.reference_ids) != (fingerprint, scope, ids)):
        raise ArtifactError("report does not match the model and reference")
    versions = _fields(versions, "/versions", ("anf", "python", "numpy"))
    if any(not isinstance(value, str) or not value.strip() for value in versions.values()):
        raise ArtifactError("versions must be nonempty strings")
    if report.solve_status == "solved":
        energy = _number(energy, "/strain_energy")
        residual = _number(residual, "/relative_residual")
        check = report.checks[0]
        if check.id != "free_dof_relative_residual" or check.observed != residual:
            raise ArtifactError("relative residual does not match the report")
    elif energy is not None or residual is not None or not report.warnings:
        raise ArtifactError("numerical failure requires diagnostics and null measurements")
    return versions, energy, residual


@dataclass(frozen=True, slots=True, eq=False)
class ResultBundle:
    """A solve outcome with a normalized model, report and named SI arrays."""

    model: ModelSpec
    report: VerificationReport
    reference: str | None
    versions: Mapping[str, str]
    arrays: Mapping[str, np.ndarray]
    strain_energy: float | None = None
    relative_residual: float | None = None

    def __post_init__(self):
        model = validate_model(self.model)
        versions, energy, residual = _validate_context(
            model, self.report, self.reference, self.versions, self.strain_energy, self.relative_residual
        )
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "versions", MappingProxyType(versions))
        object.__setattr__(self, "strain_energy", energy)
        object.__setattr__(self, "relative_residual", residual)

        layout = _array_layout(model) if self.solve_status == "solved" else {}
        if not isinstance(self.arrays, Mapping) or self.arrays.keys() != layout.keys():
            raise ArtifactError("array names do not match the model and solve status")
        arrays = {}
        for name, (shape, _, _) in layout.items():
            array = self.arrays[name]
            if (not isinstance(array, np.ndarray) or array.dtype != _DTYPE
                    or array.shape != shape or not np.isfinite(array).all()):
                raise ArtifactError(f"invalid result array: {name}")
            arrays[name] = np.frombuffer(array.tobytes(order="C"), dtype=_DTYPE).reshape(shape)
        object.__setattr__(self, "arrays", MappingProxyType(arrays))

    @property
    def input_hash(self):
        return self.report.input_hash

    @property
    def solve_status(self):
        return self.report.solve_status

    @property
    def verification_status(self):
        return self.report.verification_status

    @property
    def diagnostics(self):
        return _diagnostics(self.report)

    @property
    def array_metadata(self):
        return _array_metadata(_array_layout(self.model)) if self.arrays else {}


def _manifest(result, digest):
    return {
        "schema_version": "1.0",
        "model": result.model.to_dict(),
        "input_hash": result.input_hash,
        "solve_status": result.solve_status,
        "verification_status": result.verification_status,
        "versions": dict(result.versions),
        "reference": result.reference,
        "report": json.loads(result.report.to_json()),
        "diagnostics": list(result.diagnostics),
        "strain_energy": result.strain_energy,
        "relative_residual": result.relative_residual,
        "arrays": result.array_metadata,
        "arrays_sha256": digest,
    }


def _read_manifest(path):
    manifest = _fields(_decode_json(_read_regular(path, _JSON_LIMIT)), "", _ROOT_FIELDS)
    if manifest["schema_version"] != "1.0":
        raise ArtifactError("unsupported result schema")
    model = validate_model(manifest["model"])
    if _json_numbers(manifest["model"]) != model.to_dict():
        raise ArtifactError("manifest model must have all normalized fields")
    # The report's own reader verifies its check schema, state and reference IDs.
    report_fields = _json_numbers(manifest["report"])
    report = VerificationReport.from_json(json.dumps(report_fields, allow_nan=False))
    if report_fields != json.loads(report.to_json()):
        raise ArtifactError("expected a complete verification report")
    status = (manifest["input_hash"], manifest["solve_status"], manifest["verification_status"])
    if status != (report.input_hash, report.solve_status, report.verification_status):
        raise ArtifactError("manifest state or input hash does not match the report")
    _validate_context(model, report, manifest["reference"], manifest["versions"],
                      manifest["strain_energy"], manifest["relative_residual"])
    if manifest["diagnostics"] != list(_diagnostics(report)):
        raise ArtifactError("diagnostics do not match the report")
    layout = _array_layout(model) if report.solve_status == "solved" else {}
    metadata = _fields(manifest["arrays"], "/arrays", layout)
    for name in layout:
        shape = _fields(metadata[name], f"/arrays/{name}", ("shape", "position", "unit", "dtype"))["shape"]
        if not isinstance(shape, list) or any(type(size) not in (int, Decimal) for size in shape):
            raise ArtifactError(f"invalid array shape: {name}")
    if metadata != _array_metadata(layout):
        raise ArtifactError("array metadata does not match the model")
    digest = manifest["arrays_sha256"]
    if layout and (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
        raise ArtifactError("invalid array digest")
    if not layout and digest is not None:
        raise ArtifactError("numerical failure cannot have an array digest")
    return manifest, model, report, layout


def _json_numbers(value):
    # Round report numbers as its existing JSON reader does, rejecting nonfinite values.
    if type(value) is Decimal:
        return _number(value, "/report")
    if isinstance(value, dict):
        return {key: _json_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_numbers(item) for item in value]
    return value


def _reparse(info):
    return bool(getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _require_directory(path):
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or _reparse(info):
        raise ArtifactError("expected a regular artifact directory")


def _regular(info):
    return stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and not _reparse(info)


def _read_regular(path, limit):
    original = path.lstat()
    if not _regular(original) or original.st_size > limit:
        raise ArtifactError(f"invalid file or size limit exceeded: {path.name}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not _regular(info) or not os.path.samestat(original, info)
                or info.st_size > limit):
            raise ArtifactError(f"invalid file or size limit exceeded: {path.name}")
        payload = stream.read(info.st_size + 1)
    if len(payload) > limit or len(payload) != info.st_size:
        raise ArtifactError(f"size limit exceeded or file changed: {path.name}")
    return payload


def _zip_directory(data):
    # Bound the true central directory count before ZipFile allocates ZipInfo objects.
    end = len(data) - _EOCD.size
    if end < 0:
        raise ArtifactError("incomplete ZIP")
    signature, disk, start_disk, on_disk, count, size, offset, comment = _EOCD.unpack_from(data, end)
    if (signature != b"PK\x05\x06" or disk or start_disk or count != on_disk or comment
            or count > _MEMBER_LIMIT or offset + size != end):
        raise ArtifactError("invalid ZIP directory or member budget exceeded")
    cursor = offset
    for _ in range(count):
        if cursor + _CENTRAL.size > end:
            raise ArtifactError("truncated ZIP directory")
        entry = _CENTRAL.unpack_from(data, cursor)
        if entry[0] != b"PK\x01\x02" or entry[13] != 0:
            raise ArtifactError("invalid ZIP directory entry")
        cursor += _CENTRAL.size + sum(entry[10:13])
    if cursor != end:
        raise ArtifactError("ZIP member count or directory length is inconsistent")
    return offset


def _member_offsets(data, members, directory):
    # V1 uses a single volume and complete local entries without ZIP data descriptors.
    cursor = 0
    for member in sorted(members, key=lambda member: member.header_offset):
        if member.header_offset != cursor or cursor + _LOCAL.size > directory:
            raise ArtifactError("missing or overlapping ZIP local entry")
        header = _LOCAL.unpack_from(data, cursor)
        flags, compression, _, _, crc, compressed, expanded, name_size, extra_size = header[2:]
        name_start = cursor + _LOCAL.size
        start = name_start + name_size + extra_size
        cursor = start + member.compress_size
        if (header[0] != b"PK\x03\x04" or flags != member.flag_bits
                or compression != member.compress_type or crc != member.CRC or cursor > directory
                or data[name_start:name_start + name_size] != member.filename.encode("ascii")
                or compressed not in (member.compress_size, 0xFFFFFFFF)
                or expanded not in (member.file_size, 0xFFFFFFFF)):
            raise ArtifactError("ZIP local metadata does not match its directory")
        yield member, start
    if cursor != directory:
        raise ArtifactError("unexpected data before ZIP directory")


def _read_header(stream, shape, declared_size):
    prefix = stream.read(10)
    if len(prefix) != 10 or prefix[:8] != b"\x93NUMPY\x01\x00":
        raise ArtifactError("expected NPY format 1.0")
    length = int.from_bytes(prefix[8:], "little")
    if not 0 < length <= _HEADER_LIMIT:
        raise ArtifactError("NPY header exceeds its byte budget")
    header = stream.read(length)
    if len(header) != length or not header.endswith(b"\n"):
        raise ArtifactError("invalid NPY header length")
    fields = literal_eval(header.decode("latin1").strip())
    if (not isinstance(fields, dict) or fields.keys() != {"descr", "fortran_order", "shape"}
            or fields["descr"] != _DTYPE.str or fields["fortran_order"] is not False
            or not isinstance(fields["shape"], tuple) or fields["shape"] != shape
            or any(type(size) is not int for size in fields["shape"])):
        raise ArtifactError("NPY dtype, order or shape does not match the model")
    if declared_size != 10 + length + math.prod(shape) * _DTYPE.itemsize:
        raise ArtifactError("NPY payload length does not match its validated shape")


def _member_payload(data, member, offset):
    # ZipExtFile stops at the declared file_size. Check raw DEFLATE completion independently
    # so bytes hidden beyond a false ZIP size cannot be mistaken for a complete member.
    source = memoryview(data)[offset:offset + member.compress_size]
    if member.compress_type == ZIP_STORED:
        payload = bytes(source)
    else:
        decoder = zlib.decompressobj(-zlib.MAX_WBITS)
        payload = bytearray()
        for start in range(0, len(source), 65536):
            budget = member.file_size - len(payload)
            payload.extend(decoder.decompress(source[start:start + 65536], budget + 1))
            if len(payload) > member.file_size or decoder.unused_data or decoder.unconsumed_tail:
                raise ArtifactError("ZIP contains data beyond the declared member")
        if not decoder.eof:
            raise ArtifactError("incomplete compressed member")
    if len(payload) != member.file_size or zlib.crc32(payload) != member.CRC:
        raise ArtifactError("ZIP member length or CRC is invalid")
    return payload


def _read_arrays(data, layout):
    directory = _zip_directory(data)
    names = {f"{name}.npy": shape for name, (shape, _, _) in layout.items()}
    with ZipFile(BytesIO(data)) as archive:
        members = archive.infolist()
        if len(members) != len(names) or {item.orig_filename for item in members} != names.keys():
            raise ArtifactError("unexpected or duplicate ZIP members")
        if sum(item.file_size for item in members) > _EXPANDED_LIMIT:
            raise ArtifactError("ZIP exceeds its expanded byte budget")
        for member in members:
            kind = stat.S_IFMT(member.external_attr >> 16)
            if (kind not in (0, stat.S_IFREG) or member.external_attr & 0x410
                    or member.flag_bits & ~0x806 or member.compress_type not in (ZIP_STORED, ZIP_DEFLATED)):
                raise ArtifactError("ZIP member must be a regular, unencrypted NPY file")
            if member.compress_type == ZIP_STORED and member.compress_size != member.file_size:
                raise ArtifactError("stored ZIP member length is inconsistent")
            shape = names[member.filename]
            elements = math.prod(shape)
            if elements > _ELEMENT_LIMIT or member.file_size > 10 + _HEADER_LIMIT + elements * _DTYPE.itemsize:
                raise ArtifactError("NPY member exceeds its model-derived budget")
            with archive.open(member) as stream:
                _read_header(stream, shape, member.file_size)
        return {
            member.filename[:-4]: np.load(BytesIO(_member_payload(data, member, start)),
                                          allow_pickle=False, max_header_size=_HEADER_LIMIT)
            for member, start in _member_offsets(data, members, directory)
        }


def _safe_name(name):
    if (not isinstance(name, str) or not name or name.endswith((".", " "))
            or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
            or re.fullmatch(r"CON|PRN|AUX|NUL|CONIN\$|CONOUT\$|(?:COM|LPT)[1-9¹²³]",
                            name.partition(".")[0].rstrip(" "), re.I)):
        raise ArtifactError("expected a single portable artifact name")


def _flush(stream):
    stream.flush()
    os.fsync(stream.fileno())


def save_result(result, root, name):
    """Create root/name exclusively and publish its complete manifest last."""
    if not isinstance(result, ResultBundle):
        raise TypeError("expected a ResultBundle")
    _safe_name(name)
    try:
        root = Path(root).resolve(strict=True)
        _require_directory(root)
        directory = root / name
        directory.mkdir(mode=0o700)
        digest = None
        if result.arrays:
            arrays = directory / _ARRAYS
            with arrays.open("xb") as stream:
                with ZipFile(stream, "w", compression=ZIP_DEFLATED, allowZip64=False) as archive:
                    for key, value in result.arrays.items():
                        with archive.open(f"{key}.npy", "w") as member:
                            np.lib.format.write_array(member, value, version=(1, 0), allow_pickle=False)
                _flush(stream)
            digest = sha256(_read_regular(arrays, _ZIP_LIMIT)).hexdigest()
        payload = (json.dumps(_manifest(result, digest), allow_nan=False, indent=2) + "\n").encode("utf-8")
        if len(payload) > _JSON_LIMIT:
            raise ArtifactError("manifest exceeds its byte budget")
        pending = directory / ".manifest.tmp"
        with pending.open("xb") as stream:
            stream.write(payload)
            _flush(stream)
        pending.replace(directory / _MANIFEST)
        return directory
    except (OSError, ValueError) as error:
        if isinstance(error, ArtifactError):
            raise
        raise ArtifactError(f"could not save result: {error}") from error


def load_result(directory):
    """Load a complete artifact using model-derived metadata and byte budgets."""
    directory = Path(directory)
    try:
        _require_directory(directory)
        manifest, model, report, layout = _read_manifest(directory / _MANIFEST)
        expected = {_MANIFEST, _ARRAYS} if layout else {_MANIFEST}
        with os.scandir(directory) as entries:
            if any(entry.name not in expected for entry in entries):
                raise ArtifactError("unexpected file in artifact directory")
        arrays = {}
        if layout:
            data = _read_regular(directory / _ARRAYS, _ZIP_LIMIT)
            if sha256(data).hexdigest() != manifest["arrays_sha256"]:
                raise ArtifactError("array digest does not match the manifest")
            arrays = _read_arrays(data, layout)
        return ResultBundle(model, report, manifest["reference"], manifest["versions"], arrays,
                            manifest["strain_energy"], manifest["relative_residual"])
    except (OSError, ValueError, SyntaxError, TypeError, RuntimeError, EOFError,
            BadZipFile, struct.error, zlib.error) as error:
        if isinstance(error, ArtifactError):
            raise
        raise ArtifactError(f"invalid result artifact: {error}") from error
