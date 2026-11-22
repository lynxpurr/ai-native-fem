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

from ._linear import MAX_RELATIVE_RESIDUAL
from .beam import Beam, BeamResult
from .loads import _case_hash, _request, _request_hash, _resolve_cases
from .model import (ModelSpec, _JSON_LIMIT, _decode_json, _fields, _number, _physical_inputs,
                    validate_load_case_model, validate_model)
from .recovery import SECTION_UNITS, _envelope, _envelope_status
from .structures import DISPLACEMENTS, FORCES, StructureResult
from .verification import VerificationReport, _STRUCTURE_FAILURES, _describe_case


_ZIP_LIMIT = 64 * 1024 * 1024
_MEMBER_LIMIT = 32
_EXPANDED_LIMIT = 64 * 1024 * 1024
_ELEMENT_LIMIT = 1_000_000
_HEADER_LIMIT = 10 * 1024
_DTYPE = np.dtype("<f8")
_MANIFEST = "manifest.json"
_ARRAYS = "arrays.npz"
_BATCH_LIMIT = 4 * _JSON_LIMIT
_BATCH_FIELDS = ("schema_version", "model", "request_hash", "selected_ids", "stations",
                 "ordering", "versions", "envelope_status", "results")
_CASE_FIELDS = ("id", "kind", "input_hash", "solve_status", "verification_status", "diagnostics",
                "strain_energy", "relative_residual", "arrays", "arrays_path", "arrays_sha256")
_ROOT_FIELDS = ("schema_version", "model", "input_hash", "solve_status", "verification_status",
                "versions", "reference", "report", "diagnostics", "strain_energy",
                "relative_residual", "arrays", "arrays_sha256")
_EOCD = struct.Struct("<4s4H2IH")
_CENTRAL = struct.Struct("<4s6H3I5H2I")
_LOCAL = struct.Struct("<4s5H3I2H")


class ArtifactError(ValueError):
    """An unsafe, incomplete or inconsistent result artifact."""


def _array_layout(model):
    if model.mesh is not None:
        mesh = model.mesh
        nodes = (len(mesh.node_order),)
        rotations = (len(mesh.frame_node_order),)
        ends = (len(mesh.member_order), 2)
        layout = {
            "node_ux": (nodes, "node", "m"), "node_uy": (nodes, "node", "m"),
            "reaction_fx": (nodes, "node", "N"), "reaction_fy": (nodes, "node", "N"),
            "member_end_axial": (ends, "member_end", "N"),
            "member_end_shear": (ends, "member_end", "N"),
            "member_end_moment": (ends, "member_end", "N·m"),
        }
        if rotations[0]:
            layout.update(node_rz=(rotations, "frame_node", "rad"),
                          reaction_mz=(rotations, "frame_node", "N·m"))
        return layout
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
    if isinstance(result, StructureResult):
        dofs = result.mesh.node_dofs
        arrays = {}
        for index, (component, reaction) in enumerate(zip(DISPLACEMENTS, FORCES, strict=True)):
            indices = [row[index] for row in dofs if len(row) > index]
            if indices:
                arrays[f"node_{component}"] = result.displacement[indices]
                arrays[f"reaction_{reaction}"] = result.reactions[indices]
        for index, name in enumerate(("axial", "shear", "moment")):
            arrays[f"member_end_{name}"] = result.member_end_actions[..., index]
        if result.frame_member_end_rotation is not None:
            arrays["frame_member_end_rotation"] = result.frame_member_end_rotation
        return {name: np.asarray(array, dtype=_DTYPE) for name, array in arrays.items()}
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


def _validated_arrays(arrays, layout):
    if not isinstance(arrays, Mapping) or arrays.keys() != layout.keys():
        raise ArtifactError("array names do not match the model and solve status")
    validated = {}
    for name, (shape, _, _) in layout.items():
        array = arrays[name]
        if (not isinstance(array, np.ndarray) or array.dtype != _DTYPE
                or array.shape != shape or not np.isfinite(array).all()):
            raise ArtifactError(f"invalid result array: {name}")
        validated[name] = np.frombuffer(array.tobytes(order="C"), dtype=_DTYPE).reshape(shape)
    return MappingProxyType(validated)


def _validated_versions(versions):
    versions = _fields(versions, "/versions", ("anf", "python", "numpy"))
    if any(not isinstance(value, str) or not value.strip() for value in versions.values()):
        raise ArtifactError("versions must be nonempty strings")
    return MappingProxyType(versions)


def _diagnostics(report):
    code = report.diagnostic_code or (
        "numerical_failure" if report.solve_status == "numerical_failure" else "verification_warning"
    )
    return tuple({"code": code, "message": warning} for warning in report.warnings)


def _validate_context(model, report, reference, versions, energy, residual):
    inputs = _physical_inputs(model)
    fingerprint, scope = _describe_case(inputs)
    ids = () if reference is None else (reference,)
    if (not isinstance(report, VerificationReport)
            or model.schema_version != report.schema_version
            or report.input_status != "valid"
            or (report.input_hash, report.scope, report.reference_ids) != (fingerprint, scope, ids)):
        raise ArtifactError("report does not match the model and reference")
    versions = _validated_versions(versions)
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
        object.__setattr__(self, "versions", versions)
        object.__setattr__(self, "strain_energy", energy)
        object.__setattr__(self, "relative_residual", residual)

        layout = _array_layout(model) if self.solve_status == "solved" else {}
        object.__setattr__(self, "arrays", _validated_arrays(self.arrays, layout))

    @property
    def schema_version(self):
        return self.model.schema_version

    @property
    def ordering(self):
        """Model-derived node and member ordering for a planar structure."""
        return self.model.mesh.ordering() if self.model.mesh is not None else None

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
    manifest = {
        "schema_version": result.schema_version,
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
    if result.ordering is not None:
        manifest["ordering"] = result.ordering
    return manifest


def _read_manifest(path):
    data = _decode_json(_read_regular(path, _JSON_LIMIT))
    version = data.get("schema_version") if isinstance(data, Mapping) else None
    if version not in ("1.0", "1.1"):
        raise ArtifactError("unsupported result schema")
    fields = (*_ROOT_FIELDS, "ordering") if version == "1.1" else _ROOT_FIELDS
    manifest = _fields(data, "", fields)
    model = validate_model(manifest["model"])
    if version != model.schema_version:
        raise ArtifactError("result schema does not match its model schema")
    if model.mesh is not None:
        _check_ordering(model, manifest["ordering"])
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


def _check_ordering(model, ordering):
    if (ordering != model.mesh.ordering()
            or any(type(node[2]) is not Decimal
                   for node in ordering["node_order"] + ordering["frame_node_order"] if len(node) == 3)):
        raise ArtifactError("structure ordering does not match the model")


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


def _read_arrays(data, layout, expanded_limit=_EXPANDED_LIMIT):
    directory = _zip_directory(data)
    names = {f"{name}.npy": shape for name, (shape, _, _) in layout.items()}
    with ZipFile(BytesIO(data)) as archive:
        members = archive.infolist()
        if len(members) != len(names) or {item.orig_filename for item in members} != names.keys():
            raise ArtifactError("unexpected or duplicate ZIP members")
        expanded = sum(item.file_size for item in members)
        if expanded > expanded_limit:
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
        arrays = {
            member.filename[:-4]: np.load(BytesIO(_member_payload(data, member, start)),
                                          allow_pickle=False, max_header_size=_HEADER_LIMIT)
            for member, start in _member_offsets(data, members, directory)
        }
        return arrays, expanded


def _safe_name(name):
    if (not isinstance(name, str) or not name or name.endswith((".", " "))
            or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
            or re.fullmatch(r"CON|PRN|AUX|NUL|CONIN\$|CONOUT\$|(?:COM|LPT)[1-9¹²³]",
                            name.partition(".")[0].rstrip(" "), re.I)):
        raise ArtifactError("expected a single portable artifact name")


def _flush(stream):
    stream.flush()
    os.fsync(stream.fileno())


def _write_arrays(path, arrays):
    with path.open("xb") as stream:
        with ZipFile(stream, "w", compression=ZIP_DEFLATED, allowZip64=False) as archive:
            for key, value in arrays.items():
                with archive.open(f"{key}.npy", "w") as member:
                    np.lib.format.write_array(member, value, version=(1, 0), allow_pickle=False)
        _flush(stream)
    return sha256(_read_regular(path, _ZIP_LIMIT)).hexdigest()


def _publish_manifest(directory, payload, limit):
    if len(payload) > limit:
        raise ArtifactError("manifest exceeds its byte budget")
    pending = directory / ".manifest.tmp"
    with pending.open("xb") as stream:
        stream.write(payload)
        _flush(stream)
    pending.replace(directory / _MANIFEST)


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
        digest = _write_arrays(directory / _ARRAYS, result.arrays) if result.arrays else None
        payload = (json.dumps(_manifest(result, digest), allow_nan=False, indent=2) + "\n").encode("utf-8")
        _publish_manifest(directory, payload, _JSON_LIMIT)
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
            arrays, _ = _read_arrays(data, layout)
        return ResultBundle(model, report, manifest["reference"], manifest["versions"], arrays,
                            manifest["strain_energy"], manifest["relative_residual"])
    except (OSError, ValueError, SyntaxError, TypeError, RuntimeError, EOFError,
            BadZipFile, struct.error, zlib.error) as error:
        if isinstance(error, ArtifactError):
            raise
        raise ArtifactError(f"invalid result artifact: {error}") from error


@dataclass(frozen=True, slots=True, eq=False)
class LoadCaseResult:
    """The outcome and named SI arrays of one selected case or combination."""

    id: str
    kind: str
    input_hash: str
    member_order: tuple
    stations: tuple
    arrays: Mapping[str, np.ndarray]
    strain_energy: float | None
    relative_residual: float | None
    _diagnostic: tuple[str, str] | None

    @property
    def solve_status(self):
        return "solved" if self._diagnostic is None else "numerical_failure"

    @property
    def verification_status(self):
        return "solved_unverified" if self._diagnostic is None else "not_run"

    @property
    def diagnostics(self):
        if self._diagnostic is None:
            return ()
        code, message = self._diagnostic
        return ({"code": code, "message": message},)


@dataclass(frozen=True, slots=True, eq=False)
class LoadCaseResults:
    """A load case request and its selected outcomes in ASCII ID order."""

    model: ModelSpec
    stations: tuple
    versions: Mapping[str, str]
    results: Mapping[str, LoadCaseResult]

    @property
    def schema_version(self):
        return self.model.schema_version

    @property
    def selected_ids(self):
        return tuple(self.results)

    @property
    def request_hash(self):
        return _request_hash(self.model, self.selected_ids, self.stations)

    @property
    def ordering(self):
        return self.model.mesh.ordering()

    @property
    def envelope_status(self):
        return _envelope_status(self.results)

    @property
    def envelope(self):
        return _envelope(self)


def _case_layout(model, stations):
    layout = _array_layout(model)
    shape = (len(model.mesh.member_order), len(stations))
    layout.update({f"member_section_{name}": (shape, "member_station", unit)
                   for name, unit in SECTION_UNITS.items()})
    if model.schema_version == "1.3":
        frames = sum(member["kind"] == "frame2d" for member in model.model["members"].values())
        if frames:
            layout["frame_member_end_rotation"] = ((frames, 2), "frame_member_end", "rad")
    return layout


def _case_result(model, stations, name, kind, input_hash, arrays, energy, residual, diagnostic=None):
    if diagnostic is None:
        energy = _number(energy, "/strain_energy")
        residual = _number(residual, "/relative_residual")
        if not 0 <= residual <= MAX_RELATIVE_RESIDUAL:
            raise ArtifactError("invalid solved-case relative residual")
    else:
        code, message = diagnostic
        if (code not in _STRUCTURE_FAILURES or not isinstance(message, str) or not message.strip()
                or energy is not None or residual is not None):
            raise ArtifactError("invalid failure diagnostics or measurements")
    layout = _case_layout(model, stations) if diagnostic is None else {}
    return LoadCaseResult(name, kind, input_hash, model.mesh.member_order, stations,
                          _validated_arrays(arrays, layout), energy, residual, diagnostic)


def _case_path(index):
    return f"cases/{index:04d}/{_ARRAYS}"


def save_load_case_results(batch, root, name):
    """Create root/name exclusively and publish the complete batch index last."""
    if not isinstance(batch, LoadCaseResults):
        raise TypeError("expected LoadCaseResults from run_load_cases or load_load_case_results")
    _safe_name(name)
    try:
        root = Path(root).resolve(strict=True)
        _require_directory(root)
        directory = root / name
        directory.mkdir(mode=0o700)
        if batch.envelope_status != "not_available":
            (directory / "cases").mkdir(mode=0o700)
        entries = []
        layout = _case_layout(batch.model, batch.stations)
        for index, result in enumerate(batch.results.values()):
            relative_path, digest, metadata = None, None, {}
            if result.solve_status == "solved":
                relative_path = _case_path(index)
                path = directory / relative_path
                path.parent.mkdir(mode=0o700)
                digest = _write_arrays(path, result.arrays)
                metadata = _array_metadata(layout)
            entries.append({
                "id": result.id, "kind": result.kind, "input_hash": result.input_hash,
                "solve_status": result.solve_status, "verification_status": result.verification_status,
                "diagnostics": list(result.diagnostics), "strain_energy": result.strain_energy,
                "relative_residual": result.relative_residual, "arrays": metadata,
                "arrays_path": relative_path, "arrays_sha256": digest,
            })
        manifest = {
            "schema_version": batch.schema_version, "model": batch.model.to_dict(),
            "request_hash": batch.request_hash, "selected_ids": batch.selected_ids,
            "stations": batch.stations, "ordering": batch.ordering, "versions": dict(batch.versions),
            "envelope_status": batch.envelope_status, "results": entries,
        }
        payload = (json.dumps(manifest, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        _publish_manifest(directory, payload, _BATCH_LIMIT)
        return directory
    except (OSError, ValueError) as error:
        if isinstance(error, ArtifactError):
            raise
        raise ArtifactError(f"could not save load case results: {error}") from error


def _check_entries(directory, expected):
    _require_directory(directory)
    with os.scandir(directory) as entries:
        if {entry.name for entry in entries} != expected:
            raise ArtifactError("unexpected or missing entry in the load case artifact")


def _read_case_entry(entry, directory, index, layout, budget):
    status = entry["solve_status"], entry["verification_status"]
    if status == ("numerical_failure", "not_run"):
        diagnostics = entry["diagnostics"]
        if (entry["arrays_path"] is not None or entry["arrays_sha256"] is not None or entry["arrays"] != {}
                or not isinstance(diagnostics, list) or len(diagnostics) != 1):
            raise ArtifactError("failure entry requires one diagnostic and no arrays")
        diagnostic = _fields(diagnostics[0], "/diagnostics/0", ("code", "message"))
        return {}, (diagnostic["code"], diagnostic["message"]), budget

    relative_path = _case_path(index)
    digest = entry["arrays_sha256"]
    if (status != ("solved", "solved_unverified") or entry["diagnostics"] != []
            or entry["arrays_path"] != relative_path or entry["arrays"] != _array_metadata(layout)
            or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
        raise ArtifactError("invalid solved case status, array path or metadata")
    for metadata in entry["arrays"].values():
        if any(type(size) is not Decimal for size in metadata["shape"]):
            raise ArtifactError("invalid array shape in load case index")
    path = directory / relative_path
    _require_directory(directory / "cases")
    _check_entries(path.parent, {_ARRAYS})
    data = _read_regular(path, _ZIP_LIMIT)
    if sha256(data).hexdigest() != digest:
        raise ArtifactError("array digest does not match the load case index")
    arrays, expanded = _read_arrays(data, layout, budget)
    return arrays, None, budget - expanded


def load_load_case_results(directory):
    """Load a complete batch index and verify all selected outcomes."""
    directory = Path(directory)
    try:
        _require_directory(directory)
        manifest = _fields(_decode_json(_read_regular(directory / _MANIFEST, _BATCH_LIMIT), _BATCH_LIMIT),
                           "", _BATCH_FIELDS)
        model = validate_load_case_model(manifest["model"])
        if manifest["schema_version"] != model.schema_version:
            raise ArtifactError("load case index schema does not match the model")
        if _json_numbers(manifest["model"]) != model.to_dict():
            raise ArtifactError("index model must have all normalized fields")
        ids, stations = _request(model, manifest["selected_ids"], manifest["stations"])
        if (manifest["selected_ids"] != list(ids)
                or _json_numbers(manifest["stations"]) != list(stations)
                or manifest["request_hash"] != _request_hash(model, ids, stations)):
            raise ArtifactError("request identity does not match the model, selected IDs and stations")
        _check_ordering(model, manifest["ordering"])
        versions = _validated_versions(manifest["versions"])
        entries = manifest["results"]
        if not isinstance(entries, list) or len(entries) != len(ids):
            raise ArtifactError("result entries do not match the selected IDs")

        layout = _case_layout(model, stations)
        results, expected_children = {}, set()
        budget = _EXPANDED_LIMIT
        for index, ((name, kind, effective), entry) in enumerate(zip(_resolve_cases(model, ids), entries, strict=True)):
            entry = _fields(entry, f"/results/{index}", _CASE_FIELDS)
            fingerprint = _case_hash(model, effective)
            if (entry["id"], entry["kind"], entry["input_hash"]) != (name, kind, fingerprint):
                raise ArtifactError("case identity does not match the effective model loads")
            arrays, diagnostic, budget = _read_case_entry(entry, directory, index, layout, budget)
            if diagnostic is None:
                expected_children.add(f"{index:04d}")
            results[name] = _case_result(model, stations, name, kind, fingerprint, arrays,
                                         entry["strain_energy"], entry["relative_residual"], diagnostic)
        expected_root = {_MANIFEST}
        if expected_children:
            expected_root.add("cases")
            _check_entries(directory / "cases", expected_children)
        _check_entries(directory, expected_root)
        batch = LoadCaseResults(model, stations, versions, MappingProxyType(results))
        if batch.envelope_status != manifest["envelope_status"]:
            raise ArtifactError("envelope status does not match the selected outcomes")
        return batch
    except (OSError, ValueError, SyntaxError, TypeError, RuntimeError, EOFError,
            BadZipFile, struct.error, zlib.error) as error:
        if isinstance(error, ArtifactError):
            raise
        raise ArtifactError(f"invalid load case artifact: {error}") from error
