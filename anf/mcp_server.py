"""Local stdio MCP tools for ANF declarations, solves and mesh studies."""

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4

from .application import run_analysis
from .artifacts import ArtifactError, _read_regular, _require_directory, load_result, save_result
from .model import ModelValidationError, _BEAM, _JSON_LIMIT, _PLATE, _SI, _STRUCTURE, _decode_json, _fields, _number
from .model import canonical_input_hash, validate_model
from .reporting import write_report
from .structures import MAX_DOFS, MAX_MEMBER_ELEMENTS, MAX_NODES, MAX_SEGMENTS, _structure_model_schema
from .studies import StudySpec, _MAX_MESHES, _MAX_OUTPUT_BYTES, _OBSERVABLES, _deflection, run_convergence
from .verification import REFERENCE_REGISTRY, _CLASSICAL_REFERENCES, _STRUCTURE_REFERENCE


_SUMMARY_LIMIT = 16 * 1024
_ARTIFACT_ID = re.compile(r"[0-9a-f]{32}")
_DEFAULT_TIMEOUT = 30
_TOOL_FIELDS = {
    "capabilities": ((), {}),
    "validate_model": (("model",), {}),
    "solve_model": (("model",), {"reference": None}),
    "convergence_study": (("study",), {}),
    "inspect_artifact": (("artifact_id",), {}),
}


def _object(properties, required=None):
    return {"type": "object", "properties": properties, "additionalProperties": False,
            "required": list(properties if required is None else required)}


def _model_schema():
    positive = {"type": "number", "exclusiveMinimum": 0}
    number = {"type": "number"}
    plate_count = {"type": "integer", "minimum": 2, "maximum": 199, "default": 8}
    beam = {
        "model": _object({"length": positive, "young_modulus": positive, "second_moment": positive,
                          "elements": {"type": "integer", "minimum": 1, "maximum": 200, "default": 8}},
                         ("length", "young_modulus", "second_moment")),
        "loads": _object({"tip_force": number, "uniform_load": number}, ()),
    }
    plate = {
        "model": _object({
            "lx": positive, "ly": positive, "thickness": positive, "young_modulus": positive,
            "poisson_ratio": {"type": "number", "exclusiveMinimum": -1, "exclusiveMaximum": 0.5},
            "nx": plate_count, "ny": plate_count, "shear_factor": positive,
            "support": {"enum": ["clamped", "hard_simply_supported"]},
            "formulation": {"enum": ["q4_full", "mitc4_rectangle"]},
        }, ("lx", "ly", "thickness", "young_modulus", "poisson_ratio")),
        "loads": _object({"pressure": number}),
    }
    plate["model"].update(
        description="The plate mesh is limited to 1800 DOFs.",
        allOf=[{
            "if": {"required": ["formulation"], "properties": {"formulation": {"const": "mitc4_rectangle"}}},
            "then": {"required": ["support"], "properties": {"support": {"const": "hard_simply_supported"}}},
        }],
    )
    return {
        **_object({"schema_version": {"const": "1.0"}, "problem_type": {"enum": [_BEAM, _PLATE]},
                   "units": _object({key: {"const": unit} for key, unit in _SI.items()}),
                   "model": {"type": "object"}, "loads": {"type": "object"},
                   "analysis": _object({"type": {"const": "static"}})}),
        "oneOf": [
            {"properties": {"problem_type": {"const": problem}, **fields}}
            for problem, fields in ((_BEAM, beam), (_PLATE, plate))
        ],
    }


def _tool_schemas():
    classical = _model_schema()
    model = {"type": "object", "oneOf": [classical, _structure_model_schema()]}
    classical_reference = {"enum": [None, *_CLASSICAL_REFERENCES]}
    solve = _object({"model": model, "reference": {"enum": [None, *REFERENCE_REGISTRY]}}, ("model",))
    solve.update({
        "if": {"properties": {"model": {"properties": {"schema_version": {"const": "1.1"}}}}},
        "then": {"properties": {"reference": {"enum": [None, _STRUCTURE_REFERENCE.id]}}},
        "else": {"properties": {"reference": classical_reference}},
    })
    meshes = {"type": "array", "minItems": 1, "maxItems": _MAX_MESHES, "items": {"oneOf": [
        _object({"elements": {"type": "integer", "minimum": 1, "maximum": 200}}),
        _object({axis: {"type": "integer", "minimum": 2, "maximum": 199} for axis in ("nx", "ny")}),
    ]}}
    study = _object({
        "model": classical, "meshes": meshes, "observable": {"enum": list(_OBSERVABLES.values())},
        "reference": classical_reference, "failure_policy": {"enum": ["stop", "record_and_continue"]},
        "max_solves": {"type": "integer", "minimum": 1, "maximum": _MAX_MESHES},
        "max_output_bytes": {"type": "integer", "minimum": 1, "maximum": _MAX_OUTPUT_BYTES},
        "base_model_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    }, ("model", "meshes", "observable"))
    return {
        "capabilities": ("Discover supported models, formulations and budgets.", _object({})),
        "validate_model": ("Validate a static SI declaration and compute its fingerprint.",
                           _object({"model": model})),
        "solve_model": ("Run one solve and return a result summary and local artifact ID.",
                        solve),
        "convergence_study": ("Run an explicit mesh sequence and save the study with its results.",
                              _object({"study": study})),
        "inspect_artifact": ("Read a schema 1.0 or 1.1 result or a schema 1.0 study by local artifact ID.",
                             _object({"artifact_id": {"type": "string", "pattern": "^[0-9a-f]{32}$"}})),
    }


def _capabilities(timeout):
    return {
        "model_schema": "1.0", "result_schema": "1.0", "study_schema": "1.0",
        "model_schemas": ["1.0", "1.1"], "result_schemas": ["1.0", "1.1"],
        "analysis_types": ["static"], "units": dict(_SI),
        "problems": {
            "cantilever_bending": {"formulations": ["euler_bernoulli_hermite"],
                                   "supports": ["cantilever"], "max_elements": 200},
            "rectangular_plate": {"formulations": ["q4_full", "mitc4_rectangle"],
                                  "supports": ["clamped", "hard_simply_supported"],
                                  "mitc4_support": "hard_simply_supported", "max_dofs": 1800},
            _STRUCTURE: {"model_schema": "1.1", "result_schema": "1.1",
                         "formulations": ["truss2d", "frame2d"], "supports": ["ux", "uy", "rz"],
                         "loads": ["nodal", "uniform_transverse"],
                         "max_nodes": MAX_NODES, "max_segments": MAX_SEGMENTS, "max_dofs": MAX_DOFS,
                         "max_frame_elements": MAX_MEMBER_ELEMENTS, "max_truss_elements": 1},
        },
        "observables": dict(_OBSERVABLES), "references": list(REFERENCE_REGISTRY),
        "limits": {"max_request_bytes": _JSON_LIMIT, "max_summary_bytes": _SUMMARY_LIMIT,
                   "max_study_solves": _MAX_MESHES, "max_study_output_bytes": _MAX_OUTPUT_BYTES,
                   "timeout_seconds": timeout},
    }


def _error(code, path, message):
    return {"error": {"code": code, "path": path, "message": message}}


def _result_summary(result):
    summary = {
        "kind": "result", "input_hash": result.input_hash, "solve_status": result.solve_status,
        "verification_status": result.verification_status, "relative_residual": result.relative_residual,
        "strain_energy": result.strain_energy, "diagnostics": list(result.diagnostics),
    }
    if result.schema_version == "1.1":
        summary["schema_version"] = result.schema_version
    if result.solve_status == "numerical_failure":
        diagnostic = result.diagnostics[0]
        summary.update(_error(diagnostic["code"], "", diagnostic["message"]))
    return summary


def _study_summary(report):
    steps = [{key: row[key] for key in ("input_hash", "solve_status", "verification_status",
                                       "observed_m", "relative_residual")} for row in report["steps"]]
    summary = {"kind": "study", "input_hash": report["study"]["base_model_hash"], "status": report["status"],
               "steps": steps, "estimate": report["estimate"]}
    if any(row["solve_status"] == "numerical_failure" for row in steps):
        summary.update(_error("numerical_failure", "", "One or more meshes failed. See the study report."))
    return summary


def _inspect_study(directory):
    report = _decode_json(_read_regular(directory / "report.json", _JSON_LIMIT))
    spec = StudySpec.from_dict(report["study"])
    if (report["schema_version"] != "1.0" or not isinstance(report["steps"], list)
            or len(report["steps"]) != len(spec.cases)):
        raise ArtifactError("invalid study report")
    for index, (row, model) in enumerate(zip(report["steps"], spec.cases, strict=True), 1):
        fingerprint = canonical_input_hash(model)
        if row["input_hash"] != fingerprint:
            raise ArtifactError("study step fingerprint differs")
        if row["solve_status"] == "not_run":
            if (row["artifact"] is not None or row["verification_status"] != "not_run"
                    or row["observed_m"] is not None or row["relative_residual"] is not None):
                raise ArtifactError("skipped mesh has a result")
            continue
        path = directory / f"mesh-{index:02}"
        _require_directory(path)
        expected = {"path": path.name, "manifest_sha256":
                    sha256(_read_regular(path / "manifest.json", _JSON_LIMIT)).hexdigest()}
        if row["artifact"] != expected:
            raise ArtifactError("study artifact digest or name differs")
        result = load_result(path)
        if (fingerprint, row["solve_status"], row["verification_status"]) != (
                result.input_hash, result.solve_status, result.verification_status):
            raise ArtifactError("study step status differs from its artifact")
        for field in ("observed_m", "relative_residual"):
            if row[field] is not None:
                row[field] = _number(row[field], f"/{field}")
        observed = _deflection(result) if result.solve_status == "solved" else None
        if (row["observed_m"], row["relative_residual"]) != (observed, result.relative_residual):
            raise ArtifactError("study observations differ from their artifact")
    summary = _study_summary(report)
    status = "complete" if all(row["solve_status"] == "solved" for row in summary["steps"]) else "partial"
    if summary["status"] != status:
        raise ArtifactError("study status differs from its steps")
    numbers = ("observed_order", "finest_discretization_estimate_m", "refinement_ratio")
    estimate = _fields(summary["estimate"], "/estimate", ("status", "reason", *numbers))
    if estimate["status"] not in ("estimated", "unavailable") or (
            estimate["reason"] is not None and not isinstance(estimate["reason"], str)):
        raise ArtifactError("invalid study estimate")
    for key in numbers:
        if estimate[key] is not None:
            estimate[key] = _number(estimate[key], f"/estimate/{key}")
    summary["estimate"] = estimate
    return summary


def _worker(staging):
    request = _decode_json(_read_regular(staging / "request.json", _JSON_LIMIT))
    name, arguments = request["name"], request["arguments"]
    try:
        if name == "solve_model":
            result = run_analysis(arguments["model"], {"reference": arguments["reference"]})
            save_result(result, staging, "artifact")
            summary = _result_summary(result)
        elif name == "convergence_study":
            study = run_convergence(StudySpec.from_dict(arguments["study"]))
            write_report(study, staging / "artifact")
            summary = _study_summary(study.to_dict())
        elif name == "inspect_artifact":
            identity = arguments["artifact_id"]
            if not isinstance(identity, str) or _ARTIFACT_ID.fullmatch(identity) is None:
                raise ModelValidationError("/artifact_id", "invalid_artifact_id", "expected a local artifact ID")
            directory = staging.parent / identity
            _require_directory(directory)
            if os.path.lexists(directory / "manifest.json"):
                summary = _result_summary(load_result(directory))
            else:
                summary = _inspect_study(directory)
            summary["artifact_id"] = identity
        else:
            raise ModelValidationError("", "unsupported_tool", "unknown ANF tool")
    except ModelValidationError as error:
        summary = _error(error.code, error.path, error.message)
    except (ArtifactError, OSError, KeyError, TypeError, ValueError):
        summary = _error("artifact_io", "/artifact_id" if name == "inspect_artifact" else "",
                         "The local artifact could not be read or written.")
    (staging / "response.json").write_text(json.dumps(summary, allow_nan=False), encoding="utf-8")
    return 0


async def _request_worker(workspace, name, arguments, timeout):
    import anyio

    request = json.dumps({"name": name, "arguments": arguments}, allow_nan=False).encode("utf-8")
    if len(request) > _JSON_LIMIT:
        return _error("size_limit", "", "Tool input is limited to 1 MiB.")
    with TemporaryDirectory(prefix=".anf-mcp-", dir=workspace) as temporary:
        staging = Path(temporary)
        (staging / "request.json").write_bytes(request)
        with anyio.fail_after(timeout):
            with anyio.CancelScope(shield=True):
                process = await anyio.open_process(
                    [sys.executable, "-I", "-m", "anf.mcp_server", "_worker", str(staging)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=staging,
                )
            async with process:
                try:
                    code = await process.wait()
                finally:
                    with anyio.CancelScope(shield=True):
                        if process.returncode is None:
                            try:
                                process.kill()
                            except ProcessLookupError:
                                pass
                            await process.wait()
            if code:
                return _error("worker_error", "", "The local worker did not complete.")
            response = json.loads(_read_regular(staging / "response.json", _JSON_LIMIT))
            await anyio.lowlevel.checkpoint()
            artifact = staging / "artifact"
            if artifact.is_dir() and response.get("kind") in ("result", "study"):
                identity = uuid4().hex
                destination = workspace / identity
                if os.path.lexists(destination):
                    raise ArtifactError("artifact ID is already in use")
                artifact.rename(destination)
                response["artifact_id"] = identity
        return response


def _bounded(payload):
    encoded = json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    if len(encoded.encode("ascii")) <= _SUMMARY_LIMIT:
        return payload, encoded
    summary = {name: payload[name][:128] for name in ("kind", "input_hash", "artifact_id",
                                                     "status", "solve_status")
               if isinstance(payload.get(name), str)}
    if "error" in payload:
        error = payload["error"]
        summary.update(_error(error["code"], error["path"][:256], error["message"][:256]))
    summary["truncated"] = True
    return summary, json.dumps(summary, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def create_server(workspace, timeout=_DEFAULT_TIMEOUT):
    """Register ANF tools for an explicit local workspace."""
    import anyio
    from mcp import types
    from mcp.server.lowlevel import Server

    workspace = Path(workspace).resolve(strict=True)
    _require_directory(workspace)
    if not math.isfinite(timeout) or not 0 < timeout <= 120:
        raise ValueError("timeout must be positive and at most 120 seconds")
    server = Server("anf")
    limiter = anyio.Semaphore(1)
    schemas = _tool_schemas()

    @server.list_tools()
    async def list_tools():
        return [types.Tool(name=name, description=description, inputSchema=schema)
                for name, (description, schema) in schemas.items()]

    @server.call_tool(validate_input=False)
    async def call_tool(name, arguments):
        try:
            if name not in _TOOL_FIELDS:
                raise ModelValidationError("", "unsupported_tool", "unknown ANF tool")
            required, defaults = _TOOL_FIELDS[name]
            args = _fields(arguments, "", required, defaults)
            if len(json.dumps(args, allow_nan=False).encode("utf-8")) > _JSON_LIMIT:
                raise ModelValidationError("", "size_limit", "Tool input is limited to 1 MiB.")
            if name == "capabilities":
                payload = _capabilities(timeout)
            elif name == "validate_model":
                model = validate_model(args["model"])
                payload = {"input_status": "valid", "input_hash": canonical_input_hash(model)}
            else:
                if name == "inspect_artifact":
                    identity = args["artifact_id"]
                    if not isinstance(identity, str) or _ARTIFACT_ID.fullmatch(identity) is None:
                        raise ModelValidationError("/artifact_id", "invalid_artifact_id", "expected a local artifact ID")
                with anyio.fail_after(timeout):
                    async with limiter:
                        payload = await _request_worker(workspace, name, args, timeout)
        except ModelValidationError as error:
            payload = _error(error.code, error.path, error.message)
        except TimeoutError:
            payload = _error("timeout", "", "The tool exceeded its time budget.")
        except (OSError, ArtifactError):
            payload = _error("artifact_io", "", "The local artifact operation failed.")
        except (TypeError, ValueError, OverflowError, RecursionError):
            payload = _error("invalid_arguments", "", "Expected finite JSON arguments.")
        payload, encoded = _bounded(payload)
        return types.CallToolResult(isError="error" in payload, structuredContent=payload,
                                    content=[types.TextContent(type="text", text=encoded)])

    return server


def main(argv=None):
    """Launch the optional ANF stdio server."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "_worker":
        return _worker(Path(argv[1]))
    parser = argparse.ArgumentParser(prog="anf-mcp", description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="existing writable artifact directory")
    parser.add_argument("--timeout", type=float, default=_DEFAULT_TIMEOUT, help="tool time budget in seconds")
    args = parser.parse_args(argv)
    try:
        import anyio
        from mcp.server.stdio import stdio_server
    except ModuleNotFoundError:
        parser.error('Install the optional dependency with pip install "ai-native-fem[mcp]".')
    try:
        server = create_server(args.workspace, args.timeout)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    async def serve():
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(serve)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
