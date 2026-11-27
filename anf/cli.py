"""Command-line interface for SI model declarations and result artifacts."""

import argparse
import json
from pathlib import Path
import sys

from .application import run_analysis
from .artifacts import ArtifactError, load_result, save_result
from .model import ModelValidationError, _JSON_LIMIT, canonical_input_hash, validate_model
from .viewer import ViewerOutputError, export_result_view, open_result_artifact


class _UsageError(ValueError):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise _UsageError(message)


def _parser():
    parser = _Parser(prog="anf", description=__doc__, allow_abbrev=False)
    parser.add_argument("--json", action="store_true", help="emit one JSON object on stdout")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate a model declaration", allow_abbrev=False)
    validate.add_argument("model", type=Path)

    solve = commands.add_parser("solve", help="solve a model and write a result directory", allow_abbrev=False)
    solve.add_argument("model", type=Path)
    solve.add_argument("--output", required=True, type=Path)
    solve.add_argument("--reference", help="registered verification reference ID")

    inspect = commands.add_parser("inspect", help="read a result directory", allow_abbrev=False)
    inspect.add_argument("artifact", type=Path)
    view = commands.add_parser("view", help="export an offline result viewer", allow_abbrev=False)
    view.add_argument("artifact", type=Path)
    view.add_argument("--output", required=True, type=Path)
    for command in (validate, solve, inspect, view):
        command.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                             help="emit one JSON object on stdout")
    return parser


def _read_model(path):
    try:
        with path.open("rb") as stream:
            payload = stream.read(_JSON_LIMIT + 1)
    except OSError as error:
        raise ModelValidationError("", "input_io", str(error)) from error
    return validate_model(payload)


def _summary(result, path):
    summary = {
        "input_hash": result.input_hash,
        "artifact": str(path.absolute()),
        "solve_status": result.solve_status,
        "verification_status": result.verification_status,
        "reference": result.reference,
        "report": json.loads(result.report.to_json()),
        "diagnostics": list(result.diagnostics),
        "versions": dict(result.versions),
        "strain_energy": result.strain_energy,
        "relative_residual": result.relative_residual,
        "arrays": result.array_metadata,
    }
    if result.ordering is not None:
        summary.update(schema_version=result.schema_version, ordering=result.ordering)
    return summary


def _dispatch(args):
    if args.command == "inspect":
        return _summary(load_result(args.artifact), args.artifact), 0
    if args.command == "view":
        result = open_result_artifact(args.artifact)
        path = export_result_view(result, args.output)
        return {**_summary(result, args.artifact), "output": str(path.absolute())}, 0
    model = _read_model(args.model)
    if args.command == "validate":
        return {"input_status": "valid", "input_hash": canonical_input_hash(model),
                "model": model.to_dict()}, 0
    result = run_analysis(model, {"reference": args.reference})
    path = save_result(result, args.output.parent, args.output.name)
    return _summary(result, path), 3 if result.solve_status == "numerical_failure" else 0


def _error(code, path, message, *, machine):
    if machine:
        print(json.dumps({"error": {"code": code, "path": path, "message": message}}, allow_nan=False))
    else:
        print(f"{path or '/'}: {code}: {message}", file=sys.stderr)


def main(argv=None):
    """Run the console command and return its process exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except _UsageError as error:
        machine = "--json" in argv
        if not machine:
            parser.print_usage(sys.stderr)
        _error("invalid_command", "", str(error), machine=machine)
        return 2

    try:
        payload, status = _dispatch(args)
    except ModelValidationError as error:
        _error(error.code, error.path, error.message, machine=args.json)
        return 2
    except ArtifactError as error:
        path = "/output" if args.command == "solve" else "/artifact"
        _error("artifact_io", path, str(error), machine=args.json)
        return 4
    except ViewerOutputError as error:
        _error("artifact_io", "/output", str(error), machine=args.json)
        return 4

    print(json.dumps(payload, allow_nan=False, indent=None if args.json else 2))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
