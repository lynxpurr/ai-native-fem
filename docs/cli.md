# Run ANF from the command line

Install the package in an active environment to get the `anf` command.
The examples below run from a copy of the source distribution or checkout.
Each solve creates a new result directory.

```sh
anf validate examples/models/cantilever.json --json
anf solve examples/models/cantilever.json --reference cantilever.closed-form.v1 --output beam-result --json
anf inspect beam-result --json
anf view beam-result --output beam-view.html
anf solve examples/models/thick-q4.json --reference plate.navier-31.v1 --output thick-result --json
anf solve examples/models/thin-q4.json --reference plate.thin-q4-navier-31.v1 --output thin-result --json
anf solve examples/structures/portal.json --output portal-result --json
anf validate examples/models/invalid-units.json --json
```

`validate` returns the normalized declaration and input fingerprint.
`solve` runs the static model and writes a result directory to a new path.
Its parent directory must already exist. `inspect` reads an existing result
and returns the report, diagnostics and array metadata. Use the
[Python artifact reader](model-format.md) to access the numeric arrays.
`view` exports an [offline HTML viewer](viewer.md) for static artifacts and
saved schema 1.4 modal artifacts.

`validate`, `solve` and `inspect` accept schema 1.0 beam/plate declarations
and schema 1.1 planar structure declarations. Named load cases,
combinations, end releases and rotational springs use schema 1.2/1.3 through
the [Python batch entry point](load-cases.md); the CLI rejects those schemas
with `unsupported_schema` and exit code 2.

Pass a registered `--reference` to compare with an independent reference.
Omitting the option evaluates the free-DOF residual. The thin Q4 example
completes the solve and records `verification_failed` in its report.

| Exit code | Outcome |
| --- | --- |
| 0 | Validation, solve, inspection or viewer export completed. Check `verification_status` for accuracy evidence. |
| 2 | Invalid command, unreadable or invalid model, or unsupported model or reference. |
| 3 | Numerical failure. The output directory contains a manifest with the diagnostic. |
| 4 | Output or inspection failed. This includes existing output and invalid artifacts. |

With `--json`, each invocation emits one JSON object to stdout. Input and
I/O error objects contain `error.code`, `error.path` and `error.message`.
The numerical-failure result includes the original solver diagnostic.
Without `--json`, successful results are indented and command and input
diagnostics go to stderr. Quote file and directory names containing spaces.
