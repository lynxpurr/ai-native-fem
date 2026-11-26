# Run ANF through local MCP

Download `ai_native_fem-0.1.0a7-py3-none-any.whl` from the [GitHub Releases
page](https://github.com/lynxpurr/ai-native-fem/releases). In an active
Python environment, run one of the following commands from the directory
containing the wheel.

For Python and the `anf` command line:

```sh
python -m pip install "./ai_native_fem-0.1.0a7-py3-none-any.whl"
```

To also enable local MCP:

```sh
python -m pip install "./ai_native_fem-0.1.0a7-py3-none-any.whl[mcp]"
```

Configure a stdio client to launch `anf-mcp --workspace PATH`, with an existing,
writable directory for local artifacts. Use the executable from the environment
where you installed ANF. The server runs without an API key.

The tools discover the supported static analysis domains, validate a
declaration, solve a model, run a mesh study and inspect an artifact by ID.
Pass a [model declaration](model-format.md) to `validate_model` and
`solve_model`. Pass a [study declaration](convergence.md) to
`convergence_study`. `capabilities` advertises supported formulations,
registered references and numerical limits.

`validate_model` and `solve_model` accept schema 1.0 beam/plate declarations
and schema 1.1 planar structure declarations. `inspect_artifact` reads result
schema 1.0 or 1.1, and `capabilities` reports both model and result schema
sets. Schema 1.2/1.3 batch load cases, sparse backend selection, parameter
studies and native mesh import are Python interfaces and are not exposed as
MCP tools.

Solve and study calls return a bounded summary and a workspace-local ID.
Use `inspect_artifact` with that ID to check the saved result. The complete
arrays and study files remain in the configured workspace. Tool errors carry
`error.code`, `error.path` and `error.message`. Numerical failures can
carry an artifact ID for the diagnostic. The default deadline is 30 seconds.
Use `--timeout` to set a deadline up to 120 seconds. Cancellation or timeout
discards the staged output. The server limits input to 1 MiB, tool summaries
to 16 KiB, and study work to the declared mesh and output budgets.

## Run a real client

The example model and study are in the matching `0.1.0a7` source distribution
and the `v0.1.0a7` tag. Run the example from the root of the unpacked source
distribution or tag checkout with the MCP extra installed.

```python
import json
from pathlib import Path
import sys

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

workspace = Path("mcp-work").resolve()
workspace.mkdir(exist_ok=True)
model = json.loads(Path("examples/models/cantilever.json").read_bytes())
study = json.loads(Path("examples/studies/thick-q4.json").read_bytes())
server = StdioServerParameters(
    command=sys.executable,
    args=["-m", "anf.mcp_server", "--workspace", str(workspace)],
)


async def demo():
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            initialized = await session.initialize()
            print(initialized.protocolVersion)
            for tool in (await session.list_tools()).tools:
                print(tool.name)

            valid = await session.call_tool("validate_model", {"model": model})
            solved = await session.call_tool("solve_model", {
                "model": model, "reference": "cantilever.closed-form.v1",
            })
            run = await session.call_tool("convergence_study", {"study": study})
            inspected = await session.call_tool("inspect_artifact", {
                "artifact_id": solved.structuredContent["artifact_id"],
            })
            assert all(not reply.isError for reply in (valid, solved, run, inspected))
            assert valid.structuredContent["input_hash"] == inspected.structuredContent["input_hash"]
            print(inspected.structuredContent["verification_status"], run.structuredContent["status"])


anyio.run(demo)
```
