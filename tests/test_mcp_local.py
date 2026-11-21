"""Exercise the optional server over real stdio with an installed client."""

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
from uuid import uuid4

import pytest

pytest.importorskip("mcp")
import anyio
from anyio.streams.buffered import BufferedByteReceiveStream
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anf import mcp_server
from anf.artifacts import load_result


_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def source(path):
    return json.loads((_EXAMPLES / path).read_bytes())


def server_command(workspace, timeout):
    executable = Path(sysconfig.get_path("scripts")) / ("anf-mcp.exe" if os.name == "nt" else "anf-mcp")
    if Path(mcp_server.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        assert executable.is_file(), "the package must install the anf-mcp command"
    base = [str(executable)] if executable.is_file() else [sys.executable, "-I", "-m", "anf.mcp_server"]
    return [*base, "--workspace", str(workspace), "--timeout", str(timeout)]


@asynccontextmanager
async def connection(workspace, timeout=30):
    command, *args = server_command(workspace, timeout)
    params = StdioServerParameters(command=command, args=args, cwd=workspace.parent,
                                  env={"OPENBLAS_NUM_THREADS": "1"})
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            with anyio.fail_after(45):
                init = await session.initialize()
                assert init.protocolVersion == "2025-11-25"
                yield session


async def call(session, name, args, *, error=False):
    response = await session.call_tool(name, args)
    assert response.isError == error
    text = response.content[0].text
    assert len(text.encode("utf-8")) <= 16 * 1024
    result = json.loads(text)
    assert response.structuredContent == result
    return result


def test_real_client_matches_cli_and_reads_local_artifacts(tmp_path):
    workspace = tmp_path / "本地 workspace"
    workspace.mkdir()
    model = source("models/cantilever.json")
    path = tmp_path / "model.json"
    path.write_text(json.dumps(model), encoding="utf-8")
    reference = "cantilever.closed-form.v1"
    cli = subprocess.run([sys.executable, "-I", "-m", "anf.cli", "solve", str(path),
                          "--output", "cli", "--reference", reference, "--json"],
                         cwd=tmp_path, check=True, capture_output=True)
    expected = json.loads(cli.stdout)

    async def run():
        async with connection(workspace) as session:
            tools = {tool.name for tool in (await session.list_tools()).tools}
            assert tools == {"capabilities", "validate_model", "solve_model",
                             "convergence_study", "inspect_artifact"}
            caps = await call(session, "capabilities", {})
            assert caps["problems"]["rectangular_plate"]["max_dofs"] == 1800
            validated = await call(session, "validate_model", {"model": model})
            solved = await call(session, "solve_model", {"model": model, "reference": reference})
            inspected = await call(session, "inspect_artifact", {"artifact_id": solved["artifact_id"]})
            assert validated["input_hash"] == solved["input_hash"] == expected["input_hash"]
            assert solved["verification_status"] == expected["verification_status"]
            assert inspected == solved
            assert load_result(workspace / solved["artifact_id"]).input_hash == solved["input_hash"]

            studied = await call(session, "convergence_study", {"study": source("studies/thick-q4.json")})
            artifact = workspace / studied["artifact_id"]
            report = json.loads((artifact / "report.json").read_bytes())
            assert [row["input_hash"] for row in studied["steps"]] == [
                row["input_hash"] for row in report["steps"]]
            assert await call(session, "inspect_artifact", {"artifact_id": studied["artifact_id"]}) == studied
            report["steps"][0]["artifact"]["path"] = "../cli"
            (artifact / "report.json").write_text(json.dumps(report), encoding="utf-8")
            corrupt = await call(session, "inspect_artifact", {"artifact_id": studied["artifact_id"]}, error=True)
            assert corrupt["error"]["code"] == "artifact_io"

    anyio.run(run)


def test_tool_errors_preserve_real_failure(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def run():
        async with connection(workspace) as session:
            bad_units = source("models/cantilever.json")
            bad_units.pop("units")
            response = await call(session, "validate_model", {"model": bad_units}, error=True)
            assert response["error"]["path"] == "/units"
            bad_support = source("models/thick-q4.json")
            bad_support["model"].pop("support")
            bad_support["model"]["formulation"] = "mitc4_rectangle"
            response = await call(session, "solve_model", {"model": bad_support}, error=True)
            assert response["error"]["path"] == "/model/support"
            bad_type = source("models/cantilever.json")
            bad_type["problem_type"] = "modal"
            response = await call(session, "solve_model", {"model": bad_type}, error=True)
            assert response["error"]["code"] == "unsupported_problem"
            for identity, code in (("../outside", "invalid_artifact_id"),
                                   (str(tmp_path), "invalid_artifact_id"), (uuid4().hex, "artifact_io")):
                response = await call(session, "inspect_artifact", {"artifact_id": identity}, error=True)
                assert response["error"]["code"] == code
            failure = source("models/cantilever.json")
            failure["model"]["length"] = 1e-280
            response = await call(session, "solve_model", {"model": failure}, error=True)
            assert response["error"]["code"] == "numerical_failure"
            assert response["solve_status"] == "numerical_failure"
            artifact = load_result(workspace / response["artifact_id"])
            assert artifact.solve_status == "numerical_failure"
            assert not artifact.arrays

    anyio.run(run)


def test_timed_out_tool_discards_staged_result(tmp_path):
    workspace = tmp_path / "timeout"
    workspace.mkdir()

    async def run():
        async with connection(workspace, timeout=0.01) as session:
            result = await call(session, "solve_model", {"model": source("models/cantilever.json")}, error=True)
            assert result["error"]["code"] == "timeout"
            assert "artifact_id" not in result
            assert not list(workspace.iterdir())
            await call(session, "capabilities", {})

    anyio.run(run)


def test_large_tool_diagnostic_is_bounded():
    payload, encoded = mcp_server._bounded({"error": {
        "code": "unknown_field", "path": "/" + "😀" * 10000, "message": "Unknown field " + "😀" * 10000,
    }})
    assert len(encoded.encode("utf-8")) <= 16 * 1024
    assert payload["truncated"]
    assert payload["error"]["code"] == "unknown_field"


def test_real_stdio_cancellation_leaves_no_completed_output(tmp_path):
    workspace = tmp_path / "cancel"
    workspace.mkdir()

    async def run():
        env = {**os.environ, "OPENBLAS_NUM_THREADS": "1"}
        async with await anyio.open_process(server_command(workspace, 30), env=env, cwd=tmp_path) as process:
            try:
                incoming = BufferedByteReceiveStream(process.stdout)

                async def send(method, params, identity=None):
                    request = {"jsonrpc": "2.0", "method": method, "params": params}
                    if identity is not None:
                        request["id"] = identity
                    await process.stdin.send(json.dumps(request).encode("ascii") + b"\n")

                async def receive(identity):
                    while True:
                        reply = json.loads(await incoming.receive_until(b"\n", 65536))
                        assert reply["jsonrpc"] == "2.0"
                        if reply.get("id") == identity:
                            return reply

                with anyio.fail_after(30):
                    await send("initialize", {"protocolVersion": "2025-11-25", "capabilities": {},
                                              "clientInfo": {"name": "ANF cancellation test", "version": "1"}}, 1)
                    assert (await receive(1))["result"]["protocolVersion"] == "2025-11-25"
                    await send("notifications/initialized", {})
                    await send("tools/call", {"name": "convergence_study",
                                              "arguments": {"study": source("studies/thick-q4.json")}}, 2)
                    while not any(workspace.glob(".anf-mcp-*")):
                        await anyio.sleep(0.005)
                    await send("notifications/cancelled", {"requestId": 2, "reason": "test cancellation"})
                    while any(workspace.glob(".anf-mcp-*")):
                        await anyio.sleep(0.01)
                    assert not list(workspace.iterdir())
                    await send("tools/call", {"name": "capabilities", "arguments": {}}, 3)
                    assert "result" in await receive(3)
            finally:
                if process.returncode is None:
                    process.kill()

    anyio.run(run)
