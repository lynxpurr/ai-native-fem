# AI-Native FEM (ANF)

ANF is a Python research package for linear static finite element analysis of
Euler–Bernoulli cantilevers and rectangular Reissner–Mindlin plates. The
`v0.1.0a4` research prerelease adds declarative SI models, result artifacts,
a command line, replayable mesh convergence studies, and optional local MCP.
[GitHub prerelease](https://github.com/lynxpurr/ai-native-fem/releases/tag/v0.1.0a4) ·
[Release notes](docs/releases/0.1.0a4.md)

## Install

Create and activate a virtual environment with Python 3.11 or newer.
Download `ai_native_fem-0.1.0a4-py3-none-any.whl` from the prerelease and
run this command from the directory containing the wheel.

```sh
python -m pip install "./ai_native_fem-0.1.0a4-py3-none-any.whl"
```

The base installation provides the Python API and the `anf` command. Pip
installs NumPy as a dependency.

## Solve a cantilever

### Python

```python
from anf import Beam, solve_cantilever

model = Beam(length=2, young_modulus=210e9, second_moment=1e-6)
result = solve_cantilever(model, tip_force=1000)
print(result.deflection[-1])  # approximately 0.0126984 m
```

### Command line

Get the example inputs from the matching source distribution on the release
page or the `v0.1.0a4` tag. Run these commands from the directory containing
`examples/`. Choose a fresh output directory each time you solve.
[Tagged cantilever input](https://github.com/lynxpurr/ai-native-fem/blob/v0.1.0a4/examples/models/cantilever.json)

```sh
anf validate examples/models/cantilever.json --json
anf solve examples/models/cantilever.json --reference cantilever.closed-form.v1 --output beam-result --json
anf inspect beam-result --json
```

Expect `verification_passed` for the analytical comparison with the bundled
cantilever reference. The result directory contains a JSON manifest with the
input fingerprint and verification report, plus the numeric arrays in
`arrays.npz`. [CLI guide](docs/cli.md) · [Model and result format](docs/model-format.md)

## Mesh studies and local MCP

- Run a mesh convergence study to compare explicit grids and save a replayable
  report with the result from each solve. [Convergence guide](docs/convergence.md)
- Scan cantilever parameters or estimate local sensitivities from Python.
  [Parameter studies](docs/parameter-studies.md)
- Use a local stdio MCP client to validate models, solve them, run studies,
  and inspect artifacts. [MCP guide](docs/mcp.md)

Install the wheel with the `mcp` extra to enable the server.

```sh
python -m pip install "./ai_native_fem-0.1.0a4-py3-none-any.whl[mcp]"
```

Configure your client to launch `anf-mcp --workspace PATH`. Use an existing
writable directory for the local artifacts.

## Supported models and verification

- **Cantilever.** One-plane Euler–Bernoulli bending under a tip force, uniform
  line load, or both.
- **Plate.** Reissner–Mindlin bending of an axis-aligned rectangle under
  uniform pressure. The default `q4_full` supports clamped and hard simply
  supported edges. Select `mitc4_rectangle` explicitly for hard simple support.

Use SI units. Dense assembly limits model size, and fine meshes can be
poorly conditioned. The direct Python solvers raise `ArithmeticError` on
numerical failure. For CLI or declarative runs, read the solve status and
diagnostics. [Methods and limits](docs/methods.md)

Check `verification_status` separately from solve completion. In the thin
Q4 shear-locking example, the CLI solve completes with exit code `0` and
reports `verification_failed`. Automatic MITC4 reports check the equilibrium
residual and report `solved_unverified` after a successful solve. The
validation record documents the independent MITC4 comparisons and tested
ranges. [Verification guide](docs/verification.md) · [Validation record](docs/validation.md)

## Development and contribution

From a source checkout with a virtual environment active, install the test
dependencies and run the suite.

```sh
python -m pip install -e ".[test]"
python -m pytest
```

Reproduce the analytical comparisons and see the planned research.
[Runnable examples](examples/README.md) · [Roadmap](docs/roadmap.md)

The project uses the MIT license. See the contribution and security guides
for source requirements and reporting.
[License](LICENSE) · [Notice](NOTICE) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)
