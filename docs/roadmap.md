# Research roadmap

The current package is a classical, isotropic beam/rectangular plate research
baseline. The sections below record the capabilities by version and outline
future research. See [CONTRIBUTING](../CONTRIBUTING.md) for contributions.

## Maintenance iteration

- Reproducible cantilever, thick-plate convergence and thin-plate locking examples.
- Wheel and source distribution installation tests on Linux Python 3.11/3.12
  and Windows Python 3.12.
- Dependency advisory checking and reviewed dependency/Actions update PRs.
- Investigate fine-mesh beam conditioning using analytical displacement, force
  balance and the original residual requirement. Model-size limits alone do not
  guarantee a solve.

The 0.1.0a3 prerelease includes verification reports and rectangular MITC4.
Both public distributions were downloaded, checked against the approved CI
build and installed again.
[Prerelease](https://github.com/lynxpurr/ai-native-fem/releases/tag/v0.1.0a3) ·
[Release policy](release-policy.md)

## Versioned models and results in 0.1.0a4

The [0.1.0a4 release notes](releases/0.1.0a4.md) describe the shared analysis
workflow for the existing beam and rectangular plate solvers. Use versioned
SI model files and JSON/NPZ result artifacts from
[Python](model-format.md) or the [installed command line](cli.md).
[Mesh convergence reports](convergence.md) record each mesh, reference
comparison and failure for replay. The [optional local MCP server](mcp.md)
exposes these operations over stdio.

## Plate research

The rectangular MITC4 implementation covers hard simply supported edges under
uniform pressure. The fully integrated Q4 remains the default and provides a
locking comparison. Extending MITC4 to clamped edges or general quadrilaterals
requires separate derivation and independent verification.

Additional material models, shells and higher-order formulations require separate
source review, mathematical specification and validation before inclusion.
