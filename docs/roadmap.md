# Research roadmap

The current package is a classical, isotropic beam/rectangular plate research
baseline. This roadmap describes intended work, not available capabilities or
a delivery date. Contributions should follow CONTRIBUTING.md.

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

## Versioned models and results

Planned work starts with SI model files for the existing beam and rectangular
plate solvers and result artifacts that preserve each input fingerprint,
solve status and physical quantities. A shared analysis entry point will
support a command-line interface and a local MCP interface. Convergence
reports will record each mesh, reference comparison and failure.

## Plate research

The rectangular MITC4 implementation covers hard simply supported edges under
uniform pressure. The fully integrated Q4 remains the default and provides a
locking comparison. Extending MITC4 to clamped edges or general quadrilaterals
requires separate derivation and independent verification.

Additional material models, shells and higher-order formulations require separate
source review, mathematical specification and validation before inclusion.
