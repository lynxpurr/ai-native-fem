# Research roadmap

The current package is a classical, isotropic beam/rectangular plate research
baseline. This roadmap describes intended work, not available capabilities or
a delivery date. Contributions should follow CONTRIBUTING.md.

## Maintenance iteration

- Reproducible cantilever, thick-plate convergence and thin-plate locking examples.
- Installed-wheel testing on Linux Python 3.11/3.12 and Windows Python 3.12.
- Dependency advisory checking and reviewed dependency/Actions update PRs.
- Investigate fine-mesh beam conditioning using analytical displacement, force
  balance and the original residual requirement. Model-size limits alone do not
  guarantee a solve.

The next prerelease target is 0.1.0a3 after artifact validation. Existing release
tags and artifacts remain fixed; CI success is required before merging changes.

## Plate research

The rectangular MITC4 implementation covers hard simply supported edges under
uniform pressure. The fully integrated Q4 remains the default and provides a
locking comparison. Extending MITC4 to clamped edges or general quadrilaterals
requires separate derivation and independent verification.

Additional material models, shells and higher-order formulations require separate
source review, mathematical specification and validation before inclusion.
