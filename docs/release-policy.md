# Release policy

## Review the candidate

Start a feature pull request against `develop`. Review the full diff and any
new commits against the public baseline. Update the
[exact file inventory](../.github/workflows/release-files.json) when paths
change. Its groups list repository-only files, installed Python sources,
licenses, and additional sdist files such as tests and examples. Review any
new model data or third-party resources individually before adding them.

Record the origin and distribution rights of each changed path, including any
third-party material or dependency and its license obligations. Keep detailed
source cards and review receipts in the maintainer's internal records. Review
the proposed public refs and run a secret scan before publication. GitHub
secret scanning and push protection and the dependency-advisory workflow
remain part of the checks.

Review numerical evidence independently. For a new or changed method, record
the independent reference, tested domain, error budget and observed errors.
Keep the installed regression suite and examples passing. A passing suite
does not establish source rights. See the
[method notes](methods.md) and [validation record](validation.md).

## Supported Python capabilities

| Entry point | Scope |
| --- | --- |
| `anf.Beam`, `anf.BeamResult`, `anf.solve_cantilever` | Linear Euler–Bernoulli bending of a cantilever in one plane, with a tip force and/or uniform line load. |
| `anf.Plate`, `anf.PlateResult`, `anf.solve_plate` | Linear Reissner–Mindlin bending under uniform pressure on axis-aligned rectangles. The default `q4_full` supports clamped and hard simply supported edges and can lock on thin plates. `mitc4_rectangle` requires hard simple support. |
| `anf.verification.verify_cantilever`, `verify_plate`, `evaluate_checks`, `input_parameters` | Opt-in verification with report schema 1.0, an input fingerprint and applicability checks. Automatic MITC4 reports check the residual and leave successful solves `solved_unverified`. |
| `anf.verification.VerificationReport`, `VerificationCheck`, `ReferenceKind`, `ReferenceRecord`, `REFERENCE_REGISTRY` | Report serialization and the bundled read-only reference registry described in the [verification guide](verification.md). |

The base installation uses NumPy 2.x. The `test` extra supplies pytest and
build tooling. CI records the resolved versions for Linux Python 3.11 and
3.12 and Windows Python 3.12. See [NOTICE](../NOTICE) for license details.
List API and report-schema migration instructions in the release notes when
behavior changes. The verification guide documents the preserved default Q4
fingerprint and the distinct MITC4 fingerprint.

## Capture candidate evidence

The three required installation jobs build a wheel and sdist from the exact
Git tree and check their file lists, source and license bytes, package
metadata, SHA-256 digests and wheel `RECORD`. Each package is installed in
its own clean environment outside the checkout. The jobs run the installed
test suite and four example commands. After they pass, each job uploads both
packages, `SHA256SUMS` and `release-manifest.json`. The manifest records the
commit, tree, version, checked paths, package hashes, installed dependencies
and CI run and attempt. For pull requests, it also records the base and
head SHAs alongside the actual tested merge commit.

To rehearse locally from the repository root with Python 3.11 or 3.12:

```bash
python -m pip install "build>=1.2,<2"
python .github/workflows/release-evidence.py build dist
```

Use a fresh output directory. Record the PR checks against the candidate head,
then compare its tree with the squash commit on `develop`. Review any tree
differences and validate the resulting tree. Keep the post-merge installation
matrix and advisory run links with the receipt.

## Publish and verify the downloads

1. Promote the integrated changes through a pull request from this
   repository's `develop` to `main`, using a merge commit. Record the
   reviewed source, numerical evidence, version, API migrations and passing
   checks. Select one successful installation job from the push workflow on
   the exact `main` commit for the release assets. Keep its CI manifest
   independently of the public downloads, along with the full matrix and
   advisory run links. Archive this evidence before Actions artifacts expire.
2. Create a new `v<version>` tag on that commit. Record the tag object SHA
   (`git rev-parse "refs/tags/<tag>"`) and its peeled commit and tree.
   Upload that job's wheel, sdist and `SHA256SUMS` to the new GitHub Release.
   Record the release URL, asset names and digests.
3. Download the three assets from their public release URLs without
   credentials into an empty directory. From a checkout of the public tag,
   run the verifier with the saved CI manifest and tag object SHA:

   ```bash
   python .github/workflows/release-evidence.py verify-download downloads \
     --manifest /path/to/approved/release-manifest.json \
     --tag "$TAG" --tag-object "$EXPECTED_TAG_OBJECT" \
     --receipt /path/to/new/download-receipt.json
   ```

   Set `TAG` and `EXPECTED_TAG_OBJECT` from the release record.
   The verifier checks the saved build identity and hashes before installing
   both packages again. Record the public URLs and download receipt with the
   generation and CI evidence. Candidate or local rehearsal evidence and
   public download evidence are retained as separate records.

Keep issued tags and assets fixed. If a release must be withdrawn, record the
affected version and digests, explain the problem and point users to a known
good version. Publish any fix using a new version and new assets.
