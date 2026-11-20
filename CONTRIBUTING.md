# Contributing

Keep contributions within the documented mathematical model. Provide specific
references and an independent verification calculation. Contribute only code,
text and data that you are entitled to distribute under MIT.

Public availability of a paper does not grant permission to copy its code,
figures, text or datasets. Record which equations inform an implementation.
Do not translate an existing codebase without resolving its license and rights.
Preserve mathematical attribution and describe limitations accurately.

Run `python -m pytest` and include convergence evidence for numerical changes.
Do not replace an analytical reference with results of the code being tested.

## Branches and pull requests

1. Start each change from the latest `origin/develop` on its own
   `feature/<short-name>` branch.
2. Open a pull request to `develop`. Resolve review discussions and wait for
   the Linux and Windows wheel tests, dependency audit and branch-source check
   before using **Squash and merge**. Dependabot also targets `develop`.
3. To promote the integrated changes, open a pull request from this repository's
   `develop` branch to `main`. Wait for the checks on the proposed merge and
   use **Create a merge commit** to preserve the integrated history.

`main` is the release branch. Protected branches are updated through pull
requests. A promotion to `main` does not create a release or a version tag.

To start a feature from the current integration branch:

```bash
git fetch origin
git switch --no-track -c feature/my-change origin/develop
git push -u origin feature/my-change
```

Select `develop` as the base when opening the feature pull request. If
`branch-source` reports the wrong base, edit the pull request's base branch
and wait for the checks on the updated comparison.
