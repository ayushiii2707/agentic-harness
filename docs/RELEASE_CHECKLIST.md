# Release checklist

**Nothing is published without the owner's explicit approval** (master prompt, rule 24). This list gets a release
ready; publishing is a separate, approved step.

## Before tagging

- [ ] **CI is green** on the release commit: Python 3.10–3.13 on Linux, 3.13 on macOS, the Docker sandbox job and
      the package job.
- [ ] **Clean checkout:**
  - [ ] `git status` is clean.
  - [ ] No untracked files inside `src/` (a wheel built from a working folder ships untracked modules; see
        evaluation finding F13).
- [ ] **Build only from a clean checkout:**
      `git archive HEAD | tar -x -C /tmp/rel && cd /tmp/rel && python -m build`.
- [ ] **Wheel contents:** only `adaptive_harness/` and its `.dist-info`; no `experiments/`, no databases, no local
      settings files.
- [ ] **sdist contents:** no `.harness/`, `*.db` or local settings files.
- [ ] **Clean install:** a fresh virtual environment installs the wheel, and the quickstart runs.
- [ ] **Credentials scan:** the scan of the release diff finds no key values.
- [ ] **Version:**
  - [ ] `src/adaptive_harness/_version.py` is bumped.
  - [ ] `CHANGELOG.md` moves "Unreleased" to the version and date.
- [ ] **Docs match the code:**
  - [ ] `docs/API.md` signatures are regenerated if the API changed.
  - [ ] README examples run.
  - [ ] `docs/PROVIDERS.md` live-verification dates are current.
- [ ] **Licence review is current** (`docs/LICENSES.md`), in particular after any dependency change.
- [ ] **Known limitations** are stated in the README and in `docs/DEVLOG.md`, including Phase 12 results with
      their caveats.

## Publishing (only after approval)

- [ ] Check that the package name is available on PyPI.
- [ ] Upload to TestPyPI first, install from it in a fresh environment, and run the quickstart.
- [ ] Upload to PyPI; create the git tag and the GitHub release with the changelog section.
