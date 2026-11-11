# Licence review (2026-11-11)

The package is licensed **Apache-2.0** (`LICENSE`, with `NOTICE`).

## Runtime dependencies

Read from the installed packages' metadata on Python 3.13; `tomli` is needed only on Python < 3.11.

| Package | Version checked | Licence | Note |
|---|---|---|---|
| httpx | 0.28.1 | BSD-3-Clause | direct dependency |
| httpcore | 1.0.9 | BSD-3-Clause | via httpx |
| h11 | 0.16.0 | MIT | via httpcore |
| anyio | 4.16.0 | MIT | via httpx |
| idna | 3.20 | BSD-3-Clause | via httpx |
| certifi | 2026.10.23 | MPL-2.0 | via httpx. File-level copyleft; used unmodified, which is compatible with Apache-2.0 distribution |
| typing_extensions | 4.16.0 | PSF-2.0 | via anyio |
| tomli | (Python < 3.11 only) | MIT | direct dependency on old Pythons |

No dependency is copied or vendored into this repository. Development tools (pytest, respx, ruff, mypy) are not
distributed.

## Data and third-party material

- **Our own data.** The calibration suites and the classifier dataset (`data/task_classification/`) were written
  for this project.
- **LeBlanc v4.** Its results enter only as cited numbers in `learning/priors.py`. No LeBlanc code, data or
  database is included.
- **External benchmarks.** CodeSecEval, IFEval, GSM8K, HumanEval and MBPP are **not** included. Their licences
  must be checked before any of them is added.
