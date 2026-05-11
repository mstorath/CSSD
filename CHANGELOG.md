# Changelog

All notable changes to `cssd` are documented here. This project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) and the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) layout.

## [Unreleased]

## [1.0.2] - 2026-05-11

Metadata maintenance release. No algorithmic changes.

### Changed

- README harmonised with the lab-repo family: badge block, TL;DR sentence, expanded "See also" section linking the five sibling repos and two external related projects, License footer.
- `CITATION.cff` updated to `version: 1.0.2`, `date-released: 2026-05-11`.
- Auto-create GitHub Release on tag push (alongside PyPI publish).
- PyPI publish step set to `skip-existing: true` so workflow re-runs are idempotent.

## [1.0.1] - 2026-05-07

First PyPI release. Continues the version line of the MATLAB reference
implementation (last MATLAB tag: `v1.0.0`); the Rust core and Python
bindings introduced here ship under the next patch version so MATLAB
and Python consumers see a single coherent release history.

### Added
- Rust core (`cssd-core`) implementing cubic smoothing splines with
  discontinuities (CSSD) per Storath & Weinmann, *Smoothing splines for
  discontinuous signals*, JCGS 2024.
- Python package (`cssd`) with PyO3 bindings via `cssd-py`. Exposes
  `cssd(x, y, p, gamma, …)` and `cssd_cvscore(...)` with FPVI / PELT
  pruning and optional local preconditioning.
- 211 Python unit tests, 6 Rust integration tests, MATLAB-parity tests
  driven by 594 pre-baked `.mat` fixtures, paper-compliance tests
  validating equations from the JCGS 2024 paper.
- CITATION.cff with Storath & Weinmann ORCIDs and the JCGS 2024
  reference.

### Security
- Built against `pyo3 >= 0.24.1` (closes GHSA-pph8-gcv7-4qj5,
  `PyString::from_object` buffer-overread).

[Unreleased]: https://github.com/mstorath/CSSD/compare/v1.0.2...HEAD
[1.0.2]: https://github.com/mstorath/CSSD/releases/tag/v1.0.2
[1.0.1]: https://github.com/mstorath/CSSD/releases/tag/v1.0.1
