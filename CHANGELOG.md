# Changelog

All notable changes to `cssd` are documented here. This project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) and the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) layout.

## [Unreleased]

## [0.1.0] - 2026-05-07

First public release.

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
