# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-10-04

Research-grade overhaul: a tested core library, a ground-truth flagship
study, and reproducible infrastructure.

### Added
- `inst/+omp`: a namespaced library, installable with `pkg install`. Every
  function carries built-in tests against analytic or published references.
  - `linalg`: Householder QR, CGS/MGS/CGS2, randomized SVD.
  - `optim`: BFGS, strong-Wolfe line search, gradient check, convergence
    order estimation, Rosenbrock.
  - `stats`: BCa/percentile bootstrap, cluster bootstrap, permutation
    tests, p-value adjustment, effect sizes.
  - `signal`: Welch PSD with degrees-of-freedom bookkeeping.
  - `ml`: ridge logistic regression, k-NN, DeLong AUC, average precision,
    calibration metrics, purged forward-chaining splits.
  - `pdm`: fleet degradation simulator, causal features, censoring-aware
    labels.
  - `bench`, `repro`, `io`: robust timing, Amdahl fit, RNG seeding,
    environment capture, Octave-native CSV reader.
- Flagship simulation study (`flagship_project/experiment`) with
  pre-registered hypotheses, baselines, a negative control, leakage
  protocols and committed results.
- Data audit of the shipped flagship CSVs.
- `tests/run_tests.m` and `tests/check_parse.m`; GitHub Actions CI on
  Octave 8.4 and 6.4; Makefile; `CITATION.cff`; `CONTRIBUTING.md`.

### Changed
- Docker image: Ubuntu 24.04 (Octave 8.4), pinned Python tooling in a
  virtual environment, non-root user, token-protected Jupyter bound to
  localhost by docker compose.

### Fixed
- Five files that did not parse: duplicate subfunctions and a C-style
  ternary.
- Mask indexing on RGB images in `create_sample_images`.
- Missing `pkg load signal` before `butter`/`chirp`.
- Helpers that were unreachable because they were subfunctions of
  function files.
- A slow stock data generator (minutes → seconds).
- `generate_all_project_data` aborting on a self-copy.

### Security
- Removed the empty Jupyter token and password, which had exposed an
  unauthenticated server on all interfaces.

## [0.1.0] - 2025-08-31

Initial release: notebooks, mini projects, flagship scripts, cheat sheets.
