# Contributing

Contributions are welcome. This project prioritizes **correctness you can
check** over feature count, so every change to the library is held to the
standards below.

## Development setup

```bash
sudo apt-get install octave octave-signal octave-statistics   # Octave >= 6.1
make check        # library tests + parse check of every .m file
```

Or use the container: `docker compose run --rm octave-cli make check`.

## Library code (`inst/+omp`)

- **One public function per file**, in the namespace that matches its
  domain (`omp.linalg`, `omp.optim`, `omp.stats`, `omp.signal`, `omp.ml`,
  `omp.pdm`, `omp.bench`, `omp.repro`, `omp.io`).
- **Texinfo help block** describing inputs, outputs, defaults, the method
  and a reference for any non-trivial algorithm.
- **Built-in tests** (`%!test`, `%!assert`, `%!error`) in the same file.
  Tests must check results against something independent of the
  implementation:
  - an analytic result (e.g. Parseval, a closed-form minimizer);
  - a published reference value (e.g. R's `p.adjust`, scikit-learn);
  - a brute-force definition (e.g. pairwise AUC);
  - a theoretical bound (e.g. orthogonality loss `O(u κ)` for MGS);
  - a Monte Carlo property with an explicit tolerance derived from its
    standard error (coverage, type-I error, bias).
- **Never assert a single random draw against a probabilistic guarantee**
  (e.g. "this 95% CI covers the truth"): such a test fails 5% of the time
  by construction. Test the property by simulation instead.
- **Seed everything** with `omp.repro.seed`, and use the `"Seed"` options,
  which restore the caller's RNG state.
- Write Octave-portable code (Octave >= 6.1). Beware: inside `[...]` and
  `{...}`, `f (x)` with a space parses as two elements, so write `f(x)`.

## Experiments

- State hypotheses, controls and the evaluation protocol in the script
  header **before** running. Report negative results.
- Fit every data-dependent transform (imputation, scaling, feature
  selection) inside the training fold.
- Treat the independent sampling unit (machine, subject) as the unit of
  resampling for uncertainty estimates.
- Commit generated results only from a clean working tree. The
  `environment.txt` file records the git revision and a dirty flag.

## Commits and pull requests

- Small, logically separated commits with descriptive messages.
- `make check` must pass. CI runs it on Octave 8.4 and 6.4, plus a quick
  run of the flagship experiment with sanity assertions on its controls.
