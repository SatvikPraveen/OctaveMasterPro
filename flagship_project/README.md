# Flagship: predictive maintenance against known ground truth

## Why a simulation study?

The original pipeline in `project_scripts/` was written for the CSV
files in `datasets/`. The audit `experiment/audit_shipped_data.m`
(output: [`results/data_audit.md`](results/data_audit.md)) found two
problems with those files:

- The sensor table covers **29 hours** of three machines.
- **No failure event falls inside that window**, even allowing a
  72-hour look-ahead.

Every label available to a failure-prediction model is therefore
negative, and no predictive performance can be estimated from these
data. That is a limit on what the data contain, and a better model
cannot fix it.

The study is instead built on a generative model whose ground truth is
known (`omp.pdm.simulate_fleet`), so every evaluation choice can be
checked against the right answer:

- **Latent health** `h(t)`: degradation follows a gamma process, with
  heterogeneous rates across machines.
- **Failure hazard**: `λ(t) = λ₀ exp(β h(t))`.
- **Renewal**: preventive maintenance resets health, which censors
  degradation paths.
- **Confounding**: an ambient temperature signal shared by all machines.
- **Machine offsets**: each machine has its own sensor offsets.
- **Missing readings**: a fraction of readings is missing at random.

Because the hazard depends on the sensors only through `h(t)`, a scorer
that sees `h(t)` itself is the *oracle*. It is the Bayes ceiling for
ranking "fails within 72 h".

## Design (pre-registered in the header of `experiment/run_pdm_experiment.m`)

| | |
|---|---|
| Fleet | 40 machines × 120 days, hourly; 5 independent seeds |
| Target | failure within the next 72 h. Rows that are offline, failure hours, or right-censored are excluded. |
| Features | trailing 24 h mean, SD and deviation from the machine's expanding mean; 72 h slope (temperature − ambient, log vibration, pressure); hours since renewal |
| Primary protocol | 4-block forward-chaining CV with label-horizon purging and a 24 h embargo |
| Fitted per training fold | median imputation, standardization, feature selection |
| Models | oracle, prevalence, best single feature (selected in-fold), ridge logistic regression, k-NN (k = 25) |
| Negative control | logistic regression trained on permuted labels |
| Leakage protocols | random row-level K-fold; centred (look-ahead) windows |
| Metrics | ROC AUC (with DeLong SE), average precision with prevalence, Brier skill, ECE |
| Uncertainty | spread across seeds, plus a cluster bootstrap over machines (paired for model differences) |

## Results

See [`results/summary.md`](results/summary.md),
[`results/metrics_by_seed.csv`](results/metrics_by_seed.csv) and
[`results/roc_reliability.png`](results/roc_reliability.png). The
interpretation is in the main [README](../README.md#flagship-study-results).

## Layout

| Path | Status |
|---|---|
| `experiment/run_pdm_experiment.m` | The study. `make experiment`, or `run_pdm_experiment ("Quick", true)` for a 1-minute smoke run. |
| `experiment/audit_shipped_data.m` | Data audit of `datasets/`. `make audit`. |
| `results/` | Committed outputs, with an `environment.txt` recording Octave, BLAS, git revision and configuration. |
| `datasets/` | Small illustrative CSVs. Usable for data-handling demos, **not** for predictive claims (see the audit). |
| `parallelized_pipeline_demo.m` | Runs under Octave and demonstrates parallel helpers from `utils/`. Without the `parallel` package it falls back to serial. |
| `project_scripts/`, `project_notebook.ipynb` | Legacy pipeline written against MATLAB's `table`/`datetime` API, which Octave lacks. Kept for reference only; superseded by `experiment/`. |
