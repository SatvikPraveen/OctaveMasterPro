# Usage examples

Every snippet below is executed by `docs/check_usage_examples.m` (`make docs-check`),
so the code cannot drift from the library; the numbers in comments are
real outputs on Octave 8.4 / OpenBLAS 0.3.26. Run from the repository root after `addpath inst`.

## Reproducibility first

```octave
addpath inst
omp.repro.seed (42);              % seeds rand, randn, randi, rande, randg, randp
omp.repro.env_info ("print");     % Octave / BLAS / LAPACK / packages / git revision
```

## Numerical linear algebra

```octave
A = hilb (12);                                    % cond(A) ~ 1.7e16
[Q, R] = omp.linalg.householder_qr (A, "econ");
[Qm, ~] = omp.linalg.gram_schmidt (A, "mgs");
printf ("Householder: %.1e   MGS: %.1e\n", ...
        norm (eye (12) - Q' * Q), norm (eye (12) - Qm' * Qm));
% Householder: 1.5e-15   MGS: 1.4e-01      (MGS loses orthogonality ~ u*kappa)

omp.repro.seed (1);
B = randn (500, 40) * randn (40, 300);            % exact rank 40
[U, S, V] = omp.linalg.randomized_svd (B, 40, "Seed", 1);
printf ("relative error %.1e\n", norm (B - U * S * V', "fro") / norm (B, "fro"));
% relative error 2.3e-15
```

## Optimization

```octave
[x, f, info] = omp.optim.bfgs (@omp.optim.rosenbrock, [-1.2; 1]);
printf ("x = [%.6f %.6f], %d iterations, %d evaluations\n", x, info.iterations, info.nfev);
% x = [1.000000 1.000000], 38 iterations, 48 evaluations

printf ("gradient check: %.1e\n", omp.optim.gradcheck (@omp.optim.rosenbrock, [0.3; -0.7]));
% gradient check: 1.8e-11

e = abs (info.trace.x(:, 1) - 1);
printf ("empirical order of convergence: %.2f\n", omp.optim.convergence_order (e(end-8:end)));
% empirical order of convergence: 1.47      (superlinear, as expected for BFGS)
```

## Statistical inference

```octave
omp.repro.seed (3);
a = randn (40, 1) + 0.6;  b = randn (40, 1);
ci = omp.stats.bootstrap_ci (a, @mean, "Seed", 1);         % BCa interval for mean(a)
p = omp.stats.permutation_test (a, b, "B", 9999, "Seed", 2);
g = omp.stats.effect_size (a, b, "hedges_g");
printf ("mean(a) 95%% CI [%.2f, %.2f]; p = %.4f; Hedges' g = %.2f\n", ci, p, g);
% mean(a) 95% CI [0.08, 0.74]; p = 0.0533; Hedges' g = 0.43
% (true shift 0.6: one n = 40 sample estimates it with substantial noise;
%  the effect size is the quantity to report, not only the p-value)

q = omp.stats.p_adjust ([0.01 0.02 0.03 0.04 0.05], "holm")
% q = 0.050000   0.080000   0.090000   0.090000   0.090000   (matches R's p.adjust)
```

## Spectral estimation

```octave
fs = 1000;  t = (0:2^14 - 1)' / fs;
x = 2 * sin (2 * pi * 50 * t) + randn (size (t));
[P, f, info] = omp.signal.welch_psd (x, fs, "SegmentLength", 1024);
[~, k] = max (P);
printf ("peak %.1f Hz, %d segments, ~%.0f degrees of freedom\n", f(k), info.segments, info.dof);
% peak 49.8 Hz, 31 segments, ~59 degrees of freedom
% (bin spacing is fs/L = 0.98 Hz, so 49.8 Hz is the bin nearest 50 Hz)
```

## Leakage-aware model evaluation

```octave
sim = omp.pdm.simulate_fleet ("Units", 20, "Hours", 24 * 90, "Seed", 1);
[X, names] = omp.pdm.causal_features (sim);       % past-only windows
[y, ok] = omp.pdm.failure_labels (sim, 72);       % fails within 72 h; censoring-aware
keep = ok & all (isfinite (X), 2) & mod (sim.t, 6) == 0;
folds = omp.ml.purged_time_splits (sim.t(keep), 3, "Horizon", 72, "Embargo", 24);
Xk = X(keep, :);  yk = y(keep);
s = NaN (numel (yk), 1);
for i = 1:numel (folds)
  m = omp.ml.logistic_fit (Xk(folds(i).train, :), yk(folds(i).train), "Lambda", 1);
  s(folds(i).test) = omp.ml.logistic_predict (m, Xk(folds(i).test, :));
end
k = ! isnan (s);
[auc, se] = omp.ml.roc_auc (yk(k), s(k));
c = omp.ml.calibration (yk(k), s(k));
printf ("AUC %.3f (DeLong SE %.3f), AP %.3f at prevalence %.3f, ECE %.3f\n", auc, se, ...
        omp.ml.average_precision (yk(k), s(k)), mean (yk(k)), c.ece);
% AUC 0.825 (DeLong SE 0.009), AP 0.476 at prevalence 0.132, ECE 0.031
```

DeLong's standard error treats rows as independent. Rows from the same
machine are correlated, so for inference use
`omp.stats.cluster_bootstrap` over machines, as the flagship study does.

## Benchmarking

```octave
M = randn (200) + 200 * eye (200);
r = omp.bench.timeit (@() M \ ones (200, 1), "Repeats", 25);
printf ("median %.2f us (IQR %.2f us)\n", 1e6 * r.median, 1e6 * r.iqr);   % machine-dependent

fit = omp.bench.amdahl_fit ([1 2 4 8], [1 1.9 3.4 5.6]);
printf ("serial fraction %.3f -> speedup ceiling %.1fx\n", fit.serial_fraction, fit.max_speedup);
% serial fraction 0.059 -> speedup ceiling 16.9x
```
