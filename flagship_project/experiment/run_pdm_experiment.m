function R = run_pdm_experiment (varargin)
  ## -*- texinfo -*-
  ## @deftypefn {} {@var{R} =} run_pdm_experiment ()
  ## @deftypefnx {} {@var{R} =} run_pdm_experiment ("Quick", true)
  ## @deftypefnx {} {@var{R} =} run_pdm_experiment ("Seeds", @var{seeds}, "OutDir", @var{dir}, @dots{})
  ## Simulation study of failure prediction for predictive maintenance,
  ## evaluated against a known ground truth.
  ##
  ## PRE-REGISTERED QUESTIONS (fixed before results were inspected)
  ## H1  A ridge logistic model on causal sensor features ranks
  ##     "fails within 72 h" better than the best single feature, where
  ##     that feature is selected inside each training fold.
  ## H2  A measurable gap remains to the oracle that observes the latent
  ##     health state (the Bayes ceiling for this generative model).
  ## H3  Evaluation protocol matters: (a) random row-level K-fold CV
  ##     inflates scores relative to purged forward-chaining CV, most for
  ##     the memorizing k-NN model; (b) centred (look-ahead) windows inflate
  ##     scores even under temporal CV.
  ## C1  Negative control: training on permuted labels gives AUC ~ 0.5.
  ##
  ## PROTOCOL
  ## - Each seed simulates an independent fleet (default 40 units x 120 d).
  ## - Rows are sub-sampled every @var{Stride} hours; labels exclude
  ##   offline, failure-hour and right-censored rows.
  ## - Primary protocol: forward-chaining CV, 4 test blocks, purge = label
  ##   horizon, embargo 24 h.  Imputation (training medians),
  ##   standardization and feature selection are fitted per training fold.
  ## - Out-of-fold predictions are pooled per seed; AUC, average precision
  ##   (with prevalence), Brier skill and ECE are reported as mean +/- SD
  ##   across seeds.  Units -- not rows -- are the independent sampling
  ##   unit, so uncertainty within a seed is quantified by a cluster
  ##   bootstrap over units, including paired model differences.
  ##
  ## Outputs (in @var{OutDir}, default flagship_project/results):
  ## metrics_by_seed.csv, summary.md, environment.txt, and
  ## roc_reliability.png when a graphics toolkit is available.
  ## @end deftypefn

  here = fileparts (mfilename ("fullpath"));
  root = fileparts (fileparts (here));
  addpath (fullfile (root, "inst"));

  C = struct ("seeds", 1:5, "units", 40, "days", 120, "stride", 6, ...
              "horizon", 72, "embargo", 24, "folds", 4, "lambda", 1, ...
              "knn_k", 25, "bootstrap", 1000, "quick", false, ...
              "outdir", fullfile (root, "flagship_project", "results"), ...
              "figure", true);
  for k = 1:2:numel (varargin)
    key = lower (varargin{k});
    if (! isfield (C, key))
      error ("run_pdm_experiment: unknown option '%s'", varargin{k});
    endif
    C.(key) = varargin{k+1};
  endfor
  if (C.quick)
    C.seeds = 1:2; C.units = 15; C.days = 60; C.bootstrap = 200;
  endif
  ensure_dir (C.outdir);

  env = omp.repro.env_info ();
  t_start = tic ();
  rows_out = {};
  boot = struct ();
  for si = 1:numel (C.seeds)
    seed = C.seeds(si);
    printf ("[seed %d] simulating %d units x %d days ...\n", seed, C.units, C.days);
    [D, X, Xc, names, y, units, t, oracle] = build_dataset (C, seed);
    printf ("[seed %d] %d eligible rows, prevalence %.3f, %d failures\n", ...
            seed, numel (y), mean (y), D.n_failures);

    F = omp.ml.purged_time_splits (t, C.folds, "Horizon", C.horizon, ...
                                   "Embargo", C.embargo);
    test_mask = any ([F.test], 2);

    ## --- Primary protocol: purged forward-chaining ---------------------
    S = struct ();
    S.oracle = oracle;
    S.prevalence = oof (F, X, y, @fit_prevalence);
    S.best_single = oof (F, X, y, @fit_best_single);
    S.logistic = oof (F, X, y, @(a, b, c) fit_logistic (a, b, c, C.lambda));
    S.knn = oof (F, X, y, @(a, b, c) fit_knn (a, b, c, C.knn_k));
    omp.repro.seed (1000 + seed);
    S.shuffled_control = oof (F, X, y, ...
      @(a, b, c) fit_logistic (a, b(randperm (numel (b))), c, C.lambda));

    ## --- H3a: random row-level K-fold (ignores time and units) ---------
    omp.repro.seed (2000 + seed);
    Fr = random_folds (numel (y), C.folds, test_mask);
    S.logistic_randomcv = oof (Fr, X, y, @(a, b, c) fit_logistic (a, b, c, C.lambda));
    S.knn_randomcv = oof (Fr, X, y, @(a, b, c) fit_knn (a, b, c, C.knn_k));

    ## --- H3b: look-ahead (centred) features under temporal CV ----------
    S.logistic_centered = oof (F, Xc, y, @(a, b, c) fit_logistic (a, b, c, C.lambda));
    S.knn_centered = oof (F, Xc, y, @(a, b, c) fit_knn (a, b, c, C.knn_k));

    models = fieldnames (S);
    for m = 1:numel (models)
      s = S.(models{m});
      k = test_mask & ! isnan (s);
      met = metrics (y(k), s(k));
      rows_out(end+1, :) = {seed, models{m}, met.auc, met.auc_se, met.ap, ...
                            mean(y(k)), met.brier_skill, met.ece, sum(k)};
    endfor

    if (si == 1)
      ## Cluster bootstrap over units for the first seed.
      k = find (test_mask);
      g = units(k);
      yy = y(k);
      auc = @(s, i) omp.ml.roc_auc (yy(i), s(k(i)));
      pairs = {"logistic", "best_single"; "oracle", "logistic"; ...
               "logistic_randomcv", "logistic"; "knn_randomcv", "knn"; ...
               "logistic_centered", "logistic"; "knn_centered", "knn"};
      boot.single = struct ();
      for m = {"oracle", "logistic", "best_single", "knn", "shuffled_control"}
        ss = S.(m{1});
        boot.single.(m{1}) = omp.stats.cluster_bootstrap (g, @(i) auc (ss, i), ...
                               "B", C.bootstrap, "Seed", 7);
      endfor
      boot.pairs = cell (rows (pairs), 3);
      for p = 1:rows (pairs)
        a = S.(pairs{p, 1}); b = S.(pairs{p, 2});
        [ci, th] = omp.stats.cluster_bootstrap (g, @(i) auc (a, i) - auc (b, i), ...
                                                "B", C.bootstrap, "Seed", 7);
        boot.pairs(p, :) = {sprintf("%s - %s", pairs{p, :}), th, ci};
      endfor
      boot.n_units = numel (unique (g));
      if (C.figure)
        make_figure (C, yy, S, k);
      endif
    endif
    ensure_dir (C.outdir);
    save ("-binary", fullfile (C.outdir, "partial_rows.octbin"), "rows_out", "boot");
  endfor
  unlink (fullfile (C.outdir, "partial_rows.octbin"));

  R = struct ("config", C, "env", env, "boot", boot, ...
              "elapsed_s", toc (t_start));
  R.table = cell2struct (rows_out, {"seed", "model", "auc", "auc_se_delong", ...
                         "ap", "prevalence", "brier_skill", "ece", "n"}, 2);
  ## Checkpoint everything before formatting outputs, so a failure while
  ## writing reports never discards a long run.
  ensure_dir (C.outdir);
  save ("-binary", fullfile (C.outdir, "results.octbin"), "R", "rows_out");
  write_outputs (R, rows_out);
  printf ("Done in %.1f s. Results in %s\n", R.elapsed_s, C.outdir);
endfunction

## ------------------------------------------------------------------------
function [D, X, Xc, names, y, units, t, oracle] = build_dataset (C, seed)
  sim = omp.pdm.simulate_fleet ("Units", C.units, "Hours", 24 * C.days, ...
                                "Seed", seed);
  [Xall, names] = omp.pdm.causal_features (sim);
  Xc_all = omp.pdm.causal_features (sim, "Centered", true);
  [yall, elig] = omp.pdm.failure_labels (sim, C.horizon);
  keep = elig & mod (sim.t, C.stride) == 0 & sim.t >= 72;   # warm-up for windows
  X = Xall(keep, :);
  Xc = Xc_all(keep, :);
  y = yall(keep);
  units = sim.unit(keep);
  t = sim.t(keep);
  oracle = sim.health(keep);
  D.n_failures = sum (sim.failure);
endfunction

function s = oof (F, X, y, fitpred)
  ## Out-of-fold scores; NaN for rows never in a test fold.
  s = NaN (numel (y), 1);
  for i = 1:numel (F)
    tr = F(i).train; te = F(i).test;
    med = median_omit (X(tr, :));
    Xtr = impute (X(tr, :), med);
    Xte = impute (X(te, :), med);
    s(te) = fitpred (Xtr, y(tr), Xte);
  endfor
endfunction

function m = median_omit (X)
  m = zeros (1, columns (X));
  for j = 1:columns (X)
    v = X(! isnan (X(:, j)), j);
    if (isempty (v))
      m(j) = 0;
    else
      m(j) = median (v);
    endif
  endfor
endfunction

function X = impute (X, med)
  for j = 1:columns (X)
    X(isnan (X(:, j)), j) = med(j);
  endfor
endfunction

function F = random_folds (n, k, eval_mask)
  ## Row-level random K-fold restricted to the same evaluation rows as the
  ## temporal protocol, so the two are scored on identical samples.
  idx = find (eval_mask);
  fold = mod (randperm (numel (idx)), k) + 1;
  F = struct ("train", {}, "test", {});
  for i = 1:k
    te = false (n, 1); te(idx(fold == i)) = true;
    F(i).train = ! te;
    F(i).test = te;
  endfor
endfunction

function p = fit_prevalence (Xtr, ytr, Xte)
  p = mean (ytr) * ones (rows (Xte), 1);
endfunction

function p = fit_best_single (Xtr, ytr, Xte)
  ## Select, on the training fold only, the single feature (and sign) with
  ## the highest training AUC; its raw value is the score.  A univariate
  ## logistic fit maps it to a probability for calibration metrics.
  best = 0.5; j = 1; sgn = 1;
  for c = 1:columns (Xtr)
    a = omp.ml.roc_auc (ytr, Xtr(:, c));
    if (max (a, 1 - a) > best)
      best = max (a, 1 - a); j = c; sgn = 2 * (a >= 0.5) - 1;
    endif
  endfor
  m = omp.ml.logistic_fit (sgn * Xtr(:, j), ytr, "Lambda", 1);
  p = omp.ml.logistic_predict (m, sgn * Xte(:, j));
endfunction

function p = fit_logistic (Xtr, ytr, Xte, lambda)
  m = omp.ml.logistic_fit (Xtr, ytr, "Lambda", lambda);
  p = omp.ml.logistic_predict (m, Xte);
endfunction

function p = fit_knn (Xtr, ytr, Xte, k)
  p = omp.ml.knn_predict (Xtr, ytr, Xte, k);
endfunction

function met = metrics (y, s)
  [met.auc, met.auc_se] = omp.ml.roc_auc (y, s);
  met.ap = omp.ml.average_precision (y, s);
  if (all (s >= 0 & s <= 1))
    c = omp.ml.calibration (y, s);
    met.brier_skill = c.brier_skill;
    met.ece = c.ece;
  else
    met.brier_skill = NaN;          # score is not a probability (oracle)
    met.ece = NaN;
  endif
endfunction

## ------------------------------------------------------------------------
function ensure_dir (d)
  if (! exist (d, "dir"))
    [ok, msg] = mkdir (d);
    if (! ok)
      error ("run_pdm_experiment: cannot create '%s': %s", d, msg);
    endif
  endif
endfunction

function fid = fopen_w (f)
  [fid, msg] = fopen (f, "w");
  if (fid < 0)
    error ("run_pdm_experiment: cannot write '%s': %s", f, msg);
  endif
endfunction

function write_outputs (R, rows_out)
  C = R.config;
  ensure_dir (C.outdir);
  fid = fopen_w (fullfile (C.outdir, "metrics_by_seed.csv"));
  fprintf (fid, "seed,model,auc,auc_se_delong,average_precision,prevalence,brier_skill,ece,n\n");
  for i = 1:rows (rows_out)
    fprintf (fid, "%d,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n", rows_out{i, :});
  endfor
  fclose (fid);

  fid = fopen_w (fullfile (C.outdir, "environment.txt"));
  e = R.env;
  fprintf (fid, "timestamp_utc: %s\noctave: %s\narch: %s\nnproc: %d\nblas: %s\nlapack: %s\npackages: %s\ngit_commit: %s\ngit_dirty: %d\n", ...
           e.timestamp_utc, e.octave_version, e.arch, e.nproc, e.blas, ...
           e.lapack, strjoin (e.packages, " "), e.git_commit, e.git_dirty);
  cfg = fieldnames (C);
  for i = 1:numel (cfg)
    v = C.(cfg{i});
    if (ischar (v))
      continue;                     # paths are machine-specific
    endif
    fprintf (fid, "config.%s: %s\n", cfg{i}, mat2str (v));
  endfor
  fclose (fid);

  models = unique (rows_out(:, 2), "stable");
  fid = fopen_w (fullfile (C.outdir, "summary.md"));
  fprintf (fid, "# Predictive-maintenance simulation study: results\n\n");
  fprintf (fid, "Generated by `flagship_project/experiment/run_pdm_experiment.m`");
  fprintf (fid, " (git `%s`). %d seeds x %d units x %d days, horizon %d h, ", ...
           R.env.git_commit(1:min(7, end)), numel (C.seeds), C.units, C.days, C.horizon);
  fprintf (fid, "stride %d h, %d purged forward-chaining folds (embargo %d h).\n\n", ...
           C.stride, C.folds, C.embargo);
  fprintf (fid, "## Out-of-fold metrics (mean ± SD across seeds)\n\n");
  fprintf (fid, "| Model / protocol | ROC AUC | Average precision | Prevalence | Brier skill | ECE |\n");
  fprintf (fid, "|---|---|---|---|---|---|\n");
  for m = 1:numel (models)
    k = strcmp (rows_out(:, 2), models{m});
    V = cell2mat (rows_out(k, [3 5 6 7 8]));
    fprintf (fid, "| `%s` | %s | %s | %s | %s | %s |\n", models{m}, ...
             ms (V(:, 1)), ms (V(:, 2)), ms (V(:, 3)), ms (V(:, 4)), ms (V(:, 5)));
  endfor
  fprintf (fid, "\n## Cluster bootstrap over units (seed %d, %d units, B = %d)\n\n", ...
           C.seeds(1), R.boot.n_units, C.bootstrap);
  fprintf (fid, "| Quantity | 95%% CI |\n|---|---|\n");
  bs = fieldnames (R.boot.single);
  for i = 1:numel (bs)
    ci = R.boot.single.(bs{i});
    fprintf (fid, "| AUC `%s` | [%.3f, %.3f] |\n", bs{i}, ci(1), ci(2));
  endfor
  for i = 1:rows (R.boot.pairs)
    fprintf (fid, "| ΔAUC `%s` = %+.3f | [%+.3f, %+.3f] |\n", R.boot.pairs{i, 1}, ...
             R.boot.pairs{i, 2}, R.boot.pairs{i, 3}(1), R.boot.pairs{i, 3}(2));
  endfor
  fprintf (fid, "\nBrier skill and ECE are undefined (blank) for the oracle, whose score is a latent state rather than a probability.\n");
  fclose (fid);
endfunction

function s = ms (v)
  v = v(isfinite (v));
  if (isempty (v))
    s = "–";
  elseif (numel (v) == 1)
    s = sprintf ("%.3f", v);
  else
    s = sprintf ("%.3f ± %.3f", mean (v), std (v));
  endif
endfunction

function make_figure (C, y, S, k)
  try
    h = figure ("visible", "off", "position", [100 100 1000 420]);
    subplot (1, 2, 1); hold on;
    list = {"oracle", "logistic", "best_single", "knn", "shuffled_control"};
    for m = 1:numel (list)
      [a, ~, ~, roc] = omp.ml.roc_auc (y, S.(list{m})(k));
      plot (roc.fpr, roc.tpr, "linewidth", 1.5, ...
            "displayname", sprintf ("%s (%.3f)", strrep (list{m}, "_", " "), a));
    endfor
    plot ([0 1], [0 1], "k:", "handlevisibility", "off");
    xlabel ("False positive rate"); ylabel ("True positive rate");
    title ("ROC, purged temporal CV (seed 1)");
    legend ("location", "southeast"); axis square; box on;
    subplot (1, 2, 2); hold on;
    for m = {"logistic", "best_single"}
      c = omp.ml.calibration (y, S.(m{1})(k), 10);
      plot (c.table(:, 1), c.table(:, 2), "o-", "linewidth", 1.5, ...
            "displayname", sprintf ("%s (ECE %.3f)", strrep (m{1}, "_", " "), c.ece));
    endfor
    plot ([0 1], [0 1], "k:", "handlevisibility", "off");
    xlabel ("Mean predicted probability"); ylabel ("Observed frequency");
    title ("Reliability (equal-mass bins)");
    legend ("location", "northwest"); axis square; box on;
    ensure_dir (C.outdir);
    print (h, fullfile (C.outdir, "roc_reliability.png"), "-dpng", "-r120");
    close (h);
  catch err
    warning ("run_pdm_experiment: figure skipped (%s)", err.message);
  end_try_catch
endfunction
