function [ci, theta, boot] = cluster_bootstrap (groups, statfn, varargin)
  ## -*- texinfo -*-
  ## @deftypefn {} {[@var{ci}, @var{theta}, @var{boot}] =} omp.stats.cluster_bootstrap (@var{groups}, @var{statfn})
  ## @deftypefnx {} {[@dots{}] =} omp.stats.cluster_bootstrap (@dots{}, "B", @var{B}, "Alpha", @var{alpha}, "Seed", @var{s})
  ## Percentile bootstrap confidence interval that resamples whole
  ## clusters (e.g. machines, patients, subjects) rather than rows.
  ##
  ## @var{groups} assigns each row to a cluster.  @var{statfn} receives a
  ## vector of row indices (with repeats) and returns a scalar statistic
  ## computed on those rows.  Each replicate draws as many clusters as
  ## there are, with replacement, and concatenates their rows.
  ##
  ## When rows within a cluster are correlated -- repeated measurements of
  ## the same machine over time -- the i.i.d. row bootstrap and analytic
  ## standard errors that assume independent rows (e.g. DeLong for AUC)
  ## are too narrow; resampling clusters respects the dependence (Field &
  ## Welsh, 2007).  The effective sample size is the number of clusters,
  ## so intervals from few clusters are wide and themselves uncertain.
  ##
  ## For a paired comparison of two models evaluated on the same rows,
  ## have @var{statfn} return the difference; a CI excluding 0 indicates a
  ## difference robust to cluster-level sampling variability.
  ##
  ## Replicates for which @var{statfn} errors or returns a non-finite value
  ## (e.g. AUC on a resample without positives) are dropped; the number
  ## kept is @code{numel (boot)}.  Defaults: B = 1000, alpha = 0.05.
  ## @end deftypefn

  opts = struct ("b", 1000, "alpha", 0.05, "seed", []);
  for k = 1:2:numel (varargin)
    key = lower (varargin{k});
    if (! isfield (opts, key))
      error ("omp.stats.cluster_bootstrap: unknown option '%s'", varargin{k});
    endif
    opts.(key) = varargin{k+1};
  endfor
  if (! isempty (opts.seed))
    state = omp.repro.seed (opts.seed);
    restore = onCleanup (@() omp.repro.seed (state));
  endif

  groups = groups(:);
  [~, ~, g] = unique (groups);
  G = max (g);
  if (G < 2)
    error ("omp.stats.cluster_bootstrap: need at least 2 clusters");
  endif
  members = accumarray (g, (1:numel (g))', [], @(v) {v});

  theta = statfn ((1:numel (g))');
  boot = NaN (opts.b, 1);
  for b = 1:opts.b
    pick = randi (G, G, 1);
    idx = vertcat (members{pick});
    try
      boot(b) = statfn (idx);
    catch
      boot(b) = NaN;
    end_try_catch
  endfor
  boot = boot(isfinite (boot));
  if (isempty (boot))
    error ("omp.stats.cluster_bootstrap: every replicate failed");
  endif
  s = sort (boot);
  n = numel (s);
  q = @(p) s(max (1, min (n, round (p * (n - 1)) + 1)));
  ci = [q(opts.alpha / 2), q(1 - opts.alpha / 2)];
endfunction

%!test
%! ## Mean of cluster means: CI contains the estimate.
%! g = kron ((1:20)', ones (5, 1));
%! x = kron (randn (20, 1), ones (5, 1)) + 0.1 * randn (100, 1);
%! [ci, th] = omp.stats.cluster_bootstrap (g, @(i) mean (x(i)), "B", 500, "Seed", 1);
%! assert (ci(1) <= th && th <= ci(2));

%!test
%! ## With strong within-cluster correlation, the cluster bootstrap
%! ## interval is much wider than the naive row bootstrap interval.
%! omp.repro.seed (2);
%! g = kron ((1:15)', ones (40, 1));
%! x = kron (randn (15, 1), ones (40, 1)) + 0.2 * randn (600, 1);
%! cc = omp.stats.cluster_bootstrap (g, @(i) mean (x(i)), "B", 800, "Seed", 3);
%! cr = omp.stats.bootstrap_ci (x, @mean, "B", 800, "Seed", 3, "Method", "percentile");
%! assert (diff (cc) > 3 * diff (cr));
%! ## ... and close to the analytic cluster-level interval.
%! m = accumarray (g, x, [], @mean);
%! assert (diff (cc), 2 * 1.96 * std (m) / sqrt (15), 0.3 * diff (cc));

%!test
%! ## Failed replicates are dropped, not propagated.
%! g = [1 1 2 2 3 3]';
%! [ci, ~, b] = omp.stats.cluster_bootstrap (g, @(i) 1 / (numel (unique (g(i))) > 1), "B", 200, "Seed", 4);
%! assert (all (isfinite (b)));

%!error <at least 2 clusters> omp.stats.cluster_bootstrap ([1 1 1], @numel)
