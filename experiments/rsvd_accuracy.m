function R = rsvd_accuracy ()
  ## -*- texinfo -*-
  ## @deftypefn {} {@var{R} =} rsvd_accuracy ()
  ## Numerical experiment: accuracy of the randomized SVD relative to the
  ## optimal rank-k approximation (Eckart--Young), as a function of power
  ## iterations q and spectral decay.
  ##
  ## HYPOTHESES (Halko, Martinsson & Tropp, 2011)
  ##   H1  The ratio err / err_opt in Frobenius norm is bounded by roughly
  ##       sqrt(1 + k/(p-1)) for q = 0.
  ##   H2  Power iterations drive the ratio towards 1, with the largest
  ##       gains for slowly decaying spectra (sigma_j = j^(-alpha), small
  ##       alpha).  For fast decay, q = 0 is already near optimal.
  ## FALSIFICATION: ratios exceeding the H1 bound for q = 0, or no
  ## improvement with q for slow decay.
  ##
  ## DESIGN: 400x300 matrices, k = 20, p = 10, alpha in {0.25, 0.5, 1, 2},
  ## q in {0, 1, 2, 3}; 10 independent sketches per cell (mean and max
  ## reported).  Spectral-norm ratios are also recorded; they are known to
  ## be less favourable than Frobenius ratios.
  ##
  ## Writes results/rsvd_accuracy.csv and .md next to this file.
  ## @end deftypefn

  here = fileparts (mfilename ("fullpath"));
  addpath (fullfile (fileparts (here), "inst"));
  out = fullfile (here, "results");
  if (! exist (out, "dir"))
    mkdir (out);
  endif
  env = omp.repro.env_info ();
  fid = fopen (fullfile (out, "rsvd_accuracy_environment.txt"), "w");
  fprintf (fid, "octave: %s\nblas: %s\nlapack: %s\ngit_commit: %s\ngit_dirty: %d\n", ...
           env.octave_version, env.blas, env.lapack, env.git_commit, env.git_dirty);
  fclose (fid);

  m = 400; n = 300; k = 20; p = 10; reps = 10;
  alphas = [0.25 0.5 1 2];
  qs = 0:3;
  omp.repro.seed (7);
  [Uo, ~] = qr (randn (m, n), 0);
  [Vo, ~] = qr (randn (n));
  rows_ = [];
  for a = alphas
    s = (1:n)' .^ (-a);
    A = Uo * diag (s) * Vo';
    fro_opt = sqrt (sum (s(k+1:end) .^ 2));
    spec_opt = s(k + 1);
    for q = qs
      rf = zeros (reps, 1); rs = rf;
      for r = 1:reps
        [U, S, V] = omp.linalg.randomized_svd (A, k, "Oversample", p, ...
                                               "PowerIters", q, "Seed", 100 * r + q);
        E = A - U * S * V';
        rf(r) = norm (E, "fro") / fro_opt;
        rs(r) = norm (E) / spec_opt;
      endfor
      rows_(end+1, :) = [a, q, mean(rf), max(rf), mean(rs), max(rs)];
    endfor
  endfor
  bound = sqrt (1 + k / (p - 1));

  fid = fopen (fullfile (out, "rsvd_accuracy.csv"), "w");
  fprintf (fid, "alpha,q,fro_ratio_mean,fro_ratio_max,spec_ratio_mean,spec_ratio_max\n");
  fprintf (fid, "%g,%d,%.6f,%.6f,%.6f,%.6f\n", rows_');
  fclose (fid);

  fid = fopen (fullfile (out, "rsvd_accuracy.md"), "w");
  fprintf (fid, "# Randomized SVD accuracy vs. optimal rank-%d approximation\n\n", k);
  fprintf (fid, "400x300, p = %d, %d sketches per cell. HMT expected-error bound for q = 0 (Frobenius): %.3f.\n\n", p, reps, bound);
  fprintf (fid, "| alpha (σ_j = j^-α) | q | Frobenius ratio mean (max) | Spectral ratio mean (max) |\n|---|---|---|---|\n");
  fprintf (fid, "| %.2f | %d | %.4f (%.4f) | %.3f (%.3f) |\n", rows_');
  fclose (fid);

  R = struct ("table", rows_, "bound_q0", bound);
endfunction
