function R = qr_orthogonality (varargin)
  ## -*- texinfo -*-
  ## @deftypefn {} {@var{R} =} qr_orthogonality ()
  ## Numerical experiment: loss of orthogonality of QR algorithms as a
  ## function of the condition number kappa(A).
  ##
  ## HYPOTHESES (from rounding-error analysis; u = unit roundoff)
  ##   CGS          ||I - Q'Q|| ~ u kappa^2   (until it saturates at O(1))
  ##   MGS          ||I - Q'Q|| ~ u kappa     (Bjorck, 1967)
  ##   CGS2         ||I - Q'Q|| ~ u           (for u kappa < 1)
  ##   Householder  ||I - Q'Q|| ~ u           (any kappa)
  ## FALSIFICATION: fitted log-log slopes of loss against kappa, over the
  ## pre-saturation range, far from {2, 1, 0, 0}.
  ##
  ## DESIGN: A = U diag(s) V' (100 x 30) with Haar-random U, V and
  ## geometrically spaced singular values s from 1 to 1/kappa;
  ## kappa = 10^1 .. 10^15; 5 random matrices per kappa (median reported).
  ## The backward error ||A - QR|| / ||A|| is recorded as a control: it
  ## should stay O(u) for every method.
  ##
  ## Writes results/qr_orthogonality.csv, .md and .png next to this file.
  ## @end deftypefn

  here = fileparts (mfilename ("fullpath"));
  addpath (fullfile (fileparts (here), "inst"));
  out = fullfile (here, "results");
  if (! exist (out, "dir"))
    mkdir (out);
  endif
  env = omp.repro.env_info ();
  fid = fopen (fullfile (out, "qr_orthogonality_environment.txt"), "w");
  fprintf (fid, "octave: %s\nblas: %s\nlapack: %s\ngit_commit: %s\ngit_dirty: %d\n", ...
           env.octave_version, env.blas, env.lapack, env.git_commit, env.git_dirty);
  fclose (fid);

  m = 100; n = 30; reps = 5;
  logk = 1:15;
  methods = {"cgs", "mgs", "cgs2", "householder"};
  loss = zeros (numel (logk), numel (methods), reps);
  berr = loss;
  omp.repro.seed (2026);
  for i = 1:numel (logk)
    for r = 1:reps
      [U, ~] = qr (randn (m, n), 0);
      [V, ~] = qr (randn (n));
      A = U * diag (logspace (0, -logk(i), n)) * V';
      for j = 1:numel (methods)
        if (strcmp (methods{j}, "householder"))
          [Q, Rr] = omp.linalg.householder_qr (A, "econ");
        else
          [Q, Rr] = omp.linalg.gram_schmidt (A, methods{j});
        endif
        loss(i, j, r) = norm (eye (n) - Q' * Q);
        berr(i, j, r) = norm (A - Q * Rr) / norm (A);
      endfor
    endfor
  endfor
  L = median (loss, 3);
  B = median (berr, 3);

  ## Fit slopes on the pre-saturation range (loss < 1e-2, and for CGS2
  ## only where u*kappa < 1e-2, the regime its bound covers).
  slope = NaN (1, numel (methods));
  for j = 1:numel (methods)
    k = L(:, j) < 1e-2;
    if (strcmp (methods{j}, "cgs2"))
      k &= (eps * 10 .^ logk' < 1e-2);
    endif
    if (sum (k) >= 3)
      c = polyfit (logk(k)', log10 (L(k, j)), 1);
      slope(j) = c(1);
    endif
  endfor

  fid = fopen (fullfile (out, "qr_orthogonality.csv"), "w");
  fprintf (fid, "log10_kappa,%s,%s\n", strjoin (strcat ("loss_", methods), ","), ...
           strjoin (strcat ("backward_error_", methods), ","));
  fprintf (fid, [repmat("%.6g,", 1, 2 * numel (methods)) "%.6g\n"], [logk' L B]');
  fclose (fid);

  expected = [2 1 0 0];
  fid = fopen (fullfile (out, "qr_orthogonality.md"), "w");
  fprintf (fid, "# Loss of orthogonality vs. condition number\n\n");
  fprintf (fid, "100x30 matrices, median of %d per kappa, kappa = 1e1..1e15.\n\n", reps);
  fprintf (fid, "| Method | Fitted slope d log(loss) / d log(kappa) | Theory | Max backward error |\n|---|---|---|---|\n");
  for j = 1:numel (methods)
    fprintf (fid, "| %s | %.2f | %d | %.1e |\n", methods{j}, slope(j), expected(j), max (B(:, j)));
  endfor
  fprintf (fid, "\nSlopes are fitted where the loss is below 1e-2, before saturation.\n");
  fclose (fid);

  try
    h = figure ("visible", "off");
    loglog (10 .^ logk, L, "o-", "linewidth", 1.5);
    hold on;
    loglog (10 .^ logk, eps * 10 .^ logk, "k:", 10 .^ logk, eps * 10 .^ (2 * logk), "k--");
    ylim ([1e-17 1e1]);
    xlabel ("\\kappa(A)"); ylabel ("||I - Q^TQ||_2");
    legend ([upper(methods), {"u \\kappa", "u \\kappa^2"}], "location", "northwest");
    title ("Loss of orthogonality (median of 5)"); grid on;
    print (h, fullfile (out, "qr_orthogonality.png"), "-dpng", "-r120");
    close (h);
  catch err
    warning ("qr_orthogonality: figure skipped (%s)", err.message);
  end_try_catch

  R = struct ("log10_kappa", logk, "methods", {methods}, "loss", L, ...
              "backward_error", B, "slope", slope);
  printf ("slopes: %s\n", mat2str (slope, 3));
endfunction
