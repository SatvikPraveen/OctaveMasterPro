function p = logistic_predict (model, X)
  ## -*- texinfo -*-
  ## @deftypefn {} {@var{p} =} omp.ml.logistic_predict (@var{model}, @var{X})
  ## Predicted probabilities P(y = 1 | x) for the rows of @var{X} using a
  ## model from @code{omp.ml.logistic_fit}.  The training-set
  ## standardization stored in @var{model} is applied to @var{X}.
  ## @end deftypefn

  if (columns (X) != numel (model.coef))
    error ("omp.ml.logistic_predict: X has %d columns, model expects %d", ...
           columns (X), numel (model.coef));
  endif
  eta = model.intercept + ((X - model.mu) ./ model.sigma) * model.coef;
  p = 1 ./ (1 + exp (-eta));
endfunction

%!test
%! m = struct ("intercept", 0, "coef", [1; -1], "mu", [0 0], "sigma", [1 1]);
%! assert (omp.ml.logistic_predict (m, [0 0; 1 1; 2 0]), [0.5; 0.5; 1/(1+exp(-2))], 1e-15);

%!error <columns> omp.ml.logistic_predict (struct ("coef", [1; 2]), ones (3, 1))
