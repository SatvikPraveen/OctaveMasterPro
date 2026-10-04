function prev = seed (s)
  ## -*- texinfo -*-
  ## @deftypefn {} {@var{prev} =} omp.repro.seed (@var{s})
  ## Seed every Octave random-number generator (@code{rand}, @code{randn},
  ## @code{rande}, @code{randg}, @code{randp}; @code{randi} and
  ## @code{randperm} draw from @code{rand}) with the non-negative integer
  ## @var{s}.
  ##
  ## Returns a struct @var{prev} holding the previous generator states so a
  ## caller can restore them with @code{omp.repro.seed (prev)}.
  ## @end deftypefn

  gens = {@rand, @randn, @rande, @randg, @randp};
  names = {"rand", "randn", "rande", "randg", "randp"};

  prev = struct ();
  for k = 1:numel (gens)
    prev.(names{k}) = gens{k} ("state");
  endfor

  if (isstruct (s))
    for k = 1:numel (gens)
      gens{k} ("state", s.(names{k}));
    endfor
    return;
  endif

  if (! (isscalar (s) && isreal (s) && s >= 0 && s == fix (s)))
    error ("omp.repro.seed: S must be a non-negative integer scalar");
  endif
  for k = 1:numel (gens)
    ## Offset per generator so streams are not trivially identical.
    gens{k} ("state", s + k - 1);
  endfor
endfunction

%!test
%! omp.repro.seed (42); a = [rand(3,1); randn(3,1); randi(10,3,1)];
%! omp.repro.seed (42); b = [rand(3,1); randn(3,1); randi(10,3,1)];
%! assert (a, b);

%!test
%! omp.repro.seed (1); a = rand (5, 1);
%! omp.repro.seed (2); b = rand (5, 1);
%! assert (! isequal (a, b));

%!test
%! omp.repro.seed (7); r1 = rand ();
%! p = omp.repro.seed (99); rand (10);
%! omp.repro.seed (p);
%! omp.repro.seed (7); r2 = rand ();
%! assert (r1, r2);

%!error <non-negative integer> omp.repro.seed (-1)
%!error <non-negative integer> omp.repro.seed (1.5)
