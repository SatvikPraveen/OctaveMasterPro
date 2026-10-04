function sim = simulate_fleet (varargin)
  ## -*- texinfo -*-
  ## @deftypefn {} {@var{sim} =} omp.pdm.simulate_fleet ()
  ## @deftypefnx {} {@var{sim} =} omp.pdm.simulate_fleet ("Units", @var{n}, "Hours", @var{T}, "Seed", @var{s}, @dots{})
  ## Simulate hourly telemetry and failures for a fleet of machines whose
  ## failure risk is driven by a latent, unobserved health state, so that
  ## predictive-maintenance methods can be evaluated against known ground
  ## truth.
  ##
  ## @strong{Generative model} (per unit u, hour t):
  ## @itemize
  ## @item Degradation h_u(t) >= 0 grows by Gamma-distributed increments
  ## with mean rate @var{DegradeRate} per hour (a monotone gamma process,
  ## van Noortwijk, 2009) and resets to 0 at failure or preventive
  ## maintenance.  The per-unit rate is heterogeneous (log-normal,
  ## CV @var{RateCV}).
  ## @item Failure hazard lambda_u(t) = @var{BaseHazard} exp(@var{Beta} h_u(t)).
  ## A failure takes the unit offline for @var{RepairHours}.
  ## @item Preventive maintenance every @var{PMInterval} hours (+/- 20%
  ## jitter) resets h without a failure, censoring degradation paths.
  ## @item Observed sensors:
  ##   ambient a(t) = 20 + 5 sin(2 pi t/24) + 3 sin(2 pi t/(24*30)) + AR(1)
  ##   (shared by all units: a common confounder);
  ##   temperature = a + c_u + 6 h + e;
  ##   vibration   = v_u (1 + 1.2 h^2) exp(e');
  ##   pressure    = 100 + p_u - 2.5 h + e'';
  ##   with unit-specific offsets c_u, v_u, p_u and a fraction
  ##   @var{MissingRate} of readings missing at random.
  ## @end itemize
  ##
  ## Returns a struct with fields @code{unit}, @code{t} (hours),
  ## @code{ambient}, @code{temperature}, @code{vibration},
  ## @code{pressure}, @code{online} (false while under repair),
  ## @code{health} (latent h), @code{hazard} (true lambda),
  ## @code{failure} (1 at the hour a failure occurs), @code{maintenance},
  ## and @code{params}.  All arrays are column vectors ordered by unit then
  ## time.
  ##
  ## Because the hazard depends on the observations only through h, the
  ## score h(t) (equivalently the true hazard) is the Bayes-optimal ranking
  ## of "fails within the next H hours" among online samples, up to the
  ## unpredictable timing of future degradation: it gives an upper bound
  ## ("oracle") against which sensor-based models are judged.
  ## @end deftypefn

  P = struct ("units", 40, "hours", 24 * 120, "seed", [], ...
              "degraderate", 1 / (24 * 30), "ratecv", 0.35, ...
              "gammashape", 0.2, "basehazard", 1 / (24 * 365), "beta", 4.5, ...
              "repairhours", 24, "pminterval", 24 * 60, ...
              "missingrate", 0.01);
  for k = 1:2:numel (varargin)
    key = lower (varargin{k});
    if (! isfield (P, key))
      error ("omp.pdm.simulate_fleet: unknown option '%s'", varargin{k});
    endif
    P.(key) = varargin{k+1};
  endfor
  if (! isempty (P.seed))
    state = omp.repro.seed (P.seed);
    restore = onCleanup (@() omp.repro.seed (state));
  endif

  U = P.units;
  T = P.hours;
  t = (0:T-1)';

  ## Shared ambient temperature: diurnal + monthly cycles + AR(1) weather.
  w = filter (1, [1 -0.97], 0.4 * randn (T, 1));
  ambient = 20 + 5 * sin (2 * pi * t / 24) + 3 * sin (2 * pi * t / (24 * 30)) + w;

  n = U * T;
  sim = struct ();
  sim.unit = kron ((1:U)', ones (T, 1));
  sim.t = repmat (t, U, 1);
  sim.ambient = repmat (ambient, U, 1);
  [sim.temperature, sim.vibration, sim.pressure, sim.health, sim.hazard] = ...
    deal (zeros (n, 1));
  sim.online = true (n, 1);
  sim.failure = false (n, 1);
  sim.maintenance = false (n, 1);

  ## Gamma(shape k, scale theta) increments with mean k*theta = rate.
  sigma_r = sqrt (log (1 + P.ratecv ^ 2));
  for u = 1:U
    rate = P.degraderate * exp (sigma_r * randn () - sigma_r ^ 2 / 2);
    k = P.gammashape;
    theta = rate / k;
    c_u = 3 * randn ();
    v_u = exp (log (0.5) + 0.25 * randn ());
    p_u = 2 * randn ();

    h = 0;
    down = 0;
    next_pm = P.pminterval * (0.8 + 0.4 * rand ()) * rand ();  # staggered start
    H = zeros (T, 1); L = H; F = false (T, 1); M = F; O = true (T, 1);
    for i = 1:T
      if (down > 0)
        O(i) = false;
        down -= 1;
        H(i) = h;
        continue;
      endif
      if (t(i) >= next_pm)
        h = 0;
        M(i) = true;
        next_pm = t(i) + P.pminterval * (0.8 + 0.4 * rand ());
      endif
      h += randg (k) * theta;
      lam = P.basehazard * exp (P.beta * h);
      H(i) = h;
      L(i) = lam;
      if (rand () < 1 - exp (-lam))
        F(i) = true;
        h = 0;
        down = P.repairhours;
        next_pm = t(i) + P.repairhours + P.pminterval * (0.8 + 0.4 * rand ());
      endif
    endfor

    idx = (u - 1) * T + (1:T)';
    sim.health(idx) = H;
    sim.hazard(idx) = L;
    sim.failure(idx) = F;
    sim.maintenance(idx) = M;
    sim.online(idx) = O;
    sim.temperature(idx) = ambient + c_u + 6 * H + 1.5 * randn (T, 1);
    sim.vibration(idx) = v_u * (1 + 1.2 * H .^ 2) .* exp (0.15 * randn (T, 1));
    sim.pressure(idx) = 100 + p_u - 2.5 * H + 1.0 * randn (T, 1);
  endfor

  ## Offline machines report nothing; additional readings missing at random.
  for f = {"temperature", "vibration", "pressure"}
    v = sim.(f{1});
    v(! sim.online | rand (n, 1) < P.missingrate) = NaN;
    sim.(f{1}) = v;
  endfor
  sim.params = P;
endfunction

%!test
%! s = omp.pdm.simulate_fleet ("Units", 5, "Hours", 24 * 20, "Seed", 1);
%! n = 5 * 24 * 20;
%! for f = {"unit", "t", "temperature", "vibration", "pressure", "health", "failure"}
%!   assert (numel (s.(f{1})), n);
%! endfor
%! assert (all (s.health >= 0));
%! assert (all (isnan (s.vibration(! s.online))));

%!test
%! ## Reproducible given a seed; does not disturb the global RNG stream.
%! rand ("state", 1); r0 = rand ();
%! rand ("state", 1);
%! a = omp.pdm.simulate_fleet ("Units", 3, "Hours", 200, "Seed", 7);
%! r1 = rand ();
%! b = omp.pdm.simulate_fleet ("Units", 3, "Hours", 200, "Seed", 7);
%! assert (isequaln (a, b));
%! assert (r0, r1);

%!test
%! ## Health resets after every failure and degradation raises hazard.
%! s = omp.pdm.simulate_fleet ("Units", 20, "Seed", 3);
%! assert (sum (s.failure) > 20);
%! k = find (s.failure);
%! k = k(k < numel (s.t) & s.unit(k + 1) == s.unit(k));
%! assert (all (s.health(k + 1) == 0));
%! on = s.online;
%! r = corr (s.health(on), log (s.hazard(on)));
%! assert (r, 1, 1e-12);

%!test
%! ## The latent state carries signal: the oracle score h(t) ranks
%! ## "fails within 72 h" well above chance but below perfection (future
%! ## degradation and failure timing are random).  Observed range over
%! ## seeds 4-8: AUC 0.83-0.86.
%! s = omp.pdm.simulate_fleet ("Units", 30, "Seed", 4);
%! [y, e] = omp.pdm.failure_labels (s, 72);
%! auc = omp.ml.roc_auc (y(e), s.health(e));
%! assert (auc > 0.75 && auc < 0.95);
