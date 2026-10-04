function ok = run_tests (varargin)
  ## -*- texinfo -*-
  ## @deftypefn {} {@var{ok} =} run_tests ()
  ## @deftypefnx {} {@var{ok} =} run_tests (@var{filter})
  ## Run every built-in test block (@code{%!test}, @code{%!assert},
  ## @code{%!error}) of the OctaveMasterPro library under @file{inst/}.
  ##
  ## @var{filter} is an optional substring; only files whose path contains it
  ## are tested (e.g. @code{run_tests ("+stats")}).
  ##
  ## Returns true when every test passes.  When run
  ## with the environment variable @env{CI} set, the Octave process exits
  ## with status 1 on any failure.
  ## @end deftypefn

  filter = "";
  if (nargin > 0)
    filter = varargin{1};
  endif

  root = fileparts (fileparts (mfilename ("fullpath")));
  inst = fullfile (root, "inst");
  addpath (inst);
  cleanup = onCleanup (@() rmpath (inst));

  files = list_m_files (fullfile (inst, "+omp"));
  if (! isempty (filter))
    files = files(! cellfun (@isempty, strfind (files, filter)));
  endif

  n_total = 0;
  n_pass = 0;
  failed = {};
  t0 = tic ();
  for i = 1:numel (files)
    f = files{i};
    rel = strrep (f, [root filesep], "");
    logfile = tempname ();
    fid = fopen (logfile, "w");
    [n, m] = test (f, "quiet", fid);
    fclose (fid);
    n_pass += n;
    n_total += m;
    if (m == 0)
      status = "NO TESTS";
      failed{end+1} = [rel " (no tests)"];
    elseif (n == m)
      status = "PASS";
    else
      status = "FAIL";
      failed{end+1} = rel;
    endif
    printf ("  %-8s %3d/%-3d  %s\n", status, n, m, rel);
    if (strcmp (status, "FAIL"))
      printf ("%s\n", fileread (logfile));
    endif
    unlink (logfile);
  endfor

  printf ("\n%d/%d tests passed in %d files (%.1f s)\n", n_pass, n_total, ...
          numel (files), toc (t0));
  ok = isempty (failed);
  if (! ok)
    printf ("Failures:\n");
    printf ("  %s\n", failed{:});
    if (! isempty (getenv ("CI")))
      exit (1);  # propagate failure to the CI job
    endif
  endif
endfunction

function files = list_m_files (d)
  files = {};
  entries = dir (d);
  for k = 1:numel (entries)
    e = entries(k);
    if (any (strcmp (e.name, {".", ".."})))
      continue;
    endif
    p = fullfile (d, e.name);
    if (e.isdir)
      files = [files, list_m_files(p)];
    elseif (numel (e.name) > 2 && strcmp (e.name(end-1:end), ".m"))
      files{end+1} = p;
    endif
  endfor
  files = sort (files);
endfunction
