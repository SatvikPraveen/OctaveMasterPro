function ok = check_parse ()
  ## -*- texinfo -*-
  ## @deftypefn {} {@var{ok} =} check_parse ()
  ## Parse every tracked @file{.m} file in the repository without executing
  ## it and report syntax errors.  Complements @code{run_tests}, which only
  ## covers the tested library under @file{inst/}: legacy demos, notebooks'
  ## helper scripts and experiments must at least be valid Octave.
  ##
  ## With the environment variable @env{CI} set, exits with status 1 on any
  ## failure.
  ## @end deftypefn

  root = fileparts (fileparts (mfilename ("fullpath")));
  [status, out] = system (sprintf ("git -C \"%s\" ls-files \"*.m\"", root));
  if (status != 0)
    error ("check_parse: git ls-files failed (is this a git checkout?)");
  endif
  files = strsplit (strtrim (out), "\n");
  bad = {};
  ws = warning ("off", "all");
  restore = onCleanup (@() warning (ws));
  for i = 1:numel (files)
    try
      __parse_file__ (fullfile (root, files{i}));
    catch err
      bad{end+1} = sprintf ("%s: %s", files{i}, strtrim (err.message));
    end_try_catch
  endfor
  printf ("Parsed %d files: %d failed\n", numel (files), numel (bad));
  if (! isempty (bad))
    printf ("  %s\n", bad{:});
  endif
  ok = isempty (bad);
  if (! ok && ! isempty (getenv ("CI")))
    exit (1);
  endif
endfunction
