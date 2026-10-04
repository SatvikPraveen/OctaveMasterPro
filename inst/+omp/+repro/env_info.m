function info = env_info (varargin)
  ## -*- texinfo -*-
  ## @deftypefn {} {@var{info} =} omp.repro.env_info ()
  ## @deftypefnx {} {} omp.repro.env_info ("print")
  ## Capture the computational environment needed to reproduce a numerical
  ## result: Octave version, BLAS/LAPACK builds, loaded and installed
  ## packages, logical CPU count, platform, git revision of the working tree
  ## and a UTC timestamp.
  ##
  ## Host names, user names and paths are deliberately not recorded.
  ##
  ## With the argument @qcode{"print"} a human-readable summary is written
  ## to stdout.
  ## @end deftypefn

  info = struct ();
  info.timestamp_utc = datestr (now () - utc_offset_days (), "yyyy-mm-ddTHH:MM:SSZ");
  info.octave_version = OCTAVE_VERSION ();
  info.blas = strtrim (version ("-blas"));
  info.lapack = strtrim (version ("-lapack"));
  info.arch = computer ();
  info.nproc = nproc ();
  info.packages = installed_packages ();
  info.git_commit = git_query ("rev-parse HEAD");
  info.git_dirty = ! isempty (git_query ("status --porcelain --untracked-files=no"));

  if (nargin > 0 && strcmpi (varargin{1}, "print"))
    printf ("Environment (%s)\n", info.timestamp_utc);
    printf ("  Octave   : %s on %s, %d logical CPUs\n", info.octave_version, ...
            info.arch, info.nproc);
    printf ("  BLAS     : %s\n", info.blas);
    printf ("  LAPACK   : %s\n", info.lapack);
    if (isempty (info.packages))
      printf ("  Packages : (none)\n");
    else
      printf ("  Packages : %s\n", strjoin (info.packages, ", "));
    endif
    dirty = "";
    if (info.git_dirty)
      dirty = " (uncommitted changes)";
    endif
    printf ("  Git      : %s%s\n", info.git_commit, dirty);
  endif
endfunction

function d = utc_offset_days ()
  ## Offset of local time from UTC, in days.
  t = time ();
  l = localtime (t);
  g = gmtime (t);
  d = (mktime (l) - mktime (g)) / 86400;
endfunction

function c = installed_packages ()
  c = {};
  try
    [~, desc] = pkg ("list");
    for k = 1:numel (desc)
      tag = sprintf ("%s-%s", desc{k}.name, desc{k}.version);
      if (desc{k}.loaded)
        tag = [tag "*"];
      endif
      c{end+1} = tag;
    endfor
  catch
    c = {};
  end_try_catch
endfunction

function out = git_query (args)
  out = "";
  root = fileparts (fileparts (fileparts (fileparts (mfilename ("fullpath")))));
  [status, txt] = system (sprintf ("git -C \"%s\" %s 2>/dev/null", root, args));
  if (status == 0)
    out = strtrim (txt);
  endif
endfunction

%!test
%! info = omp.repro.env_info ();
%! assert (info.octave_version, OCTAVE_VERSION ());
%! assert (info.nproc >= 1);
%! assert (ischar (info.blas) && ! isempty (info.blas));
%! assert (numel (info.timestamp_utc), 20);
%! assert (iscellstr (info.packages));
