function T = read_csv (filename, varargin)
  ## -*- texinfo -*-
  ## @deftypefn {} {@var{T} =} omp.io.read_csv (@var{filename})
  ## @deftypefnx {} {@var{T} =} omp.io.read_csv (@var{filename}, "Delimiter", @var{d})
  ## Read a delimited text file with a single header row into a struct of
  ## column vectors -- a lightweight, Octave-native substitute for MATLAB's
  ## @code{readtable}.
  ##
  ## Header names are converted to valid field names with
  ## @code{matlab.lang.makeValidName}.  A column becomes a double column
  ## vector when every non-empty entry parses as a number (empty entries
  ## become @code{NaN}); otherwise it is returned as a cellstr column.
  ##
  ## Quoted fields containing the delimiter are supported (RFC 4180 style
  ## double quotes); embedded newlines inside quotes are not.
  ##
  ## Column order is preserved: @code{fieldnames (T)} lists the columns in
  ## file order.
  ## @end deftypefn

  delim = ",";
  for k = 1:2:numel (varargin)
    switch (lower (varargin{k}))
      case "delimiter"
        delim = varargin{k+1};
      otherwise
        error ("omp.io.read_csv: unknown option '%s'", varargin{k});
    endswitch
  endfor

  fid = fopen (filename, "r");
  if (fid < 0)
    error ("omp.io.read_csv: cannot open '%s'", filename);
  endif
  txt = fread (fid, Inf, "*char")';
  fclose (fid);

  if (numel (txt) >= 3 && all (double (txt(1:3)) == [239 187 191]))
    txt = txt(4:end);  # strip UTF-8 BOM
  endif
  lines = regexp (txt, '\r?\n', "split");
  lines = lines(! cellfun (@(s) isempty (strtrim (s)), lines));
  if (isempty (lines))
    error ("omp.io.read_csv: '%s' is empty", filename);
  endif

  header = split_line (lines{1}, delim);
  nc = numel (header);
  nr = numel (lines) - 1;
  cells = cell (nr, nc);
  for i = 1:nr
    row = split_line (lines{i+1}, delim);
    if (numel (row) != nc)
      error ("omp.io.read_csv: line %d has %d fields, expected %d", ...
             i + 1, numel (row), nc);
    endif
    cells(i, :) = row;
  endfor

  T = struct ();
  for j = 1:nc
    name = matlab.lang.makeValidName (strtrim (header{j}));
    col = cells(:, j);
    vals = str2double (col);
    empty = cellfun (@isempty, strtrim (col));
    if (all (! isnan (vals) | empty))
      vals(empty) = NaN;
      T.(name) = vals;
    else
      T.(name) = col;
    endif
  endfor
endfunction

function f = split_line (s, delim)
  if (! any (s == '"'))
    f = strsplit (s, delim, "CollapseDelimiters", false);
    return;
  endif
  f = {};
  cur = "";
  inq = false;
  i = 1;
  while (i <= numel (s))
    c = s(i);
    if (inq)
      if (c == '"' && i < numel (s) && s(i+1) == '"')
        cur(end+1) = '"';
        i += 1;
      elseif (c == '"')
        inq = false;
      else
        cur(end+1) = c;
      endif
    elseif (c == '"')
      inq = true;
    elseif (c == delim)
      f{end+1} = cur;
      cur = "";
    else
      cur(end+1) = c;
    endif
    i += 1;
  endwhile
  f{end+1} = cur;
endfunction

%!test
%! fn = [tempname() ".csv"];
%! fid = fopen (fn, "w");
%! fprintf (fid, "id,Value A,name\n1,2.5,foo\n2,,\"bar, baz\"\n3,-1e3,\"q\"\"x\"\n");
%! fclose (fid);
%! T = omp.io.read_csv (fn);
%! unlink (fn);
%! assert (fieldnames (T), {"id"; "ValueA"; "name"});
%! assert (T.id, [1; 2; 3]);
%! assert (T.ValueA, [2.5; NaN; -1000]);
%! assert (T.name, {"foo"; "bar, baz"; "q\"x"});

%!test
%! fn = [tempname() ".csv"];
%! fid = fopen (fn, "w"); fprintf (fid, "a;b\r\n1;x\r\n"); fclose (fid);
%! T = omp.io.read_csv (fn, "Delimiter", ";");
%! unlink (fn);
%! assert (T.a, 1);
%! assert (T.b, {"x"});

%!error <cannot open> omp.io.read_csv ("/nonexistent/file.csv")
