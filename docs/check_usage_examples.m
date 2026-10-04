function ok = check_usage_examples ()
  ## -*- texinfo -*-
  ## @deftypefn {} {@var{ok} =} check_usage_examples ()
  ## Execute, in order and in one workspace, every @code{```octave} block of
  ## @file{docs/usage_examples.md}, printing each block's output.  Fails
  ## (and, with @env{CI} set, exits with status 1) if any block errors,
  ## so documentation cannot silently drift from the library.
  ## @end deftypefn

  here = fileparts (mfilename ("fullpath"));
  root = fileparts (here);
  txt = fileread (fullfile (here, "usage_examples.md"));
  blocks = regexp (txt, '```octave\n(.*?)```', "tokens");
  old = pwd ();
  cd (root);
  restore = onCleanup (@() cd (old));
  ok = run_blocks (blocks);
  if (! ok && ! isempty (getenv ("CI")))
    exit (1);
  endif
endfunction

function ok = run_blocks (blocks)
  ## All blocks share this function's workspace, like a reader's session.
  ok = true;
  for i__ = 1:numel (blocks)
    printf ("--- block %d ---\n", i__);
    try
      eval (blocks{i__}{1});
    catch err__
      printf ("ERROR in block %d: %s\n", i__, err__.message);
      ok = false;
      return;
    end_try_catch
  endfor
  printf ("--- all %d blocks ran ---\n", numel (blocks));
endfunction
