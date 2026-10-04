% Location: mini_projects/signal_processing_simulation/private/is_interactive_session.m
% Decide whether the demo may prompt the user with input().
%
%   tf = is_interactive_session()      true when prompting is possible
%   is_interactive_session(false)      force non-interactive mode (e.g. 'all')
%   is_interactive_session([])         clear the forced value again
%
% Octave 8 has no isatty(); a session counts as interactive when the GUI
% is running, or when stdin is a terminal and Octave was not started with
% --eval (which runs a command and exits).  `test -t 0` in a child shell
% inherits Octave's stdin, so it reports whether that is a terminal.

function tf = is_interactive_session(force)
    persistent forced;
    if nargin > 0
        forced = force;
        tf = isempty(forced) || logical(forced);
        return;
    end
    if ~isempty(forced)
        tf = logical(forced);
        return;
    end

    if isguirunning()
        tf = true;
        return;
    end
    args = argv();
    if any(strcmp(args, '--eval'))
        tf = false;
        return;
    end
    if ispc()
        tf = true;  % no cheap tty test; assume a console session
    else
        tf = (system('test -t 0') == 0);
    end
end
