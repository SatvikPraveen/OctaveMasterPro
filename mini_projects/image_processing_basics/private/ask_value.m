% Location: mini_projects/image_processing_basics/private/ask_value.m
% Prompt for a value, falling back to a default.
%
%   v = ask_value(prompt, default)        numeric input
%   v = ask_value(prompt, default, 's')   string input
%
% In a non-interactive session (see is_interactive_session) no prompt is
% shown and the default is returned, so demos also run headless.

function v = ask_value(prompt, default, varargin)
    if is_interactive_session()
        v = input(prompt, varargin{:});
        if isempty(v)
            v = default;
        end
    else
        if ischar(default)
            shown = default;
        else
            shown = mat2str(default);
        end
        fprintf('%s%s (default, non-interactive)\n', prompt, shown);
        v = default;
    end
end
