% Location: mini_projects/parallel_image_batch_processing/private/sgtitle.m
% Figure-level title for a grid of subplots.
%
% sgtitle only exists in Octave 9 and later; this local version keeps the
% demos working on Octave 8.  It draws a centred text label in an
% invisible axes across the top of the figure and leaves the current
% axes unchanged.
%
%   sgtitle(txt)
%   sgtitle(fig, txt)
%   sgtitle(..., prop, value, ...)   extra text properties
%   h = sgtitle(...)

function h = sgtitle(varargin)
    if nargin >= 1 && isscalar(varargin{1}) && ishghandle(varargin{1}) ...
            && strcmp(get(varargin{1}, 'type'), 'figure')
        fig = varargin{1};
        varargin(1) = [];
    else
        fig = gcf();
    end
    if isempty(varargin)
        error('sgtitle: missing title text');
    end
    txt = varargin{1};
    props = varargin(2:end);

    prev_ax = get(fig, 'currentaxes');
    ax = axes('parent', fig, 'position', [0 0.95 1 0.05], 'visible', 'off', ...
              'tag', 'sgtitle', 'handlevisibility', 'off', 'hittest', 'off');
    th = text(0.5, 0.5, txt, 'parent', ax, 'units', 'normalized', ...
              'horizontalalignment', 'center', 'verticalalignment', 'middle', ...
              'fontweight', 'bold', 'fontsize', 12, props{:});
    if ~isempty(prev_ax) && ishghandle(prev_ax)
        set(fig, 'currentaxes', prev_ax);
    end
    if nargout > 0
        h = th;
    end
end
