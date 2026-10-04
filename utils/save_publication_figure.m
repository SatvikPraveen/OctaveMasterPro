% File location: OctaveMasterPro/utils/save_publication_figure.m
% Plotting helper (from plot_utils.m). Kept in its own
% file so it can be called from other scripts: Octave subfunctions in
% plot_utils.m are only visible inside that file.

function save_publication_figure(filename, varargin)
    % Save high-quality figures for publications
    % Usage: save_publication_figure('my_plot', 'Format', 'both', 'DPI', 300)
    
    p = inputParser;
    addParameter(p, 'Format', 'png'); % 'png', 'pdf', 'eps', 'both'
    addParameter(p, 'DPI', 300);
    addParameter(p, 'Size', [8, 6]); % inches
    parse(p, varargin{:});
    
    % Set figure properties
    set(gcf, 'PaperUnits', 'inches');
    set(gcf, 'PaperSize', p.Results.Size);
    set(gcf, 'PaperPosition', [0, 0, p.Results.Size]);
    
    % Save in requested formats
    if strcmp(p.Results.Format, 'png') || strcmp(p.Results.Format, 'both')
        print('-dpng', sprintf('-r%d', p.Results.DPI), [filename '.png']);
    end
    
    if strcmp(p.Results.Format, 'pdf') || strcmp(p.Results.Format, 'both')
        print('-dpdf', [filename '.pdf']);
    end
    
    if strcmp(p.Results.Format, 'eps')
        print('-deps2', [filename '.eps']);
    end
    
    fprintf('Figure saved as: %s\n', filename);
end
