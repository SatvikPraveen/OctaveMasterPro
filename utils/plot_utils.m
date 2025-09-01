% File location: OctaveMasterPro/utils/plot_utils.m
% Comprehensive plotting utilities for OctaveMasterPro

function plot_utils()
    % Utility collection for enhanced plotting in Octave
    % Usage: Functions can be called individually or access help with plot_utils_help()
end

function plot_utils_help()
    % Display available plotting utilities
    fprintf('\n=== OctaveMasterPro Plot Utilities ===\n\n');
    fprintf('📊 BASIC PLOTS:\n');
    fprintf('  enhanced_plot(x, y, options)     - Enhanced line plots with styling\n');
    fprintf('  multi_subplot(data, layout)      - Create multiple subplots easily\n');
    fprintf('  save_publication_figure(name)    - Save high-quality figures\n\n');
    
    fprintf('📈 STATISTICAL PLOTS:\n');
    fprintf('  plot_histogram_with_stats(data)  - Histogram with statistics overlay\n');
    fprintf('  plot_correlation_matrix(data)    - Correlation heatmap\n');
    fprintf('  plot_box_comparison(groups)      - Multi-group box plots\n\n');
    
    fprintf('📉 TIME SERIES:\n');
    fprintf('  plot_time_series(dates, data)    - Time series with trend lines\n');
    fprintf('  plot_stock_chart(stock_data)     - OHLC candlestick charts\n');
    fprintf('  plot_seasonal_decomp(ts_data)    - Seasonal decomposition\n\n');
    
    fprintf('🎛️ SIGNAL PROCESSING:\n');
    fprintf('  plot_signal_analysis(signal, fs) - Time and frequency domain\n');
    fprintf('  plot_spectrogram_enhanced(sig)   - Enhanced spectrogram\n');
    fprintf('  plot_filter_response(b, a, fs)   - Filter frequency response\n\n');
    
    fprintf('🖼️ IMAGE PROCESSING:\n');
    fprintf('  plot_image_analysis(img)         - Multi-panel image analysis\n');
    fprintf('  plot_image_histogram_rgb(img)    - RGB histogram analysis\n');
    fprintf('  plot_edge_detection_comparison(img) - Edge detection methods\n\n');
    
    fprintf('🎨 3D VISUALIZATION:\n');
    fprintf('  plot_surface_enhanced(X, Y, Z)   - Enhanced surface plots\n');
    fprintf('  plot_3d_scatter(data)            - 3D scatter with coloring\n');
    fprintf('  animate_3d_rotation(fig_handle)  - Animated 3D rotation\n\n');
    
    fprintf('⚙️ UTILITIES:\n');
    fprintf('  set_plot_theme(theme_name)       - Apply consistent themes\n');
    fprintf('  export_plot_formats(basename)    - Export to multiple formats\n');
    fprintf('  create_subplot_grid(rows, cols)  - Smart subplot layouts\n\n');
end

function enhanced_plot(x, y, varargin)
    % Enhanced line plot with professional styling
    % Usage: enhanced_plot(x, y, 'PropertyName', PropertyValue, ...)
    
    % Parse optional arguments
    p = inputParser;
    addParameter(p, 'LineWidth', 2);
    addParameter(p, 'Color', 'auto');
    addParameter(p, 'Style', '-');
    addParameter(p, 'Title', '');
    addParameter(p, 'XLabel', '');
    addParameter(p, 'YLabel', '');
    addParameter(p, 'Grid', true);
    addParameter(p, 'Legend', '');
    parse(p, varargin{:});
    
    % Create plot
    if strcmp(p.Results.Color, 'auto')
        plot(x, y, p.Results.Style, 'LineWidth', p.Results.LineWidth);
    else
        plot(x, y, p.Results.Style, 'LineWidth', p.Results.LineWidth, 'Color', p.Results.Color);
    end
    
    % Styling
    if p.Results.Grid
        grid on;
        set(gca, 'GridAlpha', 0.3);
    end
    
    if ~isempty(p.Results.Title)
        title(p.Results.Title, 'FontSize', 14, 'FontWeight', 'bold');
    end
    
    if ~isempty(p.Results.XLabel)
        xlabel(p.Results.XLabel, 'FontSize', 12);
    end
    
    if ~isempty(p.Results.YLabel)
        ylabel(p.Results.YLabel, 'FontSize', 12);
    end
    
    if ~isempty(p.Results.Legend)
        legend(p.Results.Legend, 'Location', 'best');
    end
    
    % Professional styling
    set(gca, 'FontSize', 10);
    set(gca, 'LineWidth', 1.2);
    box on;
end

function multi_subplot(data_cell, layout, titles)
    % Create multiple subplots with consistent styling
    % Usage: multi_subplot({x1,y1; x2,y2; x3,y3}, [2,2], {'Title1','Title2','Title3'})
    
    [rows, cols] = deal(layout(1), layout(2));
    
    for i = 1:size(data_cell, 1)
        subplot(rows, cols, i);
        
        if size(data_cell{i}, 2) >= 2
            x_data = data_cell{i}(:, 1);
            y_data = data_cell{i}(:, 2);
        else
            x_data = 1:length(data_cell{i});
            y_data = data_cell{i};
        end
        
        plot(x_data, y_data, 'LineWidth', 1.5);
        grid on;
        
        if exist('titles', 'var') && i <= length(titles)
            title(titles{i}, 'FontSize', 12);
        end
        
        set(gca, 'FontSize', 9);
    end
    
    % Adjust spacing
    set(gcf, 'Position', [100, 100, 800, 600]);
end

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

function plot_histogram_with_stats(data, varargin)
    % Histogram with overlaid statistical information
    % Usage: plot_histogram_with_stats(data, 'Bins', 20, 'ShowStats', true)
    
    p = inputParser;
    addParameter(p, 'Bins', 15);
    addParameter(p, 'ShowStats', true);
    addParameter(p, 'Title', 'Data Distribution');
    parse(p, varargin{:});
    
    % Create histogram
    [n, x] = hist(data, p.Results.Bins);
    bar(x, n, 'FaceColor', [0.3, 0.6, 0.9], 'EdgeColor', 'black', 'FaceAlpha', 0.7);
    
    % Add statistics
    if p.Results.ShowStats
        hold on;
        
        % Mean line
        mean_val = mean(data);
        ylim_vals = ylim;
        plot([mean_val, mean_val], ylim_vals, 'r-', 'LineWidth', 2);
        
        % Standard deviation lines
        std_val = std(data);
        plot([mean_val-std_val, mean_val-std_val], ylim_vals, 'r--', 'LineWidth', 1);
        plot([mean_val+std_val, mean_val+std_val], ylim_vals, 'r--', 'LineWidth', 1);
        
        % Add text annotations
        text_y = max(ylim_vals) * 0.8;
        text(mean_val + 0.1*range(xlim), text_y, sprintf('μ = %.2f', mean_val), 'FontSize', 10);
        text(mean_val + 0.1*range(xlim), text_y*0.9, sprintf('σ = %.2f', std_val), 'FontSize', 10);
        text(mean_val + 0.1*range(xlim), text_y*0.8, sprintf('n = %d', length(data)), 'FontSize', 10);
        
        hold off;
        legend('Data', 'Mean', '±1 SD', 'Location', 'best');
    end
    
    title(p.Results.Title, 'FontSize', 14);
    xlabel('Value', 'FontSize', 12);
    ylabel('Frequency', 'FontSize', 12);
    grid on;
end

function plot_correlation_matrix(data, var_names)
    % Plot correlation matrix as heatmap
    % Usage: plot_correlation_matrix(data_matrix, {'Var1', 'Var2', 'Var3'})
    
    % Calculate correlation matrix
    corr_matrix = corrcoef(data);
    
    % Create heatmap
    imagesc(corr_matrix);
    colormap(jet);
    colorbar;
    
    % Set axis properties
    if exist('var_names', 'var')
        set(gca, 'XTick', 1:length(var_names));
        set(gca, 'XTickLabel', var_names);
        set(gca, 'YTick', 1:length(var_names));
        set(gca, 'YTickLabel', var_names);
    end
    
    % Add correlation values as text
    [rows, cols] = size(corr_matrix);
    for i = 1:rows
        for j = 1:cols
            if abs(corr_matrix(i,j)) > 0.3
                text_color = 'white';
            else
                text_color = 'black';
            end
            text(j, i, sprintf('%.2f', corr_matrix(i,j)), ...
                'HorizontalAlignment', 'center', 'Color', text_color, 'FontWeight', 'bold');
        end
    end
    
    title('Correlation Matrix', 'FontSize', 14);
    axis equal tight;
    caxis([-1, 1]);
end

function plot_time_series(dates, data, varargin)
    % Enhanced time series plotting with trend analysis
    % Usage: plot_time_series(date_vector, data_vector, 'ShowTrend', true)
    
    p = inputParser;
    addParameter(p, 'ShowTrend', false);
    addParameter(p, 'TrendColor', 'red');
    addParameter(p, 'Title', 'Time Series Analysis');
    parse(p, varargin{:});
    
    % Main time series plot
    plot(dates, data, 'b-', 'LineWidth', 1.5);
    hold on;
    
    % Add trend line
    if p.Results.ShowTrend
        trend_coeffs = polyfit(dates, data, 1);
        trend_line = polyval(trend_coeffs, dates);
        plot(dates, trend_line, '--', 'Color', p.Results.TrendColor, 'LineWidth', 2);
        legend('Data', 'Trend', 'Location', 'best');
    end
    
    % Styling
    grid on;
    xlabel('Date', 'FontSize', 12);
    ylabel('Value', 'FontSize', 12);
    title(p.Results.Title, 'FontSize', 14);
    
    % Format x-axis for dates
    if isnumeric(dates(1)) && dates(1) > 700000 % datenum format
        datetick('x', 'yyyy-mm-dd', 'keepticks');
    end
    
    hold off;
end

function plot_signal_analysis(signal, fs, varargin)
    % Comprehensive signal analysis plot (time + frequency domain)
    % Usage: plot_signal_analysis(signal_vector, sampling_rate, 'Title', 'My Signal')
    
    p = inputParser;
    addParameter(p, 'Title', 'Signal Analysis');
    addParameter(p, 'ShowPhase', false);
    parse(p, varargin{:});
    
    % Time vector
    t = (0:length(signal)-1) / fs;
    
    % Create subplots
    figure;
    
    % Time domain
    subplot(2, 2, 1);
    plot(t, signal, 'LineWidth', 1.2);
    xlabel('Time (s)');
    ylabel('Amplitude');
    title('Time Domain');
    grid on;
    
    % Frequency domain
    subplot(2, 2, 2);
    Y = fft(signal);
    f = (0:length(Y)-1) * fs / length(Y);
    plot(f(1:floor(end/2)), abs(Y(1:floor(end/2))));
    xlabel('Frequency (Hz)');
    ylabel('Magnitude');
    title('Frequency Domain');
    grid on;
    
    % Power spectral density
    subplot(2, 2, 3);
    psd = abs(Y).^2 / (fs * length(signal));
    semilogy(f(1:floor(end/2)), psd(1:floor(end/2)));
    xlabel('Frequency (Hz)');
    ylabel('Power Spectral Density');
    title('Power Spectrum');
    grid on;
    
    % Phase plot (if requested)
    subplot(2, 2, 4);
    if p.Results.ShowPhase
        phase = angle(Y);
        plot(f(1:floor(end/2)), phase(1:floor(end/2)));
        xlabel('Frequency (Hz)');
        ylabel('Phase (rad)');
        title('Phase Spectrum');
    else
        % Spectrogram instead
        spectrogram(signal, 256, 128, 256, fs, 'yaxis');
        title('Spectrogram');
    end
    grid on;
    
    % Overall title
    suptitle(p.Results.Title);
end

function plot_surface_enhanced(X, Y, Z, varargin)
    % Enhanced 3D surface plot with professional styling
    % Usage: plot_surface_enhanced(X, Y, Z, 'Colormap', 'jet', 'Lighting', true)
    
    p = inputParser;
    addParameter(p, 'Colormap', 'parula');
    addParameter(p, 'Lighting', true);
    addParameter(p, 'Transparency', 1.0);
    addParameter(p, 'Title', '3D Surface');
    parse(p, varargin{:});
    
    % Create surface
    surf(X, Y, Z, 'FaceAlpha', p.Results.Transparency, 'EdgeColor', 'none');
    
    % Apply colormap
    colormap(p.Results.Colormap);
    colorbar;
    
    % Lighting effects
    if p.Results.Lighting
        shading interp;
        lighting gouraud;
        camlight;
    end
    
    % Labels and title
    xlabel('X axis', 'FontSize', 12);
    ylabel('Y axis', 'FontSize', 12);
    zlabel('Z axis', 'FontSize', 12);
    title(p.Results.Title, 'FontSize', 14);
    
    % Set viewing angle
    view(45, 30);
    axis tight;
end

function plot_image_analysis(img)
    % Multi-panel image analysis display
    % Usage: plot_image_analysis(image_matrix)
    
    figure;
    
    % Original image
    subplot(2, 3, 1);
    imshow(img);
    title('Original Image', 'FontSize', 12);
    
    % Convert to grayscale if RGB
    if size(img, 3) == 3
        gray_img = rgb2gray(img);
    else
        gray_img = img;
    end
    
    % Grayscale
    subplot(2, 3, 2);
    imshow(gray_img);
    title('Grayscale', 'FontSize', 12);
    
    % Histogram
    subplot(2, 3, 3);
    imhist(gray_img);
    title('Histogram', 'FontSize', 12);
    
    % Edge detection
    subplot(2, 3, 4);
    edges = edge(gray_img, 'canny');
    imshow(edges);
    title('Edge Detection', 'FontSize', 12);
    
    % Enhanced contrast
    subplot(2, 3, 5);
    enhanced = imadjust(gray_img);
    imshow(enhanced);
    title('Enhanced Contrast', 'FontSize', 12);
    
    % Image statistics
    subplot(2, 3, 6);
    axis off;
    stats_text = sprintf(['Image Statistics:\n\n' ...
                         'Size: %d × %d\n' ...
                         'Channels: %d\n' ...
                         'Data type: %s\n' ...
                         'Min value: %.3f\n' ...
                         'Max value: %.3f\n' ...
                         'Mean: %.3f\n' ...
                         'Std: %.3f'], ...
                        size(img, 1), size(img, 2), size(img, 3), ...
                        class(img), min(gray_img(:)), max(gray_img(:)), ...
                        mean(gray_img(:)), std(double(gray_img(:))));
    text(0.1, 0.8, stats_text, 'FontSize', 10, 'VerticalAlignment', 'top');
    
    suptitle('Image Analysis Dashboard');
end

function plot_stock_chart(stock_data)
    % OHLC candlestick chart for stock data
    % Usage: plot_stock_chart(stock_table)
    
    dates = datenum(stock_data.Date);
    opens = stock_data.Open;
    highs = stock_data.High;
    lows = stock_data.Low;
    closes = stock_data.Close;
    volumes = stock_data.Volume;
    
    % Main price chart
    subplot(2, 1, 1);
    
    % Simple OHLC representation
    for i = 1:length(dates)
        % Determine color (green for up, red for down)
        if closes(i) >= opens(i)
            color = 'green';
        else
            color = 'red';
        end
        
        % Draw high-low line
        plot([dates(i), dates(i)], [lows(i), highs(i)], 'k-', 'LineWidth', 1);
        hold on;
        
        % Draw open-close box
        box_height = abs(closes(i) - opens(i));
        box_bottom = min(opens(i), closes(i));
        
        rectangle('Position', [dates(i)-0.3, box_bottom, 0.6, box_height], ...
                 'FaceColor', color, 'EdgeColor', 'black');
    end
    
    % Styling
    datetick('x', 'yyyy-mm');
    xlabel('Date');
    ylabel('Price ($)');
    title(sprintf('%s Stock Price', stock_data.Symbol{1}));
    grid on;
    hold off;
    
    % Volume chart
    subplot(2, 1, 2);
    bar(dates, volumes/1e6, 'FaceColor', [0.7, 0.7, 0.7]);
    datetick('x', 'yyyy-mm');
    xlabel('Date');
    ylabel('Volume (Millions)');
    title('Trading Volume');
    grid on;
end

function set_plot_theme(theme_name)
    % Apply consistent plot themes
    % Usage: set_plot_theme('professional') % 'professional', 'dark', 'minimal'
    
    switch lower(theme_name)
        case 'professional'
            set(0, 'DefaultAxesFontSize', 11);
            set(0, 'DefaultAxesFontWeight', 'normal');
            set(0, 'DefaultAxesLineWidth', 1.2);
            set(0, 'DefaultLineLineWidth', 1.5);
            set(0, 'DefaultAxesColor', 'white');
            set(0, 'DefaultFigureColor', 'white');
            
        case 'dark'
            set(0, 'DefaultAxesColor', [0.15, 0.15, 0.15]);
            set(0, 'DefaultFigureColor', [0.1, 0.1, 0.1]);
            set(0, 'DefaultAxesXColor', 'white');
            set(0, 'DefaultAxesYColor', 'white');
            set(0, 'DefaultTextColor', 'white');
            
        case 'minimal'
            set(0, 'DefaultAxesLineWidth', 0.8);
            set(0, 'DefaultLineLineWidth', 1.0);
            set(0, 'DefaultAxesFontSize', 10);
            set(0, 'DefaultAxesBox', 'off');
            
        otherwise
            fprintf('Unknown theme: %s\n', theme_name);
            fprintf('Available themes: professional, dark, minimal\n');
    end
    
    fprintf('Applied %s theme\n', theme_name);
end

function create_subplot_grid(rows, cols, spacing)
    % Create subplot grid with custom spacing
    % Usage: handles = create_subplot_grid(2, 3, 0.05)
    
    if nargin < 3
        spacing = 0.05;
    end
    
    handles = zeros(rows, cols);
    
    % Calculate subplot positions
    width = (1 - (cols + 1) * spacing) / cols;
    height = (1 - (rows + 1) * spacing) / rows;
    
    for i = 1:rows
        for j = 1:cols
            left = spacing + (j - 1) * (width + spacing);
            bottom = 1 - spacing - i * (height + spacing);
            
            handles(i, j) = subplot('Position', [left, bottom, width, height]);
        end
    end
    
    fprintf('Created %dx%d subplot grid\n', rows, cols);
end

function export_plot_formats(basename)
    % Export current figure to multiple formats
    % Usage: export_plot_formats('my_analysis')
    
    formats = {'png', 'pdf', 'eps'};
    
    for i = 1:length(formats)
        filename = sprintf('%s.%s', basename, formats{i});
        
        switch formats{i}
            case 'png'
                print('-dpng', '-r300', filename);
            case 'pdf'
                print('-dpdf', filename);
            case 'eps'
                print('-deps2', filename);
        end
        
        fprintf('Exported: %s\n', filename);
    end
end

function plot_filter_response(b, a, fs)
    % Plot filter frequency response
    % Usage: plot_filter_response(numerator, denominator, sampling_rate)
    
    % Calculate frequency response
    [h, w] = freqz(b, a, 512, fs);
    
    % Magnitude response
    subplot(2, 1, 1);
    semilogx(w, 20*log10(abs(h)));
    xlabel('Frequency (Hz)');
    ylabel('Magnitude (dB)');
    title('Magnitude Response');
    grid on;
    
    % Phase response
    subplot(2, 1, 2);
    semilogx(w, angle(h) * 180/pi);
    xlabel('Frequency (Hz)');
    ylabel('Phase (degrees)');
    title('Phase Response');
    grid on;
    
    suptitle('Filter Frequency Response');
end

function animate_3d_rotation(fig_handle, duration)
    % Animate 3D plot rotation
    % Usage: animate_3d_rotation(gcf, 10) % 10 second rotation
    
    if nargin < 2
        duration = 5; % Default 5 seconds
    end
    
    if nargin < 1
        fig_handle = gcf;
    end
    
    figure(fig_handle);
    
    % Animation parameters
    frames = 60;
    pause_time = duration / frames;
    
    % Get current view
    [az, el] = view;
    
    % Rotate around z-axis
    for i = 1:frames
        new_az = az + (i-1) * 360 / frames;
        view(new_az, el);
        drawnow;
        pause(pause_time);
    end
    
    fprintf('Animation complete\n');
end

function plot_box_comparison(data_groups, group_names)
    % Enhanced box plot for group comparisons
    % Usage: plot_box_comparison({group1_data, group2_data}, {'Group 1', 'Group 2'})
    
    % Prepare data for boxplot
    all_data = [];
    group_labels = [];
    
    for i = 1:length(data_groups)
        all_data = [all_data; data_groups{i}(:)];
        group_labels = [group_labels; i * ones(length(data_groups{i}(:)), 1)];
    end
    
    % Create box plot
    boxplot(all_data, group_labels);
    
    % Customize
    if exist('group_names', 'var')
        set(gca, 'XTickLabel', group_names);
    end
    
    ylabel('Value');
    title('Group Comparison');
    grid on;
    
    % Add sample sizes
    for i = 1:length(data_groups)
        n = length(data_groups{i}(:));
        text(i, min(all_data), sprintf('n=%d', n), ...
             'HorizontalAlignment', 'center', 'FontSize', 9);
    end
end

function plot_spectrogram_enhanced(signal, fs, varargin)
    % Enhanced spectrogram with better visualization
    % Usage: plot_spectrogram_enhanced(signal, fs, 'WindowSize', 256)
    
    p = inputParser;
    addParameter(p, 'WindowSize', 256);
    addParameter(p, 'Overlap', 128);
    addParameter(p, 'Title', 'Enhanced Spectrogram');
    parse(p, varargin{:});
    
    % Calculate spectrogram
    [S, f, t] = specgram(signal, p.Results.WindowSize, fs, p.Results.WindowSize, p.Results.Overlap);
    
    % Plot with enhanced visualization
    imagesc(t, f, 20*log10(abs(S) + eps));
    axis xy;
    colormap(jet);
    colorbar;
    
    xlabel('Time (s)', 'FontSize', 12);
    ylabel('Frequency (Hz)', 'FontSize', 12);
    title(p.Results.Title, 'FontSize', 14);
    
    % Set color limits for better contrast
    clim([-80, 20]);
end

% Helper function for consistent figure sizing
function resize_figure(width, height)
    % Resize current figure to specified dimensions
    % Usage: resize_figure(800, 600) % pixels
    
    if nargin < 2
        height = width * 0.75; % Default 4:3 aspect ratio
    end
    
    set(gcf, 'Position', [100, 100, width, height]);
end

% Color palette utilities
function colors = get_color_palette(palette_name, n_colors)
    % Get predefined color palettes
    % Usage: colors = get_color_palette('vibrant', 5)
    
    if nargin < 2
        n_colors = 6;
    end
    
    switch lower(palette_name)
        case 'vibrant'
            base_colors = [
                1.0, 0.0, 0.0;  % Red
                0.0, 0.7, 0.0;  % Green
                0.0, 0.0, 1.0;  % Blue
                1.0, 0.5, 0.0;  % Orange
                0.7, 0.0, 0.7;  % Purple
                0.0, 0.8, 0.8;  % Cyan
            ];
            
        case 'pastel'
            base_colors = [
                1.0, 0.7, 0.7;  % Light red
                0.7, 1.0, 0.7;  % Light green
                0.7, 0.7, 1.0;  % Light blue
                1.0, 0.9, 0.7;  % Light orange
                0.9, 0.7, 0.9;  % Light purple
                0.7, 0.9, 0.9;  % Light cyan
            ];
            
        case 'professional'
            base_colors = [
                0.2, 0.4, 0.6;  % Steel blue
                0.6, 0.2, 0.2;  % Dark red
                0.2, 0.6, 0.2;  % Forest green
                0.6, 0.4, 0.2;  % Brown
                0.4, 0.2, 0.6;  % Purple
                0.6, 0.6, 0.2;  % Olive
            ];
            
        otherwise
            base_colors = lines(6); % Default MATLAB colors
    end
    
    % Interpolate to get requested number of colors
    colors = interp1(1:size(base_colors, 1), base_colors, ...
                    linspace(1, size(base_colors, 1), n_colors));
end

function plot_image_histogram_rgb(img)
    % RGB histogram analysis for color images
    % Usage: plot_image_histogram_rgb(image_matrix)
    
    if size(img, 3) ~= 3
        error('Image must be RGB (3 channels)');
    end
    
    figure;
    
    % Original image
    subplot(2, 2, 1);
    imshow(img);
    title('Original RGB Image', 'FontSize', 12);
    
    % Red channel histogram
    subplot(2, 2, 2);
    imhist(img(:,:,1));
    title('Red Channel Histogram', 'FontSize', 12);
    set(gca, 'Color', [1, 0.9, 0.9]); % Light red background
    
    % Green channel histogram
    subplot(2, 2, 3);
    imhist(img(:,:,2));
    title('Green Channel Histogram', 'FontSize', 12);
    set(gca, 'Color', [0.9, 1, 0.9]); % Light green background
    
    % Blue channel histogram
    subplot(2, 2, 4);
    imhist(img(:,:,3));
    title('Blue Channel Histogram', 'FontSize', 12);
    set(gca, 'Color', [0.9, 0.9, 1]); % Light blue background
    
    suptitle('RGB Histogram Analysis');
end

function plot_edge_detection_comparison(img)
    % Compare different edge detection methods
    % Usage: plot_edge_detection_comparison(image_matrix)
    
    % Convert to grayscale if needed
    if size(img, 3) == 3
        gray_img = rgb2gray(img);
    else
        gray_img = img;
    end
    
    figure;
    
    % Original image
    subplot(2, 3, 1);
    imshow(gray_img);
    title('Original Image', 'FontSize', 12);
    
    % Sobel edge detection
    subplot(2, 3, 2);
    edges_sobel = edge(gray_img, 'sobel');
    imshow(edges_sobel);
    title('Sobel Edges', 'FontSize', 12);
    
    % Canny edge detection
    subplot(2, 3, 3);
    edges_canny = edge(gray_img, 'canny');
    imshow(edges_canny);
    title('Canny Edges', 'FontSize', 12);
    
    % Prewitt edge detection
    subplot(2, 3, 4);
    edges_prewitt = edge(gray_img, 'prewitt');
    imshow(edges_prewitt);
    title('Prewitt Edges', 'FontSize', 12);
    
    % Roberts edge detection
    subplot(2, 3, 5);
    edges_roberts = edge(gray_img, 'roberts');
    imshow(edges_roberts);
    title('Roberts Edges', 'FontSize', 12);
    
    % Edge statistics
    subplot(2, 3, 6);
    axis off;
    
    % Calculate edge statistics
    sobel_count = sum(edges_sobel(:));
    canny_count = sum(edges_canny(:));
    prewitt_count = sum(edges_prewitt(:));
    roberts_count = sum(edges_roberts(:));
    
    stats_text = sprintf(['Edge Detection Statistics:\n\n' ...
                         'Sobel: %d pixels (%.1f%%)\n' ...
                         'Canny: %d pixels (%.1f%%)\n' ...
                         'Prewitt: %d pixels (%.1f%%)\n' ...
                         'Roberts: %d pixels (%.1f%%)\n\n' ...
                         'Image size: %d × %d\n' ...
                         'Total pixels: %d'], ...
                        sobel_count, sobel_count/numel(gray_img)*100, ...
                        canny_count, canny_count/numel(gray_img)*100, ...
                        prewitt_count, prewitt_count/numel(gray_img)*100, ...
                        roberts_count, roberts_count/numel(gray_img)*100, ...
                        size(gray_img, 1), size(gray_img, 2), numel(gray_img));
    
    text(0.1, 0.9, stats_text, 'FontSize', 10, 'VerticalAlignment', 'top');
    
    suptitle('Edge Detection Method Comparison');
end

function plot_seasonal_decomp(ts_data, period)
    % Plot seasonal decomposition of time series
    % Usage: plot_seasonal_decomp(time_series_data, 12) % 12 for monthly data
    
    if nargin < 2
        period = 12; % Default monthly seasonality
    end
    
    % Simple seasonal decomposition
    n = length(ts_data);
    
    % Trend (moving average)
    half_window = floor(period/2);
    trend = zeros(size(ts_data));
    
    for i = (half_window+1):(n-half_window)
        trend(i) = mean(ts_data((i-half_window):(i+half_window)));
    end
    
    % Seasonal component
    detrended = ts_data - trend;
    seasonal = zeros(size(ts_data));
    
    for i = 1:period
        season_indices = i:period:n;
        if ~isempty(season_indices)
            seasonal(season_indices) = mean(detrended(season_indices), 'omitnan');
        end
    end
    
    % Residual
    residual = ts_data - trend - seasonal;
    
    % Plot decomposition
    figure;
    
    % Original series
    subplot(4, 1, 1);
    plot(1:n, ts_data, 'LineWidth', 1.5);
    title('Original Time Series', 'FontSize', 12);
    grid on;
    
    % Trend
    subplot(4, 1, 2);
    plot(1:n, trend, 'LineWidth', 1.5, 'Color', 'red');
    title('Trend Component', 'FontSize', 12);
    grid on;
    
    % Seasonal
    subplot(4, 1, 3);
    plot(1:n, seasonal, 'LineWidth', 1.5, 'Color', 'green');
    title('Seasonal Component', 'FontSize', 12);
    grid on;
    
    % Residual
    subplot(4, 1, 4);
    plot(1:n, residual, 'LineWidth', 1.5, 'Color', 'orange');
    title('Residual Component', 'FontSize', 12);
    xlabel('Time Period', 'FontSize', 12);
    grid on;
    
    suptitle('Seasonal Decomposition');
end

function plot_3d_scatter(data, varargin)
    % Enhanced 3D scatter plot with coloring
    % Usage: plot_3d_scatter(data_matrix, 'ColorBy', 'column', 'Size', 50)
    
    if size(data, 2) < 3
        error('Data must have at least 3 columns for 3D plotting');
    end
    
    p = inputParser;
    addParameter(p, 'ColorBy', 'z'); % 'z', 'column', 'constant'
    addParameter(p, 'ColorColumn', 4);
    addParameter(p, 'Size', 36);
    addParameter(p, 'Title', '3D Scatter Plot');
    parse(p, varargin{:});
    
    x = data(:, 1);
    y = data(:, 2);
    z = data(:, 3);
    
    % Determine coloring
    switch lower(p.Results.ColorBy)
        case 'z'
            colors = z;
        case 'column'
            if size(data, 2) >= p.Results.ColorColumn
                colors = data(:, p.Results.ColorColumn);
            else
                colors = z;
                warning('Color column not available, using z-values');
            end
        case 'constant'
            colors = ones(size(z));
        otherwise
            colors = z;
    end
    
    % Create 3D scatter
    scatter3(x, y, z, p.Results.Size, colors, 'filled');
    colormap(jet);
    colorbar;
    
    xlabel('X axis', 'FontSize', 12);
    ylabel('Y axis', 'FontSize', 12);
    zlabel('Z axis', 'FontSize', 12);
    title(p.Results.Title, 'FontSize', 14);
    
    grid on;
    view(45, 30);
end