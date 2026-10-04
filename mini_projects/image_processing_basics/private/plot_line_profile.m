% Location: mini_projects/image_processing_basics/private/plot_line_profile.m
% (split out of basic_filters.m so it can be called from other files)

function plot_line_profile(original, noisy, gaussian, median)
    % Plot line profiles for filter comparison
    
    [h, w] = size(original);
    row = round(h/2); % Middle row
    
    x = 1:w;
    plot(x, original(row, :), 'g', 'LineWidth', 2); hold on;
    plot(x, noisy(row, :), 'r', 'LineWidth', 1);
    plot(x, gaussian(row, :), 'b', 'LineWidth', 1.5);
    plot(x, median(row, :), 'm', 'LineWidth', 1.5);
    
    legend('Original', 'Noisy', 'Gaussian', 'Median', 'Location', 'best');
    xlabel('Pixel Position');
    ylabel('Intensity');
    grid on;
end
