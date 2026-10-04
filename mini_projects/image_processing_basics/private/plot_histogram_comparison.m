% Location: mini_projects/image_processing_basics/private/plot_histogram_comparison.m
% (split out of histogram_analysis.m so it can be called from other files)

function plot_histogram_comparison(img, img_title)
    % Plot histogram with statistics overlay
    
    [hist_counts, bin_centers] = compute_histogram(img, 64);
    
    bar(bin_centers, hist_counts, 'hist');
    title(img_title);
    xlabel('Intensity');
    ylabel('Normalized Frequency');
    grid on;
    
    % Add statistics text
    stats = analyze_histogram_statistics(img);
    stats_text = sprintf('μ=%.1f σ=%.1f', stats.mean, stats.std);
    text(0.7, 0.9, stats_text, 'Units', 'normalized', 'FontSize', 10, ...
         'BackgroundColor', 'white', 'EdgeColor', 'black');
end
