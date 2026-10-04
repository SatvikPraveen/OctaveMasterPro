% Location: mini_projects/image_processing_basics/private/compare_image_statistics.m
% (split out of histogram_analysis.m so it can be called from other files)

function compare_image_statistics(images, names)
    % Compare statistics across multiple images
    
    num_images = length(images);
    metrics = {'Mean', 'Std', 'Entropy', 'Contrast'};
    data = zeros(num_images, length(metrics));
    
    for i = 1:num_images
        stats = analyze_histogram_statistics(images{i});
        data(i, :) = [stats.mean, stats.std, stats.entropy, stats.rms_contrast];
    end
    
    % Normalize data for comparison
    data_norm = data ./ max(data, [], 1);
    
    bar(data_norm);
    set(gca, 'XTickLabel', names);
    legend(metrics, 'Location', 'best');
    title('Image Statistics Comparison');
    ylabel('Normalized Values');
    grid on;
end
