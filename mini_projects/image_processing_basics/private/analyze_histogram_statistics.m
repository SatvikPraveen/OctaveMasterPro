% Location: mini_projects/image_processing_basics/private/analyze_histogram_statistics.m
% (split out of histogram_analysis.m so it can be called from other files)

function stats = analyze_histogram_statistics(img)
    % Compute comprehensive histogram statistics
    
    if size(img, 3) > 1, img = rgb2gray_custom(img); end
    
    % Normalize to [0, 255] for consistent analysis
    if max(img(:)) <= 1
        img = img * 255;
    end
    
    pixel_values = img(:);
    
    % Basic statistics
    stats.mean = mean(pixel_values);
    stats.median = median(pixel_values);
    stats.mode = mode(pixel_values);
    stats.std = std(pixel_values);
    stats.variance = var(pixel_values);
    stats.range = [min(pixel_values), max(pixel_values)];
    
    % Higher order moments
    stats.skewness = skewness_custom(pixel_values);
    stats.kurtosis = kurtosis_custom(pixel_values);
    
    % Entropy (measure of randomness)
    [hist_counts, ~] = compute_histogram(img, 256);
    hist_counts = hist_counts(hist_counts > 0); % Remove zeros
    stats.entropy = -sum(hist_counts .* log2(hist_counts));
    
    % Contrast measures
    stats.rms_contrast = sqrt(mean((pixel_values - stats.mean).^2));
    stats.michelson_contrast = (max(pixel_values) - min(pixel_values)) / (max(pixel_values) + min(pixel_values));
    
    % Print statistics
    fprintf('\nHistogram Statistics:\n');
    fprintf('--------------------\n');
    fprintf('Mean: %.2f\n', stats.mean);
    fprintf('Median: %.2f\n', stats.median);
    fprintf('Standard Deviation: %.2f\n', stats.std);
    fprintf('Skewness: %.3f\n', stats.skewness);
    fprintf('Kurtosis: %.3f\n', stats.kurtosis);
    fprintf('Entropy: %.3f bits\n', stats.entropy);
    fprintf('RMS Contrast: %.2f\n', stats.rms_contrast);
    fprintf('Michelson Contrast: %.3f\n', stats.michelson_contrast);
end
