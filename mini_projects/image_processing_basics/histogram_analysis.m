% Location: mini_projects/image_processing_basics/histogram_analysis.m
% Histogram Processing and Equalization Functions

function [hist_counts, bin_centers] = compute_histogram(img, num_bins)
    % Compute image histogram
    
    if nargin < 2, num_bins = 256; end
    
    if size(img, 3) > 1, img = rgb2gray_custom(img); end
    
    if max(img(:)) <= 1
        img = img * 255;
    end
    
    img = uint8(img);
    
    bin_edges = 0:255/(num_bins-1):255;
    bin_centers = bin_edges(1:end-1) + diff(bin_edges)/2;
    
    hist_counts = hist(double(img(:)), bin_centers);
    hist_counts = hist_counts / sum(hist_counts); % Normalize
end

function equalized_img = histogram_equalization(img)
    % Histogram equalization for contrast enhancement
    
    if size(img, 3) > 1, img = rgb2gray_custom(img); end
    
    original_range = [min(img(:)), max(img(:))];
    
    if max(img(:)) <= 1
        img = img * 255;
        normalize_output = true;
    else
        normalize_output = false;
    end
    
    img = uint8(img);
    [h, w] = size(img);
    
    % Compute histogram
    hist_counts = hist(double(img(:)), 0:255);
    
    % Compute cumulative distribution
    cdf = cumsum(hist_counts);
    cdf = cdf / cdf(end); % Normalize to [0,1]
    
    % Create mapping function
    mapping = round(255 * cdf);
    
    % Apply equalization
    equalized_img = zeros(size(img));
    for i = 1:h
        for j = 1:w
            equalized_img(i,j) = mapping(img(i,j) + 1); % +1 for MATLAB indexing
        end
    end
    
    if normalize_output
        equalized_img = equalized_img / 255;
    end
end

function enhanced_img = adaptive_histogram_equalization(img, varargin)
    % Contrast Limited Adaptive Histogram Equalization (CLAHE)
    
    % Default parameters
    tile_size = [8, 8];
    clip_limit = 0.02;
    
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'tile_size'
                tile_size = varargin{i+1};
            case 'clip_limit'
                clip_limit = varargin{i+1};
        end
    end
    
    if size(img, 3) > 1, img = rgb2gray_custom(img); end
    
    if max(img(:)) <= 1
        img = img * 255;
        normalize_output = true;
    else
        normalize_output = false;
    end
    
    [h, w] = size(img);
    tile_h = round(h / tile_size(1));
    tile_w = round(w / tile_size(2));
    
    enhanced_img = zeros(size(img));
    
    % Process each tile
    for tile_i = 1:tile_size(1)
        for tile_j = 1:tile_size(2)
            % Define tile boundaries
            row_start = (tile_i-1) * tile_h + 1;
            row_end = min(tile_i * tile_h, h);
            col_start = (tile_j-1) * tile_w + 1;
            col_end = min(tile_j * tile_w, w);
            
            % Extract tile
            tile = img(row_start:row_end, col_start:col_end);
            
            % Compute clipped histogram
            hist_counts = hist(double(tile(:)), 0:255);
            
            % Apply clipping
            clip_value = clip_limit * numel(tile) / 256;
            clipped_hist = min(hist_counts, clip_value);
            
            % Redistribute clipped pixels
            excess = sum(hist_counts - clipped_hist);
            redistribution = excess / 256;
            clipped_hist = clipped_hist + redistribution;
            
            % Compute CDF and mapping
            cdf = cumsum(clipped_hist);
            cdf = cdf / cdf(end);
            mapping = round(255 * cdf);
            
            % Apply equalization to tile
            for i_tile = 1:size(tile, 1)
                for j_tile = 1:size(tile, 2)
                    pixel_val = tile(i_tile, j_tile);
                    enhanced_img(row_start + i_tile - 1, col_start + j_tile - 1) = mapping(pixel_val + 1);
                end
            end
        end
    end
    
    if normalize_output
        enhanced_img = enhanced_img / 255;
    end
end

function matched_img = histogram_matching(img, target_img)
    % Histogram matching (specification)
    
    if size(img, 3) > 1, img = rgb2gray_custom(img); end
    if size(target_img, 3) > 1, target_img = rgb2gray_custom(target_img); end
    
    if max(img(:)) <= 1
        img = img * 255;
        normalize_output = true;
    else
        normalize_output = false;
    end
    
    if max(target_img(:)) <= 1
        target_img = target_img * 255;
    end
    
    img = uint8(img);
    target_img = uint8(target_img);
    
    % Compute histograms
    hist_source = hist(double(img(:)), 0:255);
    hist_target = hist(double(target_img(:)), 0:255);
    
    % Compute CDFs
    cdf_source = cumsum(hist_source) / sum(hist_source);
    cdf_target = cumsum(hist_target) / sum(hist_target);
    
    % Create mapping function
    mapping = zeros(256, 1);
    for i = 1:256
        [~, idx] = min(abs(cdf_target - cdf_source(i)));
        mapping(i) = idx - 1; % -1 for 0-based indexing
    end
    
    % Apply mapping
    matched_img = zeros(size(img));
    [h, w] = size(img);
    
    for i = 1:h
        for j = 1:w
            matched_img(i,j) = mapping(img(i,j) + 1); % +1 for MATLAB indexing
        end
    end
    
    if normalize_output
        matched_img = matched_img / 255;
    end
end

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

function skew = skewness_custom(data)
    % Calculate skewness
    mu = mean(data);
    sigma = std(data);
    skew = mean(((data - mu) / sigma).^3);
end

function kurt = kurtosis_custom(data)
    % Calculate kurtosis
    mu = mean(data);
    sigma = std(data);
    kurt = mean(((data - mu) / sigma).^4) - 3;
end

function demo_histogram_processing()
    % Comprehensive histogram processing demonstration
    
    fprintf('\n--- Histogram Processing Demonstration ---\n');
    
    % Create test images with different contrast characteristics
    low_contrast = create_low_contrast_image();
    high_contrast = create_high_contrast_image();
    dark_image = create_dark_image();
    
    % Apply histogram processing
    eq_low = histogram_equalization(low_contrast);
    eq_high = histogram_equalization(high_contrast);
    eq_dark = histogram_equalization(dark_image);
    
    % Adaptive equalization
    adaptive_low = adaptive_histogram_equalization(low_contrast, 'tile_size', [4, 4]);
    adaptive_dark = adaptive_histogram_equalization(dark_image, 'tile_size', [6, 6]);
    
    % Display results
    figure('Position', [50, 50, 1400, 1000]);
    
    % Original images
    subplot(4, 4, 1); imshow(low_contrast, []); title('Low Contrast Original');
    subplot(4, 4, 2); imshow(high_contrast, []); title('High Contrast Original');
    subplot(4, 4, 3); imshow(dark_image, []); title('Dark Image Original');
    subplot(4, 4, 4); plot_histogram_comparison(low_contrast, 'Original Low Contrast');
    
    % Histogram equalized
    subplot(4, 4, 5); imshow(eq_low, []); title('HE: Low Contrast');
    subplot(4, 4, 6); imshow(eq_high, []); title('HE: High Contrast');
    subplot(4, 4, 7); imshow(eq_dark, []); title('HE: Dark Image');
    subplot(4, 4, 8); plot_histogram_comparison(eq_low, 'Equalized Low Contrast');
    
    % Adaptive equalized
    subplot(4, 4, 9); imshow(adaptive_low, []); title('CLAHE: Low Contrast');
    subplot(4, 4, 10); imshow(adaptive_dark, []); title('CLAHE: Dark Image');
    
    % Histogram matching demo
    matched = histogram_matching(dark_image, high_contrast);
    subplot(4, 4, 11); imshow(matched, []); title('Histogram Matched');
    subplot(4, 4, 12); plot_histogram_comparison(matched, 'Matched Histogram');
    
    % Statistics comparison
    subplot(4, 4, [13, 14, 15, 16]);
    compare_image_statistics({low_contrast, eq_low, adaptive_low}, ...
                           {'Original', 'Equalized', 'CLAHE'});
    
    sgtitle('Comprehensive Histogram Processing');
    
    fprintf('Histogram processing demonstration complete.\n');
end

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

function low_img = create_low_contrast_image()
    % Create low contrast test image
    [x, y] = meshgrid(1:100, 1:100);
    low_img = 0.4 + 0.2 * sin(x/10) .* cos(y/8);
    low_img = low_img + 0.05 * randn(size(x));
    low_img = max(0, min(1, low_img));
end

function high_img = create_high_contrast_image()
    % Create high contrast test image
    high_img = zeros(100, 100);
    high_img(20:40, 20:80) = 1;
    high_img(60:80, 20:40) = 1;
    high_img(60:80, 60:80) = 1;
end

function dark_img = create_dark_image()
    % Create dark test image
    [x, y] = meshgrid(1:100, 1:100);
    dark_img = 0.1 + 0.15 * exp(-((x-50).^2 + (y-50).^2)/500);
    dark_img = dark_img + 0.02 * randn(size(x));
    dark_img = max(0, min(1, dark_img));
end