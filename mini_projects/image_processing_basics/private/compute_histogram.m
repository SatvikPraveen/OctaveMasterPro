% Location: mini_projects/image_processing_basics/private/compute_histogram.m
% (split out of histogram_analysis.m so it can be called from other files)

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
