% Location: mini_projects/image_processing_basics/private/histogram_matching.m
% (split out of histogram_analysis.m so it can be called from other files)

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
            matched_img(i,j) = mapping(double(img(i,j)) + 1); % +1 for 1-based indexing (double: uint8 255+1 saturates)
        end
    end
    
    if normalize_output
        matched_img = matched_img / 255;
    end
end
