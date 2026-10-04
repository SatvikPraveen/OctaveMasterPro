% Location: mini_projects/image_processing_basics/private/histogram_equalization.m
% (split out of histogram_analysis.m so it can be called from other files)

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
            equalized_img(i,j) = mapping(double(img(i,j)) + 1); % +1 for 1-based indexing (double: uint8 255+1 saturates)
        end
    end
    
    if normalize_output
        equalized_img = equalized_img / 255;
    end
end
