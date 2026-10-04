% Location: mini_projects/image_processing_basics/private/skewness_custom.m
% (split out of histogram_analysis.m so it can be called from other files)

function skew = skewness_custom(data)
    % Calculate skewness
    mu = mean(data);
    sigma = std(data);
    skew = mean(((data - mu) / sigma).^3);
end
