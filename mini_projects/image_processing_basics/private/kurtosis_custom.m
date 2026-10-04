% Location: mini_projects/image_processing_basics/private/kurtosis_custom.m
% (split out of histogram_analysis.m so it can be called from other files)

function kurt = kurtosis_custom(data)
    % Calculate kurtosis
    mu = mean(data);
    sigma = std(data);
    kurt = mean(((data - mu) / sigma).^4) - 3;
end
