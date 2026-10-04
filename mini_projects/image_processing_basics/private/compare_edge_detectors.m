% Location: mini_projects/image_processing_basics/private/compare_edge_detectors.m
% (split out of basic_filters.m so it can be called from other files)

function compare_edge_detectors(img)
    % Compare different edge detection methods
    
    % Convert to grayscale if needed
    if size(img, 3) > 1
        img = rgb2gray_custom(img);
    end
    
    % Apply different edge detection methods
    [sobel_mag, ~] = sobel_edge_detection(img);
    canny_edges = canny_edge_detection(img, 'sigma', 1);
    
    % Simple gradient-based edge detection
    [gx, gy] = gradient(img);
    gradient_mag = sqrt(gx.^2 + gy.^2);
    
    % Laplacian edge detection
    laplacian_kernel = [0 1 0; 1 -4 1; 0 1 0];
    laplacian_edges = abs(conv2(img, laplacian_kernel, 'same'));
    
    % Display comparison
    figure('Position', [100, 100, 1200, 800]);
    
    subplot(2, 3, 1);
    imshow(img, []);
    title('Original Image');
    
    subplot(2, 3, 2);
    imshow(sobel_mag, []);
    title('Sobel Edges');
    
    subplot(2, 3, 3);
    imshow(gradient_mag, []);
    title('Gradient Magnitude');
    
    subplot(2, 3, 4);
    imshow(laplacian_edges, []);
    title('Laplacian Edges');
    
    subplot(2, 3, 5);
    imshow(canny_edges, []);
    title('Canny Edges');
    
    subplot(2, 3, 6);
    % Edge strength histogram
    edges_data = {sobel_mag(:), gradient_mag(:), laplacian_edges(:)};
    edge_names = {'Sobel', 'Gradient', 'Laplacian'};
    colors = {'r', 'g', 'b'};
    
    for i = 1:length(edges_data)
        [counts, bins] = hist(edges_data{i}, 50);
        counts = counts / sum(counts); % Normalize
        plot(bins, counts, colors{i}, 'LineWidth', 1.5); hold on;
    end
    
    legend(edge_names, 'Location', 'best');
    title('Edge Strength Distributions');
    xlabel('Edge Strength');
    ylabel('Normalized Frequency');
    grid on;
    
    sgtitle('Edge Detection Methods Comparison');
    
    fprintf('Edge detection comparison complete.\n');
end
