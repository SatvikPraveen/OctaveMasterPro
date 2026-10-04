% Location: mini_projects/image_processing_basics/private/compare_structuring_elements.m
% (split out of morphology_ops.m so it can be called from other files)

function compare_structuring_elements(img)
    % Compare effects of different structuring elements
    
    fprintf('\n--- Structuring Element Comparison ---\n');
    
    % Create different structuring elements
    se_disk3 = create_structuring_element('disk', 3);
    se_disk5 = create_structuring_element('disk', 5);
    se_square3 = create_structuring_element('square', 7);
    se_cross3 = create_structuring_element('cross', 3);
    se_line = create_structuring_element('line', 7, 45);
    
    structuring_elements = {se_disk3, se_disk5, se_square3, se_cross3, se_line};
    se_names = {'Disk r=3', 'Disk r=5', 'Square 7x7', 'Cross r=3', 'Line 45°'};
    
    figure('Position', [100, 100, 1400, 1000]);
    
    % Show original
    subplot(4, 6, 1);
    imshow(img, []);
    title('Original');
    
    % Show structuring elements
    for i = 1:5
        subplot(4, 6, i+1);
        imshow(structuring_elements{i}, []);
        title(se_names{i});
    end
    
    % Show erosion results
    for i = 1:5
        subplot(4, 6, 6+i+1);
        eroded = morphological_erosion(img, structuring_elements{i});
        imshow(eroded, []);
        title(['Erosion: ' se_names{i}]);
    end
    
    % Show dilation results
    for i = 1:5
        subplot(4, 6, 12+i+1);
        dilated = morphological_dilation(img, structuring_elements{i});
        imshow(dilated, []);
        title(['Dilation: ' se_names{i}]);
    end
    
    % Show opening results
    for i = 1:5
        subplot(4, 6, 18+i+1);
        opened = morphological_opening(img, structuring_elements{i});
        imshow(opened, []);
        title(['Opening: ' se_names{i}]);
    end
    
    sgtitle('Structuring Element Effects Comparison');
    
    fprintf('Structuring element comparison complete.\n');
end
