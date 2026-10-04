% Location: mini_projects/image_processing_basics/private/save_image.m
% (split out of image_loader.m so it can be called from other files)

function save_image(img, filename, varargin)
    % Save image with optional format conversion
    %
    % Inputs:
    %   img - image matrix
    %   filename - output filename
    %   varargin - optional parameters:
    %     'quality' - JPEG quality (1-100)
    %     'format' - force specific format
    
    % Default parameters
    quality = 95;
    format = '';
    
    % Parse optional arguments
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'quality'
                quality = varargin{i+1};
            case 'format'
                format = varargin{i+1};
        end
    end
    
    % Ensure proper data type and range
    if max(img(:)) <= 1
        img = img * 255;
    end
    img = uint8(img);
    
    try
        if ~isempty(format)
            imwrite(img, filename, format, 'Quality', quality);
        else
            imwrite(img, filename, 'Quality', quality);
        end
        fprintf('Image saved: %s\n', filename);
    catch err
        fprintf('Error saving image: %s\n', err.message);
    end
end
