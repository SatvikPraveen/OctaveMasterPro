% Location: mini_projects/image_processing_basics/private/load_image.m
% (split out of image_loader.m so it can be called from other files)

function img = load_image(filename, varargin)
    % Load and preprocess images with various options
    %
    % Inputs:
    %   filename - path to image file
    %   varargin - optional parameters:
    %     'resize' - [height, width] to resize image
    %     'grayscale' - convert to grayscale (true/false)
    %     'normalize' - normalize pixel values to [0,1] (true/false)
    %     'enhance' - apply basic enhancement (true/false)
    %
    % Output:
    %   img - processed image matrix
    
    % Default parameters
    resize_dims = [];
    convert_grayscale = false;
    normalize_img = true;
    enhance_img = false;
    
    % Parse optional arguments
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'resize'
                resize_dims = varargin{i+1};
            case 'grayscale'
                convert_grayscale = varargin{i+1};
            case 'normalize'
                normalize_img = varargin{i+1};
            case 'enhance'
                enhance_img = varargin{i+1};
        end
    end
    
    try
        % Load image
        if exist(filename, 'file')
            img = imread(filename);
        else
            % Create a synthetic test image if file doesn't exist
            fprintf('File not found. Creating synthetic test image...\n');
            img = create_test_image();
        end
        
        % Convert to double precision
        img = double(img);
        
        % Convert to grayscale if requested
        if convert_grayscale && size(img, 3) == 3
            img = rgb2gray_custom(img);
        end
        
        % Resize if requested
        if ~isempty(resize_dims)
            img = resize_image(img, resize_dims);
        end
        
        % Normalize pixel values
        if normalize_img
            if max(img(:)) > 1
                img = img / 255;
            end
        end
        
        % Apply basic enhancement
        if enhance_img
            img = enhance_image(img);
        end
        
    catch err
        fprintf('Error loading image: %s\n', err.message);
        fprintf('Creating synthetic test image instead...\n');
        img = create_test_image();
        
        if normalize_img
            img = img / 255;
        end
    end
end
