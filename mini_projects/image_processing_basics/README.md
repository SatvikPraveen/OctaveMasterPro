# Image Processing Basics

**Location:** `mini_projects/image_processing_basics/README.md`

## Overview

Comprehensive image processing mini-project covering fundamental algorithms including filtering, morphological operations, histogram analysis, and edge detection using Octave.

## Features

- Load and preprocess various image formats
- Apply Gaussian, median, and edge detection filters
- Perform morphological operations (erosion, dilation, opening, closing)
- Histogram equalization and statistical analysis
- Real-time image enhancement and restoration

## Files

- `image_loader.m` - Image loading and preprocessing utilities
- `basic_filters.m` - Gaussian, median, Sobel, Canny edge detection
- `morphology_ops.m` - Mathematical morphology operations
- `histogram_analysis.m` - Histogram processing and equalization
- `image_demo.m` - Main demonstration script

## Usage

```octave
# Run main demonstration
image_demo

# Load and process an image
img = load_image('sample_images/test_image.jpg');
filtered_img = apply_gaussian_filter(img, 2);

# Perform morphological operations
se = create_structuring_element('disk', 5);
opened_img = morphological_opening(img, se);

# Histogram equalization
eq_img = histogram_equalization(img);
```

## Requirements

- Octave with image processing package
- Sample images in `sample_images/` directory

## Sample Outputs

- Filtered and enhanced images
- Edge detection results
- Morphological operation effects
- Histogram analysis plots
