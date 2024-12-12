# GrabCut Algorithm Implementation

# collaborators: Hend Bishara, Nour Atieh 

## Overview

This repository contains an implementation of the GrabCut image segmentation algorithm, a powerful tool for foreground extraction. GrabCut uses iterative optimization techniques to refine the segmentation mask, leveraging Gaussian Mixture Models (GMMs) and graph cuts to classify pixels as either background or foreground.

## Features
- **Gaussian Mixture Models (GMMs):** For modeling foreground and background pixel distributions.
- **Graph Cuts:** To find the optimal segmentation by minimizing energy.
- **Pixel Classification:** Differentiates between hard and soft pixels for robust segmentation.
- **Customizable Parameters:** Adjustable bounding box and number of iterations.
- **Metrics Calculation:** Computes accuracy and Jaccard similarity for evaluation.

## Requirements
- Python 3.8+
- Required libraries:
  - `numpy`
  - `opencv-python`
  - `scikit-learn`
  - `python-igraph`


## Usage
### Command-Line Arguments
You can run the script with various arguments:
- `--input_name`: Name of the image (default: `teddy`).
- `--eval`: Whether to calculate evaluation metrics (default: `1`).
- `--input_img_path`: Path to the custom image.
- `--use_file_rect`: Use the bounding box from course files (default: `1`).
- `--rect`: Custom bounding box in the format `x,y,w,h` (default: `1,1,100,100`).

### Running the Algorithm
```bash
python grabcut.py --input_img_path path/to/image.jpg --rect 50,50,150,150
```

### Output
The script outputs:
- Final segmentation mask.
- Optional evaluation metrics if a ground truth mask is provided.

## References
- Rother, C., Kolmogorov, V., & Blake, A. (2004). *GrabCut: Interactive foreground extraction using iterated graph cuts.* ACM Transactions on Graphics (TOG), 23(3), 309-314.

