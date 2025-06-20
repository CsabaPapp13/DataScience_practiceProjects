# Nuclei Segmenter package by Csaba Papp (06/19/2025)

This Python module performs nuclei segmentation on high-resolution histological images. It detects and classifies nuclei based on staining intensity and outputs an image with color-coded outlines.

Features:
- Grayscale conversion and histogram equalization
- Nuclei segmentation using Otsu thresholding and watershed refinement
- Intensity-based classification (quartile groups)
- Contour outlining with OpenCV
- Optional downsampling for fast testing
- Parallel processing support (multi-core)

Dependencies:
- `scikit-image`
- `opencv-python`
- `numpy`
- `scipy`
- `Pillow`

Basic Usage:
python main.py path/to/input_image.jpg path/to/output_image.jpg

Optional arguments
--n-cores: Number of CPU cores to use for intensity calculation (default: all available)

--downsample: Add this flag to downsample the image by 10% for faster testing

Notes: The package is set up to only segment nuclei on a slice of the input image. This is due to the limited resources I had available when setting up the algorithm. Also, the overall performance of the package is mixed. In some cases it over-segments nuclei, while in other cases it fails to completely differentiate between groups of nuclei, resulting in "blobs". In summary, the performance could certainly be fine-tuned to allow for more specific detection and differentiation between nuclei. However, this is what I was able to accomplish in a limited time. 



