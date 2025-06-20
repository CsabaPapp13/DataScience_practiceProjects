import argparse
import multiprocessing
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from collections import Counter
import sys


from segmentation import (
    load_image,
    preprocess_image,
    segment_nuclei,
    label_nuclei,
    compute_nucleus_intensities,
    classify_intensities,
    downsample_image,
    downsample_rgb
)

from visualization import draw_nuclei_outlines

def main():
    parser = argparse.ArgumentParser(description="Nuclei segmentation based on staining intensity.")
    parser.add_argument("image_path", help="Path to the input histological image.")
    parser.add_argument("output_path", default="nuclei_outlined.png",
                        help="Path to save the output image with outlined nuclei.")
    parser.add_argument("--n-cores", type=int, default=multiprocessing.cpu_count(),
                        help="Number of cores to use (default: all available cores)")
    parser.add_argument("--downsample", action="store_true",
                    help="Downsample image for faster testing/debugging")
    args = parser.parse_args()

    print("Loading image...", flush=True)
    image, gray_image = load_image(args.image_path)
    print("Loading done", flush=True)

    print("Applying histogram equalization...", flush=True)
    gray_image = preprocess_image(gray_image)

    gray_image = gray_image[:2048, :2048]
    image = image[:2048, :2048]

    if args.downsample:
        print("Downsampling image for debugging...", flush=True)
        gray_image = downsample_image(gray_image, scale=0.1)
        image = downsample_rgb(image, scale=0.1)
        
    print("Starting segmentation.", flush=True)    
    binary_mask = segment_nuclei(gray_image)
    print("Segmentation using watershed applied.", flush=True)

    labeled_mask = label_nuclei(binary_mask)
    print("Labeling done", flush=True)
    print("Number of labeled nuclei:", labeled_mask.max(), flush=True)

    print("Starting intensity calculation", flush=True)
    intensity_list = compute_nucleus_intensities(labeled_mask, gray_image)
    print("Intensity calculation done", flush=True)

    classifications = classify_intensities(intensity_list)
    counts = Counter(classifications.values())
    print("Nuclei per intensity group:", counts)

    print("Drawing and saving contours...", flush=True)
    draw_nuclei_outlines(image, labeled_mask, classifications, output_path=args.output_path)
    print("Done.", flush=True)

if __name__ == "__main__":
    main()