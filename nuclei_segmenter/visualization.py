import cv2
import numpy as np

def draw_nuclei_outlines(image, labeled_mask, classifications, output_path="nuclei_outlined.png"):
    # Ensure image is in uint8 format and scaled properly
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).clip(0, 255).astype(np.uint8)

    # Convert RGB (skimage/matplotlib) to BGR (OpenCV)
    if image.ndim == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif image.ndim == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError("Unsupported image format")

    # Define BGR colors for OpenCV
    group_colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 165, 255),  # Orange
        (0, 0, 255)     # Red
    ]

    for label, group in classifications.items():
        mask = (labeled_mask == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = group_colors[group]
        cv2.drawContours(image_bgr, contours, -1, color, thickness=1)

    cv2.imwrite(output_path, image_bgr)
