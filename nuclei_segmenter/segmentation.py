from skimage import io, color, measure, exposure
from skimage.filters import threshold_otsu, unsharp_mask
from skimage.measure import regionprops_table
from skimage.transform import rescale
from skimage.morphology import remove_small_objects, h_maxima
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import numpy as np

# Downsampling utilities
def downsample_image(gray_image, scale=0.1):
    return rescale(gray_image, scale, anti_aliasing=True)

def downsample_rgb(image, scale=0.1):
    return rescale(image, scale, anti_aliasing=True, channel_axis=-1)

# Image loader
def load_image(image_path):
    image = io.imread(image_path)
    gray_image = color.rgb2gray(image)
    return image, gray_image

# Preprocessing (histogram equalization + sharpening)
def preprocess_image(gray_image):
    equalized = exposure.equalize_hist(gray_image)
    sharpened = unsharp_mask(equalized, radius=1, amount=1.5)
    return sharpened

# Nuclei segmentation using watershed with enhanced seed detection
def segment_nuclei(gray_image, block_size=101, offset=0.03, min_size=100):
    # Thresholding
    thresh_val = threshold_otsu(gray_image)
    binary_mask = gray_image < thresh_val

    # Morphological cleanup (pre-watershed)
    binary_mask = remove_small_objects(binary_mask, min_size=min_size)

    # Distance transform & watershed steps...
    distance = ndi.distance_transform_edt(binary_mask)
    local_maxi = h_maxima(distance, h=0.35)  # More conservative split
    markers = measure.label(local_maxi)
    labels = watershed(-distance, markers, mask=binary_mask)

    # Post-watershed cleanup: remove tiny/flat regions
    props = regionprops_table(labels, properties=["label", "area", "eccentricity"])
    valid_labels = {
        label for label, area, ecc in zip(props["label"], props["area"], props["eccentricity"])
        if area >= 150 and ecc < 0.95
    }
    mask = np.isin(labels, list(valid_labels))
    labels = measure.label(mask)

    return labels

# Label connected components
def label_nuclei(binary_mask):
    return measure.label(binary_mask, connectivity=2)

def compute_nucleus_intensities(labeled_mask, gray_image):
    props = regionprops_table(
        labeled_mask,
        intensity_image=gray_image,
        properties=["label", "mean_intensity"]
    )
    return list(zip(props["label"], props["mean_intensity"]))

# Classify nuclei by intensity into 4 quartile-based groups
def classify_intensities(intensity_list):
    values = np.array([i[1] for i in intensity_list])
    labels = np.array([i[0] for i in intensity_list])
    quartiles = np.quantile(values, [0.25, 0.5, 0.75])

    classifications = {}
    for label, intensity in zip(labels, values):
        if intensity <= quartiles[0]:
            group = 0
        elif intensity <= quartiles[1]:
            group = 1
        elif intensity <= quartiles[2]:
            group = 2
        else:
            group = 3
        classifications[label] = group
    return classifications
