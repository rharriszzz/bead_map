# Mask images were produced "by hand" using https://github.com/rharriszzz/hsv_tools/blob/main/hsv_picker.py

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology, filters
from scipy.interpolate import splprep, splev
from sklearn.linear_model import RANSACRegressor
from scipy.ndimage import gaussian_filter

# --- File renamed to: Bracelet Pattern Extractor ---


def detect_beads_from_mask(mask, estimated_bead_area, area_tolerance=0.3, smoothing_sigma=0.5, plot_debug=False):
    mask = (mask < 128).astype(np.uint8)

    if smoothing_sigma > 0:
        mask = gaussian_filter(mask.astype(float), sigma=smoothing_sigma)
        threshold = filters.threshold_otsu(mask)
        binary = mask > threshold
    else:
        binary = mask > 0

    labeled_mask = measure.label(binary)
    props = measure.regionprops(labeled_mask)

    area_min = estimated_bead_area * (1 - area_tolerance)
    area_max = estimated_bead_area * (1 + area_tolerance)

    bead_list = []
    for p in props:
        if area_min <= p.area <= area_max:
            bead_info = {
                'x': p.centroid[1],
                'y': p.centroid[0],
                'area': p.area,
                'major_axis': p.major_axis_length,
                'minor_axis': p.minor_axis_length,
                'orientation': p.orientation
            }
            bead_list.append(bead_info)

    if plot_debug:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(binary, cmap='gray')
        for bead in bead_list:
            ax.plot(bead['x'], bead['y'], 'ro')
        ax.set_title(f"Detected {len(bead_list)} beads")
        plt.show()

    return bead_list



def detect_beads_from_mask(mask, estimated_bead_area, area_tolerance=0.3, smoothing_sigma=0.5, plot_debug=False):
    """
    Detect bead centers from a binary mask, using area filtering.

    Args:
        mask (np.ndarray): Input binary mask (beads = black or 1, background = white or 0).
        estimated_bead_area (float): Estimated area of a bead in pixels.
        area_tolerance (float): Acceptable relative variation (e.g., 0.3 means +/-30%).
        smoothing_sigma (float): Optional Gaussian blur before processing.
        plot_debug (bool): If True, plot intermediate steps.

    Returns:
        List of bead properties: each as a dict with x, y, area, major_axis, minor_axis, orientation.
    """
    from scipy.ndimage import gaussian_filter

    # Step 1: Preprocessing
    mask = (mask < 128).astype(np.uint8)  # Ensure beads are 1, background is 0

    if smoothing_sigma > 0:
        mask = gaussian_filter(mask.astype(float), sigma=smoothing_sigma)
        threshold = filters.threshold_otsu(mask)
        binary = mask > threshold
    else:
        binary = mask > 0

    # Step 2: Label regions
    labeled_mask = measure.label(binary)
    props = measure.regionprops(labeled_mask)

    # Step 3: Area-based filtering
    area_min = estimated_bead_area * (1 - area_tolerance)
    area_max = estimated_bead_area * (1 + area_tolerance)

    bead_list = []
    for p in props:
        if area_min <= p.area <= area_max:
            bead_info = {
                'x': p.centroid[1],  # (row, col) -> (y, x)
                'y': p.centroid[0],
                'area': p.area,
                'major_axis': p.major_axis_length,
                'minor_axis': p.minor_axis_length,
                'orientation': p.orientation  # Radians, CCW from x-axis
            }
            bead_list.append(bead_info)

    if plot_debug:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(binary, cmap='gray')
        for bead in bead_list:
            ax.plot(bead['x'], bead['y'], 'ro')
        ax.set_title(f"Detected {len(bead_list)} beads")
        plt.show()

    return bead_list


# Example usage
if __name__ == "__main__":
    from skimage.io import imread

    mask_path = "beads-photo-2-yellow.jpg"  # your mask image
    mask = imread(mask_path)

    estimated_radius = 12.5  # pixels, as found before
    estimated_area = np.pi * estimated_radius**2

    beads = detect_beads_from_mask(mask, estimated_bead_area=estimated_area, plot_debug=True)

    print(f"Found {len(beads)} beads.")
    for bead in beads[:5]:
        print(bead)
