# Mask images were produced "by hand" using https://github.com/rharriszzz/hsv_tools/blob/main/hsv_picker.py

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology, filters
from skimage.io import imread
from skimage.morphology import binary_closing, remove_small_objects, disk, skeletonize
from skimage.graph import route_through_array
from scipy.spatial import distance
from scipy.interpolate import splprep, splev
from sklearn.linear_model import RANSACRegressor

# x_centerline, y_centerline = compute_bracelet_centerline_spline('yellow_mask.png', 'red_mask.png', 'black_mask.png')
def compute_bracelet_centerline_spline(yellow_path, red_path, black_path, smoothing=5.0, n_points=1000):
    yellow_mask = imread(yellow_path, as_gray=True)
    red_mask = imread(red_path, as_gray=True)
    black_mask = imread(black_path, as_gray=True)
    combined_mask = (yellow_mask < 128) | (red_mask < 128) | (black_mask < 128)
    combined_mask = binary_closing(combined_mask, disk(3))
    combined_mask = remove_small_objects(combined_mask, min_size=64)
    skeleton = skeletonize(combined_mask)
    coords = np.column_stack(np.nonzero(skeleton))
    if len(coords) < 2:
        raise ValueError("Skeletonization failed to produce enough points.")
    dist_matrix = distance.cdist(coords, coords, 'euclidean')
    i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    start, end = tuple(coords[i]), tuple(coords[j])
    cost_array = np.where(skeleton, 1, 1000)
    indices, _ = route_through_array(cost_array, start, end, fully_connected=True)
    ordered_coords = np.array(indices)
    x = ordered_coords[:, 1]
    y = ordered_coords[:, 0]
    tck, u = splprep([x, y], s=smoothing)
    u_fine = np.linspace(0, 1, n_points)
    x_fine, y_fine = splev(u_fine, tck)
    return x_fine, y_fine


def detect_beads_from_mask(mask, area_min, area_max,
                           smoothing_sigma=0, plot_debug=False,
                           bucket_size=20):
    """
    Detect bead centers from a binary mask, using area filtering.

    Args:
        mask (np.ndarray): Input binary mask (beads = black or 1, background = white or 0).
        area_min (float): Minimum area of a bead in pixels.
        area_max (float): Maximum area of a bead in pixels.
        smoothing_sigma (float): Optional Gaussian blur before processing.
        plot_debug (bool): If True, plot intermediate steps.
        bucket_size (int): Size of buckets for region size debugging.

    Returns:
        List of bead properties: each as a dict with x, y, area, major_axis, minor_axis, orientation.
    """

    # Step 1: Preprocessing
    # If the input is already a boolean mask (black=True), use it directly
    if mask.dtype == bool:
        binary = mask
    else:
        # convert image to boolean: black (<128) → True, white → False
        mask_bool = (mask < 128)
        if smoothing_sigma > 0:
            blurred = gaussian_filter(mask_bool.astype(float), sigma=smoothing_sigma)
            threshold = filters.threshold_otsu(blurred)
            binary = blurred > threshold
        else:
            binary = mask_bool

    # Step 2: Label regions
    labeled_mask = measure.label(binary)
    props = measure.regionprops(labeled_mask)

    # DEBUG: bucket region sizes into ranges and plot total pixels per bucket
    sizes = [p.area for p in props]
    if sizes:
        max_size = max(sizes)
        # create buckets: 0–(bucket_size-1), bucket_size–(2*bucket_size-1), ...
        bins = list(range(0, int(max_size) + bucket_size, bucket_size))
        bin_centers = [(b + bucket_size/2) for b in bins[:-1]]
        pixels_per_bin = []
        for start in bins[:-1]:
            end = start + bucket_size
            pixels = sum(a for a in sizes if start <= a < end)
            pixels_per_bin.append(pixels)

        # filter out empty buckets
        plot_centers = [c for c, p in zip(bin_centers, pixels_per_bin) if p > 0]
        plot_pixels = [p for p in pixels_per_bin if p > 0]
        plt.figure()
        plt.plot(plot_centers, plot_pixels,
                 marker='o', linestyle='None', markersize=3)
        plt.xlabel('Region size bucket center (pixels)')
        plt.ylabel('Total pixels in bucket')
        plt.title(f'Pixel count by region size (bucket={bucket_size})')
        plt.show()

    # Step 3: Area-based filtering (area_min and area_max are passed in)
    bead_list = []
    for p in props:
        if area_min <= p.area <= area_max:
            bead_info = {
                'x': float(p.centroid[1]),   # ensure native Python float
                'y': float(p.centroid[0]),
                'area': float(p.area),
                'major_axis': float(p.major_axis_length),
                'minor_axis': float(p.minor_axis_length),
                'orientation': float(p.orientation)  # Radians, CCW from x-axis
            }
            bead_list.append(bead_info)

    if plot_debug:
        fig, ax = plt.subplots(figsize=(8, 8))
        # display True→black, False→white
        ax.imshow(binary, cmap='gray_r', vmin=0, vmax=1)
        for bead in bead_list:
            # half the previous size
            ax.plot(bead['x'], bead['y'], 'ro', markersize=2)
        ax.set_title(f"Detected {len(bead_list)} beads")
        plt.show()

    return bead_list


def main():
    # estimated radius/area and filter bounds
    estimated_radius = 12.5  # pixels, as found before
    estimated_area = np.pi * estimated_radius**2
    area_tolerance = 0.6
    area_min = estimated_area * (1 - area_tolerance)
    area_max = estimated_area * (1 + area_tolerance)
    print(f"Area filter: min={area_min:.2f} px^2, max={area_max:.2f} px^2")

    runs = {
        "yellow": ["beads-photo-2-yellow.jpg"],
        "yellow+red": ["beads-photo-2-yellow.jpg", "beads-photo-2-red.jpg"],
        "yellow+red+black": [
            "beads-photo-2-yellow.jpg",
            "beads-photo-2-red.jpg",
            "beads-photo-2-black.jpg",
        ],
    }

    for name, paths in runs.items():
        # load & threshold each mask (<128 → True)
        imgs = [imread(p) for p in paths]
        combined = (imgs[0] < 128)
        for img in imgs[1:]:
            combined |= (img < 128)

        print(f"\n=== Running on '{name}' mask(s): {paths} ===")
        beads = detect_beads_from_mask(
            combined,
            area_min,
            area_max,
            plot_debug=True
        )
        print(f"Found {len(beads)} beads for '{name}'.")
        for bead in beads[:5]:
            print(bead)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        import sys
        sys.exit("\nInterrupted by user")
