#!/usr/bin/env python3
import sys
import numpy as np
import argparse
from scipy.spatial import KDTree
from skimage import io, img_as_bool
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.feature import blob_log
from matplotlib import pyplot as plt
from scipy.signal import find_peaks


def load_masks(mask_paths, thresh=0.5):
    """
    Read each mask image (JPEG), convert to grayscale, then threshold:
      pixels <= thresh → False (background)
      pixels >  thresh → True  (foreground)
    """
    masks = []
    for path in mask_paths:
        img = io.imread(path)
        # convert to float [0–1] grayscale
        if img.ndim == 3:
            gray = rgb2gray(img)
        else:
            gray = img.astype(float) / np.max(img)
        # threshold at half‑grey
        mask = gray > thresh
        masks.append(mask)
    return masks


def estimate_bead_radius(mask):
    # explicitly tell label that white (True=1) is background
    labels = label(mask, background=1)
    regions = [r for r in regionprops(labels) if r.area > 0]
    if not regions:
        raise RuntimeError("No bead-like regions in mask to estimate radius.")
    areas = [r.area for r in regions]
    # remove duplicates and sort to inspect unique sizes
    unique_areas = [float(a) for a in sorted(set(areas))]
    print("DEBUG: Smallest 100 unique region areas:", unique_areas[:100])
    # ensure native Python floats in the list for clean printing
    sorted_areas = [float(a) for a in sorted(areas)]
    
    # DEBUG PLOT: log10 of region areas vs region index
    idx = np.arange(1, len(areas)+1)
    log_area = np.log10(sorted_areas)
    plt.figure()
    plt.plot(idx, log_area, marker='o', linestyle='-')
    plt.xlabel('Region index')
    plt.ylabel('log10(region area)')
    plt.title('Bead region sizes')
    plt.show()

    # DEBUG: Display the largest connected region alongside the plot
    # Identify the region with the maximum area
    max_region = max(regions, key=lambda r: r.area)
    region_label = max_region.label
    # Create a boolean mask of only that region
    largest_region_mask = (labels == region_label)
    # compute ratio of largest region to total image pixels
    total_pixels = mask.size
    ratio = max_region.area / total_pixels
    print(f"DEBUG: Largest region area = {max_region.area} px, "
          f"Total pixels = {total_pixels} px, "
          f"Ratio = {ratio:.4f}")
    # Show the largest region with mask=True rendered as black
    plt.figure()
    # use inverted gray so True→0 (black) and False→1 (white)
    plt.imshow(largest_region_mask, cmap='gray_r', vmin=0, vmax=1)
    plt.title(f'Largest region (label={region_label}, area={max_region.area} px)')
    plt.axis('off')
    plt.show()
    
    # NEW DEBUG PLOT: total pixel count vs region size
    from collections import Counter
    area_counts = Counter(areas)
    unique_areas = sorted(area_counts)
    total_pixels_per_area = [area_counts[a] * a for a in unique_areas]
    log_area = np.log10(unique_areas)
    plt.figure()
    # plot log10(region area) on x-axis, total pixels on y-axis
    plt.plot(log_area, total_pixels_per_area, marker='o', linestyle='-')
    plt.xlabel('log10(region area)')
    plt.ylabel('Total pixels across regions of that area')
    plt.title('Pixel count by region size')
    plt.show()

    # Compute radius from peaks in pixel‑count curve
    logA = np.log10(unique_areas)
    counts = total_pixels_per_area

    # find all peaks in the pixel‑count curve
    peaks, _ = find_peaks(counts)
    # select only those in the bead‑expected log‑area window (e.g. 2.2–2.7)
    bead_peak_idxs = [i for i in peaks if 2.2 <= logA[i] <= 2.7]
    # ensure native Python floats for clean printing
    bead_areas = [float(unique_areas[i]) for i in bead_peak_idxs]
    # convert to radii and report
    bead_radii = [np.sqrt(a/np.pi) for a in bead_areas]
    print("DEBUG: Detected bead‑size peaks at areas:", bead_areas)
    print("DEBUG: => estimated radii (px):", [f"{r:.1f}" for r in bead_radii])

    # optionally pick the most prominent peak
    if bead_areas:
        # choose the peak with highest total_pixels_per_area
        best = bead_peak_idxs[np.argmax([counts[i] for i in bead_peak_idxs])]
        best_area, best_radius = unique_areas[best], np.sqrt(unique_areas[best]/np.pi)
        print(f"DEBUG: Auto‑picked bead radius: {best_radius:.1f} px (area={best_area})")
        radius = best_radius
    else:
        print("DEBUG: No bead‑size peaks found; falling back to median.")
        radius = np.median(areas)**0.5/np.sqrt(np.pi)

    return radius


def detect_bead_centers(combined_mask, radius):
    # LoG blob detection around estimated radius
    blobs = blob_log(
        combined_mask.astype(float),
        min_sigma=radius * 0.8,
        max_sigma=radius * 1.2,
        num_sigma=10,
        threshold=0.05,
    )
    return blobs[:, :2]  # (row, col)


def build_neighbor_tree(centers):
    return KDTree(centers)


def trace_cycle(tree, n_beads, start_idx):
    # Follow nearest-neighbor cycle of length n_beads
    order = [start_idx]
    prev = None
    curr = start_idx
    while len(order) < n_beads:
        dists, idxs = tree.query(tree.data[curr], k=4)
        # skip self and previous
        for nb in idxs:
            if nb != curr and nb != prev:
                nxt = nb
                break
        # stop if looped
        if nxt == start_idx:
            break
        order.append(nxt)
        prev, curr = curr, nxt
    # pad if missing
    if len(order) < n_beads:
        order += [-1] * (n_beads - len(order))
    return order


def assign_pixels(mask, tree, bead_ids):
    coords = np.argwhere(mask)
    bead_map = np.zeros(mask.shape, int)
    if coords.size > 0:
        _, idxs = tree.query(coords)
        bead_map[coords[:,0], coords[:,1]] = bead_ids[idxs]
    return bead_map


def main():
    p = argparse.ArgumentParser(description="Map beads and output color sequence.")
    p.add_argument('--masks', nargs=3, required=True,
                   help='>3 mask images: yellow, red, black')
    p.add_argument('--beads-per-row', type=float, required=True,
                   help='Exact beads per row (e.g. 6.5)')
    p.add_argument('--output-prefix', default='bead_map',
                   help='Prefix for outputs')
    args = p.parse_args()

    # 1) Load masks (white areas → True), then invert the black mask
    yellow_mask, red_mask, black_mask = load_masks(args.masks)
    # we want True == black pixels, not white
    black_mask = np.logical_not(black_mask)

    # DEBUG: count and compare truly‑black pixels vs largest connected black region
    total_black = int(np.sum(black_mask))
    labels_black = label(black_mask, background=0)
    regions_black = [r for r in regionprops(labels_black) if r.area > 0]
    if regions_black:
        max_b = max(regions_black, key=lambda r: r.area)
        print(f"DEBUG: Total black pixels = {total_black} px")
        print(f"DEBUG: Largest black region area = {max_b.area} px")
        print(f"DEBUG: Ratio largest/total black = {max_b.area/total_black:.4f}")
    else:
        print("DEBUG: No black regions found in mask.")

    combined = yellow_mask | red_mask | black_mask

    # 2) Estimate bead radius & detect centers
    radius = estimate_bead_radius(yellow_mask)
    print(f"Estimated bead radius: {radius:.1f} px")
    centers = detect_bead_centers(combined, radius)
    n_detected = len(centers)
    print(f"Detected {n_detected} bead centers.")

    if n_detected == 0:
        sys.exit("Error: No bead centers detected. Check your masks and blob_log threshold.")

    # 3) Infer total beads via rows
    rows = int(round(n_detected / args.beads_per_row))
    total_beads = int(round(args.beads_per_row * rows))
    print(f"Inferring {rows} rows -> total beads: {total_beads}")

    # 4) Order beads along minor-axis cycle
    tree = build_neighbor_tree(centers)
    # start at bead with smallest row coordinate
    start = int(np.argmin(centers[:,0]))
    bead_order = trace_cycle(tree, total_beads, start)

    # 5) Assign pixel-level bead indices per color
    bead_ids = np.arange(n_detected)
    ymap = assign_pixels(yellow_mask, tree, bead_ids)
    rmap = assign_pixels(red_mask,    tree, bead_ids)
    bmap = assign_pixels(black_mask,  tree, bead_ids)

    # 6) Build color sequence
    seq = []
    for bid in bead_order:
        if bid < 0 or bid >= n_detected:
            seq.append('unknown')
        else:
            yc = np.sum(ymap == bid)
            rc = np.sum(rmap == bid)
            bc = np.sum(bmap == bid)
            if max(yc,rc,bc) == 0:
                seq.append('unknown')
            elif yc >= rc and yc >= bc:
                seq.append('yellow')
            elif rc >= yc and rc >= bc:
                seq.append('red')
            else:
                seq.append('black')

    # 7) Save outputs
    np.save(f"{args.output_prefix}_centers.npy", centers)
    np.save(f"{args.output_prefix}_order.npy", np.array(bead_order))
    with open(f"{args.output_prefix}_sequence.txt", 'w') as f:
        for color in seq:
            f.write(color + '\n')
    print(f"Saved centers, bead_order, and color sequence with prefix '{args.output_prefix}'")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user")
