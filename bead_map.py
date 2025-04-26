#!/usr/bin/env python3
"""
Map pixels in color masks to bead indices by inferring bead centers,
ordering beads along the minor-axis cycle, and outputting a color sequence.

Usage:
    python map_beads.py --masks yellow_mask.png red_mask.png black_mask.png \
                       --beads-per-row 6.5 \
                       --output-prefix bead_map

Requires:
    numpy, scipy, scikit-image
"""
import numpy as np
import argparse
from scipy.spatial import KDTree
from skimage import io, img_as_bool
from skimage.measure import label, regionprops
from skimage.feature import blob_log


def load_masks(mask_paths):
    masks = []
    for path in mask_paths:
        img = io.imread(path)
        masks.append(img_as_bool(img))
    return masks


def estimate_bead_radius(mask):
    labels = label(mask)
    areas = [r.area for r in regionprops(labels) if r.area > 0]
    if not areas:
        raise RuntimeError("No bead-like regions in mask to estimate radius.")
    median_area = np.median(areas)
    return np.sqrt(median_area / np.pi)


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

    # 1) Load and combine masks
    yellow_mask, red_mask, black_mask = load_masks(args.masks)
    combined = yellow_mask | red_mask | black_mask

    # 2) Estimate bead radius & detect centers
    radius = estimate_bead_radius(yellow_mask)
    print(f"Estimated bead radius: {radius:.1f} px")
    centers = detect_bead_centers(combined, radius)
    n_detected = len(centers)
    print(f"Detected {n_detected} bead centers.")

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
    main()
