import sys
import numpy as np                       # ← add NumPy
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray

def view_combined_masks(yellow_path, red_path, black_path=None, thresh=0.5):
    """
    Load JPEG masks (yellow, red, and optional black), treat pixels <thresh as True,
    OR them together, and display with interactive zoom (VS Code Plot Viewer).
    """
    # build and threshold each mask
    paths = [yellow_path, red_path]
    if black_path:
        paths.append(black_path)
    masks = []
    for p in paths:
        img = io.imread(p)
        gray = rgb2gray(img) if img.ndim == 3 else img.astype(float) / np.max(img)
        masks.append(gray < thresh)
    # combine all masks
    combined = masks[0]
    for m in masks[1:]:
        combined = combined | m

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(combined, cmap='gray_r', vmin=0, vmax=1)
    ax.set_title('Combined masks (black=True)')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python view_combined_masks.py yellow.jpg red.jpg [black.jpg]")
    else:
        y, r = sys.argv[1], sys.argv[2]
        b = sys.argv[3] if len(sys.argv) == 4 else None
        view_combined_masks(y, r, b)