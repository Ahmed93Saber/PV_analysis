import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
import math


# --- UTILITIES ---
def natur(key):
    """Natural sort key function for file ordering"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', key)]


def get_module_coordinates(ref_image_path):
    """Auto-detects the module area using the brightest image."""
    img = cv2.imread(ref_image_path, -1)
    if img is None: raise FileNotFoundError(f"Could not load {ref_image_path}")

    # Handle bit-depth
    if img.dtype == np.uint16:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    elif len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours: raise ValueError("No contours found. Check image thresholding.")

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return (x, y, w, h)


def extract_grid_data(file_path, coords, n_slices=10):
    """
    Extracts a (n_slices x 7) matrix of mean intensities.
    Returns None if file load fails.
    """
    rx, ry, rw, rh = coords
    img = cv2.imread(file_path, -1)
    if img is None: return None

    # Crop to module
    roi = img[ry:ry + rh, rx:rx + rw]

    # Grid Setup
    n_cells = 7  # Fixed for this module type
    cell_width = rw / n_cells
    seg_height = rh / n_slices

    data_matrix = np.zeros((n_slices, n_cells))

    for c in range(n_cells):
        for r in range(n_slices):
            x1, x2 = int(c * cell_width), int((c + 1) * cell_width)
            y1, y2 = int(r * seg_height), int((r + 1) * seg_height)

            # Extract mean of this grid segment
            segment = roi[y1:y2, x1:x2]
            if segment.size > 0:
                data_matrix[r, c] = np.mean(segment)

    return data_matrix


# --- MAIN ANALYSIS ---
def analyze_hysteresis_full(forward_files, reverse_files, currents, n_slices=10):
    # 1. Calibrate Geometry
    print(f"Calibrating geometry on: {os.path.basename(forward_files[-1])}")
    coords = get_module_coordinates(forward_files[-1])

    # 2. Extract Data Matrices
    print("Processing Forward scan...")
    fwd_data = [extract_grid_data(f, coords, n_slices) for f in forward_files]

    print("Processing Reverse scan...")
    rev_data = [extract_grid_data(f, coords, n_slices) for f in reverse_files]

    # IMPORTANT: Reverse files (45->5) need to be flipped to align with Forward (5->45)
    # So index 0 corresponds to 5mA, index 1 to 10mA, etc.
    rev_data = rev_data[::-1]

    # --- PLOT 1: Cell-by-Cell I-V Curves ---
    # We average the vertical slices (axis 0) to get one value per Cell per Current

    plt.figure(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, 7))  # distinct colors for 7 cells

    for cell_idx in range(7):
        # Gather intensity for this cell across all currents
        # We perform an internal mean over the n_slices (vertical axis)
        y_fwd = [np.mean(m[:, cell_idx]) for m in fwd_data]
        y_rev = [np.mean(m[:, cell_idx]) for m in rev_data]

        # Plot
        plt.plot(currents, y_fwd, 'o-', color=colors[cell_idx], label=f'Cell {cell_idx + 1} (Fwd)')
        # Optional: Plot reverse as dashed lines to show hysteresis loops
        plt.plot(currents, y_rev, '--', color=colors[cell_idx], alpha=0.5)

    plt.title("I-V Curves by Cell (Vertical Average)")
    plt.xlabel("Current (mA)")
    plt.ylabel("Mean Intensity (counts)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- PLOT 2: Ratio Heatmaps for ALL Currents ---
    # We will display these in a grid (e.g., 3x3 for 9 steps)

    num_plots = len(currents)
    cols = 3
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten()  # easy iteration

    for i, curr in enumerate(currents):
        ax = axes[i]

        # Calculate Ratio: (Reverse - Forward) / Forward
        # Avoid division by zero by adding a tiny epsilon if needed, or masking
        f_img = fwd_data[i]
        r_img = rev_data[i]

        # Mask out very dark pixels to avoid noise explosion (optional but recommended)
        valid_mask = f_img > 5  # Arbitrary noise floor

        diff_map = np.zeros_like(f_img)
        np.divide(r_img - f_img, f_img, out=diff_map, where=valid_mask)
        diff_map = diff_map * 100  # Convert to percentage

        # Plot heatmap
        im = ax.imshow(diff_map, cmap='RdBu_r', vmin=-20, vmax=20, aspect='auto')

        ax.set_title(f"Hysteresis at {curr} mA")
        ax.set_xticks(np.arange(7))
        ax.set_xticklabels([f"C{k + 1}" for k in range(7)])
        ax.set_yticks([])  # Hide slice numbers for clarity
        if i % cols == 0:
            ax.set_ylabel("Top -> Bottom")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    # Add a shared colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    cbar.set_label('Hysteresis Magnitude (%) \n(Red = Brighter on Return)', rotation=270, labelpad=20)
    plt.show()


# --- INPUT SETUP ---
# Update this path to your folder
root_dir = r"C:\Users\ahmed\OneDrive\Desktop\tiff_imgs\EL_Increment"

# Find and sort files
files = sorted(glob.glob(os.path.join(root_dir, "*.tif")), key=natur)

# Assuming 18 files total: 9 forward (0-8) + 9 reverse (9-17)
# Forward: 5, 10 ... 45
fwd_files = files[0:9]
# Reverse: 45, 40 ... 5
rev_files = files[8::]  # Taking the rest

current_steps = [5, 10, 15, 20, 25, 30, 35, 40, 45]

# Execute
if len(fwd_files) == len(rev_files) == len(current_steps):
    analyze_hysteresis_full(fwd_files, rev_files, current_steps, n_slices=10)
else:
    print("Error: File counts do not match current steps.")
    print(f"Fwd: {len(fwd_files)}, Rev: {len(rev_files)}, Steps: {len(current_steps)}")