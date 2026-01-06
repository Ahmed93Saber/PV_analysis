import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# --- Reuse the calibration function from before ---
def get_module_coordinates(ref_image_path):
    img = cv2.imread(ref_image_path, -1)
    if img.dtype == np.uint16:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    elif len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return (x, y, w, h)


def extract_grid_data(file_path, coords, n_slices=10):
    """Extracts the 7 x n_slices matrix for a single file"""
    rx, ry, rw, rh = coords
    img = cv2.imread(file_path, -1)
    if img is None: return None

    # Crop
    roi = img[ry:ry + rh, rx:rx + rw]

    # Grid Setup
    n_cells = 7
    cell_width = rw / n_cells
    seg_height = rh / n_slices

    data_matrix = np.zeros((n_slices, n_cells))

    for c in range(n_cells):
        for r in range(n_slices):
            x1, x2 = int(c * cell_width), int((c + 1) * cell_width)
            y1, y2 = int(r * seg_height), int((r + 1) * seg_height)
            data_matrix[r, c] = np.mean(roi[y1:y2, x1:x2])

    return data_matrix


# --- MAIN ANALYSIS FUNCTION ---
def compare_scans(forward_files, reverse_files, currents, n_slices=10):
    # 1. Calibrate Grid using the brightest Forward image (usually last in list)
    print("Calibrating grid...")
    coords = get_module_coordinates(forward_files[-1])

    # 2. Extract Data for all files
    print("Extracting Forward Data...")
    fwd_data = [extract_grid_data(f, coords, n_slices) for f in forward_files]

    print("Extracting Reverse Data...")
    # Reverse files are usually 45->5, so we reverse the list to match 5->45 for plotting
    rev_data = [extract_grid_data(f, coords, n_slices) for f in reverse_files]
    rev_data = rev_data[::-1]  # Flip so index 0 is 5mA, same as forward

    # 3. Define ROI Coordinates (User can change these)
    # Example: Cell 4 (Center/Good) vs Cell 7 (Right/Weak)
    # Slices: Top (High Resistance) vs Bottom (Injector)
    rois = [
        {"name": "Good Cell (C4) - Bottom", "c": 3, "r": n_slices - 1, "color": "blue"},
        {"name": "Good Cell (C4) - Top", "c": 3, "r": 0, "color": "cyan"},
        {"name": "Weak Cell (C7) - Bottom", "c": 6, "r": n_slices - 1, "color": "red"},
    ]

    # --- PLOT 1: Local I-V Curves (Hysteresis Loops) ---
    plt.figure(figsize=(10, 6))

    for roi in rois:
        c, r = roi['c'], roi['r']

        # Extract trajectory
        y_fwd = [m[r, c] for m in fwd_data]
        y_rev = [m[r, c] for m in rev_data]

        # Plot Forward (Solid) and Reverse (Dashed)
        plt.plot(currents, y_fwd, 'o-', label=f"{roi['name']} (Fwd)", color=roi['color'])
        plt.plot(currents, y_rev, 'x--', label=f"{roi['name']} (Rev)", color=roi['color'], alpha=0.6)

    plt.title("Local Hysteresis Analysis: Good vs. Weak Regions")
    plt.xlabel("Current (mA)")
    plt.ylabel("Mean EL Intensity (a.u.)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- PLOT 2: Spatial Hysteresis Heatmap (at specific current) ---
    # We choose a mid-range current (e.g., index 3 -> 20mA) where hysteresis is usually max
    idx = 3
    if idx < len(currents):
        curr_val = currents[idx]

        # Calculate % Difference Map
        # Formula: (Reverse - Forward) / Forward
        diff_map = (rev_data[idx] - fwd_data[idx]) / fwd_data[idx] * 100

        plt.figure(figsize=(8, 6))
        plt.imshow(diff_map, cmap='RdBu_r', aspect='auto', vmin=-20, vmax=20)
        plt.colorbar(label='Hysteresis Magnitude (%)')
        plt.title(f"Hysteresis Map at {curr_val}mA\n(Red = Brighter on Return, Blue = Dimmer)")
        plt.xlabel("Cell Index (1-7)")
        plt.ylabel("Vertical Slice (Top -> Bot)")
        plt.xticks(np.arange(7), [f"C{i + 1}" for i in range(7)])
        plt.show()


# --- INPUT YOUR DATA HERE ---
import glob
import re

def natur(key):
    """Natural sort key function"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', key)]

root_dir = r"C:\Users\ahmed\OneDrive\Desktop\tiff_imgs\EL_Increment"
files = sorted(glob.glob(os.path.join(root_dir, "*.tif")), key=natur)

# List paths in order: 5mA -> 45mA
fwd_files = files[0:9]

# List paths in order: 45mA -> 5mA (Descending)
rev_files = files[8::]

current_steps = [5, 10, 15, 20, 25, 30, 35, 40, 45]  # Adjust based on your actual files

# Run Analysis
compare_scans(fwd_files, rev_files, current_steps, n_slices=8)