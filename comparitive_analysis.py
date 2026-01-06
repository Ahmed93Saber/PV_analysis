import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
import math


# --- UTILITIES ---
def natur(key):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', key)]


def get_module_coordinates(ref_image_path):
    """
    ROBUST CALIBRATION: Uses 'Master Box' logic to find the full module extent
    even if the reference image has dark spots.
    """
    img = cv2.imread(ref_image_path, -1)
    if img is None: raise FileNotFoundError(f"Could not load {ref_image_path}")

    if img.dtype == np.uint16:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    elif len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("Could not detect module! Reference image is too dark.")

    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = 0, 0
    found_valid = False

    for c in contours:
        if cv2.contourArea(c) > 100:
            x, y, w, h = cv2.boundingRect(c)
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)
            found_valid = True

    if not found_valid:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return (x, y, w, h)

    return (min_x, min_y, max_x - min_x, max_y - min_y)


def extract_grid_data(file_path, coords, n_slices=10):
    rx, ry, rw, rh = coords
    img = cv2.imread(file_path, -1)
    if img is None: return None

    roi = img[ry:ry + rh, rx:rx + rw]

    n_cells = 7
    cell_width = rw / n_cells
    seg_height = rh / n_slices

    data_matrix = np.zeros((n_slices, n_cells))

    for c in range(n_cells):
        for r in range(n_slices):
            x1, x2 = int(c * cell_width), int((c + 1) * cell_width)
            y1, y2 = int(r * seg_height), int((r + 1) * seg_height)

            if y1 >= roi.shape[0] or x1 >= roi.shape[1]: continue

            segment = roi[y1:y2, x1:x2]
            if segment.size > 0:
                data_matrix[r, c] = np.mean(segment)

    return data_matrix


# --- MAIN ANALYSIS ---
def analyze_hysteresis_split_plots(forward_files, reverse_files, currents, ref_current_val=20, n_slices=10):
    # 1. FIND REFERENCE (20mA)
    try:
        ref_idx = currents.index(ref_current_val)
        ref_image_path = forward_files[ref_idx]
        print(f"--- Calibrating Grid using {ref_current_val}mA image: {os.path.basename(ref_image_path)} ---")
    except ValueError:
        print(f"Warning: {ref_current_val}mA not found. Using last image.")
        ref_image_path = forward_files[-1]

    # 2. CALIBRATE GEOMETRY
    coords = get_module_coordinates(ref_image_path)
    rx, ry, rw, rh = coords
    print(f"Detected Module ROI: x={rx}, y={ry}, w={rw}, h={rh}")

    # 3. EXTRACT DATA
    print("Processing Data...")
    fwd_data = [extract_grid_data(f, coords, n_slices) for f in forward_files]
    rev_data = [extract_grid_data(f, coords, n_slices) for f in reverse_files]
    rev_data = rev_data[::-1]

    # ==========================================
    # --- PLOT 1: SPLIT I-V CURVES (C1-C4 | C5-C7) ---
    # ==========================================
    fig_iv, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Generate distinct colors for 7 cells
    colors = plt.cm.viridis(np.linspace(0, 1, 7))

    for cell_idx in range(7):
        # Calculate Whole Cell Average (Mean of all vertical slices)
        y_fwd = [np.mean(m[:, cell_idx]) for m in fwd_data]
        y_rev = [np.mean(m[:, cell_idx]) for m in rev_data]

        # Decide which subplot to use
        if cell_idx < 4:  # Cells 1, 2, 3, 4 (Indices 0-3)
            ax = ax1
        else:  # Cells 5, 6, 7 (Indices 4-6)
            ax = ax2

        # Plot Data
        ax.plot(currents, y_fwd, 'o-', color=colors[cell_idx], label=f'Cell {cell_idx + 1} (Fwd)')
        ax.plot(currents, y_rev, '--', color=colors[cell_idx], alpha=0.6)  # Dashed for reverse

    # Formatting Subplot 1 (Left)
    ax1.set_title("I-V Curves: Cells 1 - 4")
    ax1.set_xlabel("Current (mA)")
    ax1.set_ylabel("Mean Intensity (Whole Cell)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Formatting Subplot 2 (Right)
    ax2.set_title("I-V Curves: Cells 5 - 7")
    ax2.set_xlabel("Current (mA)")
    ax2.set_ylabel("Mean Intensity (Whole Cell)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ==========================================
    # --- PLOT 2: HYSTERESIS HEATMAPS ---
    # ==========================================
    num_plots = len(currents)
    cols = 3
    rows = math.ceil(num_plots / cols)
    fig_map, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, curr in enumerate(currents):
        ax = axes[i]

        if curr == 45:
            # Raw Image Reference with Calibration Box
            raw_img = cv2.imread(forward_files[i], -1)
            vis_img = cv2.normalize(raw_img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(vis_img, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 5)
            roi_display = vis_img[ry:ry + rh, rx:rx + rw]

            ax.imshow(roi_display, aspect='equal')
            ax.set_title(f"Ref Image ({curr} mA) w/ ROI", color='green', fontweight='bold')
            ax.axis('off')
        else:
            # Heatmaps
            f_matrix = fwd_data[i]
            r_matrix = rev_data[i]
            valid_mask = f_matrix > 5
            diff_matrix = np.zeros_like(f_matrix)
            np.divide(r_matrix - f_matrix, f_matrix, out=diff_matrix, where=valid_mask)
            diff_matrix = diff_matrix * 100

            im = ax.imshow(diff_matrix, cmap='RdBu_r', vmin=-15, vmax=15, aspect='auto')
            ax.set_title(f"Hysteresis at {curr} mA")
            ax.set_xticks(np.arange(7))
            ax.set_xticklabels([f"C{k + 1}" for k in range(7)])
            ax.set_yticks([])
            if i % cols == 0: ax.set_ylabel("Top -> Bottom")

    for j in range(i + 1, len(axes)): axes[j].axis('off')
    plt.tight_layout()
    cbar = fig_map.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    cbar.set_label('Hysteresis (%)\n(Red = Brighter on Return)', rotation=270, labelpad=20)
    plt.show()


# --- INPUT SETUP ---
root_dir = r"C:\Users\ahmed\OneDrive\Desktop\tiff_imgs\EL_Increment"
files = sorted(glob.glob(os.path.join(root_dir, "*.tif")), key=natur)

fwd_files = files[0:9]
rev_files = files[8::]
current_steps = [5, 10, 15, 20, 25, 30, 35, 40, 45]

if len(fwd_files) == len(rev_files) == len(current_steps):
    analyze_hysteresis_split_plots(fwd_files, rev_files, current_steps, ref_current_val=20, n_slices=6)
else:
    print("Error: File counts mismatch.")