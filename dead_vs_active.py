import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
import math


# --- 1. UTILITIES (STRICTLY FROM YOUR CODE) ---
def natur(key):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', key)]


def get_module_coordinates(ref_image_path):
    """
    ROBUST CALIBRATION: Uses 'Master Box' logic from YOUR working code.
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
    """
    GRID EXTRACTION (STRICTLY FROM YOUR CODE)
    """
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


# --- 2. PLOTTING FUNCTION (MODIFIED FOR SHARED Y-AXIS) ---
def plot_region_analysis(fwd_data_list, rev_data_list, currents, title_prefix, roi_display_img, ref_curr_val):
    print(f"\n--- Generating Plots for: {title_prefix} ---")

    # --- PLOT A: I-V CURVES ---
    fig_iv, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, 7))

    # --- NEW: Variables to track global min/max for shared scaling ---
    global_min = float('inf')
    global_max = float('-inf')

    for cell_idx in range(7):
        # Handle Active (2D) vs Dead (1D) dimensions
        y_fwd = []
        y_rev = []

        for m in fwd_data_list:
            # If matrix has 1 row (dead region), index directly; else take mean
            val = np.mean(m[:, cell_idx]) if m.ndim > 1 and m.shape[0] > 1 else np.mean(m.flatten()[cell_idx])
            y_fwd.append(val)

        for m in rev_data_list:
            val = np.mean(m[:, cell_idx]) if m.ndim > 1 and m.shape[0] > 1 else np.mean(m.flatten()[cell_idx])
            y_rev.append(val)

        # --- NEW: Update Global Limits ---
        # Find the min/max of the current curves and compare with global
        current_curves_min = min(min(y_fwd), min(y_rev))
        current_curves_max = max(max(y_fwd), max(y_rev))

        if current_curves_min < global_min: global_min = current_curves_min
        if current_curves_max > global_max: global_max = current_curves_max

        ax = ax1 if cell_idx < 4 else ax2
        ax.plot(currents, y_fwd, 'o-', color=colors[cell_idx], label=f'C{cell_idx + 1}')
        ax.plot(currents, y_rev, '--', color=colors[cell_idx], alpha=0.6)

    # --- NEW: Apply Shared Y-Axis Limits ---
    # Add 5% padding so curves don't touch the top/bottom edges
    y_range = global_max - global_min
    padding = y_range * 0.05 if y_range != 0 else 100

    limit_bottom = global_min - padding
    limit_top = global_max + padding

    ax1.set_ylim(limit_bottom, limit_top)
    ax2.set_ylim(limit_bottom, limit_top)

    ax1.set_title(f"{title_prefix}: Cells 1 - 4")
    ax1.set_xlabel("Current (mA)")
    ax1.set_ylabel("Mean Intensity")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title(f"{title_prefix}: Cells 5 - 7")
    ax2.set_xlabel("Current (mA)")
    # Y-label is redundant on the second plot if they share the axis, but good for clarity
    # ax2.set_ylabel("Mean Intensity")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.show()

    # --- PLOT B: HEATMAPS WITH ROI INSERT ---
    num_plots = len(currents)
    cols = 3
    rows = math.ceil(num_plots / cols)
    fig_map, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, curr in enumerate(currents):
        ax = axes[i]

        # --- ROI REPLACEMENT LOGIC ---
        # Instead of plotting data at 45mA, we show the cropped image region
        if curr == 45:
            # Normalize crop for display
            disp_norm = cv2.normalize(roi_display_img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            ax.imshow(disp_norm, cmap='gray', aspect='equal')  # 'equal' keeps aspect ratio true
            ax.set_title(f"ROI @ {ref_curr_val}mA", color='green', fontweight='bold')
            ax.axis('off')
        else:
            # Standard Hysteresis Calculation
            f_matrix = fwd_data_list[i]
            r_matrix = rev_data_list[i]

            valid_mask = f_matrix > 5
            diff_matrix = np.zeros_like(f_matrix)
            np.divide(r_matrix - f_matrix, f_matrix, out=diff_matrix, where=valid_mask)
            diff_matrix = diff_matrix * 100

            # Plot Heatmap
            im = ax.imshow(diff_matrix, cmap='RdBu_r', vmin=-15, vmax=15, aspect='auto')
            ax.set_title(f"Hysteresis at {curr} mA")
            ax.set_xticks(np.arange(7))
            ax.set_xticklabels([f"C{k + 1}" for k in range(7)])
            ax.set_yticks([])
            if i % cols == 0: ax.set_ylabel("Top -> Bottom")

    for j in range(i + 1, len(axes)): axes[j].axis('off')
    plt.tight_layout()
    plt.suptitle(f"{title_prefix}: Spatial Hysteresis Map", y=1.02)
    # Only add colorbar if we drew heatmaps
    cbar = fig_map.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    cbar.set_label('Hysteresis (%)', rotation=270, labelpad=20)
    plt.show()


# --- 3. MAIN CONTROLLER ---
def analyze_hysteresis_split_regions(forward_files, reverse_files, currents, ref_current_val=20, n_slices=10):
    # A. LOCATE REFERENCE IMAGE (20mA)
    try:
        ref_idx = currents.index(ref_current_val)
        ref_image_path = forward_files[ref_idx]
        print(f"--- Calibrating Grid using {ref_current_val}mA image: {os.path.basename(ref_image_path)} ---")
    except ValueError:
        print(f"Warning: {ref_current_val}mA not found. Using last image.")
        ref_image_path = forward_files[-1]
        ref_idx = -1

    # B. CALIBRATE COORDINATES (Using your trusted function)
    coords = get_module_coordinates(ref_image_path)
    rx, ry, rw, rh = coords
    print(f"Detected Module ROI: x={rx}, y={ry}, w={rw}, h={rh}")

    # C. PREPARE DISPLAY IMAGES (SPLIT TOP vs BOTTOM)
    # Load raw ref image
    raw_ref = cv2.imread(ref_image_path, -1)
    if raw_ref.dtype == np.uint16:
        raw_ref = cv2.normalize(raw_ref, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    # Crop to main ROI
    roi_img = raw_ref[ry:ry + rh, rx:rx + rw]

    # Calculate the Y-split coordinate based on slices
    # Active = Slices 0 to n-2 (e.g., 0-8)
    # Dead   = Slice n-1 (e.g., 9)
    # We need the pixel height of ONE slice
    seg_height = rh / n_slices
    split_y_pixel = int((n_slices - 1) * seg_height)

    # Create the visual crops
    roi_img_active = roi_img[0:split_y_pixel, :]  # Top part
    roi_img_dead = roi_img[split_y_pixel:, :]  # Bottom part

    # D. EXTRACT ALL DATA (Using your trusted function)
    print("Extracting full grid data...")
    raw_fwd = [extract_grid_data(f, coords, n_slices) for f in forward_files]
    raw_rev = [extract_grid_data(f, coords, n_slices) for f in reverse_files]
    # Reverse reverse_files order to match 5->45 sequence
    raw_rev = raw_rev[::-1]

    # E. SPLIT DATA ARRAYS
    # Active: Everything EXCEPT the last row
    active_fwd = [m[:-1, :] for m in raw_fwd]
    active_rev = [m[:-1, :] for m in raw_rev]

    # Dead: ONLY the last row
    dead_fwd = [m[-1:, :] for m in raw_fwd]
    dead_rev = [m[-1:, :] for m in raw_rev]

    # F. EXECUTE ANALYSIS TWICE
    # 1. Active Region
    plot_region_analysis(active_fwd, active_rev, currents,
                         title_prefix="ACTIVE REGION (Upper Slices)",
                         roi_display_img=roi_img_active,
                         ref_curr_val=ref_current_val)

    # 2. Dead Region
    plot_region_analysis(dead_fwd, dead_rev, currents,
                         title_prefix="DEAD REGION (Bottom Slice)",
                         roi_display_img=roi_img_dead,
                         ref_curr_val=ref_current_val)


# --- 4. EXECUTION ---
root_dir = r"C:\Users\ahmed\OneDrive\Desktop\tiff_imgs\EL_Increment"
files = sorted(glob.glob(os.path.join(root_dir, "*.tif")), key=natur)

# Assuming 9 files forward, 9 files reverse
fwd_files = files[0:9]
rev_files = files[8::]
current_steps = [5, 10, 15, 20, 25, 30, 35, 40, 45]

if len(fwd_files) == len(rev_files) == len(current_steps):
    # n_slices=10 means the bottom slice is exactly 10% of the height
    analyze_hysteresis_split_regions(fwd_files, rev_files, current_steps, ref_current_val=20, n_slices=5)
else:
    print("Error: File counts mismatch.")