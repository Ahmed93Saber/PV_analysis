import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


def get_module_coordinates(ref_image_path):
    """
    Detects the module location from a bright Reference Image.
    Returns: (x, y, w, h) of the active area.
    """
    # Load and process reference
    img = cv2.imread(ref_image_path, -1)
    # Normalize if 16-bit
    if img.dtype == np.uint16:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    elif len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("Could not detect module in reference image!")

    # Find the "Master Box"
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = 0, 0

    for c in contours:
        if cv2.contourArea(c) > 100:
            x, y, w, h = cv2.boundingRect(c)
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)

    return (min_x, min_y, max_x - min_x, max_y - min_y)


def process_batch_with_overlay(file_list, ref_index=0):
    """
    Applies the grid from file_list[ref_index] to ALL files and plots overlays.
    """
    # 1. Calibration
    ref_path = file_list[ref_index]
    print(f"--- Calibrating Grid using: {os.path.basename(ref_path)} ---")
    rx, ry, rw, rh = get_module_coordinates(ref_path)

    # Grid Settings
    n_cells = 7
    n_segments = 3
    cell_width = rw / n_cells
    seg_height = rh / n_segments

    # Storage
    batch_results = {}
    num_files = len(file_list)

    # 2. Setup Plot (One row per file, 2 columns)
    # Height is dynamic: 3 inches per file
    plt.figure(figsize=(12, 3 * num_files))

    for i, fpath in enumerate(file_list):
        if not os.path.exists(fpath):
            print(f"Skipping missing file: {fpath}")
            continue

        filename = os.path.basename(fpath)
        img = cv2.imread(fpath, -1)

        # Crop using Reference Coordinates
        roi = img[ry:ry + rh, rx:rx + rw]

        # --- PREPARE VISUALIZATION (Auto-Brightness) ---
        # Convert to 8-bit for display
        if roi.dtype == np.uint16:
            vis_roi = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        else:
            vis_roi = roi.copy()

        # Contrast Stretch: Make the max pixel 255 so we can see the dark images
        vis_roi = cv2.normalize(vis_roi, None, 0, 255, cv2.NORM_MINMAX)
        vis_img = cv2.cvtColor(vis_roi, cv2.COLOR_GRAY2BGR)

        # Calculate Data
        intensity_map = np.zeros((n_segments, n_cells))

        for c in range(n_cells):
            for r in range(n_segments):
                x1 = int(c * cell_width)
                x2 = int((c + 1) * cell_width)
                y1 = int(r * seg_height)
                y2 = int((r + 1) * seg_height)

                # Data Collection (Raw values)
                segment = roi[y1:y2, x1:x2]
                intensity_map[r, c] = np.mean(segment)

                # Draw Grid (Green) on Visualization
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        batch_results[filename] = intensity_map

        # --- PLOTTING ---
        # Plot 1: Heatmap (Left)
        plt.subplot(num_files, 2, (i * 2) + 1)
        plt.imshow(intensity_map, cmap='magma', aspect='auto')
        plt.colorbar(label='Intensity')
        plt.title(f"{filename[:10]}... - Data")
        plt.ylabel("Top -> Bot")
        plt.xticks([])  # Hide x-ticks to reduce clutter

        # Plot 2: Overlay (Right)
        plt.subplot(num_files, 2, (i * 2) + 2)
        plt.imshow(vis_img, aspect='auto')  # Stretched to match heatmap
        plt.title(f"Grid Verification (Auto-Contrast)")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    return batch_results


# List your files here. Ensure the "Reference" (clearest one) is FIRST or specify ref_index
root_dir = r"C:\Users\ahmed\OneDrive\Desktop\tiff_imgs\EL_Increment"
files = sorted(glob.glob(os.path.join(root_dir, "*.tif")))

# Run Analysis
data = process_batch_with_overlay(files, ref_index=1)
