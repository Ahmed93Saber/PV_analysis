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
    img = cv2.imread(ref_image_path, -1)
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


def process_batch_custom_slices(file_list, n_slices=10, ref_index=0):
    """
    Applies a 7-column x n-row grid to all images.
    """
    # 1. Calibration
    ref_path = file_list[ref_index]
    print(f"--- Calibrating Grid using: {os.path.basename(ref_path)} ---")
    rx, ry, rw, rh = get_module_coordinates(ref_path)

    # Grid Settings
    n_cells = 7  # Fixed by physics (module has 7 cells)
    n_rows = n_slices  # Defined by user

    cell_width = rw / n_cells
    seg_height = rh / n_rows

    # Storage
    batch_results = {}
    num_files = len(file_list)

    # Increase figure height if n_slices is huge to ensure labels fit
    plt.figure(figsize=(14, 4 * num_files))

    for i, fpath in enumerate(file_list):
        if not os.path.exists(fpath):
            continue

        filename = os.path.basename(fpath)
        img = cv2.imread(fpath, -1)

        # Crop
        roi = img[ry:ry + rh, rx:rx + rw]

        # Prepare Overlay (Auto-Contrast)
        if roi.dtype == np.uint16:
            vis_roi = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        else:
            vis_roi = roi.copy()

        # Brighten for display
        vis_roi = cv2.normalize(vis_roi, None, 0, 255, cv2.NORM_MINMAX)
        vis_img = cv2.cvtColor(vis_roi, cv2.COLOR_GRAY2BGR)

        # Analysis Loop
        intensity_map = np.zeros((n_rows, n_cells))

        for c in range(n_cells):
            for r in range(n_rows):
                x1 = int(c * cell_width)
                x2 = int((c + 1) * cell_width)
                y1 = int(r * seg_height)
                y2 = int((r + 1) * seg_height)

                # Extract Data
                segment = roi[y1:y2, x1:x2]
                intensity_map[r, c] = np.mean(segment)

                # Draw Grid
                # Only draw horizontal lines if n_slices is small (<50) to avoid solid green blocks
                if n_rows < 50:
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                else:
                    # If too many slices, just draw vertical cell dividers
                    cv2.line(vis_img, (x1, 0), (x1, int(rh)), (0, 255, 0), 1)
                    cv2.line(vis_img, (x2, 0), (x2, int(rh)), (0, 255, 0), 1)

        batch_results[filename] = intensity_map

        # --- PLOTTING ---

        # 1. Heatmap
        plt.subplot(num_files, 2, (i * 2) + 1)
        plt.imshow(intensity_map, cmap='magma', aspect='auto')
        plt.colorbar(label='Intensity')
        plt.title(f"{filename[:15]}... ({n_rows} Slices)")

        # Y-Axis formatting
        if n_rows <= 20:
            # Label every slice if count is low
            plt.yticks(np.arange(n_rows), [f"S{k + 1}" for k in range(n_rows)])
        else:
            # If high count, just label Top and Bottom
            plt.yticks([0, n_rows - 1], ["Top", "Bottom"])

        plt.xlabel("Cell Index")

        # 2. Overlay
        plt.subplot(num_files, 2, (i * 2) + 2)
        plt.imshow(vis_img, aspect='auto')
        plt.title(f"Grid Overlay")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    return batch_results


# List your files here. Ensure the "Reference" (clearest one) is FIRST or specify ref_index
root_dir = r"C:\Users\ahmed\OneDrive\Desktop\tiff_imgs\EL_Increment"
files = glob.glob(os.path.join(root_dir, "*.tif"))

# Run Analysis
data = process_batch_custom_slices(files, n_slices=5, ref_index=3)
