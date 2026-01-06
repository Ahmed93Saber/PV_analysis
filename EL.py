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
    if img.dtype == np.uint16:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    elif len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("Could not detect module in reference image!")

    # Find the "Master Box" encompassing all cells
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


def process_batch(file_list, ref_index=0):
    """
    Applies the grid from file_list[ref_index] to ALL files.
    """
    # 1. Get Coordinates from the Reference Image
    ref_path = file_list[ref_index]
    print(f"--- Calibrating Grid using: {os.path.basename(ref_path)} ---")
    rx, ry, rw, rh = get_module_coordinates(ref_path)

    # Grid Settings
    n_cells = 7
    n_segments = 3
    cell_width = rw / n_cells
    seg_height = rh / n_segments

    # Storage for all data: results[filename] = 3x7 matrix
    batch_results = {}

    # Setup Plotting Grid (Dynamic size based on number of files)
    num_files = len(file_list)
    cols = 3
    rows = (num_files // cols) + (1 if num_files % cols > 0 else 0)
    plt.figure(figsize=(15, 4 * rows))

    # 2. Loop through ALL images
    for i, fpath in enumerate(file_list):
        if not os.path.exists(fpath):
            print(f"Skipping missing file: {fpath}")
            continue

        # Load Target Image
        img = cv2.imread(fpath, -1)

        # --- CRITICAL: FORCE THE CROP ---
        # We do NOT run thresholding here. We just cut using the Reference coords.
        roi = img[ry:ry + rh, rx:rx + rw]

        # Calculate Intensity Grid
        intensity_map = np.zeros((n_segments, n_cells))

        for c in range(n_cells):
            for r in range(n_segments):
                x1 = int(c * cell_width)
                x2 = int((c + 1) * cell_width)
                y1 = int(r * seg_height)
                y2 = int((r + 1) * seg_height)

                segment = roi[y1:y2, x1:x2]
                intensity_map[r, c] = np.mean(segment)

        batch_results[os.path.basename(fpath)] = intensity_map

        # Add to Subplot
        plt.subplot(rows, cols, i + 1)
        plt.imshow(intensity_map, cmap='magma', aspect='auto')
        plt.colorbar(label='Mean Intensity')
        plt.title(f"{os.path.basename(fpath)[:15]}...")  # Shorten title
        plt.xticks(np.arange(n_cells), [f"C{k + 1}" for k in range(n_cells)])
        plt.yticks(np.arange(n_segments), ["Top", "Mid", "Bot"])

    plt.tight_layout()
    plt.show()
    return batch_results

# --- EXECUTION ---
# List your files here. Ensure the "Reference" (clearest one) is FIRST or specify ref_index
root_dir = r"C:\Users\ahmed\OneDrive\Desktop\tiff_imgs\EL_Increment"
files = sorted(glob.glob(os.path.join(root_dir, "*.tif")))

# Run Analysis
data = process_batch(files, ref_index=1)
