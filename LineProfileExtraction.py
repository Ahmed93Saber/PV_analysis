import cv2
import numpy as np
import matplotlib.pyplot as plt


def analyze_cross_sections(image_path, h_lines=[], v_lines=[]):
    """
    Plots intensity profiles for full-span horizontal or vertical lines.

    Parameters:
    - h_lines: List of Y-coordinates for Horizontal cuts (0 is Top)
    - v_lines: List of X-coordinates for Vertical cuts (0 is Left)
    """
    # 1. Load Image
    img = cv2.imread(image_path, -1)
    if img is None:
        print(f"Error: File not found at {image_path}")
        return

    h, w = img.shape

    # Visualization Image (Brightened for display)
    if img.dtype == np.uint16:
        vis_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    else:
        vis_img = img.copy()
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)

    # Prepare Plotting
    plt.figure(figsize=(14, 6))

    # --- PLOT 1: The Image with Lines ---
    plt.subplot(1, 2, 1)

    # Draw Horizontal Lines
    colors_h = plt.cm.spring(np.linspace(0, 1, len(h_lines)))  # Color palette 1
    for idx, y in enumerate(h_lines):
        # Check bounds
        if 0 <= y < h:
            color = tuple(int(c * 255) for c in colors_h[idx][:3])  # Convert to RGB
            # Draw line on image
            cv2.line(vis_img, (0, y), (w, y), color, 2)
            # Add text label on image
            cv2.putText(vis_img, f"H{y}", (10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw Vertical Lines
    colors_v = plt.cm.cool(np.linspace(0, 1, len(v_lines)))  # Color palette 2
    for idx, x in enumerate(v_lines):
        if 0 <= x < w:
            color = tuple(int(c * 255) for c in colors_v[idx][:3])
            cv2.line(vis_img, (x, 0), (x, h), color, 2)
            cv2.putText(vis_img, f"V{x}", (x + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    plt.imshow(vis_img, aspect='auto')
    plt.title("Cross-Section Locations")
    plt.axis('off')

    # --- PLOT 2: The Intensity Profiles ---
    plt.subplot(1, 2, 2)

    # Plot Horizontal Profiles
    for idx, y in enumerate(h_lines):
        if 0 <= y < h:
            # Extract Row
            profile = img[y, :]
            c = colors_h[idx]
            plt.plot(profile, label=f"Horiz (Y={y})", color=c, linewidth=1.5)

    # Plot Vertical Profiles
    for idx, x in enumerate(v_lines):
        if 0 <= x < w:
            # Extract Column
            profile = img[:, x]
            c = colors_v[idx]
            # We usually plot vertical profiles against Y (distance from top)
            plt.plot(profile, linestyle='--', label=f"Vert (X={x})", color=c, linewidth=1.5)

    plt.title("Intensity Profiles")
    plt.xlabel("Distance (Pixels)")
    plt.ylabel("EL Intensity (a.u.)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# --- USER CONFIGURATION ---
file_path = r"C:\Users\ahmed\OneDrive\Desktop\tiff_imgs\EL_Increment\5_251103_EL_M8_50pro_6000ms_25mA_PM045500_243.tif"

# DEFINE YOUR CUTS HERE
# Horizontal: Enter Y-coordinates (e.g., [100] is near top, [500] is near bottom)
horizontal_cuts = [220]

# Vertical: Enter X-coordinates (e.g., to slice through specific cells)
# Tip: If image width is ~600, center is ~300.
vertical_cuts = []

# Run
analyze_cross_sections(file_path, v_lines=vertical_cuts, h_lines=horizontal_cuts)