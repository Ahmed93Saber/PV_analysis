import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mutual_info_score

def advanced_metrics_and_plot(img1_path, img2_path):
    # 1. Load as Grayscale
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("Error: Could not load one or both images.")
        return

    # 2. Resize img2 to match img1
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # --- METRIC 1: SSIM ---
    # data_range=255 is safer for 8-bit images to ensure accurate scaling
    score_ssim, diff_map = ssim(img1, img2_resized, full=True, data_range=255)
    print(f"SSIM Score: {score_ssim:.4f} (1.0 is identical)")

    # --- METRIC 2: MUTUAL INFORMATION ---
    hist_2d, _, _ = np.histogram2d(img1.ravel(), img2_resized.ravel(), bins=20)
    mi = mutual_info_score(None, None, contingency=hist_2d)
    print(f"Mutual Information: {mi:.4f} (Higher is better)")

    # --- PLOTTING ---
    plt.figure(figsize=(15, 5))

    # Plot 1: Reference Image
    plt.subplot(1, 3, 1)
    plt.imshow(img1, cmap='gray')
    plt.title("Reference Image")
    plt.axis('off')

    # Plot 2: Resized Comparison Image
    plt.subplot(1, 3, 2)
    plt.imshow(img2_resized, cmap='gray')
    plt.title("Resized Comparison Image")
    plt.axis('off')

    # Plot 3: SSIM Difference Map
    # Note: In SSIM diff maps, brighter usually indicates HIGHER similarity.
    # Darker regions indicate structural differences.
    plt.subplot(1, 3, 3)
    im = plt.imshow(diff_map, cmap='hot', vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f"SSIM Diff Map (Score: {score_ssim:.2f})")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Run
advanced_metrics_and_plot(
    "4.tif",
    "4_M8_aged_luminescence_c.PNG"
)