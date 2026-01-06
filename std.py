import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np

# List of image filenames
image_files = [
    "4.tif",
    "4_M8_aged_luminescence_c.PNG",
    "251031_EL_M8_6000ms_12mA_PM043609_279_1e749cae.tif",
    "251113_PL_M8_BL_33_LPCF665_BP810_25000ms_centered_tiltedmax_103cmclose_PM014325_525_22bbbad5.tif",
    "251216_7_PL_M8_25s_740LP_UV_62cmfar_120hoch_20angle_PM013633_585_9bafe515.tif",
    "M8_aged_bandwidth_c.PNG",
    "M8_aged_energy_of_max_c.PNG"
]

# Function to calculate SSIM between two images
def calculate_ssim(image1_path, image2_path):
    # Load images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Convert to grayscale
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Resize images to the same dimensions for SSIM calculation
    # A reasonable size that captures the overall structure
    new_size = (512, 512)
    image1_resized = cv2.resize(image1_gray, new_size, interpolation=cv2.INTER_AREA)
    image2_resized = cv2.resize(image2_gray, new_size, interpolation=cv2.INTER_AREA)

    # Calculate SSIM
    # SSIM returns a value between -1 and 1, where 1 is perfect similarity
    ssim_value, _ = ssim(image1_resized, image2_resized, full=True)
    return ssim_value

# Calculate SSIM for all pairs of images
num_images = len(image_files)
ssim_matrix = np.zeros((num_images, num_images))

for i in range(num_images):
    for j in range(num_images):
        if i == j:
            ssim_matrix[i, j] = 1.0
        else:
            ssim_matrix[i, j] = calculate_ssim(image_files[i], image_files[j])

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Draw the heatmap with the mask and correct aspect ratio, using original filenames for labels
second_max = sorted(list(set(ssim_matrix.flatten())), reverse=True)[1]
sns.heatmap(ssim_matrix, annot=True, cmap='viridis', fmt='.2f', xticklabels=image_files, yticklabels=image_files,
            vmin=0, vmax=second_max)



# Add title and axis labels
plt.title('SSIM Correlation Heatmap')
plt.xlabel('Image Name')
plt.ylabel('Image Name')

# Rotate x-axis tick labels for better readability
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Show the plot
plt.tight_layout() # Adjust layout to prevent clipping of labels
plt.show()