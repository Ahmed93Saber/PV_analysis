import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score

# 1. Setup Files
file_names = [
    "4.tif",
    "4_M8_aged_luminescence_c.PNG",
    "251031_EL_M8_6000ms_12mA_PM043609_279_1e749cae.tif",
    "251113_PL_M8_BL_33_LPCF665_BP810_25000ms_centered_tiltedmax_103cmclose_PM014325_525_22bbbad5.tif",
    "251216_7_PL_M8_25s_740LP_UV_62cmfar_120hoch_20angle_PM013633_585_9bafe515.tif",
    "M8_aged_bandwidth_c.PNG",
    "M8_aged_energy_of_max_c.PNG"
]

def preprocess_image(path, target_size=(64, 64)):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: return None
    if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img_float = img.astype(float)
        img_norm = 255 * (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-8)
        img = img_norm.astype(np.uint8)
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img_resized

# Load images
images = {}
for f_name in file_names:
    img = preprocess_image(f_name)
    if img is not None:
        images[f_name] = img

# 2. Compute NMI Matrix
n = len(file_names)
nmi_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i == j:
            nmi_matrix[i][j] = 1.0
        else:
            name1 = file_names[i]
            name2 = file_names[j]
            if name1 in images and name2 in images:
                # Flatten arrays for NMI calculation
                # NMI treats values as categorical labels.
                # For images, intensity values (0-255) act as labels.
                score = normalized_mutual_info_score(images[name1].ravel(), images[name2].ravel())
                nmi_matrix[i][j] = score

# 3. Plotting
# Determine vmax for "temperature" effect
# If max non-diagonal is low, we clamp vmax.
max_val = np.max(nmi_matrix[~np.eye(nmi_matrix.shape[0], dtype=bool)])
vmax_adjusted = max(0.5, max_val + 0.1) # Heuristic: slightly above max, min 0.5

plt.figure(figsize=(12, 10))
# Using 'plasma' colormap which goes from blue/purple (low) to yellow (high)
ax = sns.heatmap(nmi_matrix, annot=True, cmap='plasma', fmt='.2f',
                 xticklabels=file_names, yticklabels=file_names,
                 vmin=0, vmax=vmax_adjusted)

plt.title(f'Normalized Mutual Information (NMI) Heatmap (Max Scale: {vmax_adjusted:.2f})')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('nmi_heatmap.png')
plt.show()

# Print the matrix for the user
df_nmi = pd.DataFrame(nmi_matrix, index=file_names, columns=file_names)
print(df_nmi)