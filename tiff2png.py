import os
from PIL import Image
import tifffile as tiff

# ---- CONFIG ----
input_dir = r"C:\Users\ahmed\OneDrive\Desktop\tiff_imgs\EL_Increment"
output_dir = r"C:\Users\ahmed\OneDrive\Desktop\tiff_imgs\EL_Increment"
# ----------------

os.makedirs(output_dir, exist_ok=True)

tiff_files = [f for f in os.listdir(input_dir)
              if f.lower().endswith((".tif", ".tiff"))]

print(f"Found {len(tiff_files)} TIFF files")

for fname in tiff_files:
    tiff_path = os.path.join(input_dir, fname)
    base = os.path.splitext(fname)[0]

    try:
        pages = tiff.imread(tiff_path)

        # Handle single-page and multi-page TIFFs
        if pages.ndim == 2 or pages.ndim == 3:
            pages = [pages]

        for i, page in enumerate(pages):
            img = Image.fromarray(page)

            if len(pages) > 1:
                out_name = f"{base}_page{i+1}.png"
            else:
                out_name = f"{base}.png"

            out_path = os.path.join(output_dir, out_name)
            img.save(out_path, "PNG")

        print(f"✔ Converted: {fname}")

    except Exception as e:
        print(f"✖ Failed: {fname}  → {e}")

print("Done.")
