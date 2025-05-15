import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import bin2cell as b2c

# === Paths for WT_A1
sample_id = "WT_A1"
input_dir = f"{sample_id}_square_002um"
spatial_dir = os.path.join(input_dir, "spatial")
source_image_path = "BIG_TIFF_WT/VisiumHD_002E3_slide2_38-47-49-50.btf"
output_dir = "diagnostics_outside_bins"
os.makedirs(output_dir, exist_ok=True)

# === Load full binned AnnData object with .obsm["spatial"]
print(f"📦 Loading Visium object for {sample_id}...")
adata = b2c.read_visium(
    path=input_dir,
    source_image_path=source_image_path,
    spaceranger_image_path=spatial_dir,
)

# === Load full-resolution H&E image (BTF)
print(f"🖼️ Reading full BTF image: {source_image_path}")
img = tifffile.imread(source_image_path)
img_height, img_width = img.shape[:2]

# === Extract spatial coordinates
x = adata.obsm["spatial"][:, 0]  # column (X)
y = adata.obsm["spatial"][:, 1]  # row (Y)

# === Identify out-of-image bins
outside_mask = (x < 0) | (x >= img_width) | (y < 0) | (y >= img_height)
n_outside = outside_mask.sum()
print(f"⚠️ Found {n_outside} bins outside image bounds.")

# === Optional: store as flag
adata.obs["out_of_image"] = outside_mask

# === Plot overlay
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.scatter(x, y, s=1, color="gray", alpha=0.3, label="In-image bins")
plt.scatter(x[outside_mask], y[outside_mask], s=10, color="red", label="Out-of-image bins")
plt.gca().invert_yaxis()
plt.title(f"{sample_id} – Bins Outside Image Bounds ({n_outside} total)")
plt.legend()
plt.axis("off")
plt.tight_layout()
plot_path = os.path.join(output_dir, f"{sample_id}_out_of_image_bins.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"✅ Saved plot to {plot_path}")

