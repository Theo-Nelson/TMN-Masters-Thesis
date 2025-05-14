import os
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tifffile
import bin2cell as b2c

# === Config ===
sample_id = "WT_A1"
input_dir = f"{sample_id}_square_002um"
spatial_dir = os.path.join(input_dir, "spatial")
source_image_path = "BIG_TIFF_WT/VisiumHD_002E3_slide2_38-47-49-50.btf"  # full-res BTF H&E
mpp = 0.2
img_key = f"{mpp}_mpp_150_buffer"
cropped_basis = "spatial_cropped_150_buffer"
full_basis = "spatial"
output_dir = f"spatial_patches_raw_vs_destriped_{sample_id}"
os.makedirs(output_dir, exist_ok=True)

# === Load full Visium binned data
adata = b2c.read_visium(input_dir, source_image_path=source_image_path, spaceranger_image_path=spatial_dir)
adata.var_names_make_unique()

# === Filter & destripe
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_counts=1)
b2c.destripe(adata, adjust_counts=True)

# === Save cropped HE image for patch display
b2c.scaled_he_image(adata, mpp=mpp, save_path=os.path.join(output_dir, "he_cropped.tiff"))

# === Ensure count fields
if "n_counts" not in adata.obs:
    sc.pp.calculate_qc_metrics(adata, inplace=True)

# === Load full-resolution .btf image
btf_image = tifffile.imread(source_image_path)
btf_height, btf_width = btf_image.shape[:2]

# === Patch parameters
patch_size = 300
min_row, max_row = adata.obs["array_row"].min(), adata.obs["array_row"].max()
min_col, max_col = adata.obs["array_col"].min(), adata.obs["array_col"].max()
row_starts = np.arange(min_row, max_row, patch_size)
col_starts = np.arange(min_col, max_col, patch_size)

patch_idx = 0
for r0 in row_starts:
    for c0 in col_starts:
        r1 = r0 + patch_size
        c1 = c0 + patch_size

        mask = (
            (adata.obs["array_row"] >= r0) & (adata.obs["array_row"] < r1) &
            (adata.obs["array_col"] >= c0) & (adata.obs["array_col"] < c1)
        )
        patch = adata[mask].copy()
        if patch.n_obs == 0 or (patch.obs["n_counts"] > 0).sum() == 0:
            continue

        # === 1. Side-by-side striped vs destriped expression
        sc.set_figure_params(figsize=[12, 6], dpi=150)
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        sc.pl.spatial(patch, color="n_counts", img_key=img_key, basis=cropped_basis,
                      ax=axs[0], show=False, title="Striped (n_counts)", cmap="Reds")
        sc.pl.spatial(patch, color="n_counts_adjusted", img_key=img_key, basis=cropped_basis,
                      ax=axs[1], show=False, title="Destriped (n_counts_adjusted)", cmap="Reds")
        patch_img_path = os.path.join(output_dir, f"{sample_id}_patch_{patch_idx}_r{int(r0)}_c{int(c0)}.png")
        plt.tight_layout()
        plt.savefig(patch_img_path, dpi=300)
        plt.close()

        # === 2. Overlay red dot on full-res BTF using .obsm["spatial"] (microns)
        coords = patch.obsm[full_basis]
        center_x = np.mean(coords[:, 0])
        center_y = np.mean(coords[:, 1])

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(btf_image)
        ax.scatter([center_x], [center_y], s=300, color="red", edgecolor="black", linewidth=1.5, zorder=10)
        ax.set_title(f"{sample_id} – Patch {patch_idx} Center on Full BTF")
        ax.set_xlim(0, btf_width)
        ax.set_ylim(btf_height, 0)
        ax.axis("off")
        overlay_path = os.path.join(output_dir, f"{sample_id}_patch_{patch_idx}_HE_btf_fullview.png")
        plt.savefig(overlay_path, dpi=300)
        plt.close()

        patch_idx += 1

print(f"✅ Saved {patch_idx} patch images with raw/destriped and BTF overlays to: {output_dir}")

