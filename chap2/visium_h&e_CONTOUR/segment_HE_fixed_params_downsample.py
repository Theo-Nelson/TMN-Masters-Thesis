import os
import scanpy as sc
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import pandas as pd
from skimage import color, filters, morphology
from scipy.ndimage import center_of_mass, binary_fill_holes
from skimage.transform import resize
from collections import defaultdict

# === Paths
input_folder = "split_hearts_kmeans_all"
output_folder = "HE_segmented_ab3_hole1000_morph18"
os.makedirs(output_folder, exist_ok=True)

btf_mapping = {
    "WT_A1_heart_38": "BIG_TIFF_WT/VisiumHD_002E3_slide2_38-47-49-50.btf",
    "WT_A1_heart_47": "BIG_TIFF_WT/VisiumHD_002E3_slide2_38-47-49-50.btf",
    "WT_A1_heart_49": "BIG_TIFF_WT/VisiumHD_002E3_slide2_38-47-49-50.btf",
    "WT_A1_heart_50": "BIG_TIFF_WT/VisiumHD_002E3_slide2_38-47-49-50.btf",
    "WT_D1_heart_34": "BIG_TIFF_WT/VisiumHD_007E3_slide1_34-35-36-37.btf",
    "WT_D1_heart_35": "BIG_TIFF_WT/VisiumHD_007E3_slide1_34-35-36-37.btf",
    "WT_D1_heart_36": "BIG_TIFF_WT/VisiumHD_007E3_slide1_34-35-36-37.btf",
    "WT_D1_heart_37": "BIG_TIFF_WT/VisiumHD_007E3_slide1_34-35-36-37.btf",
    "VSNL1_MUT_A1_heart_19": "BIG_TIFF_VSNL/VisiumHD_Vsnl1_E4005.btf",
    "VSNL1_MUT_A1_heart_20": "BIG_TIFF_VSNL/VisiumHD_Vsnl1_E4005.btf",
    "VSNL1_MUT_A1_heart_22": "BIG_TIFF_VSNL/VisiumHD_Vsnl1_E4005.btf",
    "VSNL1_MUT_A1_heart_23": "BIG_TIFF_VSNL/VisiumHD_Vsnl1_E4005.btf",
    "VSNL1_MUT_D1_heart_24": "BIG_TIFF_VSNL/VisiumHD_Vsnl1_E4015.btf",
    "VSNL1_MUT_D1_heart_28": "BIG_TIFF_VSNL/VisiumHD_Vsnl1_E4015.btf",
    "VSNL1_MUT_D1_heart_30": "BIG_TIFF_VSNL/VisiumHD_Vsnl1_E4015.btf",
    "VSNL1_MUT_D1_heart_31": "BIG_TIFF_VSNL/VisiumHD_Vsnl1_E4015.btf",
}

ab_thresh = 3.0
morph_radius = 9
left_expand_px = 4500
bottom_expand_px = 2500
min_tissue_area = 5000

com_records = []
h5ad_files = [f for f in os.listdir(input_folder) if f.endswith(".h5ad")]

# === Group hearts by BTF file
btf_to_hearts = defaultdict(list)
for h5ad_file in h5ad_files:
    heart_id = h5ad_file.replace(".h5ad", "")
    btf_path = btf_mapping.get(heart_id)
    if btf_path:
        btf_to_hearts[btf_path].append(heart_id)

# === Process in BTF blocks
for btf_path, heart_list in btf_to_hearts.items():
    print(f"\n🖼️ Loading BTF image once: {btf_path}")
    he_image = tifffile.imread(btf_path)

    for heart_id in heart_list:
        print(f"\n📂 Processing {heart_id}...")
        adata = sc.read_h5ad(os.path.join(input_folder, f"{heart_id}.h5ad"))
        coords = adata.obsm["spatial"]

        xmin = max(int(coords[:, 0].min()) - left_expand_px, 0)
        xmax = min(int(coords[:, 0].max()), he_image.shape[1])
        ymin = max(int(coords[:, 1].min()), 0)
        ymax = min(int(coords[:, 1].max()) + bottom_expand_px, he_image.shape[0])
        cropped = he_image[ymin:ymax, xmin:xmax]

        lab = color.rgb2lab(cropped)
        L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        ab_std = np.std(np.stack([A, B], axis=-1), axis=-1)
        non_gray = ab_std > ab_thresh

        otsu_thresh = filters.threshold_otsu(L[non_gray])
        tissue_candidate = (L < otsu_thresh) & non_gray
        base_mask = morphology.remove_small_objects(tissue_candidate, min_size=min_tissue_area)

        # Downsample for fast morphology
        base_mask_ds = resize(base_mask.astype(float), (base_mask.shape[0] // 2, base_mask.shape[1] // 2), order=0).astype(bool)
        cleaned_ds = morphology.binary_closing(base_mask_ds, morphology.disk(morph_radius))
        cleaned_ds = morphology.binary_opening(cleaned_ds, morphology.disk(morph_radius))
        cleaned = resize(cleaned_ds.astype(float), base_mask.shape, order=0).astype(bool)

        filled = binary_fill_holes(cleaned)
        com_yx = np.array(center_of_mass(filled))
        com_x = xmin + com_yx[1]
        com_y = ymin + com_yx[0]
        com_records.append({"heart_id": heart_id, "COM_x": com_x, "COM_y": com_y})

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(cropped)
        ax.imshow(filled, cmap="Blues", alpha=0.4)
        ax.plot(com_yx[1], com_yx[0], "ro")
        ax.set_title(f"{heart_id} | ab>{ab_thresh} | morph=ds{morph_radius}")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{heart_id}_mask_overlay.png"), dpi=150)
        plt.close()

        mask_lowres = resize(filled.astype(float), (filled.shape[0] // 20, filled.shape[1] // 20), order=0,
                             preserve_range=True, anti_aliasing=False).astype(bool)
        plt.imsave(os.path.join(output_folder, f"{heart_id}_mask_lowres.png"), mask_lowres, cmap="gray")

        np.savez_compressed(os.path.join(output_folder, f"{heart_id}_mask_fullres.npz"), mask=filled)

        plt.figure(figsize=(6, 4))
        plt.hist(ab_std.ravel(), bins=100, density=True, color="gray")
        plt.axvline(ab_thresh, color="red", linestyle="--", label=f"Threshold = {ab_thresh}")
        plt.title(f"{heart_id}: ab_std dist")
        plt.xlabel("ab_std")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{heart_id}_ab_std_distribution.png"), dpi=100)
        plt.close()

# Save COMs to CSV
com_df = pd.DataFrame(com_records)
com_df.to_csv(os.path.join(output_folder, "COM_coordinates.csv"), index=False)

print("\n✅ Finished tissue mask extraction with memory-efficient batching.")

