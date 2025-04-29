import os
import scanpy as sc
import numpy as np
import tifffile
import matplotlib.pyplot as plt

# === Paths
input_folder = "split_hearts_kmeans_all"
output_folder = "HE_crops_asymmetric_expand"
os.makedirs(output_folder, exist_ok=True)

# === Mapping hearts to BTF images
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

# === Cache loaded BTFs
btf_cache = {}

# === Custom expansion config
left_expand_px = 4500
bottom_expand_px = 2500

# === Process
h5ad_files = [f for f in os.listdir(input_folder) if f.endswith(".h5ad")]

for h5ad_file in h5ad_files:
    heart_id = h5ad_file.replace(".h5ad", "")
    print(f"\n📂 Processing {heart_id}...")

    adata = sc.read_h5ad(os.path.join(input_folder, h5ad_file))
    coords = adata.obsm["spatial"]
    
    btf_path = btf_mapping.get(heart_id)
    if btf_path is None:
        print(f"⚠️ Skipping {heart_id}: no BTF mapping.")
        continue

    if btf_path not in btf_cache:
        print(f"🖼️ Loading BTF image: {btf_path}")
        btf_cache[btf_path] = tifffile.imread(btf_path)
    he_image = btf_cache[btf_path]

    # Base bounding box from spatial coords
    xmin = int(np.floor(coords[:,0].min()))
    xmax = int(np.ceil(coords[:,0].max()))
    ymin = int(np.floor(coords[:,1].min()))
    ymax = int(np.ceil(coords[:,1].max()))

    # Apply asymmetric expansions
    xmin = max(xmin - left_expand_px, 0)
    ymax = min(ymax + bottom_expand_px, he_image.shape[0])
    
    # xmax and ymin stay as they are
    xmax = min(xmax, he_image.shape[1])
    ymin = max(ymin, 0)

    print(f"🔍 Asymmetric bounding box: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")

    cropped_he = he_image[ymin:ymax, xmin:xmax]

    # Save
    plt.figure(figsize=(10, 10))
    plt.imshow(cropped_he)
    plt.axis("off")
    plt.title(heart_id, fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{heart_id}_asymmetric_crop.png"), dpi=100)
    plt.close()

print("\n✅ Finished asymmetric H&E crops!")

