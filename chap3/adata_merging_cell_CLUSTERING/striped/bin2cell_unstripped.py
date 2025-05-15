import os
import bin2cell as b2c
import scanpy as sc

# === Global config ===
mpp = 0.2
samples = {
    "WT_A1": "BIG_TIFF_WT/VisiumHD_002E3_slide2_38-47-49-50.btf",
    "WT_D1": "BIG_TIFF_WT/VisiumHD_007E3_slide1_34-35-36-37.btf",
    "VSNL1_MUT_A1": "BIG_TIFF_VSNL/VisiumHD_Vsnl1_E4005.btf",
    "VSNL1_MUT_D1": "BIG_TIFF_VSNL/VisiumHD_Vsnl1_E4015.btf",
}

# === Unified output base directory
base_output_dir = "bin2cell_output_unstripped"
os.makedirs(base_output_dir, exist_ok=True)

for sample_id, btf_path in samples.items():
    print(f"\n🚀 Running Bin2Cell (no destripe) for {sample_id}...")

    input_dir = f"{sample_id}_square_002um"
    spatial_dir = os.path.join(input_dir, "spatial")
    sample_output_dir = os.path.join(base_output_dir, sample_id)
    stardist_dir = os.path.join(sample_output_dir, "stardist")
    os.makedirs(stardist_dir, exist_ok=True)

    # Load and filter
    adata = b2c.read_visium(input_dir, source_image_path=btf_path, spaceranger_image_path=spatial_dir)
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_counts=1)

    # === Skip destripe ===

    # H&E segmentation
    he_path = os.path.join(stardist_dir, "he.tiff")
    b2c.scaled_he_image(adata, mpp=mpp, save_path=he_path)
    b2c.stardist(image_path=he_path,
                 labels_npz_path=os.path.join(stardist_dir, "he.npz"),
                 stardist_model="2D_versatile_he",
                 prob_thresh=0.01)
    b2c.insert_labels(adata,
                      labels_npz_path=os.path.join(stardist_dir, "he.npz"),
                      basis="spatial",
                      spatial_key="spatial_cropped_150_buffer",
                      mpp=mpp,
                      labels_key="labels_he")
    b2c.expand_labels(adata, labels_key="labels_he", expanded_labels_key="labels_he_expanded")

    # GEX segmentation
    gex_img_path = os.path.join(stardist_dir, "gex.tiff")
    b2c.grid_image(adata, "n_counts", mpp=mpp, sigma=5, save_path=gex_img_path)
    b2c.stardist(image_path=gex_img_path,
                 labels_npz_path=os.path.join(stardist_dir, "gex.npz"),
                 stardist_model="2D_versatile_fluo",
                 prob_thresh=0.05,
                 nms_thresh=0.5)
    b2c.insert_labels(adata,
                      labels_npz_path=os.path.join(stardist_dir, "gex.npz"),
                      basis="array",
                      mpp=mpp,
                      labels_key="labels_gex")

    # Combine labels
    b2c.salvage_secondary_labels(adata,
                                 primary_label="labels_he_expanded",
                                 secondary_label="labels_gex",
                                 labels_key="labels_joint")

    # Run bin2cell
    cdata = b2c.bin_to_cell(adata,
                            labels_key="labels_joint",
                            spatial_keys=["spatial", "spatial_cropped_150_buffer"])

    # Save outputs
    adata.write(os.path.join(sample_output_dir, f"{sample_id}_bins.h5ad"))
    cdata.write(os.path.join(sample_output_dir, f"{sample_id}_cells.h5ad"))

    print(f"✅ Finished Bin2Cell (no destripe) for {sample_id}. Output saved to: {sample_output_dir}")

