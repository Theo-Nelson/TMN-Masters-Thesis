import os
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

# === Config ===
samples = ["WT_A1", "WT_D1", "VSNL1_MUT_A1", "VSNL1_MUT_D1"]
base_dir = "bin2cell_output_unstripped"
input_merged = "summary_plots/qc_filtered_unstripped/adata_qc_filtered_clustered.h5ad"
mpp = 0.2
img_key = f"{mpp}_mpp_150_buffer"
basis = "spatial_cropped_150_buffer"

# === Load merged, clustered object
adata_combined = sc.read(input_merged)
adata_combined.obs["array_row_r"] = adata_combined.obs["array_row"].round(2)
adata_combined.obs["array_col_r"] = adata_combined.obs["array_col"].round(2)

# === Plot per-sample clusters
for sample_id in samples:
    print(f"\n🗺 Plotting clusters for {sample_id}...")
    path = f"{base_dir}/{sample_id}/{sample_id}_cells.h5ad"
    adata_sample = sc.read(path)

    # Fix spatial key if needed
    if "spatial" in adata_sample.uns and sample_id not in adata_sample.uns["spatial"]:
        old_key = list(adata_sample.uns["spatial"].keys())[0]
        adata_sample.uns["spatial"][sample_id] = adata_sample.uns["spatial"].pop(old_key)
        print(f"🔧 Reassigned spatial key from '{old_key}' to '{sample_id}'")

    # Round for merging
    adata_sample.obs["array_row_r"] = adata_sample.obs["array_row"].round(2)
    adata_sample.obs["array_col_r"] = adata_sample.obs["array_col"].round(2)

    # Merge leiden_qc
    cluster_df = adata_combined[adata_combined.obs["sample"] == sample_id].obs[
        ["array_row_r", "array_col_r", "leiden_qc"]
    ]
    merged = pd.merge(
        adata_sample.obs.reset_index(),
        cluster_df,
        on=["array_row_r", "array_col_r"],
        how="left"
    ).drop_duplicates(subset="index", keep="first").set_index("index")

    # Assign cluster labels
    adata_sample.obs["leiden_qc"] = merged["leiden_qc"].astype(str).fillna("NA")
    adata_sample.obs["leiden_qc"] = adata_sample.obs["leiden_qc"].astype("category")  # Required for plotting

    # Create output folder
    plot_dir = os.path.join(base_dir, sample_id, "spatial_qc_filtered_clusters")
    os.makedirs(plot_dir, exist_ok=True)

    # Plot per cluster
    for cluster in sorted(adata_sample.obs["leiden_qc"].cat.categories):
        if cluster == "NA":
            continue
        sub = adata_sample[adata_sample.obs["leiden_qc"] == cluster]
        if sub.n_obs == 0:
            continue
        sc.pl.spatial(
            sub,
            color="leiden_qc",
            img_key=img_key,
            basis=basis,
            title=f"{sample_id} — Cluster {cluster}",
            show=False,
            legend_loc=None
        )
        out_path = os.path.join(plot_dir, f"{sample_id}_cluster_{cluster}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"✅ Saved plots to: {plot_dir}")

