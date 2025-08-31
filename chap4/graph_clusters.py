#!/usr/bin/env python3
# ---------------------------------------------------------------------
# graph_clusters_by_segmentation.py
#
# • copies `leiden_qc` from the combined object to every *_cells.h5ad
#   by matching rounded slide coordinates (array_row/array_col)
# • draws one colour-coded cell-segmentation overlay per sample
#   with bin2cell.view_cell_labels()
# ---------------------------------------------------------------------

import os
from pathlib import Path
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import bin2cell as b2c

# ============ user paths ============ #
combined_path = "summary_plots/qc_filtered_unstripped/adata_qc_filtered_clustered.h5ad"

samples = {
    "WT_A1": "BIG_TIFF_WT/VisiumHD_002E3_slide2_38-47-49-50.btf",
    "WT_D1": "BIG_TIFF_WT/VisiumHD_007E3_slide1_34-35-36-37.btf",
    "VSNL1_MUT_A1": "BIG_TIFF_VSNL/VisiumHD_Vsnl1_E4005.btf",
    "VSNL1_MUT_D1": "BIG_TIFF_VSNL/VisiumHD_Vsnl1_E4015.btf",
}

base_output_dir = Path("bin2cell_output_unstripped")   # where *_cells.h5ad live
overlay_dir     = Path("cluster_overlays")
overlay_dir.mkdir(exist_ok=True)

# ===== 0. load combined & prep helper columns ===== #
print("\n📥 Loading combined object …")
adata_comb = sc.read(combined_path)          # in-memory; fine for one column
adata_comb.obs["array_row_r"] = adata_comb.obs["array_row"].round(2)
adata_comb.obs["array_col_r"] = adata_comb.obs["array_col"].round(2)
adata_comb.obs["leiden_qc"]   = adata_comb.obs["leiden_qc"].astype("category")
cluster_cats = adata_comb.obs["leiden_qc"].cat.categories
n_clusters   = len(cluster_cats)
print(f"🔢 {n_clusters} clusters detected")

# colour palette
palette = sns.color_palette("tab20")
if n_clusters > 20:
    palette = sns.color_palette("husl", n_clusters)

# ===== 1. loop over samples ===== #
for sample_id in samples:
    print(f"\n🚀 Processing {sample_id}")

    # ---------- paths ---------- #
    sample_dir   = base_output_dir / sample_id
    cells_path   = sample_dir / f"{sample_id}_cells.h5ad"
    stardist_dir = sample_dir / "stardist"
    he_img_path  = stardist_dir / "he.tiff"
    he_npz_path  = stardist_dir / "he.npz"

    if not cells_path.exists():
        raise FileNotFoundError(f"{cells_path} not found")

    # ---------- load per-sample AnnData ---------- #
    cdata = sc.read(cells_path)

    # ---------- fix spatial key (if needed) ---------- #
    if "spatial" in cdata.uns and sample_id not in cdata.uns["spatial"]:
        old = list(cdata.uns["spatial"].keys())[0]
        cdata.uns["spatial"][sample_id] = cdata.uns["spatial"].pop(old)
        print(f"🔧 spatial key '{old}' → '{sample_id}'")

    # ---------- helper coords ---------- #
    cdata.obs["array_row_r"] = cdata.obs["array_row"].round(2)
    cdata.obs["array_col_r"] = cdata.obs["array_col"].round(2)

    # ---------- bring in cluster labels via merge ---------- #
    cluster_df = (
        adata_comb[adata_comb.obs["sample"] == sample_id]
        .obs[["array_row_r", "array_col_r", "leiden_qc"]]
        .copy()
    )

    merged = (
        pd.merge(
            cdata.obs.reset_index(),
            cluster_df,
            on=["array_row_r", "array_col_r"],
            how="left",
        )
        .drop_duplicates(subset="index", keep="first")
        .set_index("index")
    )

    # assign & categorise
    cdata.obs["leiden_qc"] = (
        merged["leiden_qc"]
        .astype("category")
        .cat.set_categories(list(cluster_cats) + ["NA"])
        .fillna("NA")
    )

    # ---------- save updated AnnData ---------- #
    cdata.write(cells_path)        # overwrite with cluster column
    print("💾 cluster labels written back to", cells_path.name)

    # ---------- render overlay ---------- #
    img_rgb, legends = b2c.view_cell_labels(
        image_path=str(he_img_path),
        labels_npz_path=str(he_npz_path),
        cdata=cdata,
        fill_key="leiden_qc",
        border_key=None,
        stardist_normalize=True,
        thicken_border=True,
        cat_cmap=palette,
        fill_label_weight=1,
    )

    out_png = overlay_dir / f"{sample_id}_clusters.png"
    plt.imsave(out_png, img_rgb)
    print("🖼  overlay saved →", out_png)

    # legend
    legends["leiden_qc"].savefig(
        overlay_dir / f"{sample_id}_clusters_legend.png",
        bbox_inches="tight",
    )
    plt.close(legends["leiden_qc"])

print("\n✅ Finished – updated AnnData files and overlays are in:", overlay_dir)

