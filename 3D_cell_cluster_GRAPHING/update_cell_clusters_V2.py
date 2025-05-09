import os
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import plotly.express as px

# === Config
input_folder = "WT_COM_centered_alignment_cells"
combined_path = "summary_plots/qc_filtered_recombined/adata_qc_filtered_clustered.h5ad"
output_3d_key = "spatial_3D"
cluster_key = "leiden_qc"

# Load combined clustered object
print("📂 Loading joint clustered object...")
adata_combined = sc.read_h5ad(combined_path)
combined_index = adata_combined.obs.index

print("🔄 Assigning cluster labels to individual objects...")

for fname in sorted(os.listdir(input_folder)):
    if not fname.endswith("_cells_3D_aligned.h5ad"):
        continue

    fpath = os.path.join(input_folder, fname)
    ad = sc.read_h5ad(fpath)

    # Strip prefix: WT_A1_heart_38_WT_A1_123 → WT_A1_123
    clean_index = [x.split("_", maxsplit=4)[-1] for x in ad.obs_names]
    print(clean_index)
    # Assign cluster labels with fallback
    cluster_labels = []
    for cid in clean_index:
        if cid in adata_combined.obs.index:
            cluster_labels.append(adata_combined.obs.at[cid, cluster_key])
        else:
            cluster_labels.append("NA")

    ad.obs[cluster_key] = cluster_labels
    ad.write(fpath)
    print(f"✅ Annotated and saved: {fname}")

# Load all 3D-aligned adatas with clusters
combined = []
for fname in sorted(os.listdir(input_folder)):
    if not fname.endswith("_cells_3D_aligned.h5ad"):
        continue

    ad = sc.read_h5ad(os.path.join(input_folder, fname))
    coords = ad.obsm[output_3d_key]
    clusters = ad.obs[cluster_key].values

    for i in range(len(coords)):
        combined.append({
            "x": coords[i, 0],
            "y": coords[i, 1],
            "z": coords[i, 2],
            "cluster": clusters[i]
        })

# Convert to DataFrame
df = pd.DataFrame(combined)

# Plot each cluster
cluster_dir = os.path.join(input_folder, "cluster_3D_views")
os.makedirs(cluster_dir, exist_ok=True)

for cluster_id in sorted(df["cluster"].unique()):
    df_subset = df[df["cluster"] == cluster_id]
    if df_subset.empty or cluster_id == "NA":
        continue
    fig = px.scatter_3d(df_subset, x="x", y="y", z="z", color_discrete_sequence=["red"], opacity=0.7)
    fig.update_traces(marker=dict(size=2))
    fig.update_layout(title=f"3D view of Cluster {cluster_id}", scene=dict(zaxis_title="Z (microns)"))
    fig.write_html(os.path.join(cluster_dir, f"cluster_{cluster_id}_3D.html"))

print("\n✅ 3D views saved for clusters")

