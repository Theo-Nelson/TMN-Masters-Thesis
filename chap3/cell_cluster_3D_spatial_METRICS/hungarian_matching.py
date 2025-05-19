# Hungarian matching-based 1-to-1 cell displacement vector analysis (WT only)
import os
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# === Config ===
data_dir = "WT_COM_centered_alignment_cells_unstripped"
output_dir = "WT_hungarian_vector_alignment"
os.makedirs(output_dir, exist_ok=True)

slice_order = [
    "WT_D1_heart_34",
    "WT_D1_heart_35",
    "WT_D1_heart_36",
    "WT_D1_heart_37",
    "WT_A1_heart_38",
    "WT_A1_heart_47",
    "WT_A1_heart_49",
    "WT_A1_heart_50",
]

spatial_key = "spatial_3D"
cluster_key = "leiden_qc"
z_map = {sid: i * 5 for i, sid in enumerate(slice_order)}
pair_order = [f"{slice_order[i]}→{slice_order[i+1]}" for i in range(len(slice_order)-1)]

# === Load all slices
adata_slices = {sid: sc.read(os.path.join(data_dir, f"{sid}_cells_3D_aligned.h5ad")) for sid in slice_order}

# === Per-cluster per-pair optimal assignments using Hungarian matching
match_records = []
vector_plot_dir = os.path.join(output_dir, "matched_vectors")
os.makedirs(vector_plot_dir, exist_ok=True)

for i in range(len(slice_order) - 1):
    sid_a, sid_b = slice_order[i], slice_order[i + 1]
    adata_a, adata_b = adata_slices[sid_a], adata_slices[sid_b]
    z_diff = abs(z_map[sid_b] - z_map[sid_a])
    pair_label = f"{sid_a}→{sid_b}"

    clusters = sorted(set(adata_a.obs[cluster_key].unique()) & set(adata_b.obs[cluster_key].unique()))

    for cluster in clusters:
        mask_a = adata_a.obs[cluster_key] == cluster
        mask_b = adata_b.obs[cluster_key] == cluster
        coords_a = adata_a.obsm[spatial_key][mask_a, :2]
        coords_b = adata_b.obsm[spatial_key][mask_b, :2]

        if coords_a.shape[0] == 0 or coords_b.shape[0] == 0:
            continue

        n_a, n_b = coords_a.shape[0], coords_b.shape[0]
        n = min(n_a, n_b)

        # Compute pairwise distance matrix
        D = cdist(coords_a, coords_b, metric="euclidean")
        row_ind, col_ind = linear_sum_assignment(D)

        if len(row_ind) > n:
            row_ind = row_ind[:n]
            col_ind = col_ind[:n]

        for i_r, i_c in zip(row_ind, col_ind):
            x0, y0 = coords_a[i_r]
            x1, y1 = coords_b[i_c]
            dx = (x1 - x0) / z_diff
            dy = (y1 - y0) / z_diff
            mag = np.sqrt(dx**2 + dy**2)
            angle = np.arctan2(dy, dx)  # radians
            match_records.append({
                "slice_pair": pair_label,
                "cluster": cluster,
                "x": x0,
                "y": y0,
                "dx": dx,
                "dy": dy,
                "magnitude": mag,
                "angle": angle
            })

        # Plot vectors
        plt.figure(figsize=(6, 6))
        for i_r, i_c in zip(row_ind, col_ind):
            x0, y0 = coords_a[i_r]
            x1, y1 = coords_b[i_c]
            dx = (x1 - x0) / z_diff
            dy = (y1 - y0) / z_diff
            plt.arrow(x0, y0, dx, dy, head_width=3, head_length=3, color='blue', alpha=0.5)
        plt.title(f"{pair_label} — Cluster {cluster}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(os.path.join(vector_plot_dir, f"{pair_label.replace('→','_')}_cluster_{cluster}.png"), dpi=300)
        plt.close()

# === Save vector data
match_df = pd.DataFrame(match_records)
match_df.to_csv(os.path.join(output_dir, "matched_vectors_summary.csv"), index=False)

# === Summary plot: mean + std of vector magnitudes per cluster/pair
summary = match_df.groupby(["slice_pair", "cluster"])\
    .agg(mean_mag=("magnitude", "mean"),
         std_mag=("magnitude", "std"),
         count=("magnitude", "count"),
         mean_angle=("angle", "mean"),
         std_angle=("angle", "std"))\
    .reset_index()

# === Plot mean displacement magnitude per cluster (separate plots)
mean_dir = os.path.join(output_dir, "mean_displacement_lineplots")
os.makedirs(mean_dir, exist_ok=True)
summary["slice_pair"] = pd.Categorical(summary["slice_pair"], categories=pair_order, ordered=True)
summary = summary.sort_values(["cluster", "slice_pair"])

for cluster_id, group in summary.groupby("cluster"):
    plt.figure(figsize=(8, 4))
    plt.plot(group["slice_pair"], group["mean_mag"], marker='o')
    plt.fill_between(group["slice_pair"], group["mean_mag"] - group["std_mag"], group["mean_mag"] + group["std_mag"], alpha=0.2)
    plt.xticks(rotation=45)
    plt.ylabel("Mean Displacement (µm/ΔZ)")
    plt.title(f"Cluster {cluster_id} — Mean Displacement")
    plt.tight_layout()
    plt.savefig(os.path.join(mean_dir, f"cluster_{cluster_id}_mean_displacement.png"), dpi=300)
    plt.close()

# === Plot mean vector angle per cluster (separate plots)
angle_dir = os.path.join(output_dir, "angle_direction_lineplots")
os.makedirs(angle_dir, exist_ok=True)

for cluster_id, group in summary.groupby("cluster"):
    plt.figure(figsize=(8, 4))
    plt.plot(group["slice_pair"], np.degrees(group["mean_angle"]), marker='o')
    plt.fill_between(group["slice_pair"],
                     np.degrees(group["mean_angle"] - group["std_angle"]),
                     np.degrees(group["mean_angle"] + group["std_angle"]),
                     alpha=0.2)
    plt.xticks(rotation=45)
    plt.ylabel("Average Angle (degrees)")
    plt.title(f"Cluster {cluster_id} — Direction Consistency")
    plt.tight_layout()
    plt.savefig(os.path.join(angle_dir, f"cluster_{cluster_id}_angle_consistency.png"), dpi=300)
    plt.close()

# === Plot number of vectors per cluster/pair (separate plots)
count_dir = os.path.join(output_dir, "vector_count_lineplots")
os.makedirs(count_dir, exist_ok=True)

for cluster_id, group in summary.groupby("cluster"):
    plt.figure(figsize=(8, 4))
    plt.plot(group["slice_pair"], group["count"], marker='o')
    plt.xticks(rotation=45)
    plt.ylabel("Matched Cell Pairs")
    plt.title(f"Cluster {cluster_id} — Pair Count")
    plt.tight_layout()
    plt.savefig(os.path.join(count_dir, f"cluster_{cluster_id}_pair_counts.png"), dpi=300)
    plt.close()

print("\n✅ Hungarian-matched displacement vectors and direction metrics calculated and visualized.")
