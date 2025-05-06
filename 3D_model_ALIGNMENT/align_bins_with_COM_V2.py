import os
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import json
from math import radians, cos, sin

# === Config
input_folder = "split_hearts_kmeans_all"
com_csv = "HE_segmented_ab3_hole1000_morph18/COM_coordinates.csv"
registration_folder = "registration_outputs"
output_plot_folder = "WT_COM_centered_alignment"
spatial_key = "spatial"
os.makedirs(output_plot_folder, exist_ok=True)

# WT slices in order
slices = [
    "WT_D1_heart_34",
    "WT_D1_heart_35",
    "WT_D1_heart_36",
    "WT_D1_heart_37",
    "WT_A1_heart_38",
    "WT_A1_heart_47",
    "WT_A1_heart_49",
    "WT_A1_heart_50"
]

# Load COMs and find nearest bin per heart
com_df = pd.read_csv(com_csv).set_index("heart_id")
com_bins = {}
com_bin_indices = {}

coords_raw = {}
aligned_coords = {}

for sid in slices:
    adata = sc.read_h5ad(os.path.join(input_folder, f"{sid}.h5ad"))
    coords = adata.obsm[spatial_key]
    coords_raw[sid] = coords
    com = np.array([com_df.loc[sid, "COM_x"], com_df.loc[sid, "COM_y"]])
    nearest_idx = np.argmin(np.linalg.norm(coords - com, axis=1))
    com_bins[sid] = coords[nearest_idx]
    com_bin_indices[sid] = nearest_idx

# Helper: apply rotation around origin
def rotate_coords(coords, angle_deg, origin):
    theta = radians(angle_deg)
    R = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    return (R @ (coords - origin).T).T + origin

# Apply chained rotations
root = slices[0]
aligned_coords[root] = coords_raw[root]

pairwise_folders = {
    ("WT_D1_heart_35", "WT_D1_heart_34"): "WT_D1_WT_D1_heart_34_to_WT_D1_heart_35",
    ("WT_D1_heart_36", "WT_D1_heart_35"): "WT_D1_WT_D1_heart_35_to_WT_D1_heart_36",
    ("WT_D1_heart_37", "WT_D1_heart_36"): "WT_D1_WT_D1_heart_36_to_WT_D1_heart_37",
    ("WT_A1_heart_38", "WT_D1_heart_37"): "WT_cross_37_to_38",
    ("WT_A1_heart_47", "WT_A1_heart_38"): "WT_A1_WT_A1_heart_38_to_WT_A1_heart_47",
    ("WT_A1_heart_49", "WT_A1_heart_47"): "WT_A1_WT_A1_heart_47_to_WT_A1_heart_49",
    ("WT_A1_heart_50", "WT_A1_heart_49"): "WT_A1_WT_A1_heart_49_to_WT_A1_heart_50"
}

for sid in slices[1:]:
    coords = coords_raw[sid]
    chain = []
    current = sid

    while current != root:
        for (child, parent), folder in pairwise_folders.items():
            if child == current:
                with open(os.path.join(registration_folder, folder, "alignment_params.json")) as f:
                    params = json.load(f)
                angle = -params["rotation_deg"]
                origin = np.array(params["raster_origin"])
                chain.append((angle, origin))
                current = parent
                break

    for angle, origin in reversed(chain):
        coords = rotate_coords(coords, angle, origin)

    aligned_coords[sid] = coords

# Plot 1: Rotated only with COM bin markers
plt.figure(figsize=(16, 16))
colors = plt.cm.tab20(np.linspace(0, 1, len(slices)))
for i, sid in enumerate(slices):
    xy = aligned_coords[sid]
    plt.scatter(xy[:, 0], xy[:, 1], s=6, color=colors[i], label=sid.split("_")[-1], alpha=0.6)
    com_bin = xy[com_bin_indices[sid]]
    plt.plot(com_bin[0], com_bin[1], "ro", markersize=8)
    plt.text(np.mean(xy[:, 0]), np.mean(xy[:, 1]), sid.split("_")[-1], fontsize=20, ha='center', va='center', weight='bold')

plt.gca().invert_yaxis()
plt.axis("equal")
plt.legend(markerscale=2, fontsize=10)
plt.title("WT rotated only — with COM bin markers")
plt.tight_layout()
plt.savefig(os.path.join(output_plot_folder, "WT_rotated_only_COM_bins.png"), dpi=300)
plt.close()

# Plot 2: Translated to align on shared COM bin
ref_com_bin = com_bins[root]
plt.figure(figsize=(16, 16))
for i, sid in enumerate(slices):
    rotated_com_bin = aligned_coords[sid][com_bin_indices[sid]]
    delta = rotated_com_bin - aligned_coords[root][com_bin_indices[root]]
    xy = aligned_coords[sid] - delta
    plt.scatter(xy[:, 0], xy[:, 1], s=6, color=colors[i], label=sid.split("_")[-1], alpha=0.6)
    com_bin_aligned = xy[com_bin_indices[sid]]
    plt.plot(com_bin_aligned[0], com_bin_aligned[1], "ro", markersize=8)
    plt.text(np.mean(xy[:, 0]), np.mean(xy[:, 1]), sid.split("_")[-1], fontsize=20, ha='center', va='center', weight='bold')

plt.gca().invert_yaxis()
plt.axis("equal")
plt.legend(markerscale=2, fontsize=10)
plt.title("WT rotated + translated to align COM bins")
plt.tight_layout()
plt.savefig(os.path.join(output_plot_folder, "WT_COM_bin_aligned.png"), dpi=300)
plt.show()

print("\n✅ Plotted WT hearts before and after COM-bin alignment.")
