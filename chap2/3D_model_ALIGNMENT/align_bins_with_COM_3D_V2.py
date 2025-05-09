import os
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import json
from math import radians, cos, sin
import plotly.express as px

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

# Convert to z microns using numeric slice number
def extract_slice_number(s):
    return int(s.split("_")[-1])

slice_z = {sid: extract_slice_number(sid) * 5 for sid in slices}

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

# Align by COM bin and collect 3D points
ref_com_bin = com_bins[root]
points = []

for sid in slices:
    delta = aligned_coords[sid][com_bin_indices[sid]] - aligned_coords[root][com_bin_indices[root]]
    xy = aligned_coords[sid] - delta
    z = slice_z[sid]

    idx = np.random.choice(len(xy), size=max(1, len(xy) // 100), replace=False)
    for i in idx:
        points.append({"x": xy[i, 0], "y": xy[i, 1], "z": z, "slice": sid.split("_")[-1]})

df = pd.DataFrame(points)
fig = px.scatter_3d(df, x="x", y="y", z="z", color="slice", opacity=0.6, height=800)
fig.update_traces(marker=dict(size=2))
fig.update_layout(title="3D Bin Positions Aligned by COM Bin", scene=dict(zaxis_title="Slice Z (microns)"))
fig.write_html(os.path.join(output_plot_folder, "WT_COM_bin_3D_alignment.html"))

print("\n✅ 3D interactive plot created: WT_COM_bin_3D_alignment.html")

