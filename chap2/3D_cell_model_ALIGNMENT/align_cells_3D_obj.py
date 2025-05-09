import os
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import json
from math import radians, cos, sin
import plotly.express as px

# === Config
input_folder = "split_cells_kmeans_all"
com_csv = "HE_segmented_ab3_hole1000_morph18/COM_coordinates.csv"
registration_folder = "registration_outputs"
output_plot_folder = "WT_COM_centered_alignment_cells"
os.makedirs(output_plot_folder, exist_ok=True)
spatial_key = "spatial"
output_3d_key = "spatial_3D"

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
com_df = pd.read_csv(com_csv).set_index("heart_id")

# Initialize
com_cells = {}
com_cell_indices = {}
coords_raw = {}
aligned_coords = {}
adatas = {}

for sid in slices:
    adata = sc.read_h5ad(os.path.join(input_folder, f"{sid}_cells.h5ad"))
    coords = adata.obsm[spatial_key]
    coords_raw[sid] = coords
    adatas[sid] = adata

    # Find COM cell
    com = np.array([com_df.loc[sid, "COM_x"], com_df.loc[sid, "COM_y"]])
    nearest_idx = np.argmin(np.linalg.norm(coords - com, axis=1))
    com_cells[sid] = coords[nearest_idx]
    com_cell_indices[sid] = nearest_idx

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

# Translate by COM cell and save spatial_3D
ref_com_cell = aligned_coords[root][com_cell_indices[root]]

for sid in slices:
    coords = aligned_coords[sid]
    delta = coords[com_cell_indices[sid]] - ref_com_cell
    aligned_3d = np.column_stack((coords - delta, np.full(len(coords), slice_z[sid])))

    adata = adatas[sid]
    adata.obsm[output_3d_key] = aligned_3d

    # ✅ Fix: ensure full prefixed IDs remain
    adata.obs_names = [f"{sid}_{oid}" for oid in adata.obs_names]

    out_path = os.path.join(output_plot_folder, f"{sid}_cells_3D_aligned.h5ad")
    adata.write(out_path)

# Visualize 1% of cells in 3D
points = []
for sid in slices:
    ad = sc.read_h5ad(os.path.join(output_plot_folder, f"{sid}_cells_3D_aligned.h5ad"))
    coords = ad.obsm[output_3d_key]
    idx = np.random.choice(len(coords), size=max(1, len(coords) // 100), replace=False)
    for i in idx:
        points.append({"x": coords[i, 0], "y": coords[i, 1], "z": coords[i, 2], "slice": sid.split("_")[-1]})

df = pd.DataFrame(points)
fig = px.scatter_3d(df, x="x", y="y", z="z", color="slice", opacity=0.6, height=800)
fig.update_traces(marker=dict(size=2))
fig.update_layout(title="WT 3D-registered Cell Positions", scene=dict(zaxis_title="Z (microns)"))
fig.write_html(os.path.join(output_plot_folder, "WT_COM_cell_3D_alignment_confirmed.html"))

print("\n✅ Saved updated 3D spatial adatas for cells and confirmed with interactive 3D plot.")

