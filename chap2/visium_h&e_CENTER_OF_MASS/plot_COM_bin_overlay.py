import os
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

# === Paths
input_folder = "split_hearts_kmeans_all"
com_csv = "HE_segmented_ab3_hole1000_morph18/COM_coordinates.csv"
output_folder = "HE_COM_bin_overlays"
os.makedirs(output_folder, exist_ok=True)

# Load COMs
com_df = pd.read_csv(com_csv)

for _, row in com_df.iterrows():
    heart_id = row["heart_id"]
    com_x = row["COM_x"]
    com_y = row["COM_y"]

    adata = sc.read_h5ad(os.path.join(input_folder, f"{heart_id}.h5ad"))
    coords = adata.obsm["spatial"]

    # Find nearest bin
    dists = np.linalg.norm(coords - np.array([[com_x, com_y]]), axis=1)
    nearest_idx = np.argmin(dists)
    com_bin = coords[nearest_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(coords[:, 0], coords[:, 1], s=3, c="gray", alpha=0.5)
    ax.plot(com_bin[0], com_bin[1], "ro", label="Closest Bin to COM")
    ax.set_title(f"{heart_id} — Bins + COM bin")
    ax.axis("equal")
    ax.invert_yaxis()
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{heart_id}_bin_overlay.png"), dpi=100)
    plt.close()

print("\n✅ Generated COM bin overlay plots.")

