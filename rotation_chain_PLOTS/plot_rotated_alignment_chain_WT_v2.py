import os
import numpy as np
import scanpy as sc
import json
import matplotlib.pyplot as plt
from math import radians, cos, sin

# === Config ===
input_folder = "split_hearts_kmeans_all"
registration_folder = "registration_outputs"
output_plot_folder = "rotation_chain_plots_WT_v2"
spatial_key = "spatial_cropped_150_buffer"

# Slices in order
slices = [
    "WT_D1_heart_34",  # root
    "WT_D1_heart_35",
    "WT_D1_heart_36",
    "WT_D1_heart_37",
    "WT_A1_heart_38",
    "WT_A1_heart_47",
    "WT_A1_heart_49",
    "WT_A1_heart_50"
]

# Pairwise registration folders (child -> parent)
pairwise_registrations = {
    ("WT_D1_heart_35", "WT_D1_heart_34"): "WT_D1_WT_D1_heart_34_to_WT_D1_heart_35",
    ("WT_D1_heart_36", "WT_D1_heart_35"): "WT_D1_WT_D1_heart_35_to_WT_D1_heart_36",
    ("WT_D1_heart_37", "WT_D1_heart_36"): "WT_D1_WT_D1_heart_36_to_WT_D1_heart_37",
    ("WT_A1_heart_38", "WT_D1_heart_37"): "WT_cross_37_to_38",
    ("WT_A1_heart_47", "WT_A1_heart_38"): "WT_A1_WT_A1_heart_38_to_WT_A1_heart_47",
    ("WT_A1_heart_49", "WT_A1_heart_47"): "WT_A1_WT_A1_heart_47_to_WT_A1_heart_49",
    ("WT_A1_heart_50", "WT_A1_heart_49"): "WT_A1_WT_A1_heart_49_to_WT_A1_heart_50"
}


os.makedirs(output_plot_folder, exist_ok=True)

# === Load all slices
coords_raw = {}
for s in slices:
    adata = sc.read_h5ad(os.path.join(input_folder, f"{s}.h5ad"))
    coords_raw[s] = adata.obsm[spatial_key]

# === Helper to rotate coordinates
def rotate_coords(coords, angle_deg, center):
    theta = radians(angle_deg)
    R = np.array([[cos(theta), -sin(theta)],
                  [sin(theta),  cos(theta)]])
    centered = coords - center
    rotated = (R @ centered.T).T + center
    return rotated

# === Compute full cumulative rotation for each slice
aligned_coords = {}

# Set reference slice (no rotation needed)
aligned_coords[slices[0]] = coords_raw[slices[0]]

# Now, for each other slice, find the chain of rotations needed
for target_slice in slices[1:]:
    print(f"\n🔄 Processing {target_slice}...")
    
    # Start with raw coordinates
    coords = coords_raw[target_slice]
    
    # Trace back chain
    current = target_slice
    rotations = []
    
    while current != slices[0]:
        # Find parent
        parent = None
        for (child, par), folder in pairwise_registrations.items():
            if child == current:
                parent = par
                reg_folder = folder
                break
        if parent is None:
            raise ValueError(f"Parent not found for {current}")
        
        # Load rotation
        with open(os.path.join(registration_folder, reg_folder, "alignment_params.json"), "r") as f:
            params = json.load(f)
        angle = -params["rotation_deg"]  # Negative rotation
        raster_origin = np.array(params["raster_origin"])
        
        rotations.append((angle, raster_origin))
        
        current = parent  # Go up one level
    
    # Apply all rotations in reverse order (from closest parent up to root)
    for angle, center in reversed(rotations):
        coords = rotate_coords(coords, angle, center)
    
    # Save aligned coordinates
    aligned_coords[target_slice] = coords

# === Plot all slices together
plt.figure(figsize=(16, 16))
colors = plt.cm.tab20(np.linspace(0, 1, len(slices)))

for i, sl in enumerate(slices):
    coords = aligned_coords[sl]
    plt.scatter(coords[:, 0], coords[:, 1], s=8, color=colors[i], label=sl.split("_")[-1], alpha=0.7)
    # Label center of mass
    com = coords.mean(axis=0)
    plt.text(com[0], com[1], sl.split("_")[-1], fontsize=28, ha='center', va='center', weight='bold')

plt.gca().invert_yaxis()
plt.axis("equal")
plt.legend(markerscale=4, fontsize=28)
plt.title("WT Hearts — After Full Cumulative Rotations", fontsize=28)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.tight_layout()
plt.savefig(os.path.join(output_plot_folder, "full_alignment.png"), dpi=300)
plt.show()

print("\n✅ Full hierarchical rotation and plotting complete for WT!")

