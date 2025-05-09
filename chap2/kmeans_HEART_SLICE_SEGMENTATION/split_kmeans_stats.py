import os
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# === Configuration ===
samples = ["WT_A1", "WT_D1","VSNL1_MUT_A1","VSNL1_MUT_D1"]
spatial_key = "spatial_cropped_150_buffer"
output_dir = "split_hearts_kmeans_all"
os.makedirs(output_dir, exist_ok=True)

# Initialize dataframe to collect stats
stats_list = []

def split_and_save_hearts(adata, sample_id):
    coords = adata.obsm[spatial_key]
    
    # Run KMeans
    kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
    adata.obs["heart_cluster"] = kmeans.fit_predict(coords)

    for i in range(4):
        heart_mask = adata.obs["heart_cluster"] == i
        heart_adata = adata[heart_mask].copy()
        heart_coords = coords[heart_mask]
        
        # === Save AnnData ===
        out_h5ad = os.path.join(output_dir, f"{sample_id}_heart_{i}.h5ad")
        heart_adata.write(out_h5ad)

        # === Plotting ===
        plt.figure(figsize=(6, 6))
        plt.scatter(coords[:, 0], coords[:, 1], s=1, c="lightgray")
        plt.scatter(heart_coords[:, 0], heart_coords[:, 1], s=1.5, c="red")
        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.title(f"{sample_id} — Heart Cluster {i}")
        plt.tight_layout()
        out_png = os.path.join(output_dir, f"{sample_id}_heart_{i}.png")
        plt.savefig(out_png, dpi=300)
        plt.close()

        # === Compute Statistics ===
        n_bins = heart_coords.shape[0]
        x_mean, y_mean = heart_coords.mean(axis=0)
        x_std, y_std = heart_coords.std(axis=0)
        x_min, y_min = heart_coords.min(axis=0)
        x_max, y_max = heart_coords.max(axis=0)

        stats = {
            "sample_id": sample_id,
            "heart_cluster": i,
            "n_bins": n_bins,
            "x_mean": x_mean,
            "y_mean": y_mean,
            "x_std": x_std,
            "y_std": y_std,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        }
        stats_list.append(stats)

        print(f"✅ Saved: {out_h5ad}, {out_png}")
        print(f"📊 Stats for {sample_id} — Heart Cluster {i}:")
        print(f"   Bins: {n_bins}")
        print(f"   Mean (x, y): ({x_mean:.1f}, {y_mean:.1f})")
        print(f"   Std  (x, y): ({x_std:.1f}, {y_std:.1f})")
        print(f"   Bounds: x[{x_min:.1f}, {x_max:.1f}], y[{y_min:.1f}, {y_max:.1f}]\n")

# === Main loop
for sample_id in samples:
    input_path = f"/Users/theo/Downloads/bin2cell_atlas/{sample_id}_bin2cell_output/{sample_id}_bins.h5ad"
    print(f"\n📂 Loading {sample_id} from {input_path}")
    adata = sc.read_h5ad(input_path)
    split_and_save_hearts(adata, sample_id)

# === Save stats summary
stats_df = pd.DataFrame(stats_list)
csv_path = os.path.join(output_dir, "split_heart_stats.csv")
stats_df.to_csv(csv_path, index=False)
print(f"\n📄 Saved summary stats to: {csv_path}")
print("\n🏁 Done! Split hearts and stats saved to:", output_dir)

