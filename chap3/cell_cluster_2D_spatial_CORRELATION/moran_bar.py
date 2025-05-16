import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Config ===
input_path = "cluster_qc_summary_striped/moran_i_per_cluster.csv"
output_dir = "cluster_qc_summary_striped"
os.makedirs(output_dir, exist_ok=True)

# === Load data
df = pd.read_csv(input_path)

# === Sort clusters by Moran's I
df_sorted = df.sort_values("moran_I", ascending=False)

# === Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=df_sorted, x="cluster", y="moran_I", color="cornflowerblue")
plt.title("Moran’s I per Cluster (Spatial Autocorrelation)")
plt.xlabel("Leiden Cluster")
plt.ylabel("Moran’s I")
plt.xticks(rotation=90)
plt.tight_layout()

# === Save
plot_path = os.path.join(output_dir, "barplot_moran_i_per_cluster.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"✅ Saved Moran’s I bar plot to: {plot_path}")

