# Plot top 100 marker gene scores for each cluster
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Config ===
marker_dir = "summary_plots/qc_filtered_unstripped/top_markers"
output_path = "top100_marker_scores_per_cluster.png"
num_genes = 100

# === Read all marker CSVs
cluster_marker_scores = {}
for fname in sorted(os.listdir(marker_dir)):
    if not fname.startswith("top100_markers_cluster_") or not fname.endswith(".csv"):
        continue
    cluster_id = fname.split("_")[-1].split(".")[0]
    df = pd.read_csv(os.path.join(marker_dir, fname))
    cluster_marker_scores[cluster_id] = df["scores"].values[:num_genes]

# === Prepare DataFrame for plotting
plot_df = pd.DataFrame(cluster_marker_scores)
plot_df["rank"] = range(1, num_genes + 1)
plot_df = plot_df.melt(id_vars="rank", var_name="cluster", value_name="score")

# === Plot
plt.figure(figsize=(14, 8))
plot_df["cluster"] = plot_df["cluster"].astype(int)
plot_df = plot_df.sort_values("cluster")
sns.lineplot(data=plot_df, x="rank", y="score", hue="cluster", palette="tab20", linewidth=1.5)
plt.xlabel("Top marker gene rank")
plt.ylabel("Marker score")
plt.title("Top 100 Marker Gene Scores per Cluster")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print("✅ Saved plot of top 100 marker gene scores per cluster:", output_path)

