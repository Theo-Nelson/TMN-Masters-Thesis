import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import os

# === Config ===
samples = ["WT_A1", "WT_D1", "VSNL1_MUT_A1", "VSNL1_MUT_D1"]
striped_dir = "bin2cell_output_unstripped"
destriped_dir = "."
output_dir = "qc_summary_striped_vs_destriped"
os.makedirs(output_dir, exist_ok=True)

excluded_genes = {
    "Alb", "Fabp1", "Fga", "Fgb", "Plg", "Apoc3", "C9", "Itih1", "Pigr",
    "Hamp2", "Apoa5", "Ces3a", "Vnn3", "Trf", "Apoc1", "Ahsg", "Adh1"
}

adata_list = []
gene_sets = []

# === Load, filter, and tag ===
for sample in samples:
    for source in ["striped", "destriped"]:
        path = (
            os.path.join(striped_dir, sample, f"{sample}_cells.h5ad")
            if source == "striped"
            else os.path.join(f"{sample}_bin2cell_output", f"{sample}_cells.h5ad")
        )
        if not os.path.exists(path):
            print(f"⚠️ Missing: {source} {sample}")
            continue

        adata = sc.read(path)
        adata.obs["sample"] = sample
        adata.obs["source"] = source
        adata.obs_names = [f"{sample}_{source}_{oid}" for oid in adata.obs["object_id"]]

        # Calculate QC metrics + filter
        sc.pp.calculate_qc_metrics(adata, inplace=True)
        adata = adata[(adata.obs["n_genes_by_counts"] > 200) & (adata.obs["total_counts"] > 400)].copy()
        print(f"✅ Loaded {sample} ({source}) after QC: shape {adata.shape}")

        gene_sets.append(set(adata.var_names))
        adata_list.append(adata)

# === Find shared genes ===
shared_genes = sorted(set.intersection(*gene_sets) - excluded_genes)
print(f"\n🔗 Shared genes after exclusion: {len(shared_genes)}")

# === Subset to shared genes and combine ===
adata_subs = [a[:, shared_genes].copy() for a in adata_list]
adata_combined = adata_subs[0].concatenate(
    *adata_subs[1:],
    batch_key="batch",
    batch_categories=[f"{a.obs['sample'][0]}_{a.obs['source'][0]}" for a in adata_subs],
    index_unique=None
)
print(f"🧬 Combined shape: {adata_combined.shape}")

# === Joint embedding pipeline ===
sc.pp.normalize_total(adata_combined, target_sum=1e4)
sc.pp.log1p(adata_combined)
sc.pp.highly_variable_genes(adata_combined, n_top_genes=2000, subset=True)
sc.pp.scale(adata_combined)
sc.tl.pca(adata_combined)
sc.pp.neighbors(adata_combined, n_neighbors=15, n_pcs=30)
sc.tl.umap(adata_combined)

# === Save UMAP plot colored by source ===
sc.pl.umap(
    adata_combined,
    color="source",
    title="UMAP – Striped vs Destriped (QC + Exclusion Applied)",
    show=False
)
plt.savefig(os.path.join(output_dir, "umap_striped_vs_destriped.png"), dpi=300, bbox_inches="tight")
plt.close()

# === Save final object
adata_combined.write(os.path.join(output_dir, "adata_joint_striped_destriped.h5ad"))

print("✅ Joint UMAP and object saved.")

