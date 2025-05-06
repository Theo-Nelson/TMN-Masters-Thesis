import os
import numpy as np
import json
import random
import re
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgba2rgb
from skimage.transform import rotate, resize
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
from tqdm import tqdm

# === Configuration ===
input_folder = "HE_segmented_ab3_hole1000_morph18"
output_base = "registration_outputs_HE_masks"
os.makedirs(output_base, exist_ok=True)

rotation_range = np.arange(0, 360, 0.2)

# === Utility: load and downsample full-res masks

def load_and_downsample_masks(heart1, heart2):
    path1 = os.path.join(input_folder, f"{heart1}_mask_fullres.npz")
    path2 = os.path.join(input_folder, f"{heart2}_mask_fullres.npz")
    m1 = np.load(path1)["mask"]
    m2 = np.load(path2)["mask"]
    m1_ds = resize(m1.astype(float), (m1.shape[0] // 20, m1.shape[1] // 20), order=0, anti_aliasing=False).astype(bool)
    m2_ds = resize(m2.astype(float), (m2.shape[0] // 20, m2.shape[1] // 20), order=0, anti_aliasing=False).astype(bool)
    h = max(m1_ds.shape[0], m2_ds.shape[0])
    w = max(m1_ds.shape[1], m2_ds.shape[1])
    padded1 = np.zeros((h, w), dtype=bool)
    padded2 = np.zeros((h, w), dtype=bool)
    padded1[:m1_ds.shape[0], :m1_ds.shape[1]] = m1_ds
    padded2[:m2_ds.shape[0], :m2_ds.shape[1]] = m2_ds
    return padded1, padded2

# === Register downsampled masks

def register_masks(heart1, heart2, output_dir):
    print(f"\n📂 Registering {heart1} ➔ {heart2}")

    mask1, mask2 = load_and_downsample_masks(heart1, heart2)
    os.makedirs(output_dir, exist_ok=True)

    # Show the first registration pair overlay
    if heart1.endswith("_19") and heart2.endswith("_20"):
        plt.figure(figsize=(6, 6))
        plt.imshow(mask1, cmap="gray")
        plt.imshow(mask2, cmap="Reds", alpha=0.4)
        plt.title("Preview: First Pair Mask Overlay")
        plt.axis("off")
        plt.show()

    best_angle = 0
    best_shift = (0, 0)
    best_score = -np.inf
    rotated_best = None
    scores = []

    for angle in tqdm(rotation_range):
        rotated = rotate(mask2.astype(float), angle, resize=False, preserve_range=True)
        shift, error, _ = phase_cross_correlation(mask1.astype(float), rotated, upsample_factor=10)
        shifted = np.fft.ifftn(fourier_shift(np.fft.fftn(rotated), shift)).real
        score = np.sum((shifted > 0.5) & mask1)
        scores.append(score)

        if score > best_score:
            best_score = score
            best_angle = angle
            best_shift = shift
            rotated_best = shifted

    plt.figure(figsize=(8, 5))
    plt.plot(rotation_range, scores, c="blue")
    plt.axvline(best_angle, color="red", linestyle="--", label=f"Best: {best_angle:.1f}°")
    plt.xlabel("Rotation angle (°)")
    plt.ylabel("Overlap score")
    plt.title("Alignment Score vs Rotation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_curve.png"), dpi=300)
    plt.close()

    for idx, angle in enumerate([best_angle] + random.sample(list(rotation_range), 2)):
        rotated = rotate(mask2.astype(float), angle, resize=False, preserve_range=True)
        shift, _, _ = phase_cross_correlation(mask1.astype(float), rotated, upsample_factor=10)
        shifted = np.fft.ifftn(fourier_shift(np.fft.fftn(rotated), shift)).real

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(mask1, cmap="gray")
        plt.title("Reference")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(shifted, cmap="Reds", alpha=0.6)
        plt.imshow(mask1, cmap="gray", alpha=0.5)
        plt.title(f"Target (Rotated {angle:.1f}°)")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"alignment_overlay_{idx}.png"), dpi=300)
        plt.close()

    shift_xy = best_shift[::-1]
    param_dict = {
        "rotation_deg": float(best_angle),
        "shift_pixels_yx": [float(s) for s in best_shift],
        "shift_pixels_xy": [float(s) for s in shift_xy],
    }
    with open(os.path.join(output_dir, "alignment_params.json"), "w") as f:
        json.dump(param_dict, f, indent=2)

    print(f"✅ Alignment complete. Best angle = {best_angle:.2f} degrees.")

# === Get slice groups

def group_heart_series():
    heart_groups = {"VSNL1_MUT_A1": [], "VSNL1_MUT_D1": [], "WT_A1": [], "WT_D1": []}
    for f in sorted(os.listdir(input_folder)):
        if f.endswith("_mask_fullres.npz"):
            for g in heart_groups:
                if f.startswith(g):
                    heart_groups[g].append(f.replace("_mask_fullres.npz", ""))
    for g in heart_groups:
        heart_groups[g].sort(key=lambda x: int(re.search(r"heart_(\d+)", x).group(1)))
    return heart_groups

# === Run registration across heart groups
groups = group_heart_series()

for gname, flist in groups.items():
    for i in range(len(flist) - 1):
        h1 = flist[i]
        h2 = flist[i + 1]
        out_dir = os.path.join(output_base, f"{gname}_{h1}_to_{h2}")
        register_masks(h1, h2, out_dir)

print("\n🎯 Finished registration of downsampled full-res H&E contour masks.")
