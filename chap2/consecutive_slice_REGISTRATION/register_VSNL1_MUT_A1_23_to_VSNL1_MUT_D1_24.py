import os
import json
import random
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk
from skimage.transform import rotate
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
from skimage.morphology import remove_small_objects
from tqdm import tqdm

# === Configuration ===
input_folder = "split_hearts_kmeans_all"
output_dir = "registration_outputs/VSNL1_MUT_cross_23_to_24"
spatial_key = "spatial_cropped_150_buffer"
slice1_filename = "VSNL1_MUT_A1_heart_23.h5ad"
slice2_filename = "VSNL1_MUT_D1_heart_24.h5ad"
downsample_factor = 20
mpp = 1.0 * downsample_factor  # microns per pixel
bin_radius_um = 1.0
rotation_range = np.arange(0, 360, 0.2)
canvas_shape = (3000, 3000)

os.makedirs(output_dir, exist_ok=True)

def rasterize(coords, shape, origin, mpp, radius_um):
    scale = 1.0 / mpp
    canvas = np.zeros(shape, dtype=bool)
    coords_scaled = np.round((coords - origin) * scale).astype(int)
    for x, y in coords_scaled:
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            rr, cc = disk((y, x), radius=radius_um * scale)
            rr = rr[(rr >= 0) & (rr < shape[0])]
            cc = cc[(cc >= 0) & (cc < shape[1])]
            canvas[rr, cc] = True
    canvas = remove_small_objects(canvas, min_size=5)
    return canvas

def register_slices(slice1_path, slice2_path, output_dir):
    print(f"\n📂 Registering {os.path.basename(slice1_path)} ➔ {os.path.basename(slice2_path)}")

    adata1 = sc.read_h5ad(slice1_path)
    adata2 = sc.read_h5ad(slice2_path)

    coords1 = adata1.obsm[spatial_key]
    coords2 = adata2.obsm[spatial_key]

    real_world_min = np.floor(np.vstack([coords1, coords2]).min(axis=0))

    mask1 = rasterize(coords1, canvas_shape, real_world_min, mpp, bin_radius_um)
    mask2 = rasterize(coords2, canvas_shape, real_world_min, mpp, bin_radius_um)

    # Save raster previews
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(mask1, cmap="gray")
    plt.title("Reference")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask2, cmap="gray")
    plt.title("Target")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "raster_masks.png"), dpi=300)
    plt.close()

    # Alignment
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

    # Save score curve
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

    # Random angles
    random_angles = random.sample(list(rotation_range), 2)
    selected_angles = [best_angle] + random_angles

    for idx, angle in enumerate(selected_angles):
        rotated = rotate(mask2.astype(float), angle, resize=False, preserve_range=True)
        shift, error, _ = phase_cross_correlation(mask1.astype(float), rotated, upsample_factor=10)
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

    # Save parameters
    shift_xy_pixels = best_shift[::-1]
    shift_xy_microns = shift_xy_pixels * mpp
    param_dict = {
        "rotation_deg": float(best_angle),
        "shift_pixels_yx": [float(s) for s in best_shift],
        "shift_pixels_xy": [float(s) for s in shift_xy_pixels],
        "shift_microns_xy": [float(s) for s in shift_xy_microns],
        "downsample_factor": downsample_factor,
        "mpp": mpp,
        "raster_origin": [float(x) for x in real_world_min]
    }

    with open(os.path.join(output_dir, "alignment_params.json"), "w") as f:
        json.dump(param_dict, f, indent=2)

    print(f"✅ Alignment complete. Best angle = {best_angle:.2f} degrees.")

# === Run ===
slice1_path = os.path.join(input_folder, slice1_filename)
slice2_path = os.path.join(input_folder, slice2_filename)
register_slices(slice1_path, slice2_path, output_dir)

