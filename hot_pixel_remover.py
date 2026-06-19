from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tif
from scipy.ndimage import median_filter, label

# =========================
# Settings
# =========================

spatial_size = 3  # 3 or 5
replace_size = 3  # 3 or 5

spatial_k = 4.0  # lower = more aggressive
temporal_k = 4.0  # lower = more aggressive
persistence_fraction = 0.40

max_component_size = 4  # keep only isolated/small hot-pixel clusters


# =========================
# Functions
# =========================

def robust_sigma(x):
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    sigma = 1.4826 * mad

    if sigma == 0 or not np.isfinite(sigma):
        sigma = np.nanstd(x)

    return med, sigma


def ensure_tyx(stack):
    if stack.ndim == 2:
        stack = stack[None, :, :]

    if stack.ndim != 3:
        raise ValueError(
            f"Expected image shape (T, Y, X), but got {stack.shape}. "
            "Please split channel/z dimensions first."
        )

    return stack


def sample_frames(stack, max_frames=300):
    t = stack.shape[0]

    if t <= max_frames:
        return stack.astype(np.float32), np.arange(t)

    idx = np.linspace(0, t - 1, max_frames).astype(int)
    return stack[idx].astype(np.float32), idx


def remove_large_components(mask, max_component_size=4):
    labeled, n = label(mask)
    clean = np.zeros_like(mask, dtype=bool)

    for i in range(1, n + 1):
        component = labeled == i

        if component.sum() <= max_component_size:
            clean |= component

    return clean


def detect_hot_pixels_from_stack(
        stack,
        max_detection_frames=300,
        spatial_size=3,
        spatial_k=8.0,
        temporal_k=8.0,
        persistence_fraction=0.60,
        max_component_size=4,
):
    stack = ensure_tyx(stack)
    sample, used_idx = sample_frames(stack, max_detection_frames)

    print(f"Using {sample.shape[0]} frames for detection.")

    # Temporal median suppresses transient biological signal
    temporal_median = np.median(sample, axis=0)

    # Compare each pixel with local neighborhood
    local_median_static = median_filter(
        temporal_median,
        size=spatial_size,
        mode="reflect",
    )

    static_residual = temporal_median - local_median_static
    static_med, static_sigma = robust_sigma(static_residual)

    static_candidates = static_residual > (
            static_med + spatial_k * static_sigma
    )

    # Persistence detection frame by frame
    local_median_stack = median_filter(
        sample,
        size=(1, spatial_size, spatial_size),
        mode="reflect",
    )

    residual_stack = sample - local_median_stack

    persistent_hits = np.zeros_like(sample, dtype=bool)

    for i in range(sample.shape[0]):
        frame_residual = residual_stack[i]
        med_i, sigma_i = robust_sigma(frame_residual)

        persistent_hits[i] = frame_residual > (
                med_i + temporal_k * sigma_i
        )

    persistence = persistent_hits.mean(axis=0)

    persistent_candidates = persistence >= persistence_fraction

    hot_mask = static_candidates & persistent_candidates

    # Keep only isolated hot pixels or very small clusters
    hot_mask = remove_large_components(
        hot_mask,
        max_component_size=max_component_size,
    )

    qc = {
        "temporal_median": temporal_median,
        "static_residual": static_residual,
        "persistence": persistence,
        "used_frame_indices": used_idx,
    }

    return hot_mask, qc


def detect_hot_pixels_from_dark_stack(
        dark_stack,
        spatial_size=3,
        spatial_k=8.0,
        max_component_size=4,
):
    dark_stack = ensure_tyx(dark_stack).astype(np.float32)

    dark_median = np.median(dark_stack, axis=0)

    local_dark = median_filter(
        dark_median,
        size=spatial_size,
        mode="reflect",
    )

    residual = dark_median - local_dark
    med, sigma = robust_sigma(residual)

    hot_mask = residual > (med + spatial_k * sigma)

    hot_mask = remove_large_components(
        hot_mask,
        max_component_size=max_component_size,
    )

    qc = {
        "dark_median": dark_median,
        "dark_residual": residual,
    }

    return hot_mask, qc


def correct_hot_pixels(stack, hot_mask, replacement_size=3):
    stack = ensure_tyx(stack)

    corrected = np.empty_like(stack)

    for t in range(stack.shape[0]):
        frame = stack[t]

        replacement = median_filter(
            frame,
            size=replacement_size,
            mode="reflect",
        )

        out = frame.copy()
        out[hot_mask] = replacement[hot_mask]

        corrected[t] = out

    return corrected


def replace_hot_pixels(stack, hot_mask):
    """
    Replace only hot pixels using the median of the 8 surrounding pixels,
    excluding the center pixel.
    """

    stack = ensure_tyx(stack)

    if hot_mask.shape != stack.shape[1:]:
        raise ValueError(
            f"hot_mask shape {hot_mask.shape} does not match image shape {stack.shape[1:]}"
        )

    corrected = stack.copy()

    y, x = np.where(hot_mask)

    if len(y) == 0:
        print("No hot pixels found.")
        return corrected

    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ]

    for t in range(stack.shape[0]):
        frame = stack[t]

        # Pad image so edge hot pixels can also be corrected
        padded = np.pad(frame, pad_width=1, mode="reflect")

        yp = y + 1
        xp = x + 1

        neighbors = np.stack(
            [
                padded[yp + dy, xp + dx]
                for dy, dx in neighbor_offsets
            ],
            axis=0,
        )

        replacement_values = np.median(neighbors, axis=0)

        corrected[t, y, x] = replacement_values.astype(stack.dtype)

    return corrected


def save_qc_figures(hot_mask, qc, output_prefix):
    output_prefix = Path(output_prefix)

    plt.figure(figsize=(6, 6))
    plt.imshow(hot_mask, cmap="gray")
    plt.title(f"Detected hot pixels: {hot_mask.sum()}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(output_prefix) + "_hot_pixel_mask.png", dpi=200)
    plt.close()

    if "static_residual" in qc:
        plt.figure(figsize=(6, 5))
        plt.imshow(qc["static_residual"], cmap="gray")
        plt.title("Temporal-median local residual")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(str(output_prefix) + "_static_residual.png", dpi=200)
        plt.close()

    if "persistence" in qc:
        plt.figure(figsize=(6, 5))
        plt.imshow(qc["persistence"], cmap="magma", vmin=0, vmax=1)
        plt.title("Persistence fraction")
        plt.colorbar(label="Fraction of sampled frames")
        plt.tight_layout()
        plt.savefig(str(output_prefix) + "_persistence.png", dpi=200)
        plt.close()


def run_hot_pixel_removal(img_stack):
    stack = ensure_tyx(img_stack)
    original_dtype = stack.dtype
    hot_mask, qc = detect_hot_pixels_from_stack(stack,
                                                max_detection_frames=max_detection_frames,
                                                spatial_size=spatial_size,
                                                spatial_k=spatial_k,
                                                temporal_k=temporal_k,
                                                persistence_fraction=persistence_fraction,
                                                max_component_size=max_component_size)
    corrected = replace_hot_pixels(stack, hot_mask)
    return corrected.astype(original_dtype, copy=False), hot_mask.astype(float)


if __name__ == '__main__':

    input_tif = r""
    output_tif = r""
    dark_tif = None

    max_detection_frames = 300

    print("Reading image stack...")
    stack = tif.imread(input_tif)
    stack = ensure_tyx(stack)

    original_dtype = stack.dtype

    print("Input shape:", stack.shape)
    print("Input dtype:", original_dtype)

    if dark_tif is not None:
        print("Using dark stack for hot-pixel detection...")
        dark_stack = tif.imread(dark_tif)

        hot_mask, qc = detect_hot_pixels_from_dark_stack(
            dark_stack,
            spatial_size=spatial_size,
            spatial_k=spatial_k,
            max_component_size=max_component_size,
        )

    else:
        print("Using fluorescence stack itself for hot-pixel detection...")

        hot_mask, qc = detect_hot_pixels_from_stack(
            stack,
            max_detection_frames=max_detection_frames,
            spatial_size=spatial_size,
            spatial_k=spatial_k,
            temporal_k=temporal_k,
            persistence_fraction=persistence_fraction,
            max_component_size=max_component_size,
        )

    n_hot = int(hot_mask.sum())
    total_pixels = hot_mask.size

    print(f"Detected hot pixels: {n_hot}")
    print(f"Fraction of pixels: {100 * n_hot / total_pixels:.5f}%")

    print("Correcting stack...")
    # corrected = correct_hot_pixels(
    #     stack,
    #     hot_mask,
    #     replacement_size=replace_size
    # )
    corrected = replace_hot_pixels(stack, hot_mask)

    corrected = corrected.astype(original_dtype, copy=False)

    unchanged = np.array_equal(
        stack[:, ~hot_mask],
        corrected[:, ~hot_mask],
    )
    print("Non-hot pixels unchanged:", unchanged)

    print("Saving corrected stack...")
    tif.imwrite(
        output_tif,
        corrected,
        imagej=True,
        metadata={"axes": "TYX"},
    )

    output_prefix = Path(output_tif).with_suffix("")
    tif.imwrite(str(output_prefix) + "_hot_pixel_mask.tif", hot_mask.astype(float))
