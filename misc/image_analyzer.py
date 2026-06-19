"""
Spatial Resolution Analysis for Fluorescence Microscopy Images
==============================================================
Methods:
  1. Fourier Ring Correlation (FRC) — global resolution estimate via
     odd/even row half-dataset splitting.
  2. Rolling FRC (rFRC)   — pixel-level local resolution map via a sliding
                            window FRC.  Requires two independent images of
                            the same field (e.g. odd / even frame split).
                            Zhao et al., Light Sci. Appl. 2023.
"""
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf
from numpy import dtype, float64, ndarray
from scipy.signal import savgol_filter


def load_image(path: str) -> np.ndarray:
    """Load image as float64, normalized to [0, 1]."""
    arr = tf.imread(path).astype(np.float64)
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo)


# ============================================================================
# Method 1 — Fourier Ring Correlation
# ============================================================================

def compute_frc(img_a: np.ndarray, img_b: np.ndarray) -> tuple[float, ndarray[tuple[int], dtype[float64]]]:
    """
    Compute the Fourier Ring Correlation between two half-dataset images.

    Both inputs must have the same shape.

    Returns
    -------
    freqs   : 1D array of spatial frequencies (cycles / pixel), range [0, 0.5]
    frc     : 1D array of FRC values in [−1, 1]
    """
    F1 = np.fft.fftshift(np.fft.fft2(img_a))
    F2 = np.fft.fftshift(np.fft.fft2(img_b))

    ny, nx = img_a.shape
    cy, cx = ny // 2, nx // 2
    r = np.sqrt(((np.arange(nx) - cx)[None, :]) ** 2 +
                ((np.arange(ny) - cy)[:, None]) ** 2).astype(int)

    max_r = min(cy, cx)
    frc = np.zeros(max_r)

    for ri in range(max_r):
        mask = r == ri
        f1_r, f2_r = F1[mask], F2[mask]
        num = np.real(np.sum(f1_r * np.conj(f2_r)))
        den = np.sqrt(np.sum(np.abs(f1_r) ** 2) * np.sum(np.abs(f2_r) ** 2))
        frc[ri] = num / den if den > 0 else 0.0

    freqs = np.arange(max_r) / max_r * 0.5  # 0 … 0.5 cycles/pixel
    return freqs, frc


def frc_resolution(freqs: np.ndarray,
                   frc: np.ndarray,
                   threshold: float = 0.5,
                   smooth_window: int = 15) -> ndarray[tuple[Any, ...], dtype[float64]] | None:
    """
    Find the spatial frequency where the smoothed FRC drops below `threshold`.

    Returns resolution in pixels, or None if the threshold is never crossed.
    """
    frc_s = savgol_filter(frc, smooth_window, 3)
    below = np.where(frc_s < threshold)[0]
    if len(below) == 0:
        return None
    f_cross = freqs[below[0]]
    return 1.0 / (f_cross * 2) if f_cross > 0 else None


def global_frc_analysis(img: np.ndarray,
                        thresholds: list[float] | None = None,
                        pixel_size_nm: float | None = None) -> dict:
    """
    Split image into odd/even rows and compute FRC.

    Parameters
    ----------
    img : 2D float array, normalized [0,1]
    thresholds : list of FRC threshold values (default: [0.5, 1/7])
    pixel_size_nm : physical pixel size in nm. If provided, resolutions are
                    returned in nm; otherwise in pixels.

    Returns
    -------
    dict with keys:
        'freqs'       : frequency axis (cycles/pixel)
        'frc'         : raw FRC curve
        'frc_smooth'  : Savitzky-Golay smoothed FRC
        'resolutions' : dict mapping threshold → resolution in nm (or px if
                        pixel_size_nm is None), or None if not crossed
        'unit'        : 'nm' or 'px'
    """
    if thresholds is None:
        thresholds = [0.5, 1 / 7]

    img_a = img[0::2, :]
    img_b = img[1::2, :]
    n = min(img_a.shape[0], img_b.shape[0])
    freqs_ab, frc_ab = compute_frc(img_a[:n], img_b[:n])

    img_c = img[:, 0::2]
    img_d = img[:, 1::2]
    m = min(img_c.shape[1], img_d.shape[1])
    freqs_cd, frc_cd = compute_frc(img_c[:, :n], img_d[:, :n])

    frc = (frc_ab + frc_cd) / 2
    freqs = (freqs_ab + freqs_cd) / 2
    frc_s = savgol_filter(frc, 15, 3)

    scale = pixel_size_nm if pixel_size_nm is not None else 1.0
    unit = "nm" if pixel_size_nm is not None else "px"

    resolutions = {}
    for t in thresholds:
        res_px = frc_resolution(freqs, frc_s, threshold=t)
        resolutions[t] = res_px * scale if res_px is not None else None

    return {
        "freqs": freqs,
        "frc": frc,
        "frc_smooth": frc_s,
        "resolutions": resolutions,
        "unit": unit,
    }


def plot_global_frc(frc_data: dict,
                    pixel_size_nm: float | None,
                    output_path: str) -> None:

    unit = "nm" if pixel_size_nm else "px"
    scale = pixel_size_nm if pixel_size_nm else 1.0

    fig, ax = plt.subplots(figsize=(4, 2.5), dpi=400)

    freqs = frc_data["freqs"]
    ax.plot(freqs, frc_data["frc"], color="steelblue", alpha=0.35, lw=1)
    ax.plot(freqs, frc_data["frc_smooth"], color="steelblue", lw=2, label="FRC (smoothed)")

    colors = {0.5: "darkorange", 1 / 7: "crimson"}
    labels = {0.5: "0.5", 1 / 7: "1/7"}
    frc_unit = frc_data.get("unit", unit)
    for thresh, res in frc_data["resolutions"].items():
        col = colors.get(thresh, "gray")
        lbl = labels.get(thresh, f"{thresh:.3f}")
        ax.axhline(thresh, color=col, ls="--", lw=1.2, alpha=0.8)
        if res is not None:
            ax.axvline(1 / (res / scale * 2), color=col, ls=":", lw=1.2,
                        label=f"FRC({lbl}) = {2*res:.1f} {frc_unit}")
    ax.set_xlabel("Spatial frequency (cycles/pixel)")
    ax.set_ylabel("FRC")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-0.1, 1.05)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=400)
    plt.close(fig)

# ============================================================================
# Method 2 — Rolling / sliding-window FRC
# ============================================================================

def _frc_single_pair(img_a: np.ndarray, img_b: np.ndarray,
                     smooth: bool = True) -> tuple[float, ndarray[tuple[int], dtype[float64]] | Any]:
    """
    Core FRC between two same-shape patches.

    Returns (freqs, frc) where freqs are in cycles/pixel (0 … 0.5).
    """
    F1 = np.fft.fftshift(np.fft.fft2(img_a - img_a.mean()))
    F2 = np.fft.fftshift(np.fft.fft2(img_b - img_b.mean()))

    ny, nx = img_a.shape
    cy, cx = ny // 2, nx // 2
    r = np.hypot(
        np.arange(nx)[None, :] - cx,
        np.arange(ny)[:, None] - cy
    ).astype(int)

    max_r = min(cy, cx)
    frc = np.zeros(max_r)
    for ri in range(max_r):
        m = r == ri
        f1r, f2r = F1[m], F2[m]
        num = np.real(np.sum(f1r * np.conj(f2r)))
        den = np.sqrt(np.sum(np.abs(f1r) ** 2) * np.sum(np.abs(f2r) ** 2))
        frc[ri] = num / den if den > 0 else 0.0

    freqs = np.arange(max_r) / max_r * 0.5
    if smooth and max_r > 15:
        win = min(15, max_r // 4 * 2 + 1)  # must be odd
        frc = savgol_filter(frc, win, 3)
    return freqs, frc


def _frc_resolution(freqs: np.ndarray, frc: np.ndarray,
                    threshold: float = 0.5) -> ndarray[tuple[Any, ...], dtype[float64]] | None:
    """Return resolution in pixels at threshold crossing, or None."""
    below = np.where(frc < threshold)[0]
    if len(below) == 0:
        return None
    f = freqs[below[0]]
    return 1.0 / (f * 2) if f > 0 else None


def rolling_frc(img1: np.ndarray,
                img2: np.ndarray,
                window: int = 64,
                step: int = 16,
                threshold: float = 0.5,
                pixel_size_nm: float | None = None,
                min_signal: float = 0.02) -> dict:
    """
    Compute a local resolution map by applying FRC to overlapping square
    windows slid across two half-dataset images.

    Parameters
    ----------
    img1, img2 : 2D float arrays (same shape, normalized [0,1])
    window     : side length of each sliding window in pixels
    step       : stride in pixels (smaller = finer map, slower)
    threshold  : FRC threshold (0.5 or 1/7)
    pixel_size_nm : physical pixel size; if given, map is in nm
    min_signal : skip windows whose std < min_signal (dark / empty tiles)

    Returns
    -------
    dict with keys
        'res_map_px'  : 2D float array, resolution in pixels (NaN = skipped)
        'res_map'     : same but in nm if pixel_size_nm given, else pixels
        'unit'        : 'nm' or 'px'
        'tile_cy'     : 1D array, row centres of each tile row
        'tile_cx'     : 1D array, col centres of each tile column
        'global_res'  : median non-NaN resolution value (in output unit)
        'coverage'    : fraction of tiles successfully fitted
    """
    assert img1.shape == img2.shape, "img1 and img2 must have the same shape."
    H, W = img1.shape
    half = window // 2

    row_centres = np.arange(half, H - half + 1, step)
    col_centres = np.arange(half, W - half + 1, step)
    nrow, ncol = len(row_centres), len(col_centres)

    res_map_px = np.full((nrow, ncol), np.nan)
    n_valid = 0

    for i, rc in enumerate(row_centres):
        for j, cc in enumerate(col_centres):
            r0, r1 = rc - half, rc + half
            c0, c1 = cc - half, cc + half

            p1 = img1[r0:r1, c0:c1]
            p2 = img2[r0:r1, c0:c1]

            # Skip featureless / empty tiles
            if p1.std() < min_signal or p2.std() < min_signal:
                continue

            freqs, frc = _frc_single_pair(p1, p2, smooth=(window >= 32))
            res_px = _frc_resolution(freqs, frc, threshold)
            if res_px is not None and 1.5 < res_px < window * 0.9:
                res_map_px[i, j] = res_px
                n_valid += 1

    scale = pixel_size_nm if pixel_size_nm is not None else 1.0
    unit = "nm" if pixel_size_nm is not None else "px"
    res_map = res_map_px * scale

    valid = res_map[~np.isnan(res_map)]
    global_res = float(np.median(valid)) if len(valid) > 0 else np.nan
    coverage = n_valid / (nrow * ncol)

    return {
        "res_map_px": res_map_px,
        "res_map": res_map,
        "unit": unit,
        "tile_cy": row_centres,
        "tile_cx": col_centres,
        "global_res": global_res,
        "coverage": coverage,
        "window": window,
        "step": step,
        "threshold": threshold,
        "img_shape": img1.shape,
    }


def plot_rolling_frc(result: dict,
                     img_bg: np.ndarray) -> None:
    """Save a four-panel Rolling FRC figure."""
    H, W = result["img_shape"]
    cy = result["tile_cy"]
    cx = result["tile_cx"]
    res_map = result["res_map"]
    unit = result["unit"]

    # Upsample map to image coordinates for overlay using bilinear interpolation
    from scipy.ndimage import zoom
    zy = H / len(cy)
    zx = W / len(cx)
    # Clamp non-finite before zoom
    map_filled = res_map.copy()
    finite_mask = np.isfinite(map_filled)
    if finite_mask.any():
        map_filled[~finite_mask] = np.nanmedian(map_filled)
    else:
        map_filled[:] = 0
    res_upsampled = zoom(map_filled, (zy, zx), order=1)
    # Re-apply NaN mask (nearest-neighbour to preserve coverage)
    nan_mask_up = zoom((~finite_mask).astype(float), (zy, zx), order=0) > 0.5
    res_upsampled[nan_mask_up] = np.nan

    valid = res_map[np.isfinite(res_map)]
    vmin = np.percentile(valid, 5) if len(valid) > 0 else 0
    vmax = np.percentile(valid, 95) if len(valid) > 0 else 1

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel 1 — raw background image
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img_bg, cmap="gray", interpolation="nearest")
    ax0.set_title("Input image")
    ax0.axis("off")

    # Panel 2 — resolution overlay
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(img_bg, cmap="gray", alpha=0.4, interpolation="nearest")
    im = ax1.imshow(res_upsampled, cmap="jet", alpha=0.7,
                    vmin=vmin, vmax=vmax,
                    extent=[0, W, H, 0], interpolation="bilinear",
                    aspect="auto")
    plt.colorbar(im, ax=ax1, label=f"Resolution ({unit})",
                 fraction=0.046, pad=0.04)
    ax1.set_title(f"Rolling FRC resolution map\n"
                  f"window={result['window']} px, step={result['step']} px, "
                  f"thresh={result['threshold']:.3f}")
    ax1.axis("off")

    # Panel 3 — histogram of local resolution values
    ax2 = fig.add_subplot(gs[1, 0])
    if len(valid) > 0:
        ax2.hist(valid, bins=40, color="steelblue", edgecolor="white", lw=0.4)
        ax2.axvline(np.median(valid), color="crimson", lw=2,
                    label=f"Median {np.median(valid):.1f} {unit}")
        ax2.axvline(np.mean(valid), color="darkorange", lw=2, ls="--",
                    label=f"Mean {np.mean(valid):.1f} {unit}")
        ax2.legend(fontsize=9)
    ax2.set_xlabel(f"Local resolution ({unit})")
    ax2.set_ylabel("Tile count")
    ax2.set_title("Distribution of local FRC resolution")
    ax2.grid(alpha=0.3)

    # Panel 4 — coverage map (which tiles were fitted)
    ax3 = fig.add_subplot(gs[1, 1])
    coverage_map = np.isfinite(res_map).astype(float)
    ax3.imshow(coverage_map, cmap="RdYlGn", vmin=0, vmax=1,
               extent=[cx[0], cx[-1], cy[-1], cy[0]],
               aspect="auto", interpolation="nearest")
    ax3.set_title(f"Tile coverage map\n"
                  f"({result['coverage'] * 100:.1f}% tiles successfully fitted)")
    ax3.set_xlabel("Column (px)")
    ax3.set_ylabel("Row (px)")

    fig.suptitle(f"Rolling FRC  |  global median resolution: "
                 f"{result['global_res']:.1f} {unit}",
                 fontsize=13, y=0.99)

    plt.show()


# ============================================================================
# Method 3 — Laplacian variance (local sharpness map)
# ============================================================================

def laplacian_variance(img: np.ndarray,
                       window: int = 32,
                       step: int = 8,
                       pixel_size_nm: float | None = None) -> dict:
    """
    Compute a local sharpness map using the variance of the Laplacian
    within a sliding window.

    The Laplacian acts as a high-pass filter; its local variance correlates
    with the amount of high-frequency content (sharpness / resolution).
    Sharper regions give higher values; blurred / empty regions give lower
    values.

    This method produces a *relative* sharpness score, not an absolute
    resolution in nm.  Use it to:
      - identify the sharpest sub-region of an image
      - compare relative sharpness between acquisitions / settings
      - detect focus gradients across the field

    Parameters
    ----------
    img           : 2D float array normalized [0, 1]
    window        : sliding window side length in pixels
    step          : stride in pixels
    pixel_size_nm : used only for axis labels in the figure

    Returns
    -------
    dict with keys
        'lap_map'      : full-resolution Laplacian image
        'local_map'    : 2D array of per-tile Laplacian variance (sharpness)
        'local_map_up' : local_map bilinearly upsampled to img dimensions
        'tile_cy'      : tile row centres
        'tile_cx'      : tile col centres
        'global_score' : Laplacian variance of the full image
        'unit'         : 'nm' or 'px'
        'img_shape'    : (H, W)
    """
    H, W = img.shape

    # Full-image Laplacian (4-neighbor discrete kernel)
    lap = (
            np.roll(img, 1, axis=0) + np.roll(img, -1, axis=0) +
            np.roll(img, 1, axis=1) + np.roll(img, -1, axis=1) -
            4 * img
    )
    global_score = float(lap.var())

    half = window // 2
    row_centres = np.arange(half, H - half + 1, step)
    col_centres = np.arange(half, W - half + 1, step)
    nrow, ncol = len(row_centres), len(col_centres)

    local_map = np.zeros((nrow, ncol))
    for i, rc in enumerate(row_centres):
        for j, cc in enumerate(col_centres):
            patch = lap[rc - half:rc + half, cc - half:cc + half]
            local_map[i, j] = float(patch.var())

    # Upsample to image resolution for display
    from scipy.ndimage import zoom
    zy = H / nrow
    zx = W / ncol
    local_map_up = zoom(local_map, (zy, zx), order=1)
    # Clamp to [0, 1] after scaling
    local_map_up = np.clip(local_map_up, 0, None)

    return {
        "lap_map": lap,
        "local_map": local_map,
        "local_map_up": local_map_up,
        "tile_cy": row_centres,
        "tile_cx": col_centres,
        "global_score": global_score,
        "unit": "nm" if pixel_size_nm else "px",
        "img_shape": (H, W),
        "window": window,
        "step": step,
    }


def plot_lapvar(result: dict,
                img: np.ndarray) -> None:
    """Save a four-panel Laplacian variance figure."""
    H, W = result["img_shape"]
    lap_map = result["local_map_up"]

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel 1 — raw image
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img, cmap="gray", interpolation="nearest")
    ax0.set_title("Input image")
    ax0.axis("off")

    # Panel 2 — Laplacian (high-frequency content)
    ax1 = fig.add_subplot(gs[0, 1])
    lap = result["lap_map"]
    vabs = np.percentile(np.abs(lap), 99)
    ax1.imshow(lap, cmap="RdBu_r", vmin=-vabs, vmax=vabs,
               interpolation="nearest")
    ax1.set_title("Laplacian (high-frequency edges)")
    ax1.axis("off")

    # Panel 3 — local sharpness overlay
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(img, cmap="gray", alpha=0.4, interpolation="nearest")
    im = ax2.imshow(lap_map, cmap="hot",
                    vmin=np.percentile(lap_map, 5),
                    vmax=np.percentile(lap_map, 95),
                    extent=[0, W, H, 0],
                    alpha=0.7, aspect="auto", interpolation="bilinear")
    plt.colorbar(im, ax=ax2, label="Laplacian variance (a.u.)",
                 fraction=0.046, pad=0.04)
    ax2.set_title(f"Local sharpness map\n"
                  f"window={result['window']} px, step={result['step']} px")
    ax2.axis("off")

    # Panel 4 — histogram of per-tile scores
    ax3 = fig.add_subplot(gs[1, 1])
    flat = result["local_map"].ravel()
    ax3.hist(flat, bins=40, color="steelblue", edgecolor="white", lw=0.4)
    ax3.axvline(np.median(flat), color="crimson", lw=2,
                label=f"Median {np.median(flat):.4f}")
    ax3.axvline(np.mean(flat), color="darkorange", lw=2, ls="--",
                label=f"Mean {np.mean(flat):.4f}")
    ax3.legend(fontsize=9)
    ax3.set_xlabel("Laplacian variance (a.u.)")
    ax3.set_ylabel("Tile count")
    ax3.set_title("Distribution of local sharpness scores")
    ax3.grid(alpha=0.3)

    fig.suptitle(
        f"Laplacian Variance Sharpness  |  global score: "
        f"{result['global_score']:.5f}",
        fontsize=13, y=0.99
    )

    plt.show()


if __name__ == "__main__":
    # fd = [r"C:\Users\Ruiz\Desktop\manuscript\figures\20241128151717_dot_scanning_wao_crop_recon.tiff",
    #       r"C:\Users\Ruiz\Desktop\manuscript\figures\20241128153927_dot_scanning_woao_crop_recon.tif"]
    fd = [r"C:\Users\Ruiz\Desktop\manuscript\figures\20240605215333_dot_scanning_reconstruction_afterao_crop.tif",
          r"C:\Users\Ruiz\Desktop\manuscript\figures\20240605220227_dot_scanning_reconstruction_beforeao_crop.tif"]
    pixel_size = 40
    img = load_image(fd[1])

    frc_data = global_frc_analysis(img, thresholds=[0.5, 1 / 7], pixel_size_nm=pixel_size)
    plot_global_frc(frc_data, pixel_size_nm=pixel_size,
                    output_path=r"C:\Users\Ruiz\Desktop\20240605220227_dot_scanning_reconstruction_beforeao_crop_frc.png")

    # img1 = img[0::2, :]
    # img2 = img[1::2, :]
    # n = min(img1.shape[0], img2.shape[0])
    # img1, img2 = img1[:n], img2[:n]
    # rfrc_result = rolling_frc(img1, img2,
    #                           window=64,
    #                           step=16,
    #                           threshold=0.16,
    #                           pixel_size_nm=40)
    # bg = (img1 + img2) / 2
    # plot_rolling_frc(rfrc_result, bg)
    #
    # lapres = laplacian_variance(img,
    #                             window=6,
    #                             step=1,
    #                             pixel_size_nm=40)
    # plot_lapvar(lapres, img)
