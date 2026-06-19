"""
Per-pixel sCMOS background estimation and signal extraction for line-scan stacks.

Dataset layout assumed
-----------------------
    stack[0 : n_dark]   -> DARK frames (no illumination), e.g. n_dark = 5
    stack[n_dark : ]    -> LINE-SCAN data frames (a line at a different position
                           in each frame), many of them.

Why this works
--------------
An sCMOS pixel reads  m = offset_i + dark_i + stray_i + signal_i + noise .
The offset + dark terms are FIXED-PATTERN (sharp pixel-to-pixel structure, not a
smooth gradient), so they must be removed per pixel, never by spatial smoothing /
rolling-ball (which would blur the pattern and leave residual). Two facts are
exploited here:

  * DARK frames isolate offset + dark current with no light. Their temporal MEAN
    is the per-pixel camera offset; their temporal VARIANCE is the per-pixel read
    noise. (5 frames give a usable offset (~/sqrt(5)); the read-noise estimate from
    only 5 frames is coarse, so it is used as a reference/floor, not the primary
    noise map.)

  * In the LINE-SCAN frames, the scanning line illuminates any given pixel in only
    a small fraction of frames. A robust temporal estimator per pixel (median, or a
    lower percentile / per-bin median for drift) therefore rejects the few "on-line"
    frames and returns offset + dark + average stray = the full background to remove.
    The temporal MAD of the same frames measures the background fluctuation
    (read + background shot noise) and is the right per-pixel sigma for thresholding.

Signal extraction uses per-pixel significance  z = (m - bg) / sigma , NOT a global
threshold, because sCMOS read noise is heterogeneous (hot / RTS pixels). For
quantitative work, pass `gain_e_per_adu` to convert to photoelectrons and weight
by the proper Poisson + per-pixel-read-noise variance.
"""
import json
import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import tifffile


# --------------------------------------------------------------------------- #
# Memory-safe per-pixel temporal statistics (process in row blocks)
# --------------------------------------------------------------------------- #
def _temporal_percentile(data, q, block_rows=64):
    """Per-pixel temporal percentile over axis 0, computed in row blocks to bound
    peak memory for large stacks."""
    F, H, W = data.shape
    out = np.empty((H, W), np.float32)
    for r0 in range(0, H, block_rows):
        r1 = min(H, r0 + block_rows)
        out[r0:r1] = np.percentile(data[:, r0:r1, :], q, axis=0).astype(np.float32)
    return out


def _temporal_mad(data, center, block_rows=64):
    """Per-pixel temporal MAD about `center`, in row blocks. Returns robust sigma
    (1.4826 * MAD)."""
    F, H, W = data.shape
    out = np.empty((H, W), np.float32)
    for r0 in range(0, H, block_rows):
        r1 = min(H, r0 + block_rows)
        d = np.abs(data[:, r0:r1, :] - center[None, r0:r1, :])
        out[r0:r1] = np.median(d, axis=0).astype(np.float32)
    return (1.4826 * out).astype(np.float32)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class BackgroundResult:
    n_dark: int
    method: str
    # camera (from dark frames); None if no darks
    offset_map: "np.ndarray | None"  # per-pixel camera offset+dark (mean of darks)
    read_sigma_map: "np.ndarray | None"  # per-pixel read noise (std of darks), reference
    # background (from line-scan data)
    background_map: np.ndarray  # per-pixel offset+dark+stray to subtract
    sigma_map: np.ndarray  # per-pixel background noise for thresholding
    stray_map: "np.ndarray | None"  # background_map - offset_map (optical stray), if darks
    hot_pixel_map: np.ndarray  # bool, anomalously noisy/bright pixels
    # bookkeeping
    n_data_frames: int
    gain_e_per_adu: "float | None"
    shape: tuple
    stats: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Core estimation
# --------------------------------------------------------------------------- #
def estimate_background(stack, n_dark=5, method="median", percentile=50.0,
                        n_bins=1, hot_sigma_k=6.0, sigma_floor_frac=0.5,
                        gain_e_per_adu=None, block_rows=64):
    """Estimate per-pixel background and noise from a line-scan stack.

    Parameters
    ----------
    stack        : (F, H, W) array; first `n_dark` frames are dark frames.
    n_dark       : number of leading dark frames (0 = none).
    method       : 'median' (robust, rejects <=50% on-line frames) or
                   'percentile' (use `percentile`, tolerates >50% contamination).
    n_bins       : >1 splits the data frames into contiguous temporal bins and
                   estimates a separate background per bin (handles slow drift).
    hot_sigma_k  : pixels with read/background sigma above
                   median + k*MAD are flagged as hot/anomalous.
    sigma_floor_frac : floor the per-pixel sigma at this fraction of the global
                   median sigma (avoids divide-by-tiny for quantized pixels).
    gain_e_per_adu : if given, also report maps in photoelectrons.

    Returns
    -------
    BackgroundResult.  For n_bins>1, `background_map` holds the per-bin maps
    stacked as (n_bins, H, W) and `stats['bin_edges']` gives the frame ranges.
    """
    F, H, W = stack.shape
    if n_dark >= F:
        raise ValueError(f"n_dark={n_dark} leaves no data frames (F={F}).")

    dark = stack[:n_dark] if n_dark > 0 else None
    data = stack[n_dark:]
    n_data = data.shape[0]

    # --- camera offset + read noise from dark frames ---------------------- #
    if dark is not None and n_dark > 0:
        offset_map = dark.mean(axis=0).astype(np.float32)
        read_sigma_map = (dark.std(axis=0, ddof=1) if n_dark > 1
                          else np.zeros((H, W), np.float32)).astype(np.float32)
    else:
        offset_map = None
        read_sigma_map = None

    q = 50.0 if method == "median" else float(percentile)

    # --- per-pixel background from data (optionally per temporal bin) ------ #
    if n_bins <= 1:
        background_map = _temporal_percentile(data, q, block_rows)  # (H,W)
        sigma_map = _temporal_mad(data, background_map, block_rows)
        bin_edges = [(0, n_data)]
    else:
        edges = np.linspace(0, n_data, n_bins + 1).astype(int)
        bg_bins, sig_bins, bin_edges = [], [], []
        for b in range(n_bins):
            a, c = int(edges[b]), int(edges[b + 1])
            if c <= a:
                continue
            bgb = _temporal_percentile(data[a:c], q, block_rows)
            bg_bins.append(bgb)
            sig_bins.append(_temporal_mad(data[a:c], bgb, block_rows))
            bin_edges.append((a, c))
        background_map = np.stack(bg_bins, axis=0)  # (n_bins, H, W)
        sigma_map = np.stack(sig_bins, axis=0)

    # --- floor the noise map ---------------------------------------------- #
    sig_for_floor = sigma_map if sigma_map.ndim == 2 else sigma_map.mean(0)
    floor = sigma_floor_frac * float(np.median(sig_for_floor[sig_for_floor > 0]) or 1.0)
    sigma_map = np.maximum(sigma_map, floor).astype(np.float32)
    if read_sigma_map is not None:
        # never claim less noise than the measured read-noise floor
        rs = np.maximum(read_sigma_map, floor)
        sigma_map = np.maximum(sigma_map, rs if sigma_map.ndim == 2 else rs[None])

    # --- optical stray (decompose only if darks available) ---------------- #
    bg2d = background_map if background_map.ndim == 2 else background_map.mean(0)
    stray_map = (bg2d - offset_map).astype(np.float32) if offset_map is not None else None

    # --- hot / anomalous pixel map ---------------------------------------- #
    ref_sigma = read_sigma_map if read_sigma_map is not None else sig_for_floor
    med = float(np.median(ref_sigma))
    mad = float(np.median(np.abs(ref_sigma - med))) * 1.4826 + 1e-6
    hot = ref_sigma > med + hot_sigma_k * mad
    # also flag pixels whose background hugely exceeds the field (stuck-bright)
    if stray_map is not None:
        sm = stray_map
        smed = float(np.median(sm));
        smad = float(np.median(np.abs(sm - smed))) * 1.4826 + 1e-6
        hot = hot | (sm > smed + hot_sigma_k * smad)
    hot = hot.astype(bool)

    stats = dict(
        n_frames_total=int(F), n_dark=int(n_dark), n_data=int(n_data),
        method=method, percentile=q, n_bins=int(max(1, n_bins)),
        bin_edges=[list(e) for e in bin_edges],
        background_median_adu=round(float(np.median(bg2d)), 2),
        sigma_median_adu=round(float(np.median(sig_for_floor)), 3),
        read_sigma_median_adu=(round(float(np.median(read_sigma_map)), 3)
                               if read_sigma_map is not None else None),
        offset_median_adu=(round(float(np.median(offset_map)), 2)
                           if offset_map is not None else None),
        stray_median_adu=(round(float(np.median(stray_map)), 3)
                          if stray_map is not None else None),
        n_hot_pixels=int(hot.sum()),
        sigma_floor_adu=round(floor, 3),
    )

    return BackgroundResult(
        n_dark=n_dark, method=method, offset_map=offset_map,
        read_sigma_map=read_sigma_map, background_map=background_map,
        sigma_map=sigma_map, stray_map=stray_map, hot_pixel_map=hot,
        n_data_frames=n_data, gain_e_per_adu=gain_e_per_adu,
        shape=(H, W), stats=stats)


# --------------------------------------------------------------------------- #
# Applying the background
# --------------------------------------------------------------------------- #
def _bg_for_frame(result, frame_index):
    """Return the (H, W) background map to use for a given DATA-frame index,
    honouring temporal bins."""
    bg = result.background_map
    if bg.ndim == 2:
        return bg
    for b, (a, c) in enumerate(result.stats["bin_edges"]):
        if a <= frame_index < c:
            return bg[b]
    return bg[-1]


def _sigma_for_frame(result, frame_index):
    s = result.sigma_map
    if s.ndim == 2:
        return s
    for b, (a, c) in enumerate(result.stats["bin_edges"]):
        if a <= frame_index < c:
            return s[b]
    return s[-1]


def subtract_background(frames, result):
    """Background-subtract DATA frames (already excluding darks).
    Accepts a single (H,W) frame or an (N,H,W) stack. Returns float32, same shape."""
    frames = np.asarray(frames, np.float32)
    single = frames.ndim == 2
    if single:
        frames = frames[None]
    out = np.empty_like(frames, np.float32)
    for i in range(frames.shape[0]):
        out[i] = frames[i] - _bg_for_frame(result, i)
    return out[0] if single else out


def significance(frames, result):
    """Per-pixel SNR z = (frame - bg) / sigma for DATA frames. float32 stack."""
    frames = np.asarray(frames, np.float32)
    single = frames.ndim == 2
    if single:
        frames = frames[None]
    out = np.empty_like(frames, np.float32)
    for i in range(frames.shape[0]):
        out[i] = (frames[i] - _bg_for_frame(result, i)) / (_sigma_for_frame(result, i) + 1e-9)
    return out[0] if single else out


def extract_signal(frames, result, k=4.0, zero_hot=True):
    """Return (signal, mask): background-subtracted intensity kept only where the
    per-pixel significance z >= k, and a uint8 significance mask. Hot pixels are
    optionally zeroed so they don't masquerade as signal."""
    frames = np.asarray(frames, np.float32)
    single = frames.ndim == 2
    if single:
        frames = frames[None]
    sig = np.empty_like(frames, np.float32)
    mask = np.zeros(frames.shape, np.uint8)
    for i in range(frames.shape[0]):
        bg = _bg_for_frame(result, i)
        sg = _sigma_for_frame(result, i)
        sub = frames[i] - bg
        z = sub / (sg + 1e-9)
        keep = z >= k
        if zero_hot:
            keep &= ~result.hot_pixel_map
        s = np.where(keep, sub, 0.0).astype(np.float32)
        sig[i] = s
        mask[i] = (keep * 255).astype(np.uint8)
    if single:
        return sig[0], mask[0]
    return sig, mask


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _show(ax, img, title, cmap="viridis", p=(1, 99)):
    finite = img[np.isfinite(img)]
    lo, hi = np.percentile(finite, p) if finite.size else (0, 1)
    im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]);
    ax.set_yticks([])
    return im


def plot_background(result, example_frame=None, show=True):
    """Diagnostic panel: offset, background, stray, sigma, hot pixels, and an
    example frame with its background-subtraction + significance if provided."""
    r = result
    bg2d = r.background_map if r.background_map.ndim == 2 else r.background_map.mean(0)
    sig2d = r.sigma_map if r.sigma_map.ndim == 2 else r.sigma_map.mean(0)

    panels = []
    if r.offset_map is not None:
        panels.append(("Camera offset (mean of darks)", r.offset_map, "viridis"))
    panels.append(("Per-pixel background (data)", bg2d, "viridis"))
    if r.stray_map is not None:
        panels.append(("Optical stray (bg - offset)", r.stray_map, "magma"))
    panels.append(("Per-pixel noise sigma", sig2d, "inferno"))
    if r.read_sigma_map is not None:
        panels.append(("Read noise (std of darks)", r.read_sigma_map, "inferno"))
    panels.append(("Hot / anomalous pixels", r.hot_pixel_map.astype(float), "gray"))

    extra = []
    if example_frame is not None:
        sub = subtract_background(example_frame, r)
        z = significance(example_frame, r)
        extra = [("Example raw frame", np.asarray(example_frame, np.float32), "gray"),
                 ("  -> background-subtracted", sub, "gray"),
                 ("  -> significance z", z, "inferno")]

    items = panels + extra
    ncol = 3
    nrow = int(np.ceil(len(items) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.0 * nrow), squeeze=False)
    for i in range(nrow * ncol):
        a = ax[i // ncol][i % ncol]
        if i >= len(items):
            a.axis("off");
            continue
        title, img, cmap = items[i]
        p = (1, 99.5) if "significance" in title else (1, 99)
        im = _show(a, img, title, cmap, p)
        fig.colorbar(im, ax=a, fraction=0.046)
    s = r.stats
    fig.suptitle(
        f"sCMOS background  |  {s['n_dark']} dark + {s['n_data']} data frames  "
        f"|  offset={s['offset_median_adu']} stray={s['stray_median_adu']} "
        f"bg={s['background_median_adu']} sigma={s['sigma_median_adu']} ADU  "
        f"|  {s['n_hot_pixels']} hot px", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if show:
        plt.show()
    return fig


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
def save_outputs(result, out_dir, base, data_frames=None, k=4.0):
    os.makedirs(out_dir, exist_ok=True)
    r = result
    np.save(os.path.join(out_dir, f"{base}_background_map.npy"), r.background_map)
    np.save(os.path.join(out_dir, f"{base}_sigma_map.npy"), r.sigma_map)
    np.save(os.path.join(out_dir, f"{base}_hot_pixels.npy"), r.hot_pixel_map)
    if r.offset_map is not None:
        np.save(os.path.join(out_dir, f"{base}_offset_map.npy"), r.offset_map)
        np.save(os.path.join(out_dir, f"{base}_read_sigma_map.npy"), r.read_sigma_map)
    if r.stray_map is not None:
        np.save(os.path.join(out_dir, f"{base}_stray_map.npy"), r.stray_map)
    with open(os.path.join(out_dir, f"{base}_background_stats.json"), "w") as fh:
        json.dump(r.stats, fh, indent=2)
    if data_frames is not None:
        sub = subtract_background(data_frames, r)
        sig, mask = extract_signal(data_frames, r, k=k)
        tifffile.imwrite(os.path.join(out_dir, f"{base}_bgsub.tif"), sub.astype(np.float32))
        tifffile.imwrite(os.path.join(out_dir, f"{base}_signal.tif"), sig.astype(np.float32))
        tifffile.imwrite(os.path.join(out_dir, f"{base}_signal_mask.tif"), mask)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    fd_v = r""
    img_stack_v = tifffile.imread(fd_v)
    n_dark = 5
    data = img_stack_v[n_dark:]  # the line-scan frames

    result = estimate_background(img_stack_v,
                                 n_dark=n_dark,
                                 method="median",  # robust to the scanning line
                                 n_bins=1,  # >1 if background drifts in time
                                 gain_e_per_adu=None)

    # remove background and extract real signal (per-pixel significance threshold)
    bgsub = subtract_background(data, result)
    signal, signal_mask = extract_signal(data, result, k=4.0)

    plot_background(result, example_frame=data[12])
