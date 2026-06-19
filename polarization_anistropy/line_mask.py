"""
Extract the illuminated line from each frame of a line-scan polarization-microscopy
substack and rasterize it as a fixed-width (default 3 px) binary mask. Output is a
stack of masks, one per input frame.

Why this method
---------------
Each frame holds a single, faint, near-horizontal line on a noisy ~uniform
background, broken into segments, and surrounded by exact-zero borders left by the
upstream dual-view registration. Edge detectors would return a *double* response
(one line per side); the accepted tool for a *single* centraline of a curvilinear
structure is RIDGE detection via the image Hessian (Steger / Sato / Frangi family).
See Steger, "An Unbiased Detector of Curvilinear Structures", IEEE TPAMI 1998, and
the Sato/Frangi vesselness filters. The pipeline here is the standard one
(enhance -> localize -> centralize -> link/fit -> rasterize), hardened for this data:

  1. VALID-REGION mask: the registration fill is exact zero. Erode the non-zero
     region inward so the hard zero->signal border cannot create filter halos.
  2. BACKGROUND removal: median filter (edge-preserving, no halo) of the frame with
     invalid pixels filled by the in-frame median; subtract and clip at zero.
  3. RIDGE enhancement: Sato tubeness at scales matched to the measured line width
     (FWHM ~8 px -> sigmas ~2-4 px), giving a single bright response on the line.
  4. LINE-BAND localisation: the line is near-horizontal, so the ridge response has
     a clear peak in its row-energy profile. Find that band, then estimate the
     centraline only within a +/- window around it -> immune to scattered noise
     ridges elsewhere in the column.
  5. SUB-PIXEL centraline: per-column intensity-weighted centroid of the ridge
     response inside the band, with a per-column confidence (summed response).
  6. ROBUST FIT + LINKING: a confidence-weighted low-order polynomial through the
     confident columns bridges the broken segments into one continuous centraline.
     This is the "gap linking" step the literature flags as the hard part.
  7. rasterize at fixed width: stamp the centraline to exactly `line_width` px,
     measured perpendicular to the local line direction, clipped to the valid
     region and (optionally) to the columns that actually carried signal.

"""
import json
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import tifffile
from skimage.filters import sato


# --------------------------------------------------------------------------- #
# Per-frame building blocks
# --------------------------------------------------------------------------- #
def valid_region(frame, erode_iter=8):
    """Mask of usable pixels: the non-zero data region eroded inward so the
    registration zero-fill border cannot leak into filters."""
    m = frame > 0
    m = ndi.binary_fill_holes(m)
    if erode_iter > 0:
        m = ndi.binary_erosion(m, iterations=erode_iter)
    return m


def subtract_background(frame, vmask, med_size=25):
    """Edge-preserving background removal. Invalid pixels are filled with the
    in-frame median before the median filter so no halo forms at the border."""
    if vmask.any():
        med = float(np.median(frame[vmask]))
    else:
        med = float(np.median(frame))
    filled = np.where(vmask, frame, med)
    bg = ndi.median_filter(filled, size=med_size)
    g = frame - bg
    g[~vmask] = 0.0
    return np.clip(g, 0.0, None)


def ridge_response(g, sigmas=(2, 3, 4), presmooth=1.0):
    """Sato tubeness (bright ridges) at the given scales."""
    if presmooth > 0:
        g = ndi.gaussian_filter(g, presmooth)
    r = sato(g, sigmas=list(sigmas), black_ridges=False)
    return r.astype(np.float32)


def locate_band(ridge, smooth=3.0):
    """Row index of the dominant near-horizontal line, from the ridge row-energy."""
    re = ndi.gaussian_filter1d(ridge.sum(axis=1), smooth)
    return int(re.argmax()), re


def column_centerline(ridge, band, half_window=12, rel_thresh=0.15):
    """Per-column sub-pixel centraline (weighted centroid) within +/- half_window
    rows of `band`, plus a per-column confidence (summed ridge response)."""
    H, W = ridge.shape
    lo, hi = max(0, band - half_window), min(H, band + half_window + 1)
    sub = ridge[lo:hi].copy()
    rows = np.arange(lo, hi)[:, None]
    thr = sub.max() * rel_thresh if sub.max() > 0 else 0.0
    sub[sub < thr] = 0.0
    conf = sub.sum(axis=0)  # (W,) per-column confidence
    yc = (sub * rows).sum(axis=0) / (conf + 1e-9)
    return yc, conf


def fit_centerline(yc, conf, shape, deg=2, conf_pct=75.0, min_cols=8):
    """Confidence-weighted polynomial centraline overconfident columns, evaluated
    for every column. Returns (y_of_x, signal_cols_mask)."""
    H, W = shape
    x = np.arange(W)
    pos = conf > 0
    if pos.sum() < min_cols:
        # fall back to the flat band median
        return np.full(W, float(np.median(yc[pos])) if pos.any() else H / 2), pos
    thr = np.percentile(conf[pos], conf_pct)
    sig = conf >= max(thr, 1e-9)
    if sig.sum() < min_cols:
        sig = pos
    c = np.polyfit(x[sig], yc[sig], deg, w=conf[sig])
    y_of_x = np.polyval(c, x)
    return y_of_x.astype(np.float32), sig


def rasterize_line(y_of_x, shape, line_width=3, cols_mask=None, vmask=None,
                   pad_cols=0, continuous=True):
    """Stamp the centraline to an EXACT integer width along each column.

    line_width : total mask thickness in pixels. For each drawn column the mask
                 spans round(y) - (line_width//2) .. + the rest, giving exactly
                 `line_width` set pixels per column (e.g. 3 -> round(y)-1..+1).
    cols_mask  : columns to draw. If `continuous`, the span between the first and
                 last signal column is filled solid (gaps bridged by the fit);
                 otherwise only the signal columns are drawn, widened by `pad_cols`.
    vmask      : restrict the final mask to the valid region.
    """
    H, W = shape
    if cols_mask is None:
        draw_cols = np.ones(W, bool)
    elif continuous:
        xs = np.where(cols_mask)[0]
        draw_cols = np.zeros(W, bool)
        if len(xs):
            draw_cols[xs.min():xs.max() + 1] = True  # solid span, gaps bridged
    else:
        draw_cols = cols_mask.copy()
        if pad_cols > 0:
            draw_cols = ndi.binary_dilation(draw_cols, iterations=int(pad_cols))

    mask = np.zeros((H, W), np.uint8)
    yr = np.rint(y_of_x).astype(int)
    lo_off = line_width // 2  # width 3 -> 1 above, centre, 1 below
    for x in np.where(draw_cols)[0]:
        y0 = yr[x] - lo_off
        y1 = y0 + line_width  # exclusive -> exactly line_width rows
        y0c, y1c = max(0, y0), min(H, y1)
        if y1c > y0c:
            mask[y0c:y1c, x] = 255

    if vmask is not None:
        mask[~vmask] = 0
    return mask


# --------------------------------------------------------------------------- #
# Per-frame and stack drivers
# --------------------------------------------------------------------------- #
@dataclass
class LineInfo:
    band: int
    coef_deg: int
    n_signal_cols: int
    col_extent: tuple
    centerline: np.ndarray  # (W,) fitted row per column
    confidence: np.ndarray  # (W,) per-column confidence
    signal_cols: np.ndarray  # (W,) bool
    tilt_deg: float
    mask_pixels: int


def extract_line_mask(frame, *, line_width=3, erode_iter=8, med_size=25,
                      sigmas=(2, 3, 4), presmooth=1.0, half_window=12,
                      rel_thresh=0.15, fit_deg=2, conf_pct=75.0,
                      restrict_to_signal=True, continuous=True, pad_cols=6):
    """Extract a fixed-width line mask from one frame. Returns (mask uint8, LineInfo).

    restrict_to_signal : limit the line to the column span that carried signal
                         (True) rather than the whole frame width (False).
    continuous         : within that span, draw a solid line bridging noise gaps
                         (True) or only the confident columns widened by pad_cols
                         (False).
    """
    frame = np.asarray(frame, np.float32)
    H, W = frame.shape
    vmask = valid_region(frame, erode_iter)
    g = subtract_background(frame, vmask, med_size)
    ridge = ridge_response(g, sigmas, presmooth)
    ridge[~vmask] = 0.0

    band, _ = locate_band(ridge)
    yc, conf = column_centerline(ridge, band, half_window, rel_thresh)
    y_of_x, sig = fit_centerline(yc, conf, (H, W), fit_deg, conf_pct)

    cols_mask = sig if restrict_to_signal else None
    mask = rasterize_line(y_of_x, (H, W), line_width, cols_mask, vmask,
                          pad_cols, continuous)

    xs = np.where(sig)[0]
    extent = (int(xs.min()), int(xs.max())) if len(xs) else (0, 0)
    # tilt from a straight fit over signal columns
    if sig.sum() >= 2:
        c1 = np.polyfit(np.arange(W)[sig], y_of_x[sig], 1, w=conf[sig] + 1e-9)
        tilt = float(np.degrees(np.arctan(c1[0])))
    else:
        tilt = 0.0
    info = LineInfo(band=band, coef_deg=fit_deg, n_signal_cols=int(sig.sum()),
                    col_extent=extent, centerline=y_of_x, confidence=conf,
                    signal_cols=sig, tilt_deg=round(tilt, 3),
                    mask_pixels=int((mask > 0).sum()))
    return mask, info


def extract_line_mask_stack(stack, **params):
    """Apply extract_line_mask to every frame. Returns (mask_stack uint8, [LineInfo])."""
    masks = np.zeros(stack.shape, np.uint8)
    infos = []
    for i in range(stack.shape[0]):
        m, info = extract_line_mask(stack[i], **params)
        masks[i] = m
        infos.append(info)
    return masks, infos


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _norm(x, lo=1, hi=99.9):
    a, b = np.percentile(x, [lo, hi])
    return np.clip((x - a) / (b - a + 1e-9), 0, 1)


def plot_line_masks(stack, mask_stack, infos, show=True, max_cols=3):
    """Per-frame overlay of the extracted line mask (red) on the raw frame."""
    F = stack.shape[0]
    ncol = min(max_cols, F)
    nrow = int(np.ceil(F / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 2.8 * nrow), squeeze=False)
    for i in range(nrow * ncol):
        r, c = divmod(i, ncol)
        a = ax[r][c]
        a.set_xticks([]);
        a.set_yticks([])
        if i >= F:
            a.axis("off");
            continue
        base = _norm(stack[i])
        rgb = np.dstack([base, base, base])
        m = mask_stack[i] > 0
        rgb[m] = [1.0, 0.15, 0.15]
        a.imshow(rgb)
        info = infos[i]
        a.set_title(f"frame {i}  row≈{info.band}  tilt {info.tilt_deg:+.2f}°  "
                    f"{info.n_signal_cols} cols", fontsize=9)
    fig.suptitle(f"Extracted {mask_stack[0].max() and ''}line masks "
                 f"(width set per call)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if show:
        plt.show()
    return fig


def save_outputs(mask_stack, infos, out_dir, base, stack=None):
    os.makedirs(out_dir, exist_ok=True)
    tifffile.imwrite(os.path.join(out_dir, f"{base}_line_masks.tif"), mask_stack)
    meta = [dict(frame=i, band=info.band, tilt_deg=info.tilt_deg,
                 n_signal_cols=info.n_signal_cols, col_extent=list(info.col_extent),
                 mask_pixels=info.mask_pixels) for i, info in enumerate(infos)]
    with open(os.path.join(out_dir, f"{base}_line_masks.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    if stack is not None:
        fig = plot_line_masks(stack, mask_stack, infos, show=False)
        fig.savefig(os.path.join(out_dir, f"{base}_overlay.png"), dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    fd_v = r""
    img_stack_v = tifffile.imread(fd_v)

    fd_h = r""
    img_stack_h = tifffile.imread(fd_h)

    mask_stack = np.zeros_like(img_stack_h)

    for i in range(img_stack_h.shape[0]):
        msk_v, info_v = extract_line_mask(img_stack_v[i], line_width=5, continuous=False)
        msk_h, info_h = extract_line_mask(img_stack_h[i], line_width=5, continuous=False)
        mask_stack[i] = msk_v & msk_h

    tifffile.imwrite(r"",
                     mask_stack.astype(float))
