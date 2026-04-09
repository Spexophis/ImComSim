"""
=============================================================================
  Fluorescence Microscopy — Image Formation Metrics
  ---------------------------------------------------
  MTF · NPS · NEQ · DQE · Shannon Capacity

  Designed for real fluorescence microscope images (uint16 TIFF) where the
  photon conversion gain g [DN/photon] may not be known a priori.

  Gain estimation strategy (automatic, in priority order):
    1. TIFF / OME-XML / µManager metadata
    2. Paired-frame difference  (two frames of same field)
    3. Multi-level PTC          (several flat images at different intensities)
    4. Single-frame var/mean    (valid only when shot noise >> read noise)
    5. Relative DQE             (fallback — normalized, no absolute units)

  Usage examples:
    python fluoro_metrics.py flat.tif                  # single flat image
    python fluoro_metrics.py flat1.tif flat2.tif       # paired frames
    python fluoro_metrics.py f1.tif f2.tif f3.tif ...  # PTC multi-level

    With an edge ROI for MTF (column offsets into the image):
    python fluoro_metrics.py --edge_roi r0 r1 c0 c1 flat.tif
=============================================================================
"""

import sys
import argparse
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from scipy.signal import windows
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
import tifffile

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TIFF I/O AND METADATA
# ═════════════════════════════════════════════════════════════════════════════

def load_tiff(path):
    """
    Load a TIFF image and return (array uint16, metadata_dict).

    Handles:
      - Plain grayscale TIFF
      - Multi-frame TIFF (returns first frame)
      - OME-TIFF (parses OME-XML for pixel size, channel name)
      - µManager TIFF (parses JSON summary metadata for camera gain)
      - ImageJ hyperstack TIFF
    """
    with tifffile.TiffFile(path) as tif:
        # ── Raw pixel data ─────────────────────────────────────────────────
        data = tif.asarray()

        # Collapse to single 2-D frame if needed
        while data.ndim > 2:
            data = data[0]          # take first z / channel / time
        img = data.astype(np.uint16)

        # ── Metadata ──────────────────────────────────────────────────────
        meta = {}

        # ---- µManager: JSON in ImageDescription tag ----------------------
        try:
            for page in tif.pages:
                desc = page.description
                if desc and "Summary" in desc:
                    d = json.loads(desc)
                    summary = d.get("Summary", d)
                    # µManager stores CameraGain as a direct key or inside channels
                    for key in ("CameraGain", "Gain", "CameraEMGain",
                                "EMGain", "HardwareGain"):
                        if key in summary:
                            meta["gain_dn_per_photon"] = float(summary[key])
                            meta["gain_source"] = f"µManager/{key}"
                            break
                    # Pixel size
                    for key in ("PixelSizeUm", "PixelSize_um"):
                        if key in summary:
                            meta["pixel_size_um"] = float(summary[key])
                    break
        except Exception:
            pass

        # ---- OME-XML embedded in TIFF ------------------------------------
        if "gain_dn_per_photon" not in meta:
            try:
                ome = tif.ome_metadata
                if ome:
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(ome)
                    ns   = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
                    # DetectorSettings/Gain
                    for ds in root.iter("{http://www.openmicroscopy.org/Schemas/OME/2016-06}DetectorSettings"):
                        g = ds.get("Gain") or ds.get("Voltage")
                        if g:
                            meta["gain_dn_per_photon"] = float(g)
                            meta["gain_source"] = "OME-XML/DetectorSettings/Gain"
                            break
                    # Pixels/PhysicalSizeX
                    for px in root.iter("{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels"):
                        ps = px.get("PhysicalSizeX")
                        if ps:
                            meta["pixel_size_um"] = float(ps)
            except Exception:
                pass

        # ---- ImageJ / Fiji metadata -------------------------------------
        if "gain_dn_per_photon" not in meta:
            try:
                ij = tif.imagej_metadata
                if ij:
                    for key in ("Gain", "gain", "CameraGain"):
                        if key in ij:
                            meta["gain_dn_per_photon"] = float(ij[key])
                            meta["gain_source"] = f"ImageJ/{key}"
                            break
            except Exception:
                pass

    return img, meta


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PHOTON TRANSFER / GAIN ESTIMATION
# ═════════════════════════════════════════════════════════════════════════════

def estimate_gain_from_pair(img1, img2, roi=None):
    """
    Paired-frame difference method.

    Theory
    ------
    For two independent frames of the same Poisson source:
        Var(img1 − img2) = 2 * g * μ_photons * g²   →
        Var((img1 − img2)/2) = g²*μ/2 + σ²_read

    More directly:
        Var_diff = Var(img1 − img2) = 2*g*μ_dn + 2*σ²_read_dn
        μ_dn     = (mean(img1) + mean(img2)) / 2

    Rearranging:
        g = Var_diff / (2 * μ_dn)      [valid when σ_read << σ_shot]

    To also recover read noise, use two exposures at different intensities
    (see estimate_gain_ptc).  Here we make the common approximation that
    read noise is included in the variance estimate, so g is slightly
    over-estimated for dim images.  For typical fluorescence (μ > 1000 DN,
    read noise < 5e⁻), the error is < 1%.

    Returns
    -------
    g     : estimated gain (DN / photon)
    sigma_read : estimated read noise (DN)
    """
    H, W = img1.shape
    if roi is None:
        # Use central quarter as flat region
        r0, r1 = H // 4, 3 * H // 4
        c0, c1 = W // 4, 3 * W // 4
    else:
        r0, r1, c0, c1 = roi

    a = img1[r0:r1, c0:c1].astype(np.float64)
    b = img2[r0:r1, c0:c1].astype(np.float64)

    mu         = 0.5 * (a.mean() + b.mean())
    var_diff   = 0.5 * (a - b).var()          # = g*mu + sigma_read²

    if mu < 10:
        raise ValueError("Images are too dark to estimate gain reliably.")

    g          = var_diff / mu                 # DN / photon
    # Rough read noise: for a single dark frame we'd need var_dark, but
    # without one we can only bound it.  Set to zero as conservative default.
    sigma_read = 0.0

    return float(g), float(sigma_read)


def estimate_gain_ptc(images, rois=None):
    """
    Multi-level Photon Transfer Curve (PTC) gain estimation.

    For each image (assumed to be a flat field at a different illumination
    level), extract mean and variance from an ROI.  Fit:

        variance = g * mean + sigma_read²

    to get gain (slope) and read noise variance (intercept).

    Parameters
    ----------
    images : list of 2-D uint16 arrays, each at a different intensity level
    rois   : optional list of (r0,r1,c0,c1) per image; same ROI used for all
             if None

    Returns
    -------
    g          : gain  (DN / photon)
    sigma_read : read noise (DN)
    means      : array of mean DN values per image (for plotting)
    variances  : array of variance DN² values per image
    """
    means, variances = [], []
    H, W = images[0].shape

    for i, img in enumerate(images):
        if rois is not None:
            r0, r1, c0, c1 = rois[i]
        else:
            r0, r1 = H // 4, 3 * H // 4
            c0, c1 = W // 4, 3 * W // 4

        patch = img[r0:r1, c0:c1].astype(np.float64)
        means.append(patch.mean())
        variances.append(patch.var())

    means     = np.array(means)
    variances = np.array(variances)

    # Weighted linear fit (exclude DC offset)
    if len(means) >= 2:
        coeffs    = np.polyfit(means, variances, 1)
        g         = float(coeffs[0])           # slope  = gain
        var_read  = float(coeffs[1])           # intercept = read noise variance
        sigma_read = float(np.sqrt(max(var_read, 0.0)))
    else:
        g          = float(variances[0] / means[0])
        sigma_read = 0.0

    return g, sigma_read, means, variances


def estimate_gain_single_frame(img, flat_roi=None):
    """
    Single-frame var/mean (Shot Noise Limit approximation).

    Assumption: shot noise dominates read noise, i.e.
        σ²_total ≈ g * μ_dn  →  g ≈ σ²_total / μ_dn

    This over-estimates g by  σ²_read / μ_dn  (usually < 5% for bright
    fluorescence images, but can be > 50% for dim samples or high read noise).

    Returns g, and a reliability flag:
        'reliable'   if μ_dn > 5000 DN  (shot noise likely dominates)
        'uncertain'  if 1000 < μ_dn <= 5000
        'unreliable' if μ_dn <= 1000 DN
    """
    H, W = img.shape
    if flat_roi is None:
        flat_roi = (H // 4, 3 * H // 4, W // 4, 3 * W // 4)
    r0, r1, c0, c1 = flat_roi

    patch = img[r0:r1, c0:c1].astype(np.float64)
    mu    = patch.mean()
    var   = patch.var()

    if mu < 10:
        raise ValueError("Signal too low for var/mean gain estimate.")

    g = var / mu

    if   mu > 5000:  reliability = "reliable"
    elif mu > 1000:  reliability = "uncertain"
    else:            reliability = "unreliable"

    return float(g), reliability


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — AUTO FLAT-FIELD ROI FINDER
# ═════════════════════════════════════════════════════════════════════════════

def find_flat_roi(img, tile_size=128, n_candidates=5):
    """
    Automatically locate the most uniform region of a fluorescence image
    for use as the NPS / DQE flat-field.

    Scans non-overlapping tiles and selects the one with the lowest
    coefficient of variation (std/mean), excluding tiles that are too dark
    (< 10th percentile of the image) or saturated (> 99th percentile).

    Returns (r0, r1, c0, c1, cv) for the best tile.
    """
    H, W   = img.shape
    f      = img.astype(np.float64)
    p10    = np.percentile(f, 10)
    p99    = np.percentile(f, 99)

    best_cv   = np.inf
    best_roi  = None

    for r in range(0, H - tile_size + 1, tile_size // 2):
        for c in range(0, W - tile_size + 1, tile_size // 2):
            patch = f[r:r+tile_size, c:c+tile_size]
            mu    = patch.mean()
            if mu < p10 or mu > p99:
                continue
            cv = patch.std() / (mu + 1e-9)
            if cv < best_cv:
                best_cv  = cv
                best_roi = (r, r + tile_size, c, c + tile_size)

    if best_roi is None:
        # Fallback: central tile
        r0 = H // 2 - tile_size // 2
        c0 = W // 2 - tile_size // 2
        best_roi = (r0, r0 + tile_size, c0, c0 + tile_size)
        best_cv  = float('nan')

    return (*best_roi, best_cv)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MTF (slanted edge, same as before)
# ═════════════════════════════════════════════════════════════════════════════

def compute_mtf(img, edge_roi=None, oversample=4):
    """
    Slanted-edge MTF.  If no edge_roi is provided, attempts to auto-detect
    the sharpest edge in the image using Sobel magnitude.
    """
    H, W = img.shape

    if edge_roi is None:
        edge_roi = _auto_detect_edge_roi(img)

    r0, r1, c0, c1 = edge_roi
    roi   = img[r0:r1, c0:c1].astype(np.float64)
    rows, cols = roi.shape
    col_idx    = np.arange(cols, dtype=np.float64)

    # Sub-pixel edge position per row
    grad     = np.gradient(roi, axis=1)
    abs_grad = np.abs(grad)
    edge_pos = np.empty(rows)
    for r in range(rows):
        peak = int(np.argmax(abs_grad[r]))
        lo   = max(0, peak - 6)
        hi   = min(cols, peak + 7)
        w    = abs_grad[r, lo:hi]
        edge_pos[r] = lo + (w * np.arange(len(w))).sum() / (w.sum() + 1e-12)

    # Linear fit
    row_idx  = np.arange(rows, dtype=np.float64)
    coeffs   = np.polyfit(row_idx, edge_pos, 1)
    edge_fit = np.polyval(coeffs, row_idx)

    angle_deg = abs(np.degrees(np.arctan(coeffs[0])))
    if angle_deg < 0.5 or angle_deg > 30:
        warnings.warn(
            f"Edge angle is {angle_deg:.1f} deg. "
            "Slanted-edge MTF works best for 2–15 deg tilts. "
            "Consider specifying a better edge_roi.")

    # Oversampled ESF
    n_bins  = cols * oversample * 2
    esf_sum = np.zeros(n_bins)
    esf_cnt = np.zeros(n_bins, dtype=int)
    centre  = n_bins // 2

    for r in range(rows):
        offsets = col_idx - edge_fit[r]
        bins    = (np.round(offsets * oversample) + centre).astype(int)
        valid   = (bins >= 0) & (bins < n_bins)
        np.add.at(esf_sum, bins[valid], roi[r, valid])
        np.add.at(esf_cnt, bins[valid], 1)

    pop  = np.where(esf_cnt > 0)[0]
    b0, b1 = pop[0], pop[-1] + 1
    es, ec = esf_sum[b0:b1], esf_cnt[b0:b1]
    ok   = ec > 0
    x    = np.arange(len(es), dtype=np.float64)
    esf  = np.interp(x, x[ok], es[ok] / ec[ok])

    # LSF
    lsf  = gaussian_filter(np.gradient(esf), sigma=oversample * 0.6)
    peak = int(np.argmax(np.abs(lsf)))
    half = cols * oversample // 2
    lsf  = lsf[max(0, peak - half): min(len(lsf), peak + half)]

    N    = int(2 ** np.ceil(np.log2(len(lsf))))
    lsf_p = np.zeros(N)
    lsf_p[:len(lsf)] = lsf

    win  = windows.hann(N)
    OTF  = np.fft.fft(lsf_p * win)
    MTF  = np.abs(OTF)
    freq = np.fft.fftfreq(N, d=1.0 / oversample)

    pos  = (freq >= 0) & (freq <= 0.5)
    f_o, m_o = freq[pos], MTF[pos]
    dc = MTF[0] if MTF[0] > 1e-12 else m_o.max()
    return f_o, np.clip(m_o / dc, 0.0, 1.1)


def _auto_detect_edge_roi(img, min_contrast=0.1):
    """
    Scan the image for a region with a strong, relatively straight edge
    suitable for MTF measurement.  Returns (r0, r1, c0, c1).
    """
    from scipy.ndimage import sobel as ndsobel
    H, W = img.shape
    f    = img.astype(np.float64)
    Gx   = ndsobel(f, axis=1)
    Gy   = ndsobel(f, axis=0)
    Gm   = np.sqrt(Gx**2 + Gy**2)

    tile = min(256, H // 2, W // 2)
    best_energy = 0.0
    best_roi    = (0, tile, 0, tile)

    for r in range(0, H - tile + 1, tile // 2):
        for c in range(0, W - tile + 1, tile // 2):
            patch = Gm[r:r+tile, c:c+tile]
            # Prefer regions with high peak gradient but not a uniform field
            energy = patch.max() * (patch > patch.mean()).mean()
            if energy > best_energy:
                best_energy = energy
                best_roi    = (r, r + tile, c, c + tile)

    return best_roi


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — NPS, NEQ, DQE, SHANNON (same equations, see comments below)
# ═════════════════════════════════════════════════════════════════════════════

def compute_nps(img, flat_roi, tile_size=64, overlap=0.5):
    """NPS from a flat-field ROI using Welch's method."""
    r0, r1, c0, c1 = flat_roi
    flat = img[r0:r1, c0:c1].astype(np.float64)

    step  = max(1, int(tile_size * (1.0 - overlap)))
    win2d = np.outer(windows.hann(tile_size), windows.hann(tile_size))
    win2d /= np.sqrt((win2d ** 2).mean())

    accum, count = np.zeros((tile_size, tile_size)), 0
    for r in range(0, flat.shape[0] - tile_size + 1, step):
        for c in range(0, flat.shape[1] - tile_size + 1, step):
            tile   = flat[r:r+tile_size, c:c+tile_size]
            tiled  = (tile - tile.mean()) * win2d
            F      = np.fft.fft2(tiled)
            accum += np.abs(F) ** 2
            count += 1
    if count == 0:
        raise ValueError("No tiles fit in flat-field ROI — choose a larger region.")

    nps2d  = np.fft.fftshift(accum / count) / (tile_size ** 2)
    freq1  = np.fft.fftshift(np.fft.fftfreq(tile_size, d=1.0))
    fu, fv = np.meshgrid(freq1, freq1)
    fr     = np.sqrt(fu ** 2 + fv ** 2)

    n_bins    = tile_size // 2
    edges     = np.linspace(0.0, 0.5, n_bins + 1)
    centers   = 0.5 * (edges[:-1] + edges[1:])
    nps1d     = np.array([
        nps2d[(fr >= edges[i]) & (fr < edges[i+1])].mean()
        if ((fr >= edges[i]) & (fr < edges[i+1])).any() else 0.0
        for i in range(n_bins)
    ])
    return fu, fv, nps2d, centers, nps1d


def compute_neq(freq_mtf, mtf, freq_nps, nps1d, mean_signal_dn):
    """NEQ(f) = S² × MTF²(f) / NPS(f)"""
    fc  = np.linspace(0.002, 0.498, 512)
    mi  = interp1d(freq_mtf, mtf, kind='linear',
                   bounds_error=False, fill_value=0.0)(fc)
    fl  = max(nps1d[nps1d > 0].min() * 0.01, 1.0) if (nps1d > 0).any() else 1.0
    ni  = interp1d(freq_nps, np.maximum(nps1d, fl), kind='linear',
                   bounds_error=False,
                   fill_value=(nps1d[0], nps1d[-1]))(fc)
    return fc, (mean_signal_dn ** 2) * (mi ** 2) / ni


def compute_dqe(freq_neq, neq, q, normalized=False):
    """
    DQE(f) = NEQ(f) / q

    If normalized=True (no gain info), returns DQE / DQE(0)  — the shape
    only, showing how DQE rolls off with frequency.
    """
    if normalized:
        dqe = neq / (neq[0] + 1e-12)
    else:
        dqe = neq / q
    return freq_neq, np.clip(dqe, 0.0, 1.0 if not normalized else None)


def compute_shannon_capacity(freq, dqe, image_size=1024):
    """C = A × ∫ log₂(1+DQE) × 2πf df"""
    integrand = np.log2(1.0 + dqe) * 2.0 * np.pi * freq
    cap_bpx   = float(trapezoid(integrand, freq))
    return cap_bpx * image_size ** 2, cap_bpx, integrand


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline(image_paths, edge_roi=None):

    print("=" * 65)
    print("  Fluorescence Microscopy — Image Formation Metrics")
    print("=" * 65)

    # ── Load images ────────────────────────────────────────────────────────
    images, metas = [], []
    for p in image_paths:
        img, meta = load_tiff(p)
        images.append(img)
        metas.append(meta)
        print(f"\n  Loaded: {p}")
        print(f"    shape={img.shape}  dtype={img.dtype}  "
              f"min={img.min()}  max={img.max()}  mean={img.mean():.1f}")
        if meta:
            print(f"    Metadata: {meta}")

    img_primary = images[0]
    H, W = img_primary.shape

    # ── Gain estimation ────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  GAIN ESTIMATION")
    print("─" * 50)

    gain      = None
    gain_src  = None
    sigma_read = 0.0
    ptc_data   = None

    # Tier 1: metadata
    for meta in metas:
        if "gain_dn_per_photon" in meta:
            gain     = meta["gain_dn_per_photon"]
            gain_src = meta["gain_source"]
            break

    # Tier 2: paired frames
    if gain is None and len(images) == 2:
        try:
            gain, sigma_read = estimate_gain_from_pair(images[0], images[1])
            gain_src = "paired-frame difference (var/mean of diff)"
        except Exception as e:
            print(f"  [!] Pair method failed: {e}")

    # Tier 3: multi-level PTC
    if gain is None and len(images) >= 3:
        try:
            gain, sigma_read, ptc_means, ptc_vars = estimate_gain_ptc(images)
            ptc_data = (ptc_means, ptc_vars)
            gain_src = "multi-level Photon Transfer Curve (linear fit)"
        except Exception as e:
            print(f"  [!] PTC method failed: {e}")

    # Tier 4: single frame var/mean
    if gain is None:
        try:
            flat_info = find_flat_roi(img_primary)
            flat_roi_est = flat_info[:4]
            gain, reliability = estimate_gain_single_frame(
                img_primary, flat_roi=flat_roi_est)
            gain_src = f"single-frame var/mean  [reliability: {reliability}]"
            if reliability == "unreliable":
                print(f"  [!] WARNING: signal is too dim for reliable var/mean gain "
                      f"estimation.  Consider using paired frames or a brighter image.")
        except Exception as e:
            print(f"  [!] Single-frame method failed: {e}")

    # Tier 5: normalized (no gain)
    normalized_dqe = (gain is None)
    if normalized_dqe:
        gain_src = "FALLBACK: relative DQE (DQE normalised to DQE(0)=1)"
        gain     = 1.0   # dummy for downstream code
        print("  [!] No gain could be estimated. DQE will be relative only.")
    else:
        sigma_read_dn = sigma_read
        print(f"  Gain  g = {gain:.4f} DN/photon   [{gain_src}]")
        if sigma_read > 0:
            print(f"  Read noise σ_read = {sigma_read:.2f} DN  "
                  f"(≈ {sigma_read/gain:.2f} e⁻)")

    # ── Auto flat-field ROI ────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  FLAT-FIELD ROI")
    print("─" * 50)

    flat_r0, flat_r1, flat_c0, flat_c1, cv = find_flat_roi(img_primary)
    print(f"  Best flat tile: rows [{flat_r0}:{flat_r1}]  cols [{flat_c0}:{flat_c1}]")
    print(f"  Coefficient of variation = {cv:.4f}  "
          f"(lower = more uniform;  < 0.05 is good for fluorescence)")
    flat_roi = (flat_r0, flat_r1, flat_c0, flat_c1)

    # ── MTF ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  MTF  (slanted-edge method)")
    print("─" * 50)

    if edge_roi is not None:
        print(f"  Using user-specified edge ROI: {edge_roi}")
    else:
        print("  No edge ROI specified — auto-detecting sharpest edge region.")
        print("  TIP: for best results, image a fluorescent bead, sharp cell boundary,")
        print("       or a USAF target and specify --edge_roi r0 r1 c0 c1")

    try:
        freq_mtf, mtf = compute_mtf(img_primary, edge_roi=edge_roi, oversample=4)
        mtf_ok = True
    except Exception as e:
        print(f"  [!] MTF failed: {e}")
        freq_mtf = np.linspace(0, 0.5, 256)
        mtf      = np.ones_like(freq_mtf)
        mtf_ok   = False

    def find_level(target, f, m):
        try:
            idx = np.where(m <= target)[0]
            if not len(idx):
                return float('nan')
            i0 = max(0, idx[0] - 1)
            i1 = min(len(f)-1, idx[0] + 1)
            return float(interp1d(m[i0:i1+1], f[i0:i1+1],
                                  bounds_error=False, fill_value=np.nan)(target))
        except Exception:
            return float('nan')

    mtf50 = find_level(0.5, freq_mtf, mtf)
    mtf10 = find_level(0.1, freq_mtf, mtf)
    print(f"  MTF50 = {mtf50:.4f} cy/px  |  MTF10 = {mtf10:.4f} cy/px")

    # ── NPS ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  NPS  (flat-field Welch method)")
    print("─" * 50)

    freq_u, freq_v, nps2d, freq_nps, nps1d = compute_nps(
        img_primary, flat_roi, tile_size=64)
    flat_patch = img_primary[flat_r0:flat_r1, flat_c0:flat_c1].astype(np.float64)
    flat_mean  = flat_patch.mean()
    flat_std   = flat_patch.std()
    print(f"  Flat mean = {flat_mean:.1f} DN  |  std = {flat_std:.2f} DN")
    print(f"  NPS(0+)   = {nps1d[1]:.2e} DN²")

    # ── NEQ ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  NEQ")
    print("─" * 50)

    freq_neq, neq = compute_neq(freq_mtf, mtf, freq_nps, nps1d, flat_mean)
    print(f"  NEQ(0+) = {neq[0]:.2e}")

    # ── DQE & Shannon ─────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  DQE & Shannon Capacity")
    print("─" * 50)

    q = flat_mean / gain           # photons/pixel (or DN/DN=1 in fallback)
    freq_dqe, dqe = compute_dqe(freq_neq, neq, q, normalized=normalized_dqe)
    dqe0 = float(dqe[0])

    if normalized_dqe:
        print("  DQE is RELATIVE (normalised): DQE(0) = 1.0 by definition.")
        print("  To get absolute DQE, provide gain via metadata or paired frames.")
        dqe_label = "Relative DQE  (DQE / DQE(0))"
    else:
        print(f"  Incident fluence q = {q:.1f} ph/px")
        print(f"  DQE(0) = {dqe0:.4f}")
        dqe_label = "DQE  (absolute)"

    cap_bits, cap_bpx, integrand = compute_shannon_capacity(
        freq_dqe, dqe, image_size=max(H, W))

    shannon_note = "" if not normalized_dqe else " [relative — needs abs. DQE for calibrated value]"
    print(f"\n  +---------------------------------------------------------+")
    print(f"  |  Shannon Capacity = {cap_bits/1e6:8.3f} Mbits / image{' ':7}|")
    print(f"  |                   = {cap_bpx:8.4f} bits / pixel{' ':7}|")
    if normalized_dqe:
        print(f"  |  NOTE: relative DQE used — absolute value not calibrated |")
    print(f"  +---------------------------------------------------------+")

    # ── FIGURE ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    title_str = (
        f"Fluorescence Microscopy — Image Formation Metrics\n"
        f"Image: {image_paths[0].split('/')[-1]}   "
        f"({H}×{W} px)   "
        f"Gain: {gain:.4f} DN/ph [{gain_src[:50]}]"
    )
    fig.suptitle(title_str, fontsize=10, fontweight='bold', y=1.01)
    gs = GridSpec(4, 3, figure=fig, hspace=0.58, wspace=0.38)

    # Row 0: image, 2D-NPS, flat-ROI overlay
    ax0 = fig.add_subplot(gs[0, 0])
    lo, hi = np.percentile(img_primary, [1, 99])
    ax0.imshow(img_primary, cmap='gray', vmin=lo, vmax=hi,
               aspect='auto', interpolation='nearest')
    from matplotlib.patches import Rectangle
    ax0.add_patch(Rectangle((flat_c0, flat_r0),
                             flat_c1-flat_c0, flat_r1-flat_r0,
                             lw=1.5, edgecolor='lime', facecolor='none',
                             label='Flat ROI'))
    if edge_roi:
        er0, er1, ec0, ec1 = edge_roi
        ax0.add_patch(Rectangle((ec0, er0), ec1-ec0, er1-er0,
                                 lw=1.5, edgecolor='cyan', facecolor='none',
                                 label='Edge ROI'))
    ax0.legend(fontsize=7, loc='upper right')
    ax0.set_title("Input Image\n(green=flat ROI, cyan=edge ROI)", fontsize=9)
    ax0.set_xlabel("Col (px)"); ax0.set_ylabel("Row (px)")

    ax1 = fig.add_subplot(gs[0, 1])
    nps_log = np.log10(np.maximum(nps2d, 1.0))
    im1 = ax1.imshow(nps_log, cmap='inferno', origin='lower',
                     extent=[-0.5, 0.5, -0.5, 0.5])
    ax1.set_title("2-D NPS  (log10 scale)", fontsize=9)
    ax1.set_xlabel("u (cy/px)"); ax1.set_ylabel("v (cy/px)")
    plt.colorbar(im1, ax=ax1, label="log10(NPS) [DN²]", shrink=0.85)

    # PTC or flat histogram
    ax2 = fig.add_subplot(gs[0, 2])
    if ptc_data is not None:
        pm, pv = ptc_data
        f_line = np.linspace(0, pm.max() * 1.05, 200)
        ax2.scatter(pm, pv, s=60, color='steelblue', zorder=5, label='Measured')
        ax2.plot(f_line, gain * f_line + sigma_read**2, 'r--', lw=1.5,
                 label=f'Fit: g={gain:.3f}, σ_r={sigma_read:.1f} DN')
        ax2.set_xlabel("Mean signal (DN)"); ax2.set_ylabel("Variance (DN²)")
        ax2.set_title("Photon Transfer Curve\n(PTC)", fontsize=9)
        ax2.legend(fontsize=8)
    else:
        ax2.hist(flat_patch.ravel(), bins=80, color='steelblue', alpha=0.75)
        ax2.axvline(flat_mean, color='r', ls='--', lw=1.2,
                    label=f'Mean = {flat_mean:.0f} DN')
        ax2.set_xlabel("Pixel value (DN)"); ax2.set_ylabel("Count")
        ax2.set_title(f"Flat-ROI Histogram\nGain (var/mean) = {gain:.4f} DN/ph", fontsize=9)
        ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)

    # Row 1: MTF (full width)
    ax3 = fig.add_subplot(gs[1, :])
    if mtf_ok:
        ax3.plot(freq_mtf, mtf, color='royalblue', lw=2.2, label='Measured MTF')
    ax3.axhline(0.50, color='gray',      ls='--', lw=0.8)
    ax3.axhline(0.10, color='lightgray', ls='--', lw=0.8)
    if not np.isnan(mtf50):
        ax3.axvline(mtf50, color='crimson',    ls='--', lw=1.1,
                    label=f'MTF50 = {mtf50:.4f} cy/px')
    if not np.isnan(mtf10):
        ax3.axvline(mtf10, color='darkorange', ls='--', lw=1.1,
                    label=f'MTF10 = {mtf10:.4f} cy/px')
    ax3.axvline(0.5, color='k', ls=':', lw=0.9, label='Nyquist = 0.5 cy/px')
    ax3.set_title("MTF — Modulation Transfer Function  (slanted-edge method)",
                  fontsize=11)
    ax3.set_xlabel("Spatial frequency (cy/px)")
    ax3.set_ylabel("MTF  (DC = 1)")
    ax3.set_xlim(0, 0.5); ax3.set_ylim(-0.02, 1.12)
    ax3.legend(loc='upper right', fontsize=9); ax3.grid(True, alpha=0.25)
    if not mtf_ok:
        ax3.text(0.5, 0.5, "MTF not computed\n(no usable edge found)",
                 transform=ax3.transAxes, ha='center', va='center',
                 fontsize=12, color='red')

    # Row 2: NPS 1-D, NEQ, DQE
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.semilogy(freq_nps, np.maximum(nps1d, 1e-9), color='seagreen', lw=2)
    ax4.set_title("NPS — Noise Power Spectrum\n(1-D radial average)", fontsize=9)
    ax4.set_xlabel("f (cy/px)"); ax4.set_ylabel("NPS (DN²)")
    ax4.set_xlim(0, 0.5); ax4.grid(True, alpha=0.2, which='both')
    ax4.text(0.98, 0.97, f"NPS(0+)\n{nps1d[1]:.1e} DN²",
             transform=ax4.transAxes, fontsize=7.5, va='top', ha='right')

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.semilogy(freq_neq, np.maximum(neq, 1e-9), color='darkorange', lw=2)
    ax5.set_title("NEQ — Noise Equivalent Quanta\nNEQ = S² · MTF² / NPS", fontsize=9)
    ax5.set_xlabel("f (cy/px)"); ax5.set_ylabel("NEQ")
    ax5.set_xlim(0, 0.5); ax5.grid(True, alpha=0.2, which='both')
    ax5.text(0.98, 0.97, f"NEQ(0+)\n{neq[0]:.1e}",
             transform=ax5.transAxes, fontsize=7.5, va='top', ha='right')

    ax6 = fig.add_subplot(gs[2, 2])
    ax6.plot(freq_dqe, dqe, color='crimson', lw=2)
    ax6.axhline(dqe0, color='gray', ls='--', lw=1.0,
                label=f'DQE(0) = {dqe0:.4f}')
    ax6.set_title(f"{dqe_label}", fontsize=9)
    ax6.set_xlabel("f (cy/px)"); ax6.set_ylabel("DQE")
    ax6.set_xlim(0, 0.5); ax6.set_ylim(-0.02, max(1.05, dqe.max() * 1.1))
    ax6.legend(fontsize=9); ax6.grid(True, alpha=0.25)
    if normalized_dqe:
        ax6.text(0.5, 0.5, "RELATIVE DQE\n(gain unknown)",
                 transform=ax6.transAxes, ha='center', va='center',
                 fontsize=9, color='crimson', alpha=0.35)

    # Row 3: Shannon integrand (full width)
    ax7 = fig.add_subplot(gs[3, :])
    ax7.fill_between(freq_dqe, integrand, alpha=0.28, color='purple')
    ax7.plot(freq_dqe, integrand, color='purple', lw=1.8)
    ax7.set_title(
        f"Shannon Information Capacity Density\n"
        f"C = {cap_bits/1e6:.3f} Mbits / image   "
        f"({cap_bpx:.4f} bits/px)"
        + ("  [relative]" if normalized_dqe else ""),
        fontsize=10)
    ax7.set_xlabel("Spatial frequency (cy/px)")
    ax7.set_ylabel("log₂(1+DQE) × 2πf")
    ax7.set_xlim(0, 0.5); ax7.grid(True, alpha=0.25)

    out_png = r"C:\Users\Ruiz\Desktop\fluoro_metrics.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"\n  Figure saved: {out_png}")
    plt.close()

    return dict(
        gain=gain, gain_source=gain_src, sigma_read=sigma_read,
        flat_mean=flat_mean, flat_std=flat_std, flat_roi=flat_roi,
        mtf50=mtf50, mtf10=mtf10,
        nps0=nps1d[1], neq0=neq[0],
        dqe0=dqe0, normalized_dqe=normalized_dqe,
        shannon_mbits=cap_bits/1e6, shannon_bpx=cap_bpx,
    )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():


    results  = run_pipeline(images)

    print("\n  SUMMARY")
    print("  " + "─" * 45)
    for k, v in results.items():
        print(f"  {k:25s}: {v}")


if __name__ == "__main__":
    main()