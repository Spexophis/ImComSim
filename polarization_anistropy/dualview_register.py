"""
Pixel-to-pixel registration of a split-channel (dual-view) polarization image.

The detector frame is split into two halves (TOP / BOTTOM) that image the same
field of view through two polarization channels.

This module:

  * estimates the GLOBAL transform between the halves
      phase correlation (+ optional flip search) -> masked-NCC Powell refinement
      over translation -> Euclidean -> similarity -> affine (gated by NCC gain);
  * optionally adds a smooth NONRIGID residual warp (polynomial / thin-plate
      spline), confidence-weighted per axis and accepted only if it improves NCC
      (so it corrects genuine field distortion and stays inert otherwise);
  * builds dense float correspondence maps (map_x, map_y) for interpolated
      resampling, AND an INTEGER pixel-index map (idx_x, idx_y, valid) that pairs
      every top pixel with a single bottom pixel using NO intensity interpolation;
  * produces per-pixel residual diagnostics (local-NCC map, residual vectors,
      observability) so you can see where the alignment is weakest.
"""

import json
import os
from dataclasses import dataclass, asdict, field

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.interpolate import RBFInterpolator, griddata
from scipy.optimize import minimize


# =========================================================================== #
# Pre-processing
# =========================================================================== #
def split_halves(img, split_row=None):
    """
    Split a frame into equal-height (top, bottom).
    Default split is the middle row.
    """
    h = img.shape[0]
    if split_row is None:
        split_row = h // 2
    top, bot = img[:split_row], img[split_row:]
    n = min(top.shape[0], bot.shape[0])
    return top[:n].copy(), bot[:n].copy()


def normalize(img, lo_pct=1.0, hi_pct=99.5):
    """
    Robust percentile contrast stretch to float32 in [0, 1].
    """
    lo, hi = np.percentile(img, [lo_pct, hi_pct])
    return np.clip((img - lo) / (hi - lo + 1e-9), 0.0, 1.0).astype(np.float32)


# =========================================================================== #
# Similarity metrics
# =========================================================================== #
def weighted_ncc(a, b, w):
    """
    Weighted normalized cross-correlation in [-1, 1].
    """
    w = w / (w.sum() + 1e-12)
    ac = a - (w * a).sum()
    bc = b - (w * b).sum()
    num = (w * ac * bc).sum()
    den = np.sqrt((w * ac * ac).sum() * (w * bc * bc).sum()) + 1e-12
    return float(num / den)


def informative_mask(ref, pct=60.0):
    """
    Weight map emphasizing the bright, structured region of the frame.
    """
    return (ref > np.percentile(ref, pct)).astype(np.float32)


def local_ncc_map(a, b, win=33):
    """
    Sliding-window Pearson correlation;
    low values mark weak local agreement.
    """
    k = (win, win)
    ma = cv2.boxFilter(a, -1, k)
    mb = cv2.boxFilter(b, -1, k)
    maa = cv2.boxFilter(a * a, -1, k)
    mbb = cv2.boxFilter(b * b, -1, k)
    mab = cv2.boxFilter(a * b, -1, k)
    cov = mab - ma * mb
    va = np.clip(maa - ma * ma, 0, None)
    vb = np.clip(mbb - mb * mb, 0, None)
    return (cov / (np.sqrt(va * vb) + 1e-6)).astype(np.float32)


# =========================================================================== #
# Global motion models  ->  2x3 affine M with warpAffine(moving, M) ~ ref
# =========================================================================== #
MODELS = ("translation", "euclidean", "similarity", "affine")


def _sim_matrix(tx, ty, ang, scale, center):
    cx, cy = center
    c, s = np.cos(ang) * scale, np.sin(ang) * scale
    M = np.array([[c, -s, 0.0], [s, c, 0.0]], dtype=np.float64)
    M[0, 2] = cx - (c * cx - s * cy) + tx
    M[1, 2] = cy - (s * cx + c * cy) + ty
    return M.astype(np.float32)


def _params_to_M(p, model, center):
    if model == "affine":
        return np.asarray(p, dtype=np.float32).reshape(2, 3)
    tx, ty = p[0], p[1]
    ang = p[2] if model in ("euclidean", "similarity") else 0.0
    scale = p[3] if model == "similarity" else 1.0
    return _sim_matrix(tx, ty, ang, scale, center)


def _initial_params(model, tx, ty):
    return {"translation": [tx, ty], "euclidean": [tx, ty, 0.0],
            "similarity": [tx, ty, 0.0, 1.0],
            "affine": [1.0, 0.0, tx, 0.0, 1.0, ty]}[model]


@dataclass
class FlipState:
    flipud: bool = False
    fliplr: bool = False

    def apply(self, img):
        if self.flipud:
            img = img[::-1]
        if self.fliplr:
            img = img[:, ::-1]
        return np.ascontiguousarray(img)


def coarse_align(ref, moving, search_flips=True):
    """
    Sub-pixel translation via phase correlation,
    choosing the best of the four flip states
    (handles channels mirrored by the splitter).
    """
    h, w = ref.shape
    mask = informative_mask(ref)
    states = ([FlipState()] if not search_flips
              else [FlipState(a, b) for a in (False, True) for b in (False, True)])
    best = None
    for st in states:
        mv = st.apply(moving)
        (sx, sy), _ = cv2.phaseCorrelate(mv.astype(np.float32), ref.astype(np.float32))
        warped = cv2.warpAffine(mv, np.array([[1, 0, sx], [0, 1, sy]], np.float32), (w, h))
        score = weighted_ncc(ref, warped, mask)
        if best is None or score > best[3]:
            best = (sx, sy, st, score)
    return best


@dataclass
class RegResult:
    M: np.ndarray
    M_inv: np.ndarray
    model: str
    flip: FlipState
    ncc_initial: float
    ncc_coarse: float
    ncc_final: float
    per_model_ncc: dict = field(default_factory=dict)

    def to_json(self):
        d = asdict(self)
        d["M"] = self.M.tolist()
        d["M_inv"] = self.M_inv.tolist()
        d["flip"] = {"flipud": self.flip.flipud, "fliplr": self.flip.fliplr}
        return d


def estimate_transform(top, bottom, max_model="similarity", min_gain=2e-3,
                       search_flips=True, smooth_sigma=0.0):
    """
    Estimate the global transform mapping the BOTTOM half onto the TOP half.
    `top`/`bottom` are normalized float32 in [0, 1].
    """
    h, w = top.shape
    center = (w / 2.0, h / 2.0)
    mask = informative_mask(top)
    ref = cv2.GaussianBlur(top, (0, 0), smooth_sigma) if smooth_sigma > 0 else top
    base_moving = cv2.GaussianBlur(bottom, (0, 0), smooth_sigma) if smooth_sigma > 0 else bottom

    ncc_initial = weighted_ncc(top, bottom, mask)
    tx, ty, flip, ncc_coarse = coarse_align(ref, base_moving, search_flips)
    moving = flip.apply(bottom)
    moving_fit = flip.apply(base_moving)

    def score_for(M):
        return weighted_ncc(ref, cv2.warpAffine(moving_fit, M, (w, h)), mask)

    wanted = MODELS[: MODELS.index(max_model) + 1]
    best_M = np.array([[1, 0, tx], [0, 1, ty]], np.float32)
    best_score, best_model = score_for(best_M), "translation"
    per_model = {"translation": round(best_score, 5)}
    for model in wanted:
        if model == "translation":
            continue
        p0 = _initial_params(model, best_M[0, 2], best_M[1, 2])
        r = minimize(lambda p: -score_for(_params_to_M(p, model, center)), p0,
                     method="Powell", options={"xtol": 1e-4, "ftol": 1e-5, "maxiter": 4000})
        cand_M, cand_score = _params_to_M(r.x, model, center), -r.fun
        per_model[model] = round(cand_score, 5)
        if cand_score >= best_score + min_gain:
            best_M, best_score, best_model = cand_M, cand_score, model

    ncc_final = weighted_ncc(top, cv2.warpAffine(moving, best_M, (w, h)), mask)
    return RegResult(best_M.astype(np.float32),
                     cv2.invertAffineTransform(best_M).astype(np.float32),
                     best_model, flip, round(ncc_initial, 5),
                     round(ncc_coarse, 5), round(ncc_final, 5), per_model)


def build_correspondence_maps(res, shape):
    """
    Dense TOP->BOTTOM float maps for the global affine, in ORIGINAL bottom coords.
    """
    h, w = shape
    iM = res.M_inv
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = iM[0, 0] * xs + iM[0, 1] * ys + iM[0, 2]
    map_y = iM[1, 0] * xs + iM[1, 1] * ys + iM[1, 2]
    if res.flip.fliplr:
        map_x = (w - 1) - map_x
    if res.flip.flipud:
        map_y = (h - 1) - map_y
    return map_x.astype(np.float32), map_y.astype(np.float32)


def resample_bottom_to_top(bottom_raw, map_x, map_y):
    """
    Bilinear resample of the bottom half onto the top grid (for scoring/overlay).
    """
    return cv2.remap(bottom_raw.astype(np.float32), map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


# =========================================================================== #
# Nonrigid residual refinement
# =========================================================================== #
@dataclass
class ResidualField:
    pts: np.ndarray
    du: np.ndarray
    dv: np.ndarray
    conf_x: np.ndarray
    conf_y: np.ndarray


def measure_residual_field(top, aligned, win=64, step=24, min_sig=0.15, max_shift=None):
    """
    Windowed phase-correlation residuals between top (ref) and the aligned bottom,
    with PER-AXIS confidence from the local gradient structure tensor.
    """
    H, W = top.shape
    if max_shift is None:
        max_shift = win / 2
    Iy, Ix = np.gradient(cv2.GaussianBlur(top, (0, 0), 1.0))
    han = cv2.createHanningWindow((win, win), cv2.CV_32F)
    P, DU, DV, CX, CY = [], [], [], [], []
    for cy in range(win // 2, H - win // 2, step):
        for cx in range(win // 2, W - win // 2, step):
            sy, sx = slice(cy - win // 2, cy + win // 2), slice(cx - win // 2, cx + win // 2)
            a = top[sy, sx]
            if a.mean() < min_sig:
                continue
            b = aligned[sy, sx]
            (du, dv), pk = cv2.phaseCorrelate(b.copy(), a.copy(), han)
            pk = max(pk, 0.0)
            if abs(du) > max_shift or abs(dv) > max_shift:
                continue
            P.append((cx, cy))
            DU.append(du)
            DV.append(dv)
            CX.append(pk * float((Ix[sy, sx] ** 2).sum()))
            CY.append(pk * float((Iy[sy, sx] ** 2).sum()))
    return ResidualField(np.array(P, float), np.array(DU), np.array(DV),
                         np.clip(np.array(CX), 0, None), np.clip(np.array(CY), 0, None))


def _wpolyfit(P, z, w, deg, xn, yn):
    x, y = P[:, 0] / xn, P[:, 1] / yn
    terms = [(i, j) for i in range(deg + 1) for j in range(deg + 1 - i)]
    A = np.column_stack([x ** i * y ** j for i, j in terms])
    sw = np.sqrt(w / (w.max() + 1e-12))
    coef, *_ = np.linalg.lstsq(A * sw[:, None], z * sw, rcond=None)
    return terms, coef


def _wpolyeval(terms, coef, X, Y, xn, yn):
    x, y = X / xn, Y / yn
    return np.asarray(sum(c * (x ** i) * (y ** j) for c, (i, j) in zip(coef, terms)), np.float32)


def _fit_axis(P, z, conf, shape, method, deg, conf_pct):
    H, W = shape
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    xn, yn = max(P[:, 0].max(), 1.0), max(P[:, 1].max(), 1.0)
    keep = conf >= np.percentile(conf, conf_pct)
    if keep.sum() < 6:
        return np.zeros((H, W), np.float32)
    if method == "polynomial":
        terms, coef = _wpolyfit(P[keep], z[keep], conf[keep], deg, xn, yn)
        return _wpolyeval(terms, coef, xs, ys, xn, yn)
    rbf = RBFInterpolator(P[keep], z[keep], kernel="thin_plate_spline",
                          smoothing=max(1e-3, 1.0 / (conf[keep].mean() + 1e-9)))
    grid = np.column_stack([xs.ravel(), ys.ravel()])
    return rbf(grid).reshape(H, W).astype(np.float32)


@dataclass
class NonrigidResult:
    applied: bool
    method: str
    ncc_global: float
    ncc_nonrigid: float
    du: np.ndarray
    dv: np.ndarray
    rms_correction: float
    max_correction: float
    rf: ResidualField


def refine_nonrigid(top, bottom_oriented, map_x, map_y, mask, method="polynomial",
                    x_deg=1, y_deg=2, conf_pct=40.0, taper_sigma=20.0, min_gain=2e-3,
                    win=64, step=24):
    """
    Smooth residual warp on top of the global maps, gated on masked-NCC gain.
    `bottom_oriented` already has the global flip applied (normalized).
    """
    H, W = top.shape
    aligned = resample_bottom_to_top(bottom_oriented, map_x, map_y)
    rf = measure_residual_field(top, aligned, win=win, step=step)
    ncc_global = weighted_ncc(top, aligned, mask)
    z0 = np.zeros((H, W), np.float32)
    if len(rf.pts) < 12:
        return NonrigidResult(False, method, round(ncc_global, 5), round(ncc_global, 5),
                              z0, z0, 0.0, 0.0, rf)
    du = _fit_axis(rf.pts, rf.du, rf.conf_x, (H, W), method, x_deg, conf_pct)
    dv = _fit_axis(rf.pts, rf.dv, rf.conf_y, (H, W), method, y_deg, conf_pct)
    taper = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), taper_sigma)
    taper /= taper.max() + 1e-9
    du *= taper;
    dv *= taper

    best = (ncc_global, None, None)
    for s in (+1.0, -1.0):
        al = resample_bottom_to_top(bottom_oriented, map_x + s * du, map_y + s * dv)
        n = weighted_ncc(top, al, mask)
        if n > best[0]:
            best = (n, du * s, dv * s)
    ncc_nr, dux, dvy = best
    if dux is None or ncc_nr < ncc_global + min_gain:
        return NonrigidResult(False, method, round(ncc_global, 5), round(ncc_nr, 5),
                              z0, z0, 0.0, 0.0, rf)
    mag = np.sqrt(dux ** 2 + dvy ** 2)
    return NonrigidResult(True, method, round(ncc_global, 5), round(ncc_nr, 5),
                          dux.astype(np.float32), dvy.astype(np.float32),
                          float(np.sqrt((mag[mask > 0] ** 2).mean())),
                          float(mag[mask > 0].max()), rf)


# =========================================================================== #
# Integer pixel-index correspondence (NO interpolation)
# =========================================================================== #
def integer_index_maps(map_x, map_y, shape):
    """
    Round float partner coords to nearest integer bottom-pixel indices.
    Returns (idx_x, idx_y, valid); out-of-bounds partners -> -1 / invalid.
    """
    h, w = shape
    idx_x = np.rint(map_x).astype(np.int32)
    idx_y = np.rint(map_y).astype(np.int32)
    valid = (idx_x >= 0) & (idx_x < w) & (idx_y >= 0) & (idx_y < h)
    idx_x[~valid] = -1
    idx_y[~valid] = -1
    return idx_x, idx_y, valid


def gather_nearest(bottom_raw, idx_x, idx_y, valid, fill=0.0):
    """
    Transfer bottom intensities onto the top grid by pure integer indexing
    (no interpolation: each output pixel copies exactly one bottom pixel).
    """
    out = np.full(idx_x.shape, fill, dtype=np.float32)
    yy, xx = np.where(valid)
    out[yy, xx] = bottom_raw[idx_y[yy, xx], idx_x[yy, xx]]
    return out


def pairs_list(idx_x, idx_y, valid):
    """
    Flat (N, 4) table of correspondences: [top_y, top_x, bottom_y, bottom_x].
    """
    ty, tx = np.where(valid)
    return np.stack([ty, tx, idx_y[ty, tx], idx_x[ty, tx]], axis=1).astype(np.int32)


# =========================================================================== #
# High-level driver + result container
# =========================================================================== #
@dataclass
class DualViewResult:
    # geometry
    res: RegResult
    nr: "NonrigidResult | None"
    map_x: np.ndarray  # float dense maps (global [+ nonrigid]) for interp
    map_y: np.ndarray
    idx_x: np.ndarray  # integer index maps (no interp); -1 where invalid
    idx_y: np.ndarray
    valid: np.ndarray  # bool coverage mask
    # imagery (normalized, for display/scoring)
    top_n: np.ndarray
    bottom_n: np.ndarray
    aligned_interp_n: np.ndarray  # bottom resampled onto top grid (bilinear)
    aligned_nn_n: np.ndarray  # bottom gathered onto top grid (nearest, no interp)
    mask: np.ndarray
    # raw halves (original intensities)
    top_raw: np.ndarray
    bottom_raw: np.ndarray
    # metrics
    local_ncc: np.ndarray
    ncc_interp: float
    ncc_nn: float
    coverage_total: float
    coverage_signal: float
    pixel_size_nm: "float | None" = None

    def aligned_nn_raw(self):
        """
        Nearest-neighbour aligned bottom in ORIGINAL intensity units.
        """
        return gather_nearest(self.bottom_raw, self.idx_x, self.idx_y, self.valid)

    def partner(self, x, y):
        """
        Integer bottom-pixel partner of top pixel (x, y), or None if invalid.
        """
        if not self.valid[y, x]:
            return None
        return int(self.idx_x[y, x]), int(self.idx_y[y, x])


def register_dualview(top, bottom, *, model="similarity", refine="polynomial",
                      x_deg=1, y_deg=2, conf_pct=40.0, taper=20.0, min_gain=2e-3,
                      search_flips=True, smooth=0.0, build_index_map=True,
                      pixel_size_nm=None):
    """
    Register the BOTTOM half onto the TOP half and build correspondence maps.

    Parameters
    ----------
    top, bottom     : 2-D arrays (raw intensities), equal shape.
    model           : richest global model to attempt ('translation'..'affine').
    refine          : 'none' | 'polynomial' | 'tps' (gated nonrigid residual warp).
    build_index_map : also produce the integer, interpolation-free index maps.
    pixel_size_nm   : optional; only used to report shifts in nm.

    Returns
    -------
    DualViewResult
    """
    top_raw = np.asarray(top, np.float32)
    bottom_raw = np.asarray(bottom, np.float32)
    top_n, bottom_n = normalize(top_raw), normalize(bottom_raw)
    mask = informative_mask(top_n)

    res = estimate_transform(top_n, bottom_n, max_model=model, min_gain=min_gain,
                             search_flips=search_flips, smooth_sigma=smooth)
    map_x, map_y = build_correspondence_maps(res, top_n.shape)

    nr = None
    if refine != "none":
        oriented_n = res.flip.apply(bottom_n)
        nr = refine_nonrigid(top_n, oriented_n, map_x, map_y, mask, method=refine,
                             x_deg=x_deg, y_deg=y_deg, conf_pct=conf_pct,
                             taper_sigma=taper, min_gain=min_gain)
        if nr.applied:
            map_x = map_x + nr.du
            map_y = map_y + nr.dv

    if build_index_map:
        idx_x, idx_y, valid = integer_index_maps(map_x, map_y, top_n.shape)
        aligned_nn_n = gather_nearest(bottom_raw, idx_x, idx_y, valid)
    else:
        idx_x = idx_y = np.zeros(top_n.shape, np.int32)
        valid = np.zeros(top_n.shape, bool)
        aligned_nn_n = np.zeros_like(top_n)

    aligned_interp_n = normalize(resample_bottom_to_top(bottom_raw, map_x, map_y))
    lncc = local_ncc_map(top_n, aligned_interp_n, win=33)
    ncc_interp = weighted_ncc(top_n, aligned_interp_n, mask)
    if build_index_map:
        smask = ((mask > 0) & valid).astype(np.float32)
        ncc_nn = weighted_ncc(top_n, aligned_nn_n, smask)
    else:
        ncc_nn = float("nan")

    return DualViewResult(
        res=res, nr=nr, map_x=map_x, map_y=map_y,
        idx_x=idx_x, idx_y=idx_y, valid=valid,
        top_n=top_n, bottom_n=bottom_n,
        aligned_interp_n=aligned_interp_n, aligned_nn_n=aligned_nn_n, mask=mask,
        top_raw=top_raw, bottom_raw=bottom_raw,
        local_ncc=lncc, ncc_interp=round(ncc_interp, 4),
        ncc_nn=round(ncc_nn, 4) if build_index_map else float("nan"),
        coverage_total=round(float(valid.mean()), 4) if build_index_map else 0.0,
        coverage_signal=round(float(valid[mask > 0].mean()), 4) if build_index_map else 0.0,
        pixel_size_nm=pixel_size_nm,
    )


# =========================================================================== #
# Plotting (mirrors the plot_rolling_frc(result, bg) idiom)
# =========================================================================== #
def _overlay(top_n, moving_n):
    h, w = top_n.shape
    rgb = np.zeros((h, w, 3), np.float32)
    rgb[..., 0] = top_n
    rgb[..., 2] = top_n
    rgb[..., 1] = moving_n
    return np.clip(rgb, 0, 1)


def plot_dualview(result, show=True):
    """
    Six-panel summary: overlays before/after, local-NCC, residual magnitude,
    observability and coverage, with a text readout of the transform.
    """
    r = result
    H, W = r.top_n.shape
    bg = r.mask < 0.5
    M = r.res.M
    sx, sy = np.hypot(M[0, 0], M[1, 0]), np.hypot(M[0, 1], M[1, 1])
    ang = np.degrees(np.arctan2(M[1, 0], M[0, 0]))

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    ax[0, 0].imshow(_overlay(r.top_n, r.bottom_n))
    ax[0, 0].set_title("Before (top=magenta, bottom=green)")
    ax[0, 1].imshow(_overlay(r.top_n, r.aligned_nn_n))
    ax[0, 1].set_title("After — integer index map (no interp)")

    im2 = ax[0, 2].imshow(np.where(bg, np.nan, r.local_ncc), cmap="magma", vmin=0, vmax=1)
    ax[0, 2].set_title("Local NCC of alignment\n(low = weakest)")
    fig.colorbar(im2, ax=ax[0, 2], fraction=0.046)

    rf = r.nr.rf if r.nr is not None else ResidualField(np.empty((0, 2)), *(np.array([]),) * 4)
    if r.nr is not None and r.nr.applied:
        mag = np.where(bg, np.nan, np.sqrt(r.nr.du ** 2 + r.nr.dv ** 2))
        t3 = f"Applied nonrigid |Δ| (px)\n{r.nr.method}, RMS={r.nr.rms_correction:.2f}"
    elif len(rf.pts):
        gy, gx = np.mgrid[0:H, 0:W]
        comb = rf.conf_x + rf.conf_y
        rel = comb >= np.percentile(comb, 40)
        sel = rel if rel.sum() >= 4 else np.ones(len(rf.pts), bool)
        mm = np.sqrt(rf.du ** 2 + rf.dv ** 2)
        mag = np.where(bg, np.nan, griddata(rf.pts[sel], mm[sel], (gx, gy), method="linear"))
        t3 = "Measured residual |Δ| (px)\n(nonrigid gave no gain)"
    else:
        mag = np.full((H, W), np.nan)
        t3 = "Residual |Δ| (px)"
    im3 = ax[1, 0].imshow(mag, cmap="inferno")
    ax[1, 0].set_title(t3)
    fig.colorbar(im3, ax=ax[1, 0], fraction=0.046, label="pixels")

    if len(rf.pts):
        gy, gx = np.mgrid[0:H, 0:W]
        ratio = np.log10((rf.conf_x + 1e-6) / (rf.conf_y + 1e-6))
        obs = np.where(bg, np.nan, griddata(rf.pts, ratio, (gx, gy), method="linear"))
    else:
        obs = np.full((H, W), np.nan)
    im4 = ax[1, 1].imshow(obs, cmap="coolwarm", vmin=-2, vmax=2)
    ax[1, 1].set_title("Observability log10(conf_x/conf_y)\n(blue = x weak)")
    fig.colorbar(im4, ax=ax[1, 1], fraction=0.046)

    ax[1, 2].imshow(r.valid, cmap="gray")
    ax[1, 2].set_title(f"Valid correspondence mask\n{int(r.valid.sum()):,} pairs "
                       f"({r.coverage_signal * 100:.1f}% of signal)")

    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])

    nm = ""
    if r.pixel_size_nm:
        nm = (f"  =  ({M[0, 2] * r.pixel_size_nm:+.0f}, {M[1, 2] * r.pixel_size_nm:+.0f}) nm"
              f" @ {r.pixel_size_nm} nm/px")
    nrtxt = ("nonrigid: off" if r.nr is None else
             (f"nonrigid {r.nr.method}: APPLIED (RMS {r.nr.rms_correction:.2f}px)"
              if r.nr.applied else f"nonrigid {r.nr.method}: rejected (no gain)"))
    fig.suptitle(
        f"Dual-view registration   |   model={r.res.model}   "
        f"shift=({M[0, 2]:+.2f}, {M[1, 2]:+.2f}) px{nm}   rot={ang:+.3f}°   "
        f"scale=({sx:.4f}, {sy:.4f})\n"
        f"NCC interp={r.ncc_interp:.4f}   NCC nearest={r.ncc_nn:.4f}   "
        f"coverage={r.coverage_signal * 100:.1f}% of signal   |   {nrtxt}",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if show:
        plt.show()
    return fig


# =========================================================================== #
# Saving
# =========================================================================== #
def save_results(result, out_dir, base):
    """Write index maps, dense maps, nearest-neighbour aligned TIFF, pairs, json,
    overlay and the summary figure."""
    os.makedirs(out_dir, exist_ok=True)
    r = result
    np.save(os.path.join(out_dir, f"{base}_idx_x.npy"), r.idx_x)
    np.save(os.path.join(out_dir, f"{base}_idx_y.npy"), r.idx_y)
    np.save(os.path.join(out_dir, f"{base}_valid.npy"), r.valid)
    np.save(os.path.join(out_dir, f"{base}_map_x.npy"), r.map_x)
    np.save(os.path.join(out_dir, f"{base}_map_y.npy"), r.map_y)
    np.save(os.path.join(out_dir, f"{base}_pairs.npy"),
            pairs_list(r.idx_x, r.idx_y, r.valid))
    np.save(os.path.join(out_dir, f"{base}_local_ncc.npy"), r.local_ncc)
    tifffile.imwrite(os.path.join(out_dir, f"{base}_bottom_aligned_nn.tif"), r.aligned_nn_raw())
    cv2.imwrite(os.path.join(out_dir, f"{base}_overlay_after.png"),
                cv2.cvtColor((_overlay(r.top_n, r.aligned_nn_n) * 255).astype(np.uint8),
                             cv2.COLOR_RGB2BGR))
    meta = r.res.to_json()
    meta.update(dict(
        interpolation_for_index_map="none (nearest integer index)",
        ncc_interp=r.ncc_interp, ncc_nearest=r.ncc_nn,
        n_valid_pairs=int(r.valid.sum()),
        coverage_total=r.coverage_total, coverage_in_signal=r.coverage_signal,
        pixel_size_nm=r.pixel_size_nm,
        nonrigid=(None if r.nr is None else dict(
            applied=r.nr.applied, method=r.nr.method,
            ncc_global=r.nr.ncc_global, ncc_nonrigid=r.nr.ncc_nonrigid,
            rms_correction_px=r.nr.rms_correction, max_correction_px=r.nr.max_correction)),
    ))
    fig = plot_dualview(r, show=False)
    fig.savefig(os.path.join(out_dir, f"{base}_summary.png"), dpi=120)
    plt.close(fig)
    with open(os.path.join(out_dir, f"{base}_index_map.json"), "w") as fh:
        json.dump(meta, fh, indent=2)


if __name__ == "__main__":
    fdh = r"C:\Users\Ruiz\Desktop\20260519115441_line_scanning_beads500nm_h_corrected_stack.tif"
    fdv = r"C:\Users\Ruiz\Desktop\20260519115953_line_scanning_beads500nm_v_corrected_stack.tif"
    pixel_size = 162.5
    img_stack_h = tifffile.imread(fdh)
    img_stack_v = tifffile.imread(fdv)
    img_stack = np.concatenate((img_stack_v, img_stack_h), axis=0)
    img = img_stack.max(axis=0)
    top = img[: img.shape[0] // 2, :]
    bottom = img[img.shape[0] // 2:, :]
    n = min(top.shape[0], bottom.shape[0])
    top, bottom = top[:n], bottom[:n]
    result = register_dualview(top, bottom,
                               model="similarity",
                               refine="polynomial",
                               build_index_map=True,
                               pixel_size_nm=pixel_size)
    # plot_dualview(result)

    top_stack_h = img_stack_h[:, : img.shape[0] // 2, :]
    bottom_stack_h = img_stack_h[:, img.shape[0] // 2:, :]
    bot_stack_h = np.zeros(bottom_stack_h.shape)
    for i in range(319):
        bot_stack_h[i] = gather_nearest(bottom_stack_h[i], result.idx_x, result.idx_y, result.valid)
        top_stack_h[i] *= result.valid

    top_stack_v = img_stack_v[:, : img.shape[0] // 2, :]
    bottom_stack_v = img_stack_v[:, img.shape[0] // 2:, :]
    bot_stack_v = np.zeros(bottom_stack_v.shape)
    for i in range(319):
        bot_stack_v[i] = gather_nearest(bottom_stack_v[i], result.idx_x, result.idx_y, result.valid)
        top_stack_v[i] *= result.valid

    tifffile.imwrite(r"C:\Users\Ruiz\Desktop\20260519115441_line_scanning_beads500nm_h_corrected_stack_top.tif",
                     top_stack_h)
    tifffile.imwrite(r"C:\Users\Ruiz\Desktop\20260519115441_line_scanning_beads500nm_h_corrected_stack_bot.tif",
                     bot_stack_h.astype(np.uint16))

    tifffile.imwrite(r"C:\Users\Ruiz\Desktop\20260519115953_line_scanning_beads500nm_v_corrected_stack_top.tif",
                     top_stack_v)
    tifffile.imwrite(r"C:\Users\Ruiz\Desktop\20260519115953_line_scanning_beads500nm_v_corrected_stack_bot.tif",
                     bot_stack_v.astype(np.uint16))
