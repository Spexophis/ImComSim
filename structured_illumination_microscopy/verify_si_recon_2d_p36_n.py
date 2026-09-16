"""
Equivalence checks for the fast paths in si_recon_2d_p36_n.py / heissian_deconv.py.

Each of those files trades work for speed without changing what is computed.
This script re-evaluates the literal expression that every fast path replaced
and reports the difference.  "bitwise" means the two results were equal to the
last bit; anything else is printed with its largest absolute and relative
deviation.

    python verify_si_recon_2d_p36_n.py            # 128 -> 256 grid
    python verify_si_recon_2d_p36_n.py --nx 512   # 512 -> 1024 grid (slower)

The only expected non-bitwise row is the separable modulation used inside
getoverlap, exp(a)*exp(b) instead of exp(a+b), and the fitted magnitude and
phase that follow from it (~1e-14 relative).  Construct SIM_RECON with
exact_modulation=True to remove that one as well.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')                      # the modules plot on import/use
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import heissian_deconv as hd
from si_recon_2d_p36_n import SIM_RECON, fft2, ifft2

NANG, NPH, NORDERS = 2, 6, 5
DX, WL, NA = 0.054, 0.505, 1.3
ANGS = [1.57 + 1.538, 1.57 - 0.034]
SPACING = 0.246 * 2

_rows = []


def check(name, a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    same = a.shape == b.shape and np.array_equal(a, b)
    if a.shape != b.shape:
        _rows.append((name, 'SHAPE', np.nan, np.nan))
        return
    d = np.abs(a - b)
    s = np.abs(a).max()
    _rows.append((name, 'bitwise' if same else 'differs',
                  float(d.max()), float(d.max() / s) if s else 0.0))


def synthetic_stack(nx):
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:nx, 0:nx].astype(np.float64)
    obj = np.zeros((nx, nx))
    for _ in range(60):
        cy, cx = rng.uniform(8, nx - 8, 2)
        obj += rng.uniform(0.5, 1.5) * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 2.0)
    fr = np.hypot(np.fft.fftfreq(nx)[None, :], np.fft.fftfreq(nx)[:, None])
    otf = (fr < 0.25) * 1.0
    frames = []
    for a in range(NANG):
        kx = np.cos(ANGS[a]) / SPACING * DX
        ky = np.sin(ANGS[a]) / SPACING * DX
        for p in range(NPH):
            ph = 2 * np.pi * p / NPH
            ill = (1.0 + 0.8 * np.cos(2 * np.pi * (kx * x + ky * y) + ph)
                   + 0.3 * np.cos(4 * np.pi * (kx * x + ky * y) + 2 * ph))
            im = np.real(np.fft.ifft2(np.fft.fft2(obj * ill) * otf))
            frames.append(np.clip(im, 0, None) * 100.0 + 10.0)
    return np.array(frames)


# --- the literal expressions the fast paths replaced ------------------------

def ref_modulation(p, kx, ky):
    return np.exp(2j * np.pi * (kx * p.xv + ky * p.yv))


def ref_peak_index(p, kx, ky):
    yshf = np.abs(fft2(ref_modulation(p, kx, ky)))
    sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
    Nw = p.nx
    sx = sx if sx < Nw else sx - 2 * Nw
    sy = sy if sy < Nw else sy - 2 * Nw
    return int(sx), int(sy)


def ref_zero_suppression(p, sx, sy):
    return (np.sqrt((p.xv - sx) ** 2 + (p.yv - sy) ** 2) < p.radius) * 1.0


def ref_interp(p, arr):
    nx, ny = arr.shape
    px, py = int(nx / 2), int(ny / 2)
    arrf = fft2(arr)
    arro = np.pad(np.fft.fftshift(arrf), ((px, px), (py, py)), 'constant',
                  constant_values=(0, 0))
    return ifft2(np.fft.fftshift(arro))


def ref_getoverlap(p, order, angle, spacing):
    dx = p.dx / 2
    kx = dx * np.cos(angle) / spacing
    ky = dx * np.sin(angle) / spacing
    ysh = ref_modulation(p, kx, ky)
    otf = fft2(p.psf * ysh)
    sx, sy = ref_peak_index(p, kx, ky)
    zsp = ref_zero_suppression(p, sx, sy)
    otf = otf * zsp
    imgf = fft2(p.separr[order] * ysh)
    msk = zsp * ref_zero_suppression(p, 0, 0)
    wimgf0 = otf * p.imgf0
    wimgf1 = p.otf0 * imgf
    a = np.sum(msk * wimgf1 * wimgf0.conj()) / np.sum(msk * wimgf0 * wimgf0.conj())
    return np.abs(a), np.angle(a)


def ref_hessian_kernels(shape):
    ny, nx = shape
    kx = 2 * np.pi * np.fft.fftfreq(nx)[None, :]
    ky = 2 * np.pi * np.fft.fftfreq(ny)[:, None]
    ex, ey = np.exp(1j * kx), np.exp(1j * ky)
    dxx = (ex - 2.0 + np.conj(ex)) * np.ones((ny, 1))
    dyy = (ey - 2.0 + np.conj(ey)) * np.ones((1, nx))
    dxy = np.sqrt(2.0) * (ex - 1.0) * (ey - 1.0)
    return np.stack([dxx, dyy, dxy])


def ref_hessian_deconv(rec, lam, rho, n_iter, prior, mode):
    g, H = hd.forward_model(rec, mode=mode)
    scale = g.max()
    g = g / scale
    D = ref_hessian_kernels(g.shape)
    G = hd._f2(g)
    denom = np.abs(H) ** 2 + rho * np.sum(np.abs(D) ** 2, axis=0) + rho
    f = g.copy()
    z = np.zeros((3,) + g.shape)
    a = np.zeros_like(z)
    u = np.maximum(g, 0.0)
    b = np.zeros_like(g)
    for _ in range(n_iter):
        num = np.conj(H) * G + rho * hd._f2(u - b)
        num = num + rho * np.sum(np.conj(D) * hd._f2(z - a), axis=0)
        f = np.real(hd._i2(num / denom))
        v = np.real(hd._i2(D * hd._f2(f))) + a
        if prior == 'iso':
            nrm = np.sqrt(np.sum(v ** 2, axis=0))
            z = v * np.maximum(1.0 - lam / (rho * np.maximum(nrm, 1e-12)), 0.0)
        else:
            z = np.sign(v) * np.maximum(np.abs(v) - lam / rho, 0.0)
        a = v - z
        w = f + b
        u = np.maximum(w, 0.0)
        b = w - u
    return np.fft.fftshift(np.maximum(f, 0.0) * scale)


def ref_rl_deconv(rec, n_iter, mode, eps=1e-9):
    g, H = hd.forward_model(rec, mode=mode)
    scale = g.max()
    g = np.clip(g / scale, 0.0, None)
    Hc = np.conj(H)
    f = np.maximum(g.copy(), eps)
    for _ in range(n_iter):
        blur = np.real(hd._i2(H * hd._f2(f)))
        ratio = g / np.maximum(blur, eps)
        corr = np.real(hd._i2(Hc * hd._f2(ratio)))
        f = np.maximum(f * corr, 0.0)
    return np.fft.fftshift(f * scale)


def main(nx):
    hd.attach(SIM_RECON)
    stack = synthetic_stack(nx)
    p = SIM_RECON(stack, NANG, NPH, NORDERS, DX, WL, NA)
    p.radius = 4.0 * nx * DX
    p.init_recon()
    p.separate(0, True)
    p.shift0()

    dx = p.dx / 2
    sweep = [(ANGS[0] + da, SPACING + ds)
             for da in (-0.04, 0.0, 0.017) for ds in (-0.04, 0.0, 0.023)]
    sweep += [(ANGS[1], SPACING), (ANGS[1], SPACING / 2), (ANGS[0], SPACING / 2)]

    # 1. separable modulation
    ang, sp = sweep[0]
    kx, ky = dx * np.cos(ang) / sp, dx * np.sin(ang) / sp
    check('modulation (separable)', ref_modulation(p, kx, ky),
          p._modulation(kx, ky, np.complex128))
    check('modulation (exact=True)', ref_modulation(p, kx, ky),
          p._modulation(kx, ky, np.complex128, exact=True))

    # 2. spectral peak from two 1-D transforms
    bad = 0
    for ang, sp in sweep:
        kx, ky = dx * np.cos(ang) / sp, dx * np.sin(ang) / sp
        for kk in ((kx, ky), (-kx, -ky)):
            bad += ref_peak_index(p, *kk) != p._peak_index(*kk)
    _rows.append(('peak index (%d cases)' % (2 * len(sweep)),
                  'bitwise' if not bad else '%d MISMATCH' % bad, 0.0, 0.0))

    # 3. disc mask without the square root
    for sx, sy in ((0, 0), p._peak_index(dx * np.cos(ANGS[0]) / SPACING,
                                        dx * np.sin(ANGS[0]) / SPACING)):
        check('zero_suppression(%d,%d)' % (sx, sy),
              ref_zero_suppression(p, sx, sy), p.zero_suppression(sx, sy))

    # 4. overlap support == non-zero pixels of the mask product
    kx, ky = dx * np.cos(ANGS[0]) / SPACING, dx * np.sin(ANGS[0]) / SPACING
    sx, sy = p._peak_index(kx, ky)
    msk = ref_zero_suppression(p, sx, sy) * ref_zero_suppression(p, 0, 0)
    check('overlap support', np.flatnonzero(msk.ravel() != 0.0), p._support(sx, sy))

    # 5. Fourier interpolation
    rng = np.random.default_rng(3)
    arr = rng.standard_normal((nx, nx))
    check('interp', ref_interp(p, arr), p.interp(arr))

    # 6. the overlap integral itself
    worst = (0.0, 0.0)
    for ang, sp in sweep:
        r = ref_getoverlap(p, 1, ang, sp)
        f = p.getoverlap(1, ang, sp)
        worst = max(worst, (abs(r[0] - f[0]) / abs(r[0]), abs(r[1] - f[1])))
    _rows.append(('getoverlap (%d cases)' % len(sweep), 'differs',
                  worst[1], worst[0]))

    # 7. reconstruction end to end
    x1 = p.mapoverlap(1, ANGS[0], SPACING, nps=4, r_ang=0.04, r_sp=0.04, verbose=False)
    a1 = p.getoverlap(1, x1[0], x1[1])
    p.add_recon(1, x1[0], x1[1], -a1[1], 0.8, False, True)
    p.finish_recon()

    # 8. the deconvolution add-on
    check('hessian kernels', ref_hessian_kernels((2 * nx, 2 * nx)),
          np.stack(np.broadcast_arrays(*hd._hessian_kernels((2 * nx, 2 * nx)))))
    for prior in ('iso', 'l1'):
        check('hessian_deconv (%s)' % prior,
              ref_hessian_deconv(p, 1e-4, 1.0, 6, prior, 'matched'),
              p.hessian_deconv(lam=1e-4, rho=1.0, n_iter=6, prior=prior,
                               mode='matched', verbose=False))
    check('rl_deconv', ref_rl_deconv(p, 6, 'matched'),
          p.rl_deconv(n_iter=6, mode='matched', verbose=False))

    print('\ngrid %d -> %d\n' % (nx, 2 * nx))
    print('%-28s %-12s %12s %12s' % ('check', 'result', 'max abs', 'max rel'))
    for name, res, ad, rd in _rows:
        print('%-28s %-12s %12.3e %12.3e' % (name, res, ad, rd))
    worst_rel = max(r[3] for r in _rows)
    print('\nworst relative deviation: %.3e' % worst_rel)
    return 0 if worst_rel < 1e-10 and not any('MISMATCH' in r[1] or 'SHAPE' in r[1]
                                              for r in _rows) else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nx', type=int, default=128)
    sys.exit(main(ap.parse_args().nx))
