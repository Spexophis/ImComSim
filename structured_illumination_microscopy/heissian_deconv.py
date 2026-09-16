"""
Hessian-regularized, positivity-constrained deconvolution as an add-on to
SIM_RECON.  Nothing in the reconstruction module is modified; this file only
reads attributes that recon() has already produced.

Usage
-----
    import hessian_sim
    hessian_sim.attach(SIM_RECON)          # adds the methods to the class

    r = SIM_RECON(stack, nang, nph, dx, wl, na)
    r.recon(nang, ang, spacing, phase, mag)
    f = r.hessian_deconv(lam=3e-3, n_iter=60)
    r.save_hessian('hessian_image.tif')

or, without touching the class at all:

    f = hessian_sim.hessian_deconv(r, lam=3e-3)

Conventions
-----------
recon() leaves self.S (object spectrum estimate) and self.Sden in *unshifted*
FFT order on the 2N x 2N interpolated grid.  Everything below stays in that
order and only fftshifts at the very end, matching self.finalimage.
"""

import numpy as np

try:
    from scipy.fft import fft2 as _fft2, ifft2 as _ifft2
    _KW = {"workers": -1}
except ImportError:
    from numpy.fft import fft2 as _fft2, ifft2 as _ifft2
    _KW = {}


def _f2(a):
    return _fft2(a, **_KW)


def _i2(a):
    return _ifft2(a, **_KW)


# ---------------------------------------------------------------------------
# quantities recovered from the reconstruction
# ---------------------------------------------------------------------------

def forward_model(rec, mode='matched', apodize=False, r_apod=None):
    """
    Return (g, H) such that the forward model  g = H * f  is the one that
    actually generated g.  Two choices:

    mode='matched'  (recommended)
        Snum = sum_i OTF_i^* D_i = (sum_i |OTF_i|^2) F,  so

            g = ifft2(Snum),   H = sum_i |OTF_i|^2 = Sden - mu^2

        H rolls off smoothly to zero at the band edge, so the ADMM does the
        inversion and REPLACES the Wiener step.  Do not use Snum/Sden as well.

    mode='wiener'
        g = ifft2(S) = the reconstruction you already have, and

            H = (Sden - mu^2) / Sden

        With a small mu this H is ~1 across the whole passband, i.e. very
        nearly the indicator of the support.  The data term then pins f = g
        wherever the data lives and the solver is close to a no-op.  Only
        useful with apodize=True, where the window supplies the roll-off.

    r_apod : cutoff in pixels of the 2N grid.  Default is the inscribed
             radius of the support, kp/sqrt(2) + sqrt(k0^2 - kp^2/2), the
             widest isotropic passband the data support.
    """
    Sden = np.real(rec.Sden).astype(np.float64)
    OTF2 = np.clip(Sden - rec.mu ** 2, 0.0, None)      # sum_i |OTF_i|^2

    if mode == 'matched':
        norm = OTF2.max()
        H = OTF2 / norm
        g = np.real(_i2(rec.Snum)) / norm
    elif mode == 'wiener':
        H = OTF2 / np.maximum(Sden, 1e-12)
        H = H / H.max()
        g = np.real(_i2(rec.S))
    else:
        raise ValueError("mode must be 'matched' or 'wiener'")

    if apodize:
        if r_apod is None:
            k0 = float(rec.radius)
            kp = 2.0 * float(np.max(rec.eh)) * k0 if len(rec.eh) else 0.0
            r_apod = kp / np.sqrt(2.0) + np.sqrt(max(k0 ** 2 - kp ** 2 / 2.0, 0.0))
        r = np.hypot(rec.xv, rec.yv)                   # unshifted order
        A = np.clip(1.0 - r / float(r_apod), 0.0, 1.0)
        H = H * A
        g = np.real(_i2(A * _f2(g)))

    return g, H


def effective_otf(rec, mode='matched', apodize=False, r_apod=None):
    """The transfer function of the chosen forward model, unshifted order.

    Note this is NOT what save_results() writes to 'effective_OTF.tif' --
    that file is |fftshift(S)|, the magnitude spectrum of the final image.
    """
    return forward_model(rec, mode=mode, apodize=apodize, r_apod=r_apod)[1]


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------

def _hessian_kernels(shape):
    """Fourier multipliers of (d_xx, d_yy, sqrt(2) d_xy), unshifted order."""
    ny, nx = shape
    kx = 2 * np.pi * np.fft.fftfreq(nx)[None, :]
    ky = 2 * np.pi * np.fft.fftfreq(ny)[:, None]
    ex, ey = np.exp(1j * kx), np.exp(1j * ky)
    dxx = (ex - 2.0 + np.conj(ex)) * np.ones((ny, 1))
    dyy = (ey - 2.0 + np.conj(ey)) * np.ones((1, nx))
    dxy = np.sqrt(2.0) * (ex - 1.0) * (ey - 1.0)
    return np.stack([dxx, dyy, dxy])


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------

def hessian_deconv(rec, lam=1e-4, rho=1.0, n_iter=60, prior='iso',
                   mode='matched', apodize=False, r_apod=None,
                   tol=1e-5, verbose=True):
    """
    min_f  0.5*||H f - g||^2 + lam * R(Psi f)   s.t.  f >= 0

    mode  : 'matched' inverts Snum with H = sum|OTF_i|^2 and replaces the
            Wiener step.  'wiener' post-processes Snum/Sden and is close to
            a no-op unless apodize=True -- see forward_model().

    prior : 'iso' -> sum_x ||(f_xx, f_yy, sqrt2 f_xy)(x)||_2   (no axis bias)
            'l1'  -> component-wise soft threshold (Hessian-SIM as published;
                     favours axis-aligned edges, which is the wrong bias here)

    Returns the deconvolved image, fftshifted to match rec.finalimage, and
    stores it on rec.hessian_image.
    """
    g, H = forward_model(rec, mode=mode, apodize=apodize, r_apod=r_apod)
    scale = g.max()
    g = g / scale                       # lam is not scale invariant

    D = _hessian_kernels(g.shape)
    G = _f2(g)
    denom = np.abs(H) ** 2 + rho * np.sum(np.abs(D) ** 2, axis=0) + rho

    f = g.copy()
    z = np.zeros((3,) + g.shape)
    a = np.zeros_like(z)
    u = np.maximum(g, 0.0)
    b = np.zeros_like(g)

    for it in range(n_iter):
        f_prev = f

        # f-update: quadratic subproblem, diagonal in Fourier
        num = np.conj(H) * G + rho * _f2(u - b)
        num = num + rho * np.sum(np.conj(D) * _f2(z - a), axis=0)
        f = np.real(_i2(num / denom))

        # z-update: prox of the Hessian penalty
        v = np.real(_i2(D * _f2(f))) + a
        if prior == 'iso':
            nrm = np.sqrt(np.sum(v ** 2, axis=0))
            z = v * np.maximum(1.0 - lam / (rho * np.maximum(nrm, 1e-12)), 0.0)
        elif prior == 'l1':
            z = np.sign(v) * np.maximum(np.abs(v) - lam / rho, 0.0)
        else:
            raise ValueError("prior must be 'iso' or 'l1'")
        a = v - z

        # u-update: projection onto f >= 0
        w = f + b
        u = np.maximum(w, 0.0)
        b = w - u

        rel = np.linalg.norm(f - f_prev) / (np.linalg.norm(f) + 1e-12)
        if verbose and (it % 10 == 0 or it == n_iter - 1):
            res = np.real(_i2(H * _f2(f))) - g
            print("  it %3d  resid %.3e  rel %.2e"
                  % (it, np.sqrt((res ** 2).mean()), rel))
        if rel < tol:
            break

    out = np.fft.fftshift(np.maximum(f, 0.0) * scale)
    rec.hessian_image = out
    rec.H_eff = H
    return out


def rl_deconv(rec, n_iter=50, mode='matched', apodize=False, r_apod=None,
              reg=0.0, eps=1e-9, verbose=True):
    """
    Richardson-Lucy with the cross-star effective PSF.

        f <- f * [ h^T (x) ( g / (h (x) f) ) ]

    h = ifft2(H) is the effective PSF; in mode='matched' it is

        h(x) = (psf * psf)(x) * [ n_ang + (m^2/2)(cos(2 pi kp.x) + cos(2 pi kp.y)) ]

    i.e. the widefield autocorrelation modulated by the two orthogonal
    cosines -- that IS the cross star.  The bracket stays positive for m <= 2,
    so h >= 0 and RL's kernel requirement is met.  (This is not true of the
    Wiener-flattened PSF, whose sidelobes reach -0.23.)

    reg : optional Tikhonov-Miller damping on the multiplicative factor,
          f <- f * ratio / (1 + reg * ratio_curvature).  reg=0 is plain RL.

    RL assumes Poisson data.  Snum is a signed linear combination of frames,
    so the likelihood is not exactly right -- stop early and check by eye
    rather than running to convergence.
    """
    g, H = forward_model(rec, mode=mode, apodize=apodize, r_apod=r_apod)
    scale = g.max()
    g = np.clip(g / scale, 0.0, None)

    Hc = np.conj(H)                      # H is real and even -> h^T = h
    f = np.maximum(g.copy(), eps)

    for it in range(n_iter):
        blur = np.real(_i2(H * _f2(f)))
        ratio = g / np.maximum(blur, eps)
        corr = np.real(_i2(Hc * _f2(ratio)))
        if reg > 0.0:
            corr = corr / (1.0 + reg * (corr - 1.0) ** 2)
        f = np.maximum(f * corr, 0.0)
        if verbose and (it % 10 == 0 or it == n_iter - 1):
            res = np.real(_i2(H * _f2(f))) - g
            print("  RL it %3d  resid %.3e  max %.3f"
                  % (it, np.sqrt((res ** 2).mean()), f.max()))

    out = np.fft.fftshift(f * scale)
    rec.rl_image = out
    return out


def effective_psf(rec, mode='matched', apodize=False, r_apod=None):
    """The cross-star PSF itself, fftshifted, peak-normalized."""
    H = forward_model(rec, mode=mode, apodize=apodize, r_apod=r_apod)[1]
    h = np.fft.fftshift(np.real(_i2(H)))
    return h / h.max()


def save_hessian(rec, path='hessian_image.tif'):
    import tifffile as tf
    tf.imwrite(path, rec.hessian_image.astype(np.float32),
               photometric='minisblack')


def save_true_otf(rec, path='H_eff.tif', **kw):
    import tifffile as tf
    tf.imwrite(path, np.fft.fftshift(effective_otf(rec, **kw)).astype(np.float32),
               photometric='minisblack')


# ---------------------------------------------------------------------------

def attach(cls):
    """Bolt the methods onto SIM_RECON without editing the reconstruction file."""
    cls.forward_model = forward_model
    cls.effective_otf = effective_otf
    cls.hessian_deconv = hessian_deconv
    cls.rl_deconv = rl_deconv
    cls.effective_psf = effective_psf
    cls.save_hessian = save_hessian
    cls.save_true_otf = save_true_otf
    return cls


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Smoke test on a stand-in object carrying the attributes recon() sets.
    import tifffile as tf

    class Mock:
        pass

    img = tf.imread('/mnt/user-data/uploads/final_image_08122026.tif').astype(np.float64)
    N = img.shape[0]
    m = Mock()
    m.mu = 0.02
    m.radius = 217.0
    m.eh = np.array([180.0 / (2 * 217.0)])
    x = np.arange(N); x[N // 2:] -= N
    m.xv, m.yv = x[:, None], x[None, :]

    OTF2 = np.zeros((N, N))
    for dy, dx in [(0, 0), (0, 180), (0, -180), (180, 0), (-180, 0)]:
        OTF2 += np.clip(1 - np.hypot(m.yv - dy, m.xv - dx) / 217.0, 0, 1) ** 2
    m.Sden = OTF2 + m.mu ** 2
    m.S = _f2(np.fft.ifftshift(img))
    m.Snum = m.S * m.Sden

    for mode, lam in [('wiener', 3e-3), ('matched', 1e-4)]:
        g, H = forward_model(m, mode=mode)
        print(mode, ' H along axis:',
              [round(float(H[0, r]), 3) for r in (0, 100, 200, 250, 300, 350, 390)])
        f = np.fft.ifftshift(hessian_deconv(m, lam=lam, n_iter=30, mode=mode,
                                            verbose=False))
        print('   ||f-g||/||g|| = %.3f\n'
              % (np.linalg.norm(f / f.max() - g / g.max())
                 / np.linalg.norm(g / g.max())))
