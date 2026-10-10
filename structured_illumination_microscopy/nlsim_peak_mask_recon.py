"""
Peak-masked summation reconstruction for 2D (nonlinear) SIM raw data.

Pipeline
    1. FFT-interpolate every raw frame to half the pixel size.
    2. Estimate wave vector and per-frame phase of the sinusoidal pattern of every angle from the first-order
       peak of the phase-demodulated data, and generate one sinusoidal pattern image per raw frame.
    3. Threshold each pattern (keep only the pixels around the peaks of the sine wave) and use it as a mask on
       the corresponding raw frame.
    4. Sum the masked frames of each angle (-> one image per angle).
    5. Combine the angle images with a Wiener filter in Fourier space. The effective transfer function of an
       angle image is derived analytically from the mask harmonics, the detection OTF and the measured
       modulation depth (see ``effective_otf``).

Raw stack layout: (n_angles * n_phases, ny, nx), angle-major.
"""

import json
from pathlib import Path

import numpy as np
import tifffile as tf
from scipy.fft import fft2, ifft2
from scipy.signal import resample
from scipy.signal.windows import tukey

WORKERS = -1


def fft_upsample2(stack, pad=32):
    """x2 interpolation by Fourier zero padding. Frames are mirror padded first to suppress wrap-around ringing;
    upsampled sample 2*i is exactly raw sample i."""
    n, ny, nx = stack.shape
    out = np.empty((n, 2 * ny, 2 * nx), np.float32)
    for i in range(n):
        f = np.pad(stack[i].astype(np.float64), pad, mode='reflect')
        f = resample(resample(f, 2 * f.shape[0], axis=0), 2 * f.shape[1], axis=1)
        out[i] = f[2 * pad:2 * pad + 2 * ny, 2 * pad:2 * pad + 2 * nx]
    return out


def _zoom_dft(img, f0, f1, half, step):
    """Complex DFT of ``img`` on a fine frequency grid around (f0, f1) [bins]; returns the maximum."""
    ny, nx = img.shape
    g = np.arange(-half, half + 1e-9, step)
    a = np.exp(-2j * np.pi * np.outer(f0 + g, np.arange(ny)) / ny)
    b = np.exp(-2j * np.pi * np.outer(f1 + g, np.arange(nx)) / nx)
    z = a @ img @ b.T
    k = np.unravel_index(np.abs(z).argmax(), z.shape)
    return f0 + g[k[0]], f1 + g[k[1]], z[k]


def _dft_at(img, f0, f1):
    ny, nx = img.shape
    return np.exp(-2j * np.pi * f0 * np.arange(ny) / ny) @ img @ np.exp(-2j * np.pi * f1 * np.arange(nx) / nx)


def estimate_patterns(raw, min_period=3.0, max_period=12.0):
    """raw: (nang, nph, ny, nx), offset removed, original sampling.
    Returns per angle: wave vector (cycles per RAW pixel, axis0/axis1), per-frame pattern phase [rad] (frame by frame, linear fit,
    and equally spaced with the same mean), defined for pattern(x) = 0.5 * (1 + cos(2*pi*k.x + phase_j)) with x in raw pixel units (origin = pixel 0)."""
    nang, nph, ny, nx = raw.shape
    w = np.outer(tukey(ny, 0.2), tukey(nx, 0.2))
    j = np.arange(nph)
    ks, phases, phases_eq, phases_fr, info = [], [], [], [], []
    fy = np.fft.fftfreq(ny)[:, None] * ny
    fx = np.fft.fftfreq(nx)[None, :] * nx
    fr = np.hypot(fy, fx)
    band = (fr > ny / max_period) & (fr < ny / min_period)
    for a in range(nang):
        d = np.tensordot(np.exp(-2j * np.pi * j / nph), raw[a], axes=(0, 0)) / nph
        spec = np.abs(fft2(d * w, workers=WORKERS))
        spec[~band] = 0
        k = np.unravel_index(spec.argmax(), spec.shape)
        f0, f1 = (float(kk if kk < s // 2 else kk - s) for kk, s in zip(k, spec.shape))
        f0, f1, _ = _zoom_dft(d * w, f0, f1, 1.5, 0.02)
        f0, f1, _ = _zoom_dft(d * w, f0, f1, 0.04, 0.002)
        # per-frame phase of the first order; fit  psi_j = phi0 + j*eps  on top of the ideal step 2*pi/nph
        z = np.array([_dft_at((raw[a, i] - raw[a, i].mean()) * w, f0, f1) for i in j])
        psi = np.unwrap(np.angle(z * np.exp(-2j * np.pi * j / nph)))
        eps, phi0 = np.polyfit(j, psi, 1)
        res = psi - (phi0 + eps * j)
        ks.append((f0 / ny, f1 / nx))
        phases.append(2 * np.pi * j / nph + phi0 + eps * j)                 # measured (linear fit)
        phases_eq.append(2 * np.pi * j / nph + phi0 + eps * (nph - 1) / 2)  # equally spaced, same mean phase
        phases_fr.append(2 * np.pi * j / nph + psi)                         # measured frame by frame
        info.append(dict(angle=a, k_cycles_per_px=(f0 / ny, f1 / nx), period_px=1 / np.hypot(f0 / ny, f1 / nx),
                         direction_deg=float(np.degrees(np.arctan2(f1, f0))), phi0=float(phi0),
                         step_error_deg=float(np.degrees(eps)), phase_resid_rms_deg=float(np.degrees(res.std()))))
    return np.array(ks), np.array(phases), np.array(phases_eq), np.array(phases_fr), info


def otf_circ(kr, kc):
    """Diffraction-limited incoherent OTF (circular pupil), kr / kc = radius / cutoff."""
    r = np.clip(kr / kc, 0, 1)
    return (2 / np.pi) * (np.arccos(r) - r * np.sqrt(1 - r ** 2))


class PeakMaskRecon:

    def __init__(self, stack, n_angles, n_phases, pixel_size, na, wavelength, offset=100.0, threshold=0.85,
                 mask='ramp'):
        n, ny, nx = stack.shape
        if n != n_angles * n_phases:
            raise ValueError('stack has %d frames, expected %d x %d' % (n, n_angles, n_phases))
        if ny != nx:
            raise ValueError('square images expected')
        self.nang, self.nph, self.dx0, self.na, self.wl = n_angles, n_phases, pixel_size, na, wavelength
        self.thr = threshold
        self.mask_kind = mask
        self.raw = (stack.astype(np.float64) - offset).reshape(n_angles, n_phases, ny, nx)
        self.n0 = nx
        self.n = 2 * nx
        self.dx = pixel_size / 2
        self.kc = 2 * na / wavelength                       # OTF cutoff [cycles / um]
        # coordinates on the interpolated grid, in RAW pixel units
        self.u = np.arange(self.n) / 2.0
        uw = np.fft.fftfreq(self.n) * self.n / 2.0           # wrapped (origin at index 0), for OTF / PSF
        self.uw = uw
        f = np.fft.fftfreq(self.n, self.dx)                  # cycles / um
        self.kr = np.hypot(f[:, None], f[None, :])
        self.bins = self.kr * (self.n * self.dx)             # radius in frequency bins (bin = 1/(n dx))
        self.H = otf_circ(self.kr, self.kc)

    # ---- helpers -------------------------------------------------------------------------------------------
    def shifted_otf(self, k, m):
        """H(kappa - m f) on the FFT grid for pattern wave vector k (cycles per raw px)."""
        fm = m * np.hypot(*k) / self.dx0                     # cycles / um
        ang = np.arctan2(k[1], k[0])
        f = np.fft.fftfreq(self.n, self.dx)
        ky, kx = np.meshgrid(f, f, indexing='ij')
        return otf_circ(np.hypot(ky - fm * np.cos(ang), kx - fm * np.sin(ang)), self.kc)

    def carrier(self, k):
        c = 2 * np.pi * (k[0] * self.u[:, None] + k[1] * self.u[None, :])
        return np.mod(c, 2 * np.pi).astype(np.float32)

    def mask_values(self, pat):
        """Mask from the normalised pattern (0..1): 'binary' keeps pixels with pattern >= threshold;
        'ramp' keeps the same pixels but weights them linearly from 0 (at the threshold) to 1 (at the peak)."""
        if self.mask_kind == 'binary':
            return (pat >= self.thr).astype(np.float32)
        return np.clip((pat - self.thr) / (1 - self.thr), 0, 1)

    def mask_harmonics(self, m):
        """Fourier coefficient w_m of the mask as a function of the pattern phase (numerical, even function)."""
        t = (np.arange(8192) + 0.5) * 2 * np.pi / 8192
        mk = self.mask_values(0.5 * (1 + np.cos(t)))
        return float(np.mean(mk * np.cos(m * t)))

    # ---- pipeline ------------------------------------------------------------------------------------------
    def lockin_demod(self, ang, alpha):
        """Lock-in demodulation of one angle (same as SIM_RECON.lockin_demod in sim_recon/sim2d_recon_m.py):
        removes the fraction ``alpha`` of the unmodulated (phase-averaged) signal from every frame and adds the
        pixel-wise modulation amplitude estimated from the frame-to-frame differences along the phase axis.
        Only the phase-independent (zero-order) term is changed, the modulated harmonics are untouched."""
        a = self.raw[ang]
        d = np.empty_like(a)
        d[0] = a[0] - a[-1]                     # == a - np.roll(a, 1, axis=0)
        np.subtract(a[1:], a[:-1], out=d[1:])
        idc2 = np.sqrt((2.0 / self.nph ** 2) * np.einsum('ijk,ijk->jk', d, d))
        return a - alpha * a.mean(0) + idc2

    def run(self, mu='auto', **kw):
        """prepare() (interpolation, patterns, masks, angle sums, modulation depth) followed by wiener()."""
        self.prepare(**kw)
        self.wiener(mu)
        return self

    def prepare(self, keep_stacks=False, measure_modulation=True, p1=None, phase_mode='frame',
                lockin_alpha=None, ghost_correction='subtract', otf_model='harmonic', resolution_along=0.040):
        self.otf_model, self.resolution_along = otf_model, resolution_along
        # pattern parameters come from the plain offset-subtracted data
        self.k, self.phase_lin, self.phase_eq, self.phase_frame, self.pinfo = estimate_patterns(self.raw)
        data = self.raw if lockin_alpha is None else np.stack([self.lockin_demod(a, lockin_alpha)
                                                               for a in range(self.nang)])
        self.up = fft_upsample2(data.reshape(-1, self.n0, self.n0)).reshape(self.nang, self.nph, self.n, self.n)
        # 'frame': measured phase of every frame (the phase steps are not uniform: residuals of up to +-20 deg),
        # the patterns then overlap the pattern in each frame. 'equal': equally spaced steps, which cancel the
        # zero-order ghost of the 10-phase sum exactly but misplace the mask by the step error of each frame.
        self.phase = dict(frame=self.phase_frame, equal=self.phase_eq, linear=self.phase_lin)[phase_mode]
        if phase_mode == 'frame':
            self.refine_frame_phases()
        self.angle_sum, self.patterns, self.filtered = [], [], []
        frac = []
        self.weights = np.array([self.frame_weights(a) if ghost_correction == 'weights' else np.ones(self.nph)
                                 for a in range(self.nang)])
        for a in range(self.nang):
            car = self.carrier(self.k[a])
            y = np.zeros((self.n, self.n), np.float64)
            if ghost_correction == 'subtract':
                comp = self.lsq_components(a)                                  # c0, Cr, Ci (zero and first order)
                y_model = np.zeros((self.n, self.n), np.float64)
            for j in range(self.nph):
                pat = 0.5 * (1 + np.cos(car + np.float32(self.phase[a, j])))    # sinusoidal pattern image
                msk = self.mask_values(pat)                                    # peaks of the sine wave only
                filt = self.up[a, j] * msk                                     # masked raw frame
                y += self.weights[a, j] * filt
                if ghost_correction == 'subtract':
                    nu = self.phase[a, j] - self.phase[a, 0]
                    y_model += msk * (comp[0] + 2 * (comp[1] * np.cos(nu) - comp[2] * np.sin(nu)))
                frac.append(float((msk > 0).mean()))
                if keep_stacks:
                    self.patterns.append((pat * 255).astype(np.uint8))
                    self.filtered.append(filt.astype(np.float32))
            if ghost_correction == 'subtract':
                # replace the actual masked zero + first order by the ideal one (equally spaced phases):
                # N*w0*c0 + 2*N*w1*c1, c1 = envelope of the first order with the carrier removed
                kappa = car.astype(np.float64) + self.phase[a, 0]
                c1 = np.real((comp[1] + 1j * comp[2]) * np.exp(-1j * kappa))
                y += self.nph * (self.mask_harmonics(0) * comp[0] + 2 * self.mask_harmonics(1) * c1) - y_model
            self.angle_sum.append(y)
        self.mask_fraction = float(np.mean(frac))
        self.p1 = np.array([self.estimate_modulation(a) if (measure_modulation and p1 is None) else (p1 or 0.0)
                            for a in range(self.nang)])
        self.wf = self.up.mean((0, 1))
        return self

    def lsq_components(self, a):
        """Zero order c0 and first order C = Cr + i Ci (carrier included) of angle a from the 10 frames, least squares
        with the true frame phases:  I_j = c0 + 2 Re(C exp(i nu_j)),  nu_j = phase_j - phase_0."""
        nu = self.phase[a] - self.phase[a][0]
        dmat = np.c_[np.ones(self.nph), 2 * np.cos(nu), -2 * np.sin(nu)]
        return np.tensordot(np.linalg.pinv(dmat), self.up[a].astype(np.float64), axes=(1, 0))

    def frame_weights(self, a, orders=2):
        """Per-frame weights of the sum that null the zero-order leakage. With unequally spaced frame phases
        sum_j exp(i n theta_j) != 0, so the unmodulated image (times mask harmonic n) leaves a ghost at n times the
        pattern frequency. The minimum-norm correction of the unit weights with sum_j b_j exp(i n theta_j) = 0 for
        n = 1..orders and sum_j b_j = nph is used. orders=2 removes the ghosts at f and 2f (the 2f ghost of a ramp
        mask is as large as the one at f); the weights stay within about 0.8..1.25 (0.5..1.5 for the angle with the
        largest phase errors), which costs ~4 % noise variance."""
        th = self.phase[a]
        rows = [np.ones(self.nph)]
        for n in range(1, orders + 1):
            rows += [np.cos(n * th), np.sin(n * th)]
        c = np.array(rows)
        t = np.zeros(len(rows))
        t[0] = self.nph
        return np.ones(self.nph) + c.T @ np.linalg.solve(c @ c.T, t - c @ np.ones(self.nph))

    def refine_frame_phases(self):
        """One refinement pass on the interpolated grid: the phase of every frame demodulated with its generated
        pattern is the remaining offset (pattern fitted on the windowed raw grid); add it to the pattern phase."""
        self.phase = self.phase.copy()
        for a in range(self.nang):
            car = self.carrier(self.k[a]).astype(np.float64)
            for j in range(self.nph):
                d = self.up[a, j].astype(np.float64)
                z = np.sum((d - d.mean()) * np.exp(-1j * (car + self.phase[a, j])))
                self.phase[a, j] += np.angle(z)

    def check_patterns(self, tiles=8, scan=True, shifts=np.arange(-1.0, 1.0001, 0.1)):
        """Sub-pixel check of the generated sinusoidal patterns against the pattern in the raw images.
        (1) Per frame, the frame is demodulated with the generated pattern: the phase of the result is the residual
            phase between image pattern and generated pattern (shift = phase / (2 pi |k|) along k).
        (2) Per angle, the residual phase map over a tiles x tiles grid is fitted with a plane: its gradient is the
            error of the wave vector (phase drift across the field), the residual scatter is compared with the
            scatter expected from noise (independent top/bottom and left/right half-image estimates).
        (3) scan: the actual pattern image is shifted along k by sub-pixel steps and the correlation with the frame
            is maximised (parabolic peak); independent of the demodulation in (1).
        Shifts are in pixels of the interpolated grid (half the raw pixel)."""
        n = self.n
        yy, xx = np.indices((n, n))
        halves = [yy < n // 2, yy >= n // 2, xx < n // 2, xx >= n // 2]
        ts = n // tiles
        ty, tx = np.indices((tiles, tiles))
        cy, cx = (ty + .5) * ts - n / 2, (tx + .5) * ts - n / 2
        rep = []
        for a in range(self.nang):
            kk = np.hypot(*self.k[a]) / 2                       # cycles per interpolated pixel
            car = self.carrier(self.k[a]).astype(np.float64)
            glob, scan_pk, zt, hz = [], [], np.zeros((tiles, tiles), complex), [[], []]
            for j in range(self.nph):
                d = self.up[a, j].astype(np.float64)
                d = d - d.mean()
                e = np.exp(-1j * (car + self.phase[a, j]))
                prod = d * e
                g = prod.sum()
                glob.append(g)
                zt += (prod * np.exp(-1j * np.angle(g)))[:tiles * ts, :tiles * ts].reshape(tiles, ts, tiles, ts).sum((1, 3))
                for h in range(2):                              # noise of the phase estimate from half images
                    z1 = prod[halves[2 * h]].sum()
                    z2 = prod[halves[2 * h + 1]].sum()
                    hz[h].append(np.angle(z1 * np.conj(z2)))
                if scan:
                    sig = d[::2, ::2]
                    c = []
                    for sh in shifts:
                        pat = np.cos(car + self.phase[a, j] + 2 * np.pi * kk * sh)[::2, ::2]
                        c.append(np.corrcoef(sig.ravel(), pat.ravel())[0, 1])
                    c = np.array(c)
                    i = int(np.clip(c.argmax(), 1, len(c) - 2))
                    den = c[i - 1] - 2 * c[i] + c[i + 1]
                    scan_pk.append(shifts[i] + 0.5 * (c[i - 1] - c[i + 1]) / den * (shifts[1] - shifts[0]) if den else np.nan)
            glob = np.array(glob)
            dphi = np.angle(glob)
            ph, amp = np.angle(zt), np.abs(zt)
            ok = amp > 0.3 * np.median(amp)
            A = np.c_[np.ones(ok.sum()), cy[ok], cx[ok]]
            sol, *_ = np.linalg.lstsq(A * amp[ok][:, None], ph[ok] * amp[ok], rcond=None)
            resid = ph[ok] - A @ sol
            # tile noise expected from the half-image scatter (tile = 1/tiles^2 of the area, half = 1/2)
            noise_half = np.sqrt(np.mean(np.concatenate(hz) ** 2) / 2)
            rep.append(dict(angle=a, frame_phase_residual_deg=np.degrees(dphi).round(2).tolist(),
                            frame_shift_px=(dphi / (2 * np.pi * kk)).round(3).tolist(),
                            max_abs_shift_px=float(np.abs(dphi / (2 * np.pi * kk)).max()),
                            scan_shift_px=np.round(scan_pk, 3).tolist() if scan else None,
                            drift_deg_per_fov=(float(np.degrees(sol[1]) * n), float(np.degrees(sol[2]) * n)),
                            wavevector_error_cycles_per_px=(float(sol[1] / (2 * np.pi) * 2), float(sol[2] / (2 * np.pi) * 2)),
                            tile_resid_rms_deg=float(np.degrees(resid.std())),
                            tile_noise_expected_deg=float(np.degrees(noise_half * np.sqrt(tiles ** 2 / 2)))))
        self.pattern_check = rep
        return rep

    def estimate_modulation(self, a, band=(36, 96)):
        """First-order modulation depth p1 (cosine amplitude relative to the object) from the overlap of the
        shifted-back first-order component with the zero order, restricted to a band of object frequencies where
        the object has signal (the camera offset and unmodulated haze dilute the lowest frequencies)."""
        # least squares with the true (non-uniform) phases: the zero order does not leak into the first order as it
        # does for a plain Fourier sum over unequally spaced phases
        u3 = self.lsq_components(a)
        c0 = u3[0]
        c1 = u3[1] + 1j * u3[2]
        car = 2 * np.pi * (self.k[a][0] * self.u[:, None] + self.k[a][1] * self.u[None, :]) + self.phase[a][0]
        c1b = c1 * np.exp(-1j * car)                                  # envelope, carrier removed
        w = np.outer(tukey(self.n, 0.1), tukey(self.n, 0.1))
        f0 = fft2(c0 * w, workers=WORKERS)
        f1 = fft2(2 * c1b * w, workers=WORKERS)
        hs = self.H
        hp = self.shifted_otf(self.k[a], -1)                          # H(s + f)
        w0 = hp * f0
        w1 = hs * f1
        msk = (self.bins >= band[0]) & (self.bins < band[1]) & (hs > 0.02) & (hp > 0.02)
        return float(np.abs((w1 * np.conj(w0))[msk].sum() / (w0 * np.conj(w0)).real[msk].sum()))

    def effective_otf(self, a, orders=1):
        """Transfer function of the summed masked image of angle a (normalised to T(0) = 1).
        Sum_j M_j(x) I_j(x) keeps, for every excitation harmonic m of the data, the mask harmonic -m:
            T(kappa) = sum_m w_m * r_m * H(kappa - m f),   r_m = p_m / 2  (r_0 = 1),
        w_m = Fourier coefficient of the binary mask, p_m = modulation depth of harmonic m."""
        t = self.mask_harmonics(0) * self.H
        r = {1: self.p1[a] / 2}
        for m in range(1, orders + 1):
            wr = self.mask_harmonics(m) * r.get(m, 0.0)
            if wr:
                t = t + wr * (self.shifted_otf(self.k[a], m) + self.shifted_otf(self.k[a], -m))
        return t / t.flat[0]

    def anisotropic_otf(self, a, resolution):
        """Analytic OTF of angle a with the given resolution [um] (= 1 / cutoff) along the pattern direction and the
        diffraction-limited cutoff 2NA/lambda perpendicular to it: the circular-aperture OTF with the coordinate
        along the pattern stretched by s = (1/resolution) / (2NA/lambda), i.e. an elliptical support.
        T(kappa) = H(sqrt((kappa_par / s)^2 + kappa_perp^2))."""
        s = (1.0 / resolution) / self.kc
        f = np.fft.fftfreq(self.n, self.dx)
        ky, kx = np.meshgrid(f, f, indexing='ij')
        d = np.array(self.k[a]) / np.hypot(*self.k[a])
        par = d[0] * ky + d[1] * kx
        perp = -d[1] * ky + d[0] * kx
        return otf_circ(np.hypot(par / s, perp), self.kc)

    def transfer(self, a):
        """Transfer function of the angle-a image used in the Wiener filter."""
        return self.effective_otf(a) if self.otf_model == 'harmonic' else self.anisotropic_otf(a, self.resolution_along)

    def wiener(self, mu):
        """Wiener combination of the angle images, see ``wiener_combine``."""
        win = np.outer(tukey(self.n, 0.05), tukey(self.n, 0.05))
        fs, ts = [], []
        for a in range(self.nang):
            y = self.angle_sum[a]
            y = y / y.mean()                                          # common DC level: object DC = 1
            fs.append(fft2(y * win, workers=WORKERS))
            ts.append(self.transfer(a))
        self.otf_eff = ts
        self.S, self.otf_total, noise = wiener_combine(fs, ts, self.bins, mu, per_image_noise=False)
        self.noise_power = float(np.mean(noise)) if noise is not None else None
        self.final = np.real(ifft2(self.S, workers=WORKERS))
        self.final *= self.up.mean()                                  # restore counts (offset-subtracted)


def wiener_combine(fs, ts, bins, mu='auto', per_image_noise=False):
    """Wiener combination of images with known transfer functions:
        S = sum_i conj(T_i) F_i / n_i / (sum_i |T_i|^2 / n_i + 1 / P_S(kappa)).
    F_i: Fourier transform of image i (normalised to DC = 1), T_i: its transfer function (T(0) = 1), n_i: its noise
    power per frequency bin, estimated from the bins outside every support but below the raw Nyquist frequency
    (per image if per_image_noise, else the common mean), P_S: object power, estimated from the radially averaged
    whitened spectra divided by the transfer functions (mu = 'auto'). mu = float: constant NSR = mu**2 and equal
    weights. Returns S, sum_i |T_i|^2 and the noise powers (None for a constant mu)."""
    n = fs[0].shape[0]
    t2 = sum(np.abs(t) ** 2 for t in ts)
    if mu != 'auto':
        return sum(np.conj(t) * f for t, f in zip(ts, fs)) / (t2 + mu ** 2), t2, None
    # noise: bins outside every effective support but inside the band that still carries (interpolated) noise,
    # i.e. below the raw Nyquist frequency (bin n/4); beyond it the zero-padded spectrum holds no noise at all
    far = (t2 < 1e-4) & (bins < 0.97 * n / 4)
    if far.sum() < 2000:                                               # supports cover the whole band: take the
        far = (bins > 0.9 * n / 4) & (bins < 0.97 * n / 4)             # edge below raw Nyquist
    noise = np.array([np.mean(np.abs(f[far]) ** 2) for f in fs])
    if not per_image_noise:
        noise = np.full(len(fs), noise.mean())
    g = [f / np.sqrt(z) for f, z in zip(fs, noise)]                    # whitened data and transfer functions
    u = [t / np.sqrt(z) for t, z in zip(ts, noise)]
    u2 = sum(np.abs(x) ** 2 for x in u)
    pg = np.mean([np.abs(x) ** 2 for x in g], axis=0)
    rb = np.minimum(np.round(bins).astype(int), n // 2)
    cnt = np.maximum(np.bincount(rb.ravel(), minlength=n // 2 + 1), 1)
    prof = lambda x: np.bincount(rb.ravel(), x.ravel(), minlength=n // 2 + 1) / cnt
    nm = float(noise.mean())
    ps = (prof(pg) - 1) * nm / np.maximum(prof(u2) / len(fs) * nm, 1e-6)
    ps = np.convolve(np.maximum(ps, nm / 1e3), np.ones(5) / 5, mode='same')
    ps = np.minimum.accumulate(np.maximum(ps, nm / 1e3))               # object power decreases with frequency
    s = sum(np.conj(x) * y for x, y in zip(u, g)) / (u2 + (nm / ps)[rb] / nm)
    return s, t2, noise


class MultiStackRecon:
    """Several acquisitions (stacks) of the same field. Two ways to use them:
      'combine' (recommended): every stack gets its own wave vectors, per-frame phases, patterns and masked angle
          sums; the angle sums are plain images of the object (no pattern phase left), so they are registered with
          sub-pixel accuracy (phase correlation of the widefield images, applied as a Fourier-space shift) and all
          stacks x angles are combined in one Wiener filter, every image weighted by 1 / its noise power. Bleaching
          is normalised by the DC level of each angle image.
      'raw': average the raw frames (frame i of every stack) first. Only valid if the pattern phases and wave vectors
          agree between the stacks and the sample did not move, see ``compare_patterns``."""

    def __init__(self, stacks, n_angles, n_phases, pixel_size, na, wavelength, offset=100.0, threshold=0.85,
                 mask='ramp'):
        self.args = (n_angles, n_phases, pixel_size, na, wavelength, offset, threshold, mask)
        self.stacks = stacks
        self.recs = [PeakMaskRecon(s, *self.args) for s in stacks]
        self.nstack = len(stacks)
        self.nang, self.n = n_angles, 2 * stacks[0].shape[-1]
        self.n0 = stacks[0].shape[-1]

    def compare_patterns(self, max_deg=5.0):
        """Compare wave vectors and per-frame pattern phases of stack s with stack 0 (per angle):
        drift = phase difference accumulated over the field by the wave-vector difference [deg],
        offset = mean phase difference (pattern or sample shift along k) [deg and raw px],
        scatter = rms of the per-frame phase difference about its mean [deg].
        raw averaging is only sensible if all three are below max_deg."""
        est = [estimate_patterns(r.raw) for r in self.recs]
        rep = []
        for s in range(1, self.nstack):
            for a in range(self.nang):
                dk = np.array(est[s][0][a]) - np.array(est[0][0][a])
                dphi = np.angle(np.exp(1j * (est[s][3][a] - est[0][3][a])))
                off = np.angle(np.mean(np.exp(1j * dphi)))
                scat = np.angle(np.exp(1j * (dphi - off))).std()
                kk = np.hypot(*est[0][0][a])
                rep.append(dict(stack=s, angle=a, drift_deg=float(np.degrees(2 * np.pi * np.hypot(*dk) * self.n0)),
                                offset_deg=float(np.degrees(off)), offset_px=float(off / (2 * np.pi * kk)),
                                scatter_deg=float(np.degrees(scat))))
        self.pattern_comparison = rep
        self.raw_average_ok = all(max(abs(q['drift_deg']), abs(q['offset_deg']), q['scatter_deg']) < max_deg
                                  for q in rep)
        return rep

    def run(self, mode='combine', mu='auto', register=True, upsample=50, **kw):
        if mode == 'raw':
            return self.run_raw(mu=mu, **kw)
        for r in self.recs:
            r.prepare(**kw)
        self.rec = self.recs[0]
        self.shifts = [np.zeros(2)] + [self.register(r, upsample) if register else np.zeros(2) for r in self.recs[1:]]
        n = self.n
        fy = np.fft.fftfreq(n)[:, None]
        fx = np.fft.fftfreq(n)[None, :]
        win = np.outer(tukey(n, 0.05), tukey(n, 0.05))
        fs, ts, self.angle_sum, wfs = [], [], [], []
        for r, d in zip(self.recs, self.shifts):
            ramp = np.exp(-2j * np.pi * (fy * d[0] + fx * d[1]))        # shift by +d (registers onto stack 0)
            wfs.append(np.real(ifft2(fft2(r.wf, workers=WORKERS) * ramp, workers=WORKERS)))
            for a in range(self.nang):
                y = r.angle_sum[a] / r.angle_sum[a].mean()
                self.angle_sum.append(np.real(ifft2(fft2(y, workers=WORKERS) * ramp, workers=WORKERS)))
                fs.append(fft2(y * win, workers=WORKERS) * ramp)
                ts.append(r.transfer(a))
        self.S, self.otf_total, noise = wiener_combine(fs, ts, self.rec.bins, mu, per_image_noise=True)
        self.noise_power = None if noise is None else noise.reshape(self.nstack, self.nang).mean(1).tolist()
        self.final = np.real(ifft2(self.S, workers=WORKERS)) * self.recs[0].up.mean()   # counts of stack 0 (reference)
        self.wf = np.mean(wfs, axis=0)
        self.p1 = np.array([r.p1 for r in self.recs])
        return self

    def register(self, rec, upsample=50, sigma=2.0):
        """Shift [rows, cols] of the interpolated grid that registers the widefield image of ``rec`` onto stack 0:
        plain (not phase-normalised) cross-correlation of band-passed widefield images (difference of Gaussians,
        sigma .. 6 sigma interpolated px, i.e. no low-frequency background and no pixel noise), sub-pixel by
        1 / upsample px. The phase-normalised correlation is dominated by noise for these images."""
        from skimage.registration import phase_cross_correlation
        from scipy.ndimage import gaussian_filter
        band = lambda x: gaussian_filter(x, sigma) - gaussian_filter(x, 6 * sigma)
        shift, _, _ = phase_cross_correlation(band(self.recs[0].wf), band(rec.wf), upsample_factor=upsample,
                                              normalization=None)
        return np.asarray(shift, float)

    def run_raw(self, mu='auto', normalise=True, **kw):
        """Average the raw frames (offset kept) of all stacks, then reconstruct the mean stack."""
        offset = self.args[5]
        ref = (self.stacks[0].astype(np.float64) - offset).mean()
        mean = np.mean([(s.astype(np.float64) - offset) * (ref / (s.astype(np.float64) - offset).mean() if normalise else 1)
                        for s in self.stacks], axis=0) + offset
        self.rec = PeakMaskRecon(mean, *self.args)
        self.rec.run(mu=mu, **kw)
        for name in ('final', 'wf', 'S', 'otf_total', 'noise_power', 'p1', 'angle_sum'):
            setattr(self, name, getattr(self.rec, name))
        self.shifts = None
        return self


if __name__ == '__main__':
    # ---- test on the 6-angle x 10-phase 2D NLSIM stack(s) ------------------------------------------------------
    fns = [r"812bec86-2D_NLSIM_mito_000000_ids_0_59-1.tif"]   # raw stack(s), (60, 496, 496), angle-major.
    # Several acquisitions of the same field: list all files, e.g. fns = [r"stack1.tif", r"stack2.tif", r"stack3.tif",
    # r"stack4.tif"]; they are combined in the way selected by multi_mode.
    out = Path(r"nlsim_peak_mask_results")                 # output folder
    multi_mode = 'combine'      # several stacks. 'combine': reconstruct every stack (own patterns), register the angle
                                # images and combine all of them in one Wiener filter (recommended).
                                # 'raw': average the raw frames first (only if compare_patterns says it is safe)
    register = True             # sub-pixel registration of the stacks before combining
    n_angles = 6
    n_phases = 10
    pixel_size = 0.069          # raw pixel size [um]
    na = 1.3
    wavelength = 0.5            # emission wavelength [um]
    offset = 100.0              # camera offset subtracted from the raw data
    threshold = 0.85            # keep pixels where the normalised sinusoidal pattern (0..1) >= threshold
    mask = 'ramp'               # 'binary': hard threshold (leaves a 10th-harmonic ripple in the 10-phase sum)
                                # 'ramp': threshold, weighted linearly from 0 (threshold) to 1 (pattern peak)
    mu = 'auto'                 # 'auto': noise / object power estimated from the data; float: constant Wiener parameter
    phase_mode = 'frame'        # pattern phase: 'frame' measured per frame, 'linear' linear fit, 'equal' equally spaced
    ghost_correction = 'subtract'   # unequally spaced frame phases leave a zero-order ghost in the sum of the masked frames:
                                    # 'subtract' removes it exactly, 'weights' balances the frame weights, None: plain sum
    otf_model = 'anisotropic'   # transfer function in the Wiener filter: 'harmonic' (mask harmonics x shifted OTF) or
                                # 'anisotropic' (per angle: elliptical analytic OTF with resolution_along along the pattern)
    resolution_along = 0.080    # [um] resolution along the pattern direction for otf_model='anisotropic'
    lockin_alpha = None         # lock-in demodulation of the raw data before masking, e.g. 0.8 (None: off)
    p1 = None                   # fix the first-order modulation depth instead of measuring it
    save_stacks = True          # single stack only: also write the 60 patterns and the 60 masked frames

    kw = dict(p1=p1, phase_mode=phase_mode, lockin_alpha=lockin_alpha, ghost_correction=ghost_correction,
              otf_model=otf_model, resolution_along=resolution_along)
    args = (n_angles, n_phases, pixel_size, na, wavelength, offset, threshold, mask)
    out.mkdir(parents=True, exist_ok=True)
    stacks = [tf.imread(fn) for fn in fns]
    summary = dict(files=[str(f) for f in fns], pixel_size_um=pixel_size / 2, mask=mask, threshold=threshold,
                   lockin_alpha=lockin_alpha, phase_mode=phase_mode, ghost_correction=ghost_correction,
                   otf_model=otf_model, resolution_along=resolution_along, mu=mu)
    f32 = lambda x: np.asarray(x, np.float32)

    if len(stacks) == 1:
        rec = PeakMaskRecon(stacks[0], *args)
        rec.run(keep_stacks=save_stacks, mu=mu, **kw)
        checks = [rec.check_patterns()]
        recs = [rec]
        summary.update(mask_fraction=rec.mask_fraction, frame_weights=rec.weights.round(4).tolist(),
                       p1=rec.p1.tolist(), patterns=rec.pinfo)
        if save_stacks:
            tf.imwrite(out / 'patterns_uint8.tif', np.array(rec.patterns), photometric='minisblack')
            tf.imwrite(out / 'filtered_frames.tif', np.array(rec.filtered), photometric='minisblack')
    else:
        rec = MultiStackRecon(stacks, *args)
        comparison = rec.compare_patterns()
        print('pattern comparison with stack 0 (drift over the field, mean phase offset, per-frame scatter):')
        for q in comparison:
            print('  stack %d angle %d: drift %.1f deg, offset %.1f deg (%.3f raw px), scatter %.1f deg'
                  % (q['stack'], q['angle'], q['drift_deg'], q['offset_deg'], q['offset_px'], q['scatter_deg']))
        print('raw averaging is %s' % ('safe' if rec.raw_average_ok else 'NOT safe (use multi_mode = "combine")'))
        rec.run(mode=multi_mode, mu=mu, register=register, **kw)
        recs = rec.recs if multi_mode == 'combine' else [rec.rec]
        checks = [r.check_patterns() for r in recs]
        summary.update(multi_mode=multi_mode, register=register, pattern_comparison=comparison,
                       raw_average_ok=bool(rec.raw_average_ok), registration_shifts_interp_px=
                       None if rec.shifts is None else [np.asarray(x).tolist() for x in rec.shifts],
                       noise_power_per_stack=rec.noise_power if multi_mode == 'combine' else None,
                       p1=np.asarray(rec.p1).tolist())

    for si, check in enumerate(checks):
        print('pattern overlap check%s (shifts in pixels of the interpolated grid, %.1f nm):'
              % ('' if len(checks) == 1 else ' stack %d' % si, pixel_size / 2 * 1000))
        for q in check:
            print('  angle %d: max |shift| %.3f px, correlation-scan shift rms %.3f px, field drift (y, x) %.1f / %.1f deg'
                  % (q['angle'], q['max_abs_shift_px'], np.sqrt(np.mean(np.square(q['scan_shift_px']))),
                     *q['drift_deg_per_fov']))
    summary['pattern_check'] = checks

    tf.imwrite(out / 'widefield_interp.tif', f32(rec.wf), photometric='minisblack')
    tf.imwrite(out / 'angle_sums.tif', f32(rec.angle_sum), photometric='minisblack')
    tf.imwrite(out / 'wiener_combined.tif', f32(rec.final), photometric='minisblack')
    tf.imwrite(out / 'effective_otf_total.tif', f32(np.fft.fftshift(rec.otf_total)), photometric='minisblack')
    tf.imwrite(out / 'recon_spectrum_log.tif', f32(np.log10(np.abs(np.fft.fftshift(rec.S)) + 1e-6)),
               photometric='minisblack')
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))
