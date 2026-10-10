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

import argparse
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
    Returns per angle: wave vector (cycles per RAW pixel, axis0/axis1), per-frame pattern phase [rad] (measured, and
    equally spaced with the same mean), defined for pattern(x) = 0.5 * (1 + cos(2*pi*k.x + phase_j)) with x in raw pixel units (origin = pixel 0)."""
    nang, nph, ny, nx = raw.shape
    w = np.outer(tukey(ny, 0.2), tukey(nx, 0.2))
    j = np.arange(nph)
    ks, phases, phases_eq, info = [], [], [], []
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
        info.append(dict(angle=a, k_cycles_per_px=(f0 / ny, f1 / nx), period_px=1 / np.hypot(f0 / ny, f1 / nx),
                         direction_deg=float(np.degrees(np.arctan2(f1, f0))), phi0=float(phi0),
                         step_error_deg=float(np.degrees(eps)), phase_resid_rms_deg=float(np.degrees(res.std()))))
    return np.array(ks), np.array(phases), np.array(phases_eq), info


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
    def run(self, keep_stacks=False, mu='auto', measure_modulation=True, p1=None, measured_phases=False):
        self.up = fft_upsample2(self.raw.reshape(-1, self.n0, self.n0)).reshape(self.nang, self.nph, self.n, self.n)
        self.k, self.phase_meas, self.phase_eq, self.pinfo = estimate_patterns(self.raw)
        # Equally spaced mask phases make sum_j exp(i*n*theta_j) vanish exactly, so the unmodulated (zero-order)
        # image leaves no ghost at the pattern frequency; the small step error of the data costs < 1 % of the
        # first-order signal. The measured phases can be selected instead.
        self.phase = self.phase_meas if measured_phases else self.phase_eq
        self.angle_sum, self.patterns, self.filtered = [], [], []
        frac = []
        for a in range(self.nang):
            car = self.carrier(self.k[a])
            y = np.zeros((self.n, self.n), np.float64)
            for j in range(self.nph):
                pat = 0.5 * (1 + np.cos(car + np.float32(self.phase[a, j])))    # sinusoidal pattern image
                msk = self.mask_values(pat)                                    # peaks of the sine wave only
                filt = self.up[a, j] * msk                                     # masked raw frame
                y += filt
                frac.append(float((msk > 0).mean()))
                if keep_stacks:
                    self.patterns.append((pat * 255).astype(np.uint8))
                    self.filtered.append(filt.astype(np.float32))
            self.angle_sum.append(y)
        self.mask_fraction = float(np.mean(frac))
        self.p1 = np.array([self.estimate_modulation(a) if (measure_modulation and p1 is None) else (p1 or 0.0)
                            for a in range(self.nang)])
        self.wiener(mu)
        return self

    def estimate_modulation(self, a, band=(36, 96)):
        """First-order modulation depth p1 (cosine amplitude relative to the object) from the overlap of the
        shifted-back first-order component with the zero order, restricted to a band of object frequencies where
        the object has signal (the camera offset and unmodulated haze dilute the lowest frequencies)."""
        j = np.arange(self.nph)
        nu = self.phase[a] - self.phase[a][0]                         # demodulation phases 2*pi*j/nph + j*eps
        up = self.up[a].astype(np.float64)
        c0 = up.mean(0)
        c1 = np.tensordot(np.exp(-1j * nu), up, axes=(0, 0)) / self.nph
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

    def wiener(self, mu):
        """Wiener combination  S = sum_a conj(T_a) Y_a / (sum_a |T_a|^2 + NSR).
        mu = 'auto': NSR(kappa) = noise power / object power, both estimated from the data (noise from the
        frequencies outside every effective support but below the raw Nyquist, object power from the radially averaged angle-image
        spectra divided by the effective transfer functions). mu = float: constant NSR = mu**2."""
        win = np.outer(tukey(self.n, 0.05), tukey(self.n, 0.05))
        fs, ts = [], []
        self.otf_eff = []
        for a in range(self.nang):
            y = self.angle_sum[a]
            y = y / y.mean()                                          # common DC level: object DC = 1
            fs.append(fft2(y * win, workers=WORKERS))
            ts.append(self.effective_otf(a))
        self.otf_eff = ts
        t2 = sum(np.abs(t) ** 2 for t in ts)
        if mu == 'auto':
            # noise: bins outside every effective support but inside the band that still carries (interpolated) noise,
            # i.e. below the raw Nyquist frequency (bin n/4); beyond it the zero-padded spectrum holds no noise at all
            far = (t2 < 1e-4) & (self.bins < 0.97 * self.n / 4)
            noise = float(np.mean([np.mean(np.abs(f[far]) ** 2) for f in fs]))
            pf = np.mean([np.abs(f) ** 2 for f in fs], axis=0)
            rb = np.minimum(np.round(self.bins).astype(int), self.n // 2)
            cnt = np.maximum(np.bincount(rb.ravel(), minlength=self.n // 2 + 1), 1)
            prof = lambda x: np.bincount(rb.ravel(), x.ravel(), minlength=self.n // 2 + 1) / cnt
            ps = (prof(pf) - noise) / np.maximum(prof(t2) / self.nang, 1e-6)
            ps = np.convolve(np.maximum(ps, noise / 1e3), np.ones(5) / 5, mode='same')
            ps = np.minimum.accumulate(np.maximum(ps, noise / 1e3))    # object power decreases with frequency
            nsr = (noise / ps)[rb]
            self.noise_power = noise
        else:
            nsr = mu ** 2
        den = t2 + nsr
        self.S = sum(np.conj(t) * f for t, f in zip(ts, fs)) / den
        self.final = np.real(ifft2(self.S, workers=WORKERS))
        self.final *= self.up.mean()                                  # restore counts (offset-subtracted)
        self.wf = self.up.mean((0, 1))
        self.otf_total = t2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input')
    ap.add_argument('outdir')
    ap.add_argument('--angles', type=int, default=6)
    ap.add_argument('--phases', type=int, default=10)
    ap.add_argument('--pixel-size', type=float, default=0.069, help='raw pixel size [um]')
    ap.add_argument('--na', type=float, default=1.3)
    ap.add_argument('--wavelength', type=float, default=0.5, help='emission wavelength [um]')
    ap.add_argument('--offset', type=float, default=100.0, help='camera offset subtracted from the raw data')
    ap.add_argument('--threshold', type=float, default=0.85,
                    help='keep pixels where the normalised sinusoidal pattern (0..1) >= threshold')
    ap.add_argument('--mask', choices=('ramp', 'binary'), default='ramp',
                    help='binary: hard threshold. ramp: threshold with linear weighting up to the peak. The hard mask '
                         'has a strong 10th harmonic that leaves a ripple in the 10-phase sum')
    ap.add_argument('--mu', default='auto',
                    help="'auto' (noise/object power estimated from the data) or a constant Wiener parameter")
    ap.add_argument('--measured-phases', action='store_true', help='use the fitted per-frame phases for the masks')
    ap.add_argument('--p1', type=float, default=None, help='fix the modulation depth instead of measuring it')
    ap.add_argument('--save-stacks', action='store_true', help='also write the 60 patterns and 60 filtered frames')
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rec = PeakMaskRecon(tf.imread(a.input), a.angles, a.phases, a.pixel_size, a.na, a.wavelength, a.offset, a.threshold, a.mask)
    rec.run(keep_stacks=a.save_stacks, mu=a.mu if a.mu == 'auto' else float(a.mu), p1=a.p1,
            measured_phases=a.measured_phases)

    f32 = lambda x: np.asarray(x, np.float32)
    tf.imwrite(out / 'widefield_interp.tif', f32(rec.wf), photometric='minisblack')
    tf.imwrite(out / 'angle_sums.tif', f32(rec.angle_sum), photometric='minisblack')
    tf.imwrite(out / 'wiener_combined.tif', f32(rec.final), photometric='minisblack')
    tf.imwrite(out / 'effective_otf_total.tif', f32(np.fft.fftshift(rec.otf_total)), photometric='minisblack')
    tf.imwrite(out / 'recon_spectrum_log.tif', f32(np.log10(np.abs(np.fft.fftshift(rec.S)) + 1e-6)),
               photometric='minisblack')
    if a.save_stacks:
        tf.imwrite(out / 'patterns_uint8.tif', np.array(rec.patterns), photometric='minisblack')
        tf.imwrite(out / 'filtered_frames.tif', np.array(rec.filtered), photometric='minisblack')
    summary = dict(pixel_size_um=a.pixel_size / 2, mask_fraction=rec.mask_fraction, mask=a.mask, threshold=a.threshold, mu=a.mu,
                   p1=rec.p1.tolist(), patterns=rec.pinfo)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))


if __name__ == '__main__':
    main()
