"""
Widefield image-based fluorescence anisotropy simulator
=======================================================

Forward model
-------------
1. Dipole PSF  (``DipolePSF``, built on ``psf_generator.PSF``)
   Each molecule is an emitting electric dipole. The back-focal-plane field of
   a dipole ``mu`` imaged by an aplanatic objective is

       Ex = mu_x (cos(t) cos^2(p) + sin^2(p)) + mu_y (cos(t) - 1) sin(p) cos(p) - mu_z sin(t) cos(p)
       Ey = mu_x (cos(t) - 1) sin(p) cos(p) + mu_y (cos(t) sin^2(p) + cos^2(p)) - mu_z sin(t) sin(p)

   with apodization 1/sqrt(cos(t)). The image field is the FFT of the pupil
   field (using the pre-planned FFT, grid, defocus and Zernike aberration
   machinery of ``psf_generator.PSF``). An analyzer at angle ``a`` transmits
   E_a = cos(a) Ex + sin(a) Ey, so the intensity in channel ``a`` is linear in
   the second-moment tensor M = <mu mu^T> of the emission dipole:

       I_a(r) = sum_ij M_ij B_a,ij(r),    B_a,ij = Re(E_a,i conj(E_a,j))

2. Rotational diffusion  (``RotationalDiffusion``)
   Every molecule is a rigid body carrying an absorption dipole and an
   emission dipole (angle ``beta`` between them). Its orientation performs
   isotropic rotational Brownian motion with the Stokes-Einstein-Debye
   coefficient D_r = kT / (8 pi eta r^3). Within an exposure the orientation
   trajectory is sampled; at each instant the excitation probability is the
   photoselection weight a^T E a (E = excitation polarization tensor) and the
   emission dipole additionally depolarizes during the fluorescence lifetime
   tau, which for second moments is exact:

       <m m^T>_tau = I/3 + (m m^T - I/3) / (1 + 6 D_r tau)

   Molecules that rotate much faster than the sub-step use the analytic
   ensemble average  M = I/3 + 2/5 P2(cos beta) (E - I/3) / (1 + 6 D_r tau),
   so free dyes, slowly tumbling particles and immobile dipoles are all
   handled by the same code.

3. Detection  (``AnisotropySimulator``)
   Four polarization channels (0, 45, 90, 135 deg) either as a
   division-of-amplitude setup ('split': four images, each 1/2 of the
   analyzer-projected intensity) or a division-of-focal-plane polarization
   camera ('dofp': a single mosaic with the 2x2 super-pixel layout
   [[90, 45], [135, 0]], same as ``polarization_camera_simulator.py``).
   Images are computed on an oversampled grid, binned to camera pixels, then
   background, shot noise and camera read noise are added.

Analysis helpers compute r = (I_par - G I_perp) / (I_par + 2 G I_perp), the
linear Stokes parameters and the Axelrod high-NA correction of r.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import noise_generator as ng  # noqa: E402
import zernike_polynomials as zernike  # noqa: E402
from psf_generator import PSF  # noqa: E402

KB = 1.380649e-23  # J/K
ANALYZER_ANGLES = (0.0, 45.0, 90.0, 135.0)
# (row, col) of each analyzer inside the 2x2 DoFP super-pixel
DOFP_LAYOUT = {90.0: (0, 0), 45.0: (0, 1), 135.0: (1, 0), 0.0: (1, 1)}
# upper-triangle index pairs of the symmetric second-moment tensor
PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


# ---------------------------------------------------------------------------
# Vectorial dipole PSF
# ---------------------------------------------------------------------------

class DipolePSF(PSF):
    """Polarization-resolved dipole PSF built on top of ``psf_generator.PSF``.

    Uses the parent's grid, pre-planned FFTW transform and Zernike
    aberrations; adds the vectorial pupil of a dipole emitter.
    """

    def __init__(self, wl=0.52, na=1.4, n2=1.512, dx=0.05, nx=256, analyzer_angles=ANALYZER_ANGLES):
        super().__init__(wl=wl, na=na, n2=n2, dx=dx, nx=nx)
        self.analyzer_angles = tuple(float(a) for a in analyzer_angles)
        self._basis_cache = {}

    @property
    def pupil_radius(self):
        """Pupil (amplitude) cutoff NA / wl in pupil pixels."""
        return (self.na / self.wl) / self.dp

    def pupil_geometry(self):
        """sin/cos of polar angle, azimuth and aperture mask on the pupil grid."""
        x, y = self.grid
        kx = x * self.dp
        ky = y * self.dp
        sin_t = np.clip(self.wl * np.hypot(kx, ky) / self.n2, 0.0, 1.0)
        mask = (sin_t <= self.na / self.n2).astype(np.float64)
        cos_t = np.sqrt(1.0 - np.where(mask > 0, sin_t, 0.0) ** 2)
        phi = np.arctan2(ky, kx) * np.ones_like(kx)
        return sin_t, cos_t, phi, mask

    def dipole_pupil(self):
        """Back-focal-plane field G[c, i] (c: x/y field, i: dipole x/y/z)."""
        sin_t, cos_t, phi, mask = self.pupil_geometry()
        cp, sp = np.cos(phi), np.sin(phi)
        apod = mask / np.sqrt(cos_t)
        if self.zer is not None:
            ab = mask * np.exp(1j * zernike.zernike_sum(self.zer, rad=self.pupil_radius, nx=self.nx))
        else:
            ab = mask
        g = np.empty((2, 3, self.nx, self.nx), dtype=np.complex128)
        g[0, 0] = cos_t * cp ** 2 + sp ** 2
        g[0, 1] = (cos_t - 1.0) * sp * cp
        g[0, 2] = -sin_t * cp
        g[1, 0] = g[0, 1]
        g[1, 1] = cos_t * sp ** 2 + cp ** 2
        g[1, 2] = -sin_t * sp
        g *= apod * ab
        return g, cos_t, mask

    def set_aberration(self, zer):
        """Zernike coefficients (Noll order, radians) applied to the pupil."""
        self.zer = None if zer is None else np.asarray(zer, dtype=np.float64)
        self._basis_cache.clear()

    def basis_images(self, z=0.0):
        """Intensity basis B[a, k] for analyzer a and tensor element PAIRS[k].

        Centered on the grid (index nx//2); off-diagonal terms already carry
        the factor 2 so that I_a = sum_k c_k B[a, k] with
        c = (Mxx, Myy, Mzz, Mxy, Mxz, Myz). Normalized so that an isotropic
        emitter (M = I/3) gives sum(I_0 + I_90) = 1 (all collected photons).
        """
        key = round(float(z), 6)
        if key in self._basis_cache:
            return self._basis_cache[key]
        g, cos_t, mask = self.dipole_pupil()
        if z != 0.0:
            g = g * (mask * np.exp(2j * np.pi * self.n2 * z * cos_t / self.wl))
        e = np.empty_like(g)
        for c in range(2):
            for i in range(3):
                e[c, i] = np.fft.fftshift(self._fft2(g[c, i]))
        basis = np.empty((len(self.analyzer_angles), 6, self.nx, self.nx), dtype=np.float64)
        for n, ang in enumerate(self.analyzer_angles):
            a = np.deg2rad(ang)
            ea = np.cos(a) * e[0] + np.sin(a) * e[1]
            for k, (i, j) in enumerate(PAIRS):
                b = np.real(ea[i] * np.conj(ea[j]))
                basis[n, k] = b if i == j else 2.0 * b
        # isotropic reference with the total field (independent of analyzers)
        iso = sum((np.abs(e[0, i]) ** 2 + np.abs(e[1, i]) ** 2).sum() for i in range(3)) / 3.0
        basis /= iso
        self._basis_cache[key] = basis
        return basis

    def dipole_image(self, mu, z=0.0):
        """Analyzer images (centered) of a fixed dipole ``mu`` (3,)."""
        mu = np.asarray(mu, dtype=np.float64)
        mu = mu / np.linalg.norm(mu)
        return np.tensordot(tensor_to_coeffs(np.outer(mu, mu)), self.basis_images(z), axes=(0, 1))

    def collection_factors(self, z=0.0):
        """Axelrod K factors of the channel parallel to x and to y.

        Returns ``(ka, kb, kc)``: collected intensity in the analyzer-0 channel
        from a unit x-, y- and z-dipole (by symmetry the 90 channel has
        ka <-> kb swapped).
        """
        b = self.basis_images(z)
        i0 = self.analyzer_angles.index(0.0)
        return b[i0, 0].sum(), b[i0, 1].sum(), b[i0, 2].sum()


def tensor_to_coeffs(m):
    """(..., 3, 3) symmetric tensor -> (..., 6) coefficients in PAIRS order."""
    m = np.asarray(m)
    return np.stack([m[..., i, j] for i, j in PAIRS], axis=-1)


# ---------------------------------------------------------------------------
# Rotational diffusion of rigid dipole carriers
# ---------------------------------------------------------------------------

def rotational_diffusion_coefficient(radius_m, viscosity=1.0e-3, temperature=298.0):
    """Stokes-Einstein-Debye D_r = kT / (8 pi eta r^3)  [rad^2/s]. inf radius -> 0."""
    r = np.asarray(radius_m, dtype=np.float64)
    with np.errstate(divide='ignore'):
        return np.where(np.isfinite(r), KB * temperature / (8.0 * np.pi * viscosity * r ** 3), 0.0)


def excitation_tensor(polarization=0.0):
    """Excitation polarization tensor E (trace 1) for epi-illumination.

    ``polarization``: angle in degrees (linear), or 'circular'/'unpolarized'.
    """
    if isinstance(polarization, str):
        return np.diag([0.5, 0.5, 0.0])
    a = np.deg2rad(polarization)
    e = np.array([np.cos(a), np.sin(a), 0.0])
    return np.outer(e, e)


def p2(x):
    return 1.5 * x * x - 0.5


def random_rotations(n, rng):
    """Uniformly distributed rotation matrices (n, 3, 3) from random quaternions."""
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], axis=1)


def rodrigues(omega):
    """Rotation matrices (n, 3, 3) from rotation vectors (n, 3)."""
    theta = np.linalg.norm(omega, axis=1)
    k = omega / np.where(theta > 0, theta, 1.0)[:, None]
    kx, ky, kz = k.T
    zero = np.zeros_like(kx)
    kk = np.stack([np.stack([zero, -kz, ky], -1),
                   np.stack([kz, zero, -kx], -1),
                   np.stack([-ky, kx, zero], -1)], axis=1)
    s = np.sin(theta)[:, None, None]
    c = (1.0 - np.cos(theta))[:, None, None]
    return np.eye(3)[None] + s * kk + c * (kk @ kk)


class RotationalDiffusion:
    """Orientation state of N rigid molecules under isotropic rotational diffusion.

    Body frame: absorption dipole along z, emission dipole tilted by ``beta``
    in the x-z plane. ``R`` maps body to lab frame.
    """

    max_step_var = 0.01   # max <angle^2> per axis in one Euler-Rodrigues step
    max_steps = 64        # beyond this many steps per interval, orientation is randomized

    def __init__(self, dr, beta=0.0, lifetime=4e-9, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng
        self.dr = np.atleast_1d(np.asarray(dr, dtype=np.float64))
        n = self.dr.size
        self.beta = np.broadcast_to(np.asarray(beta, dtype=np.float64), (n,)).copy()
        self.lifetime = np.broadcast_to(np.asarray(lifetime, dtype=np.float64), (n,)).copy()
        self.R = random_rotations(n, self.rng)
        self.abs_body = np.array([0.0, 0.0, 1.0])
        self.em_body = np.stack([np.sin(self.beta), np.zeros(n), np.cos(self.beta)], axis=-1)

    @property
    def n(self):
        return self.dr.size

    @property
    def absorption_dipoles(self):
        return self.R @ self.abs_body

    @property
    def emission_dipoles(self):
        return np.einsum('nij,nj->ni', self.R, self.em_body)

    @property
    def lifetime_depolarization(self):
        """f = 1 / (1 + 6 D_r tau): decay of <P2> during the excited state."""
        return 1.0 / (1.0 + 6.0 * self.dr * self.lifetime)

    @property
    def rotational_correlation_time(self):
        with np.errstate(divide='ignore'):
            return np.where(self.dr > 0, 1.0 / (6.0 * self.dr), np.inf)

    def theoretical_anisotropy(self):
        """Steady-state Perrin anisotropy r = 0.4 P2(cos beta) / (1 + tau/theta)."""
        return 0.4 * p2(np.cos(self.beta)) * self.lifetime_depolarization

    def fast_mask(self, dt):
        """Molecules that fully decorrelate within ``dt`` (use ensemble average)."""
        return 2.0 * self.dr * dt > self.max_step_var * self.max_steps

    def advance(self, dt):
        """Propagate orientations by time ``dt``."""
        if dt <= 0:
            return
        fast = self.fast_mask(dt)
        if fast.any():
            self.R[fast] = random_rotations(int(fast.sum()), self.rng)
        slow = (~fast) & (self.dr > 0)
        if not slow.any():
            return
        idx = np.flatnonzero(slow)
        var = 2.0 * self.dr[idx] * dt
        steps = np.maximum(1, np.ceil(var / self.max_step_var)).astype(int)
        sigma = np.sqrt(var / steps)
        for s in range(steps.max()):
            act = steps > s
            ii = idx[act]
            omega = sigma[act, None] * self.rng.normal(size=(ii.size, 3))
            self.R[ii] = rodrigues(omega) @ self.R[ii]

    def ensemble_tensor(self, exc_tensor):
        """Analytic photoselected emission tensor for freely, rapidly rotating molecules."""
        amp = 0.4 * p2(np.cos(self.beta)) * self.lifetime_depolarization
        return np.eye(3)[None] / 3.0 + amp[:, None, None] * (exc_tensor - np.eye(3) / 3.0)[None]

    def instantaneous_tensor(self, exc_tensor):
        """Brightness-weighted emission tensor for the current orientations.

        Returns 3 w (I/3 + f (m m^T - I/3)) with w = a^T E a, so an
        orientation average over the sphere gives trace 1.
        """
        a = self.absorption_dipoles
        m = self.emission_dipoles
        w = np.einsum('ni,ij,nj->n', a, exc_tensor, a)
        f = self.lifetime_depolarization
        mm = np.einsum('ni,nj->nij', m, m)
        eye = np.eye(3)[None] / 3.0
        return 3.0 * w[:, None, None] * (eye + f[:, None, None] * (mm - eye))

    def expose(self, exposure, exc_tensor, n_sub=100):
        """Average emission tensor (N, 3, 3) over one exposure, advancing orientations."""
        dt = exposure / n_sub
        fast = self.fast_mask(dt)
        acc = np.zeros((self.n, 3, 3))
        for _ in range(n_sub):
            self.advance(dt / 2)
            acc += self.instantaneous_tensor(exc_tensor)
            self.advance(dt / 2)
        acc /= n_sub
        if fast.any():
            acc[fast] = self.ensemble_tensor(exc_tensor)[fast]
        return acc


# ---------------------------------------------------------------------------
# Widefield acquisition with four polarization channels
# ---------------------------------------------------------------------------

class AnisotropySimulator:
    """Widefield fluorescence anisotropy imaging with 4 polarization channels."""

    def __init__(self, nx_cam=128, pixel_size=0.1, oversample=2, wl=0.52, na=1.4, n2=1.512,
                 excitation=0.0, exposure=0.05, frame_interval=None, n_sub=100,
                 detection='split', extinction_ratio=np.inf, transmission=1.0,
                 background=5.0, qe=0.8, gain=1.0, read_noise=1.5, offset=100,
                 rng=None):
        self.nx_cam = nx_cam
        self.pixel_size = pixel_size
        self.os = oversample
        self.nx = nx_cam * oversample
        self.dxs = pixel_size / oversample
        self.psf = DipolePSF(wl=wl, na=na, n2=n2, dx=self.dxs, nx=self.nx)
        self.excitation = excitation
        self.exc_tensor = excitation_tensor(excitation)
        self.exposure = exposure
        self.frame_interval = exposure if frame_interval is None else frame_interval
        self.n_sub = n_sub
        if detection not in ('split', 'dofp'):
            raise ValueError("detection must be 'split' or 'dofp'")
        self.detection = detection
        self.extinction_ratio = extinction_ratio
        self.transmission = transmission
        self.background = background
        self.qe, self.gain, self.read_noise, self.offset = qe, gain, read_noise, offset
        self.rng = np.random.default_rng() if rng is None else rng
        np.random.seed(int(self.rng.integers(2 ** 31)))  # noise_generator uses the legacy RNG
        self.x = self.y = self.z = self.photons = None
        self.rot = None
        self.table = None

    @property
    def fov(self):
        return self.nx_cam * self.pixel_size

    # --- sample -----------------------------------------------------------

    def set_molecules(self, x, y, z=0.0, photons=2000.0, radius=None, dr=None,
                      viscosity=1.0e-3, temperature=298.0, lifetime=4e-9, beta=0.0):
        """Place molecules (positions in um).

        Rotational mobility from hydrodynamic ``radius`` (m, inf = immobile)
        via Stokes-Einstein-Debye, or directly from ``dr`` (rad^2/s).
        ``photons``: expected detected photons per frame for an isotropic
        orientation distribution.
        """
        x = np.asarray(x, dtype=np.float64)
        n = x.size
        self.x = x
        self.y = np.broadcast_to(np.asarray(y, dtype=np.float64), (n,)).copy()
        self.z = np.broadcast_to(np.asarray(z, dtype=np.float64), (n,)).copy()
        self.photons = np.broadcast_to(np.asarray(photons, dtype=np.float64), (n,)).copy()
        if dr is None:
            dr = rotational_diffusion_coefficient(radius, viscosity, temperature)
        dr = np.broadcast_to(np.asarray(dr, dtype=np.float64), (n,)).copy()
        self.rot = RotationalDiffusion(dr, beta=beta, lifetime=lifetime, rng=self.rng)

    # --- image formation --------------------------------------------------

    def _optical_images(self, tensors):
        """Noise-free analyzer images (4, nx, nx) on the oversampled grid [photons]."""
        coeffs = tensor_to_coeffs(tensors) * self.photons[:, None]
        k = np.fft.fftfreq(self.nx, d=self.dxs)
        out = np.zeros((len(self.psf.analyzer_angles), self.nx, self.nx))
        # sample positions at oversampled pixel centers (j + 0.5) dxs
        xs = self.x - 0.5 * self.dxs
        ys = self.y - 0.5 * self.dxs
        for zv in np.unique(self.z):
            sel = self.z == zv
            otf = np.fft.fft2(np.fft.ifftshift(self.psf.basis_images(zv), axes=(-2, -1)))
            ex = np.exp(-2j * np.pi * np.outer(xs[sel], k))  # (n, nx) along columns
            ey = np.exp(-2j * np.pi * np.outer(ys[sel], k))  # (n, ny) along rows
            spec = np.einsum('nk,ny,nx->kyx', coeffs[sel], ey, ex, optimize=True)
            out += np.real(np.fft.ifft2(np.einsum('akyx,kyx->ayx', otf, spec)))
        return np.clip(out, 0.0, None)

    def _bin(self, img):
        s = img.shape[:-2] + (self.nx_cam, self.os, self.nx_cam, self.os)
        return img.reshape(s).sum(axis=(-3, -1))

    def _apply_extinction(self, imgs):
        if not np.isfinite(self.extinction_ratio):
            return imgs
        ang = list(self.psf.analyzer_angles)
        leak = 1.0 / self.extinction_ratio
        out = imgs.copy()
        for n, a in enumerate(ang):
            cross = ang.index((a + 90.0) % 180.0)
            out[n] = (imgs[n] + leak * imgs[cross]) / (1.0 + leak)
        return out

    def _detect(self, channel_photons):
        """Camera measurement of expected photons -> ADU (uint16)."""
        img = channel_photons * self.transmission + self.background
        return ng.add_readout_noise(img, qe=self.qe, gain=self.gain,
                                    read_noise_e=self.read_noise, offset=self.offset)

    def mosaic(self, channels):
        """Interleave 4 full-resolution channel images into a DoFP raw frame."""
        raw = np.empty(channels.shape[-2:])
        for n, a in enumerate(self.psf.analyzer_angles):
            r, c = DOFP_LAYOUT[a]
            raw[r::2, c::2] = channels[n, r::2, c::2]
        return raw

    def acquire(self, n_frames=1, verbose=True):
        """Simulate a time series.

        Returns dict with
          'ideal'   : (T, 4, ny, nx) noise-free expected photons per channel
          'data'    : (T, 4, ny, nx) ADU for 'split' or (T, ny, nx) raw mosaic for 'dofp'
          'tensors' : (T, N, 3, 3) exposure-averaged emission tensors
        """
        n_ch = len(self.psf.analyzer_angles)
        ideal = np.empty((n_frames, n_ch, self.nx_cam, self.nx_cam))
        tensors = np.empty((n_frames, self.rot.n, 3, 3))
        data = []
        for t in range(n_frames):
            t0 = time.perf_counter()
            tensors[t] = self.rot.expose(self.exposure, self.exc_tensor, self.n_sub)
            self.rot.advance(self.frame_interval - self.exposure)
            chan = self._apply_extinction(self._bin(self._optical_images(tensors[t])))
            if self.detection == 'split':
                chan = 0.5 * chan  # 50/50 split into (0, 90) and (45, 135) arms
                ideal[t] = chan
                data.append(np.stack([self._detect(c) for c in chan]))
            else:
                ideal[t] = chan
                data.append(self._detect(self.mosaic(chan)))
            if verbose:
                print(f"frame {t + 1}/{n_frames}  ({time.perf_counter() - t0:.2f} s)")
        return {'ideal': ideal, 'data': np.stack(data), 'tensors': tensors}

    # --- ground truth ------------------------------------------------------

    def ground_truth(self):
        rot = self.rot
        return {
            'x_um': self.x, 'y_um': self.y, 'z_um': self.z, 'photons': self.photons,
            'dr_per_s': rot.dr, 'theta_rot_s': rot.rotational_correlation_time,
            'lifetime_s': rot.lifetime, 'beta_rad': rot.beta,
            'r_perrin': rot.theoretical_anisotropy(),
        }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def demosaic(raw):
    """Split a DoFP mosaic (..., ny, nx) into channels (..., 4, ny/2, nx/2) in 0/45/90/135 order."""
    return np.stack([raw[..., DOFP_LAYOUT[a][0]::2, DOFP_LAYOUT[a][1]::2] for a in ANALYZER_ANGLES], axis=-3)


def channel_pair(excitation=0.0):
    """Indices (parallel, perpendicular) of the analyzer channels for the excitation angle."""
    par = float(excitation) % 180.0
    return ANALYZER_ANGLES.index(par), ANALYZER_ANGLES.index((par + 90.0) % 180.0)


def anisotropy(i_par, i_perp, g=1.0, eps=1e-9):
    return (i_par - g * i_perp) / (i_par + 2.0 * g * i_perp + eps)


def stokes(channels, eps=1e-9):
    """Linear Stokes parameters from (..., 4, ny, nx) channels at 0/45/90/135."""
    i0, i45, i90, i135 = np.moveaxis(channels, -3, 0)
    s0 = 0.5 * (i0 + i45 + i90 + i135)
    s1 = i0 - i90
    s2 = i45 - i135
    dolp = np.sqrt(s1 ** 2 + s2 ** 2) / (s0 + eps)
    aolp = 0.5 * np.rad2deg(np.arctan2(s2, s1))
    return s0, s1, s2, dolp, aolp


def axelrod_correction(i_par, i_perp, ka, kb, kc):
    """Recover the NA-free anisotropy from parallel/perpendicular intensities.

    For a photoselected distribution symmetric about the excitation axis x:
        I_par  = ka Mxx + (kb + kc) M_perp
        I_perp = kb Mxx + (ka + kc) M_perp
    """
    det = ka * (ka + kc) - kb * (kb + kc)
    mxx = ((ka + kc) * i_par - (kb + kc) * i_perp) / det
    mperp = (ka * i_perp - kb * i_par) / det
    return (mxx - mperp) / (mxx + 2.0 * mperp)


def molecule_intensities(channels, x, y, pixel_size, half_window=2):
    """Sum each channel in a (2h+1)^2 window around each molecule -> (N, 4)."""
    ny, nx = channels.shape[-2:]
    cols = np.clip(np.floor(np.asarray(x) / pixel_size).astype(int), half_window, nx - half_window - 1)
    rows = np.clip(np.floor(np.asarray(y) / pixel_size).astype(int), half_window, ny - half_window - 1)
    h = half_window
    return np.stack([channels[..., r - h:r + h + 1, c - h:c + h + 1].sum(axis=(-2, -1))
                     for r, c in zip(rows, cols)], axis=0)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

POPULATIONS = (
    # name, hydrodynamic radius [m], colour
    ('small dye (r=0.6 nm)', 0.6e-9, 'tab:blue'),
    ('protein (r=2.4 nm)', 2.4e-9, 'tab:orange'),
    ('bead-bound (r=100 nm)', 100e-9, 'tab:green'),
    ('immobilized', np.inf, 'tab:red'),
)


def build_demo(sim, n_per_population=40, margin=1.0, min_dist=0.6, photons=3000.0,
               lifetime=4e-9, viscosity=1.0e-3, beta=0.0):
    """Four populations in four vertical stripes of the field of view."""
    rng = sim.rng
    fov = sim.fov
    width = (fov - 2 * margin) / len(POPULATIONS)
    xs, ys, radii, labels = [], [], [], []
    for p, (_, radius, _) in enumerate(POPULATIONS):
        placed = []
        while len(placed) < n_per_population:
            cand = np.array([margin + (p + rng.random()) * width, margin + rng.random() * (fov - 2 * margin)])
            if all(np.hypot(*(cand - q)) > min_dist for q in placed):
                placed.append(cand)
        placed = np.array(placed)
        xs.append(placed[:, 0])
        ys.append(placed[:, 1])
        radii.append(np.full(n_per_population, radius))
        labels.append(np.full(n_per_population, p))
    sim.set_molecules(np.concatenate(xs), np.concatenate(ys), z=0.0, photons=photons,
                      radius=np.concatenate(radii), viscosity=viscosity, lifetime=lifetime, beta=beta)
    return np.concatenate(labels)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--frames', type=int, default=10)
    ap.add_argument('--detection', choices=('split', 'dofp'), default='split')
    ap.add_argument('--excitation', default='0', help="linear angle in deg, or 'circular'")
    ap.add_argument('--nx', type=int, default=128, help='camera pixels per side')
    ap.add_argument('--pixel', type=float, default=0.1, help='camera pixel size in sample [um]')
    ap.add_argument('--na', type=float, default=1.4)
    ap.add_argument('--molecules', type=int, default=30, help='molecules per population')
    ap.add_argument('--photons', type=float, default=3000.0)
    ap.add_argument('--exposure', type=float, default=0.05, help='[s]')
    ap.add_argument('--lifetime', type=float, default=4e-9, help='[s]')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default=None, help='output folder (default: no files written)')
    ap.add_argument('--show', action='store_true', help='show the summary figure')
    args = ap.parse_args()

    excitation = args.excitation if args.excitation in ('circular', 'unpolarized') else float(args.excitation)
    sim = AnisotropySimulator(nx_cam=args.nx, pixel_size=args.pixel, na=args.na, excitation=excitation,
                              exposure=args.exposure, detection=args.detection,
                              rng=np.random.default_rng(args.seed))
    labels = build_demo(sim, n_per_population=args.molecules, photons=args.photons, lifetime=args.lifetime)
    res = sim.acquire(args.frames)
    truth = sim.ground_truth()

    # --- analysis: background/offset subtraction, per-molecule anisotropy ---
    if args.detection == 'split':
        chans = res['data'].astype(np.float64)
        pix = sim.pixel_size
    else:
        chans = demosaic(res['data'].astype(np.float64))
        pix = 2 * sim.pixel_size
    chans = (chans - sim.offset) / (sim.gain * sim.qe) - sim.background
    if args.detection == 'split':
        chans = chans / 0.5
    s0, s1, s2, dolp, aolp = stokes(chans.mean(axis=0))

    ka, kb, kc = sim.psf.collection_factors()
    print(f"collection factors  Ka={ka:.3f}  Kb={kb:.3f}  Kc={kc:.3f}")
    print(f"excitation={excitation}  detection={args.detection}")
    if isinstance(excitation, str):
        print("circular excitation: no parallel/perpendicular channels; reporting Stokes/DoLP only")
        ipar = iperp = None
    else:
        ip, iq = channel_pair(excitation)
        mol = molecule_intensities(chans, sim.x, sim.y, pix)  # (N, T, 4)
        ipar, iperp = mol[..., ip], mol[..., iq]
        r_frames = anisotropy(ipar, iperp)
        r_mol = anisotropy(ipar.sum(1), iperp.sum(1))
        if excitation % 90 == 0:
            r_corr = axelrod_correction(ipar.sum(1), iperp.sum(1), ka, kb, kc)
        else:
            r_corr = np.full_like(r_mol, np.nan)  # K factors are for x/y excitation
        print(f"{'population':24s} {'theta_rot':>11s} {'r_Perrin':>9s} {'r_meas':>14s} "
              f"{'r_NA-corr':>14s} {'frame-to-frame sd':>18s}")
        for p, (name, _, _) in enumerate(POPULATIONS):
            s = labels == p
            th = truth['theta_rot_s'][s][0]
            th_s = 'inf' if not np.isfinite(th) else f"{th:.2e} s"
            print(f"{name:24s} {th_s:>11s} {truth['r_perrin'][s].mean():9.3f} "
                  f"{r_mol[s].mean():7.3f}+-{r_mol[s].std():.3f} {np.nanmean(r_corr[s]):7.3f}+-"
                  f"{np.nanstd(r_corr[s]):.3f} {r_frames[s].std(axis=1).mean():18.3f}")

    if args.out is not None:
        import tifffile as tf
        os.makedirs(args.out, exist_ok=True)
        stamp = time.strftime('%Y%m%d%H%M%S')
        base = os.path.join(args.out, f"{stamp}_wf_anisotropy_{args.detection}")
        tf.imwrite(base + '_raw.tif', res['data'])
        tf.imwrite(base + '_ideal.tif', res['ideal'].astype(np.float32))
        if ipar is not None:
            ip, iq = channel_pair(excitation)
            rmap = anisotropy(chans[:, ip].mean(0), chans[:, iq].mean(0))
            rmap[s0 < 0.1 * s0.max()] = np.nan
            tf.imwrite(base + '_anisotropy.tif', rmap.astype(np.float32))
        tf.imwrite(base + '_dolp.tif', dolp.astype(np.float32))
        cols = list(truth.keys()) + ['population']
        table = np.column_stack([truth[k] for k in truth] + [labels])
        np.savetxt(base + '_ground_truth.csv', table, delimiter=',', header=','.join(cols), comments='')
        save_figure(base + '_summary.png', sim, chans, s0, labels, truth,
                    None if ipar is None else (r_mol, r_corr), show=args.show)
        print(f"saved to {base}_*")
    elif args.show:
        save_figure(None, sim, chans, s0, labels, truth,
                    None if ipar is None else (r_mol, r_corr), show=True)


def save_figure(path, sim, chans, s0, labels, truth, r_values, show=False):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mean = chans.mean(axis=0)
    fig, axs = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    vmax = np.percentile(mean, 99.9)
    for n, a in enumerate(ANALYZER_ANGLES):
        ax = axs[0, n]
        ax.imshow(mean[n], cmap='gray', vmin=0, vmax=vmax)
        ax.set_title(f"analyzer {a:g}\N{DEGREE SIGN}")
        ax.axis('off')

    ax = axs[1, 0]
    if r_values is not None and not isinstance(sim.excitation, str):
        ip, iq = channel_pair(sim.excitation)
        rmap = anisotropy(mean[ip], mean[iq])
        rmap = np.where(s0 > 0.1 * s0.max(), rmap, np.nan)
        im = ax.imshow(rmap, cmap='viridis', vmin=-0.2, vmax=0.6)
        fig.colorbar(im, ax=ax, shrink=0.8, label='r')
        ax.set_title('anisotropy map (time-averaged)')
    ax.axis('off')

    ax = axs[1, 1]
    ax.imshow(s0, cmap='gray')
    ax.set_title('total intensity S0')
    ax.axis('off')

    ax = axs[1, 2]
    if r_values is not None:
        r_mol, r_corr = r_values
        for p, (name, _, colour) in enumerate(POPULATIONS):
            s = labels == p
            ax.scatter(truth['r_perrin'][s], r_mol[s], s=12, color=colour, label=name)
            ax.scatter(truth['r_perrin'][s], r_corr[s], s=12, color=colour, marker='x')
        ax.plot([-0.2, 0.4], [-0.2, 0.4], color='gray', lw=0.8)
        ax.set_xlabel('Perrin r (ensemble theory)')
        ax.set_ylabel('measured r  (o raw, x NA-corrected)')
        ax.legend(fontsize=7, frameon=False)
        ax.set_title('per-molecule anisotropy')

    ax = axs[1, 3]
    psf = sim.psf.basis_images(0.0)
    c = sim.nx // 2
    h = int(round(1.0 / sim.dxs))
    tiles = []
    for mu in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        img = np.tensordot(tensor_to_coeffs(np.outer(mu, mu)), psf, axes=(0, 1))[:, c - h:c + h, c - h:c + h]
        tiles.append(np.concatenate(list(img), axis=1))
    ax.imshow(np.concatenate(tiles, axis=0), cmap='magma')
    ax.set_title('x / y / z dipole PSF, analyzers 0/45/90/135')
    ax.axis('off')

    if path is not None:
        fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
