"""
Unified pupil amplitude and phase modulator using Zernike polynomials.

Generates a pupil function  P(ρ, φ) = A(ρ) · exp(i · W(ρ, φ))  on both
Cartesian (nx×nx pixel grid) and polar (arbitrary array) coordinate systems,
providing a single source-of-truth pupil for:

  · psf_generator.PSF                      — Cartesian wavefront (bpp)
  · vectorical_focusing.RichardsWolfDirect — polar quadrature grid

Zernike convention: 1-based Noll ordering matching vectorical_focusing.py.

  j=1  piston          j=5  astigmatism 0° (cos 2φ)   j=9  trefoil y
  j=2  tilt x (cos φ)  j=6  astigmatism 45° (sin 2φ)  j=10 trefoil x
  j=3  tilt y (sin φ)  j=7  coma x (cos φ)            j=11 spherical
  j=4  defocus         j=8  coma y (sin φ)

Usage
-----
    from pupil_wavefront_modulator import PupilWavefrontModulator

    # Define the pupil: 0.5 rad defocus, Gaussian amplitude
    mod = PupilWavefrontModulator(
        zernike_coeffs={4: 0.5},
        amplitude='gaussian',
        amplitude_params={'sigma': 0.7}
    )

    # --- psf_generator.PSF ---
    psf = PSF(nx=256, na=1.4, wl=0.488, dx=0.064)
    psf.bpp = mod.to_psf_wavefront(nx=psf.nx, radius=psf.radius)
    intensity = psf.get_2d_psf((0, 0, 0))

    # --- vectorical_focusing.RichardsWolfDirect ---
    sim = RichardsWolfDirect(NA=1.4, n=1.515, wavelength=0.488,
                             amplitude_func=mod.as_amplitude_func())
    psf3d, Ex, Ey, Ez = sim.compute(mod.zernike_coeffs, x, y, z)

References
----------
Noll (1976) JOSA 66, 207-211
Richards & Wolf (1959) Proc. R. Soc. A 253, 358–379
"""

from functools import lru_cache
from math import factorial
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Noll indexing  (1-based, matches vectorical_focusing.py)
# ══════════════════════════════════════════════════════════════════════════════

def noll_to_nm(j: int) -> Tuple[int, int]:
    """
    Convert 1-based Noll index j to (n, m) radial/azimuthal orders.

    Uses the same ordering as vectorical_focusing.RichardsWolfDirect so that
    zernike_coeffs dicts are directly interchangeable.

    Parameters
    ----------
    j : int  Noll index, must be ≥ 1

    Returns
    -------
    (n, m) : (int, int)  radial order n, signed azimuthal frequency m
    """
    if j < 1:
        raise ValueError(f"Noll index must be ≥ 1, got {j}.")
    n = 0
    while n * (n + 1) // 2 < j:
        n += 1
    n -= 1
    k = j - n * (n + 1) // 2 - 1
    ms: List[int] = [0] if n % 2 == 0 else []
    for m_abs in range(2 - n % 2, n + 1, 2):
        ms.append(m_abs)
        ms.append(-m_abs)
    return n, ms[k]


def noll_table(max_j: int = 15) -> Dict[int, Tuple[int, int]]:
    """
    Return {j: (n, m)} for j = 1 … max_j.

    Useful for looking up which aberration corresponds to a given index.
    """
    return {j: noll_to_nm(j) for j in range(1, max_j + 1)}


# ══════════════════════════════════════════════════════════════════════════════
# Radial polynomial  R_n^|m|(ρ)
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=256)
def _radial_coeffs(n: int, m_abs: int) -> Tuple[Tuple[float, int], ...]:
    """Cached polynomial coefficients (coeff, power) for R_n^m_abs(ρ)."""
    h = (n - m_abs) // 2
    terms = []
    for s in range(h + 1):
        c = ((-1) ** s * factorial(n - s)
             / (factorial(s)
                * factorial((n + m_abs) // 2 - s)
                * factorial((n - m_abs) // 2 - s)))
        if c != 0:
            terms.append((float(c), n - 2 * s))
    return tuple(terms)


def _eval_radial(n: int, m_abs: int, rho: np.ndarray) -> np.ndarray:
    """Evaluate radial polynomial R_n^m_abs(rho) on an arbitrary array."""
    R = np.zeros(np.shape(rho), dtype=np.float64)
    for coeff, power in _radial_coeffs(n, m_abs):
        R += coeff * (np.ones_like(rho) if power == 0 else rho ** power)
    return R


# ══════════════════════════════════════════════════════════════════════════════
# Zernike polynomial evaluation on arbitrary (ρ, φ) arrays
# ══════════════════════════════════════════════════════════════════════════════

def eval_zernike(j: int, rho: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Evaluate a single Zernike polynomial Z_j on arbitrary (rho, phi) arrays.

    No masking is applied; the polynomial is defined for all ρ ≥ 0.

    Parameters
    ----------
    j   : 1-based Noll index
    rho : array_like  normalized pupil radius  (0–1 inside pupil)
    phi : array_like  azimuthal angle [rad]

    Returns
    -------
    Z_j : ndarray  same shape as rho
    """
    n, m = noll_to_nm(j)
    rho = np.asarray(rho, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    R = _eval_radial(n, abs(m), rho)
    norm = np.sqrt(n + 1.0) if m == 0 else np.sqrt(2.0 * (n + 1))
    if m == 0:
        return norm * R
    elif m > 0:
        return norm * R * np.cos(m * phi)
    else:
        return norm * R * np.sin(-m * phi)  # -m = |m| for m < 0


def eval_zernike_sum(
        coeffs: Dict[int, float],
        rho: np.ndarray,
        phi: np.ndarray,
) -> np.ndarray:
    """
    Weighted Zernike sum  W = Σ_j  c_j · Z_j(ρ, φ)  on arbitrary arrays.

    Parameters
    ----------
    coeffs : {noll_j (1-based): float [rad]}
    rho    : array_like  normalized pupil radius
    phi    : array_like  azimuthal angle [rad]

    Returns
    -------
    W : ndarray  same shape as rho  [rad]
    """
    rho = np.asarray(rho, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    W = np.zeros(np.shape(rho), dtype=np.float64)
    for j, c in coeffs.items():
        if c != 0.0:
            W += c * eval_zernike(j, rho, phi)
    return W


# ══════════════════════════════════════════════════════════════════════════════
# Amplitude envelope builders
# ══════════════════════════════════════════════════════════════════════════════

def _build_amp_func(
        amplitude: Union[str, Callable],
        params: Optional[Dict],
) -> Callable[[np.ndarray], np.ndarray]:
    """Return A(ρ) callable from a preset name or a user-supplied callable."""
    if callable(amplitude):
        return amplitude

    p = params or {}

    if amplitude == 'uniform':
        return lambda rho: np.ones(np.shape(rho), dtype=np.float64)

    if amplitude == 'gaussian':
        sigma = float(p.get('sigma', 1.0 / np.sqrt(2)))
        return lambda rho, _s=sigma: np.exp(-rho ** 2 / (2.0 * _s ** 2))

    if amplitude == 'annular':
        inner = float(p.get('inner', 0.0))
        return lambda rho, _i=inner: np.where(
            rho >= _i, 1.0, 0.0).astype(np.float64)

    if amplitude == 'super_gaussian':
        w = float(p.get('w', 1.0))
        order = float(p.get('order', 4))
        return lambda rho, _w=w, _o=order: np.exp(-((rho / _w) ** _o))

    raise ValueError(
        f"Unknown amplitude preset {amplitude!r}. "
        "Choose: 'uniform' | 'gaussian' | 'annular' | 'super_gaussian' | callable"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PupilWavefrontModulator
# ══════════════════════════════════════════════════════════════════════════════

class PupilWavefrontModulator:
    """
    Unified pupil function  P(ρ, φ) = A(ρ) · exp(i · W(ρ, φ))

    W(ρ, φ) = Σ_j c_j · Z_j(ρ, φ)  +  l · φ    [rad]
    A(ρ)    = amplitude envelope, zero outside the unit disc (ρ > 1)

    Evaluates identically on both Cartesian pixel grids and arbitrary polar
    coordinate arrays, providing a single source-of-truth pupil for
    psf_generator.PSF and vectorical_focusing.RichardsWolfDirect.

    Parameters
    ----------
    zernike_coeffs : dict {noll_j: float [rad]} | sequence | None
        Wavefront error.  1-based Noll convention (j ≥ 1):
          j=1 piston, j=2 tilt-x, j=3 tilt-y, j=4 defocus, …
        A list or array is treated as 0-indexed; index 0 is silently ignored,
        coefficients start at index 1 (= Noll j=1).
    amplitude : 'uniform' | 'gaussian' | 'annular' | 'super_gaussian' | callable
        Amplitude envelope A(ρ) inside the unit disc.
        callable must accept an ndarray of ρ values and return an ndarray.
    amplitude_params : dict, optional
        Parameters for built-in presets:
          'gaussian'       → {'sigma': float}                 default 1/√2
          'annular'        → {'inner': float}                 default 0.0
          'super_gaussian' → {'w': float, 'order': float}    default w=1, order=4
    vortex_charge : int
        Topological charge l for the helical phase l·φ (default 0).
    """

    def __init__(
            self,
            zernike_coeffs: Union[Dict[int, float], List[float], np.ndarray, None] = None,
            amplitude: Union[str, Callable] = 'uniform',
            amplitude_params: Optional[Dict] = None,
            vortex_charge: int = 0,
    ) -> None:
        self._coeffs = self._parse_coeffs(zernike_coeffs)
        self._amp_func = _build_amp_func(amplitude, amplitude_params)
        self.vortex_charge = int(vortex_charge)

    # ── Coefficient parsing ────────────────────────────────────────────────────

    @staticmethod
    def _parse_coeffs(
            raw: Union[Dict[int, float], List[float], np.ndarray, None]
    ) -> Dict[int, float]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return {int(j): float(c) for j, c in raw.items() if c != 0}
        arr = np.asarray(raw, dtype=float)
        return {j: float(c) for j, c in enumerate(arr) if j >= 1 and c != 0}

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def zernike_coeffs(self) -> Dict[int, float]:
        """Zernike coefficients as {noll_j (1-based): float [rad]} dict."""
        return dict(self._coeffs)

    @property
    def amplitude_func(self) -> Callable[[np.ndarray], np.ndarray]:
        """Amplitude callable A(ρ).  Does not include the aperture mask."""
        return self._amp_func

    # ── Polar-coordinate evaluation ───────────────────────────────────────────

    def phase_polar(self, rho: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Wavefront  W(ρ, φ) [rad]  on arbitrary (rho, phi) coordinate arrays.

        Includes both Zernike and vortex contributions.
        No aperture mask is applied: pass only ρ ∈ [0, 1] for the pupil.

        Parameters
        ----------
        rho : array_like  normalized pupil radius  (typically [0, 1])
        phi : array_like  azimuthal angle [rad]

        Returns
        -------
        W : ndarray  same shape as rho  [rad]
        """
        rho = np.asarray(rho, dtype=np.float64)
        phi = np.asarray(phi, dtype=np.float64)
        W = eval_zernike_sum(self._coeffs, rho, phi)
        if self.vortex_charge:
            W = W + self.vortex_charge * phi
        return W

    def amplitude_polar(self, rho: np.ndarray) -> np.ndarray:
        """
        Amplitude A(ρ) on an arbitrary rho array.
        Zero outside the unit disc (ρ > 1).

        Parameters
        ----------
        rho : array_like  normalized pupil radius

        Returns
        -------
        A : ndarray  same shape as rho, values in [0, 1]
        """
        rho = np.asarray(rho, dtype=np.float64)
        return np.where(rho <= 1.0, self._amp_func(rho), 0.0)

    def pupil_polar(self, rho: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Complex pupil  P = A(ρ) · exp(i·W(ρ,φ))  on arbitrary polar coords.
        Zero outside the unit disc (ρ > 1).

        Parameters
        ----------
        rho : array_like  normalized pupil radius
        phi : array_like  azimuthal angle [rad]

        Returns
        -------
        P : complex ndarray  same shape as rho
        """
        rho = np.asarray(rho, dtype=np.float64)
        phi = np.asarray(phi, dtype=np.float64)
        mask = rho <= 1.0
        A = np.where(mask, self._amp_func(rho), 0.0)
        W = np.where(mask, self.phase_polar(rho, phi), 0.0)
        return np.where(mask, A * np.exp(1j * W), 0j)

    # ── Cartesian-grid evaluation ─────────────────────────────────────────────

    def cartesian(
            self,
            nx: int = 256,
            radius: Optional[float] = None,
            dtype_complex: type = np.complex128,
    ) -> Dict:
        """
        Evaluate the pupil on a (nx, nx) Cartesian pixel grid.

        Pixel center convention: the pupil center lies at continuous coordinate
        (nx/2, nx/2), matching zernike_polynomials.py and psf_generator.PSF.

        Parameters
        ----------
        nx           : int    grid side length in pixels
        radius       : float  pupil radius in pixels  (default nx/2)
        dtype_complex : NumPy complex dtype for 'pupil_complex'

        Returns
        -------
        dict
          'rho'          : (nx, nx)  normalized pupil radius  ρ ∈ [0, ∞)
          'phi'          : (nx, nx)  azimuthal angle  [rad]
          'mask'         : (nx, nx)  bool — True inside pupil (ρ ≤ 1)
          'phase'        : (nx, nx)  W(ρ, φ) [rad], zero outside mask
          'amplitude'    : (nx, nx)  A(ρ) ∈ [0, 1], zero outside mask
          'pupil_complex': (nx, nx)  A · exp(i·W), complex, zero outside mask
        """
        if radius is None:
            radius = nx / 2.0

        cx = nx / 2.0
        ax = np.arange(nx, dtype=np.float64) - cx
        xx, yy = np.meshgrid(ax, ax, indexing='xy')  # xx along axis-1, yy along axis-0

        rho = np.hypot(xx, yy) / radius
        phi = np.arctan2(yy, xx)
        mask = rho <= 1.0

        phase = np.where(mask, self.phase_polar(rho, phi), 0.0)
        amp = np.where(mask, self._amp_func(rho), 0.0)
        pp = np.where(mask, amp * np.exp(1j * phase), 0j).astype(dtype_complex)

        return {
            'rho': rho,
            'phi': phi,
            'mask': mask,
            'phase': phase,
            'amplitude': amp,
            'pupil_complex': pp,
        }

    # ── Compatibility shims ───────────────────────────────────────────────────

    def to_psf_wavefront(self, nx: int, radius: float) -> np.ndarray:
        """
        Return complex pupil array (nx × nx, complex64) for psf_generator.PSF.

        Drop-in replacement for PSF.flat_wavefront() and
        PSF.aberration_wavefront().  Assign the return value directly to
        psf.bpp:

            mod = PupilWavefrontModulator(zernike_coeffs={4: 0.5})
            psf = PSF(nx=256, na=1.4, wl=0.488, dx=0.064)
            psf.bpp = mod.to_psf_wavefront(nx=psf.nx, radius=psf.radius)
            img = psf.get_2d_psf((0, 0, 0))

        Parameters
        ----------
        nx     : grid size (= psf.nx)
        radius : pupil radius in pixels (= psf.radius)

        Returns
        -------
        bpp : (nx, nx) complex64
        """
        return self.cartesian(nx=nx, radius=radius,
                              dtype_complex=np.complex64)['pupil_complex']

    def as_amplitude_func(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Return the amplitude callable for RichardsWolfDirect(amplitude_func=…).

        The hard-aperture mask (ρ ≤ 1) is NOT included because
        RichardsWolfDirect only evaluates the envelope inside the pupil.

        Example
        -------
            sim = RichardsWolfDirect(
                NA=1.4, n=1.515, wavelength=0.488,
                amplitude_func=mod.as_amplitude_func(),
            )
            psf3d, Ex, Ey, Ez = sim.compute(mod.zernike_coeffs, x, y, z)
        """
        return self._amp_func

    def phase_on_quadrature(
            self,
            rho_2d: np.ndarray,
            phi_2d: np.ndarray,
    ) -> np.ndarray:
        """
        Wavefront W on a RichardsWolfDirect (N_theta × N_phi) quadrature grid.

        Replaces the combination of pupil_phase_zernike() + vortex_charge
        inside RichardsWolfDirect.compute().  Useful when you want to bypass
        the coefficient dict entirely and drive the phase directly:

            W = mod.phase_on_quadrature(sim._rho_2d, sim._phi_2d)
            P = sim._amp_2d * np.exp(1j * W)

        Parameters
        ----------
        rho_2d : (N_theta, N_phi)  normalized pupil radii from quadrature setup
        phi_2d : (N_theta, N_phi)  azimuthal angles [rad] from quadrature setup

        Returns
        -------
        W : (N_theta, N_phi)  phase [rad]
        """
        return self.phase_polar(rho_2d, phi_2d)

    # ── Metrics ───────────────────────────────────────────────────────────────

    def wavefront_rms(self, nx: int = 256) -> float:
        """
        RMS wavefront error over the pupil [rad].

        Computed on a (nx × nx) Cartesian grid, weighted uniformly inside
        the aperture (amplitude weighting is NOT applied).

        Parameters
        ----------
        nx : int  grid size for numerical integration (default 256)

        Returns
        -------
        rms : float [rad]
        """
        d = self.cartesian(nx=nx)
        W = d['phase']
        msk = d['mask']
        return float(np.sqrt(np.mean(W[msk] ** 2))) if msk.any() else 0.0

    def strehl_ratio(self, nx: int = 256) -> float:
        """
        Approximate Strehl ratio via the Maréchal approximation:
        S ≈ exp(−σ²_W), where σ_W is the RMS wavefront error.

        Valid for small aberrations (σ_W ≲ 1 rad); exact for Gaussian
        wavefront statistics.

        Parameters
        ----------
        nx : int  grid size for RMS computation

        Returns
        -------
        S : float  in (0, 1]
        """
        rms = self.wavefront_rms(nx=nx)
        return float(np.exp(-rms ** 2))
