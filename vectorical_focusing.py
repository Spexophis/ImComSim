"""
Richards–Wolf Vectorial PSF  –  Direct Gauss-Legendre Quadrature
=================================================================

Computes the 3-D vectorial PSF by evaluating the Richards–Wolf diffraction
integral directly with high-order quadrature.

Integration scheme
------------------
  θ ∈ [0, θmax]:  N_theta-point Gauss-Legendre (GL) quadrature
                   Exact for polynomials of degree ≤ 2 N_theta − 1
                   exponential convergence for analytic integrands.

  φ ∈ [0, 2π):    N_phi-point uniform / trapezoidal rule
                   Spectrally exact for trigonometric polynomials of
                   degree ≤ N_phi/2 — equivalent to Gauss for periodic fns.

Apodisation: √(cos θ)  in the angular integral (Abbe sine condition).

Main loop structure (per field component)
------------------------------------------
  Pre-compute:
    GL nodes/weights θ_i, w_i           O(N_theta)
    Phase grid W(θ_i, φ_j)              O(N_theta × N_phi)
    polarization weights T_α(θ_i, φ_j)  O(N_theta × N_phi)

  θ-loop (i = 0 … N_theta):
    exp_x[j,k] = exp(i 2π k⊥ cosφ_j  x_k)  →  (N_phi, Nx)
    exp_y[j,l] = exp(i 2π k⊥ sinφ_j  y_l)  →  (N_phi, Ny)
    xy_contrib  = (P_eff_row * exp_x)ᵀ @ exp_y  ← BLAS dgemm (Nx, Ny)
    prop_z[m]   = exp(i 2π kz_i z_m)           →  (Nz,)
    E_α[m,l,k] += prop_z[m] * xy_contrib[k,l]

  Total FLOPs:  O(N_theta × N_phi × Nx × Ny)
  Memory:       O(N_phi × max(Nx, Ny))  per θ iteration

References
----------
Richards & Wolf (1959) Proc. R. Soc. A 253, 358-379
Novotny & Hecht (2012) Principles of Nano-Optics 2nd ed., §3
Noll (1976) JOSA 66, 207-211
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import factorial


def noll_to_nm(j: int) -> Tuple[int, int]:
    if j < 1:
        raise ValueError("Noll index j must be >= 1.")
    n = 0
    while n * (n + 1) // 2 < j:
        n += 1
    n -= 1
    k = j - n * (n + 1) // 2 - 1
    ms: List[int] = []
    if n % 2 == 0:
        ms.append(0)
    for absm in range(2 - n % 2, n + 1, 2):
        ms.append(absm)
        ms.append(-absm)
    return n, ms[k]


def zernike_radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    m = abs(m)
    R = np.zeros_like(rho, dtype=float)
    for s in range((n - m) // 2 + 1):
        c = ((-1) ** s * int(factorial(n - s, exact=True))
             / (int(factorial(s, exact=True))
                * int(factorial((n + m) // 2 - s, exact=True))
                * int(factorial((n - m) // 2 - s, exact=True))))
        R += c * rho ** (n - 2 * s)
    return R


def zernike(j: int, rho: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Zernike polynomials  – Noll 1976
    """
    n, m = noll_to_nm(j)
    R = zernike_radial(n, m, rho)
    norm = np.sqrt(n + 1) if m == 0 else np.sqrt(2 * (n + 1))
    if m == 0:
        return norm * R
    elif m > 0:
        return norm * R * np.cos(m * phi)
    else:
        return norm * R * np.sin(-m * phi)


def pupil_phase_zernike(
        rho: np.ndarray, phi: np.ndarray,
        coeffs: Dict[int, float],
) -> np.ndarray:
    """Wavefront error W = Σ c_j Z_j  [rad]  (no mask applied here)."""
    W = np.zeros_like(rho)
    for j, c in coeffs.items():
        if c:
            W += c * zernike(j, rho, phi)
    return W


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Polarization weights  T_α(θ, φ)
#
#  The 3×2 Richards-Wolf rotation matrix maps input Jones vector (ax,ay) to
#  focal (Ex, Ey, Ez).  For cylindrical vector beams the Jones vector varies
#  with φ closed-form results are used.
# ══════════════════════════════════════════════════════════════════════════════

def polarization_weights_ring(
        cos_theta: float,
        sin_theta: float,
        cos_phi: np.ndarray,
        sin_phi: np.ndarray,
        polarization: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (Tx, Ty, Tz) weight arrays of shape (N_phi,) for a single θ value.

    All components are complex to support circular polarization.
    """
    ct, st = cos_theta, sin_theta
    cp, sp = cos_phi, sin_phi
    pol = polarization.lower()

    if pol == "x":
        return (ct * cp ** 2 + sp ** 2,
                (ct - 1) * sp * cp,
                -st * cp + 0j * cp)  # keep dtype consistent

    elif pol == "y":
        return ((ct - 1) * sp * cp,
                ct * sp ** 2 + cp ** 2,
                -st * sp + 0j * sp)

    elif pol in ("lc", "left_circular"):
        f = 1.0 / np.sqrt(2)
        Mxx = ct * cp ** 2 + sp ** 2
        Mxy = (ct - 1) * sp * cp
        Myx = (ct - 1) * sp * cp
        Myy = ct * sp ** 2 + cp ** 2
        Mzx = -st * cp
        Mzy = -st * sp
        return f * (Mxx + 1j * Mxy), f * (Myx + 1j * Myy), f * (Mzx + 1j * Mzy)

    elif pol in ("rc", "right_circular"):
        f = 1.0 / np.sqrt(2)
        Mxx = ct * cp ** 2 + sp ** 2
        Mxy = (ct - 1) * sp * cp
        Myx = (ct - 1) * sp * cp
        Myy = ct * sp ** 2 + cp ** 2
        Mzx = -st * cp
        Mzy = -st * sp
        return f * (Mxx - 1j * Mxy), f * (Myx - 1j * Myy), f * (Mzx - 1j * Mzy)

    elif pol == "radial":
        # ax = cosφ, ay = sinφ  →  Tx = cosθ cosφ, Ty = cosθ sinφ, Tz = -sinθ
        return ct * cp + 0j * cp, ct * sp + 0j * sp, np.full_like(cp, -st, dtype=complex)

    elif pol == "azimuthal":
        # ax = -sinφ, ay = cosφ  →  Tx = -sinφ, Ty = cosφ, Tz = 0
        return -sp + 0j * sp, cp + 0j * cp, np.zeros(len(cp), dtype=complex)

    else:
        raise ValueError(f"Unknown polarization {polarization!r}. "
                         "Choose: x | y | lc | rc | radial | azimuthal")


class RichardsWolfDirect:
    """
    Vectorial PSF via direct Gauss-Legendre quadrature.

    Parameters
    ----------
    NA            : numerical aperture
    n             : immersion refractive index
    wavelength    : free-space wavelength (consistent units, e.g. µm)
    N_theta       : GL quadrature order for θ  (50–200 gives machine precision)
    N_phi         : uniform quadrature points for φ  (64–256 is sufficient)
    polarization  : 'x' | 'y' | 'lc' | 'rc' | 'radial' | 'azimuthal'
    amplitude_func: callable(rho_norm) → amplitude envelope, or None (uniform)
    """

    def __init__(
            self,
            NA: float,
            n: float,
            wavelength: float,
            N_theta: int = 100,
            N_phi: int = 256,
            polarization: str = "x",
            amplitude_func: Optional[Callable] = None,
    ) -> None:
        self.NA = NA
        self.n = n
        self.wavelength = wavelength
        self.N_theta = N_theta
        self.N_phi = N_phi
        self.polarization = polarization.lower()
        self.amplitude_func = amplitude_func

        self.k_n = n / wavelength  # spatial frequency unit
        self.theta_max = np.arcsin(NA / n)  # maximum half-angle

        self._setup_quadrature()

    # ── Quadrature setup ──────────────────────────────────────────────────────

    def _setup_quadrature(self) -> None:
        # ── θ: Gauss-Legendre on [0, θmax] ──────────────────────────────────
        t_nodes, t_weights = leggauss(self.N_theta)  # nodes/weights on [-1, 1]
        # Linear map  t ∈ [-1,1]  →  θ ∈ [0, θmax]
        self._theta = self.theta_max / 2 * (t_nodes + 1)  # (N_theta,)
        self._w_theta = t_weights * (self.theta_max / 2)  # Jacobian of map

        self._sin_theta = np.sin(self._theta)  # (N_theta,)
        self._cos_theta = np.cos(self._theta)

        # Normalised pupil radius ρ = sinθ / sinθmax  ∈ [0, 1]
        self._rho_nodes = self._sin_theta / np.sin(self.theta_max)

        # Transverse and axial spatial frequencies  [1/length]
        self._kperp = self.k_n * self._sin_theta  # k_n sinθ
        self._kz = self.k_n * self._cos_theta  # k_n cosθ

        # Combined θ weight: w_i · sinθ_i · √(cosθ_i)
        #   sinθ dθ  is the θ part of the angular measure
        #   √(cosθ)  is the Abbe-sine-condition apodisation
        self._q_theta = (self._w_theta
                         * self._sin_theta
                         * np.sqrt(np.maximum(self._cos_theta, 0)))

        # ── φ: uniform on [0, 2π) ────────────────────────────────────────────
        self._phi = 2 * np.pi * np.arange(self.N_phi) / self.N_phi
        self._cos_phi = np.cos(self._phi)  # (N_phi,)
        self._sin_phi = np.sin(self._phi)
        self._w_phi = 2 * np.pi / self.N_phi  # scalar weight

        # ── 2D phase grid for Zernike evaluation ─────────────────────────────
        # Shape: (N_theta, N_phi)
        self._rho_2d = self._rho_nodes[:, None] * np.ones(self.N_phi)
        self._phi_2d = np.ones(self.N_theta)[:, None] * self._phi[None, :]

        # Amplitude envelope on the 2D grid
        if self.amplitude_func is not None:
            self._amp_2d = self.amplitude_func(self._rho_2d)
        else:
            self._amp_2d = np.ones((self.N_theta, self.N_phi))

    # ── Main computation ──────────────────────────────────────────────────────

    def compute(
            self,
            zernike_coeffs: Dict[int, float],
            x_out: np.ndarray,
            y_out: np.ndarray,
            z_out: np.ndarray,
            vortex_charge: int = 0,
            normalize: bool = True,
            verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the 3-D vectorial PSF at arbitrary (x, y, z) positions.

        Parameters
        ----------
        zernike_coeffs : {noll_j: coefficient [rad]}
        x_out, y_out   : 1-D arrays of lateral output positions (same unit as λ)
        z_out          : 1-D array of axial output positions
        vortex_charge  : integer topological charge l
        normalize      : peak intensity → 1

        Returns
        -------
        psf       : (Nz, Ny, Nx)  float
        Ex, Ey, Ez: (Nz, Ny, Nx)  complex
        """
        Nx, Ny, Nz = len(x_out), len(y_out), len(z_out)

        # ── Pupil phase on (θ_i, φ_j) grid ───────────────────────────────────
        W_zernike = pupil_phase_zernike(self._rho_2d, self._phi_2d, zernike_coeffs)
        W_vortex = vortex_charge * self._phi_2d
        W_total = W_zernike + W_vortex  # (N_theta, N_phi)

        # Complex scalar pupil
        P = self._amp_2d * np.exp(1j * W_total)  # (N_theta, N_phi)

        # Per-θ weight row: q_theta[i] * w_phi  ×  P[i, :]
        # (polarization weights T_α are added inside the θ loop)
        P_weighted = P * (self._q_theta[:, None] * self._w_phi)  # (N_theta, N_phi)

        # ── Initialise output ─────────────────────────────────────────────────
        Ex = np.zeros((Nz, Ny, Nx), dtype=complex)
        Ey = np.zeros((Nz, Ny, Nx), dtype=complex)
        Ez = np.zeros((Nz, Ny, Nx), dtype=complex)

        t0 = time.time()

        # ── Main θ-loop ───────────────────────────────────────────────────────
        for i in range(self.N_theta):
            ct = float(self._cos_theta[i])
            st = float(self._sin_theta[i])
            kp = float(self._kperp[i])
            kzi = float(self._kz[i])

            # polarization weights for this θ ring  →  (N_phi,) complex arrays
            Tx, Ty, Tz = polarization_weights_ring(
                ct, st, self._cos_phi, self._sin_phi, self.polarization
            )

            # Effective amplitude per φ sample for each field component
            Pw = P_weighted[i, :]  # (N_phi,)
            Ax = Pw * Tx  # (N_phi,)
            Ay = Pw * Ty
            Az = Pw * Tz

            # Transverse plane-wave factors
            # exp_x[j, k] = exp(i 2π k⊥ cosφ_j x_k)
            # exp_y[j, l] = exp(i 2π k⊥ sinφ_j y_l)
            phase_x = (2 * np.pi * kp) * np.outer(self._cos_phi, x_out)  # (N_phi, Nx)
            phase_y = (2 * np.pi * kp) * np.outer(self._sin_phi, y_out)  # (N_phi, Ny)
            exp_x = np.exp(1j * phase_x)
            exp_y = np.exp(1j * phase_y)

            # XY contribution from this θ ring:
            #   Σ_j  A_j · exp_x[j,k] · exp_y[j,l]  →  (Nx, Ny)
            #   = (A * exp_x)ᵀ @ exp_y
            #     (Nx, N_phi) @ (N_phi, Ny)  — one BLAS dgemm per component
            xy_x = (Ax[:, None] * exp_x).T @ exp_y  # (Nx, Ny)
            xy_y = (Ay[:, None] * exp_x).T @ exp_y
            xy_z = (Az[:, None] * exp_x).T @ exp_y

            # Axial propagation factor: exp(i 2π kz z_m)  →  (Nz,)
            prop_z = np.exp(1j * 2 * np.pi * kzi * z_out)

            # Accumulate into (Nz, Ny, Nx)
            # xy_x.T has shape (Ny, Nx) prop broadcast over (Nz,)
            Ex += prop_z[:, None, None] * xy_x.T[None, :, :]
            Ey += prop_z[:, None, None] * xy_y.T[None, :, :]
            Ez += prop_z[:, None, None] * xy_z.T[None, :, :]

            if verbose and (i + 1) % max(1, self.N_theta // 10) == 0:
                frac = (i + 1) / self.N_theta
                elapsed = time.time() - t0
                eta = elapsed / frac * (1 - frac)
                print(f"    θ {i + 1:3d}/{self.N_theta}  "
                      f"[{'#' * int(20 * frac):<20}]  "
                      f"{frac * 100:4.0f}%  ETA {eta:4.1f}s", end="\r")

        if verbose:
            print(f"\n    Done in {time.time() - t0:.2f} s")

        psf = np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2

        if normalize:
            peak = psf.max()
            if peak > 0:
                psf /= peak
                sc = 1.0 / np.sqrt(peak)
                Ex *= sc
                Ey *= sc
                Ez *= sc

        return psf, Ex, Ey, Ez

    # ── Pupil visualisation ───────────────────────────────────────────────────

    def pupil_map(
            self,
            zernike_coeffs: Dict[int, float],
            vortex_charge: int = 0,
            Npix: int = 256,
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate the full pupil function on a Cartesian (kx, ky) grid.

        Because amplitude and phase are analytic functions of (rho, phi), and
        rho = sin(theta) / sin(theta_max) = kr / kr_max  with kr = sqrt(kx^2+ky^2),
        we compute everything directly on the Cartesian grid — no interpolation.

        Parameters
        ----------
        zernike_coeffs : {noll_j: coeff [rad]}
        vortex_charge  : integer topological charge l
        Npix           : side length of the output square grid

        Returns
        -------
        dict with keys:
          'kx', 'ky'        : (Npix,) 1-D axes  [units of kr_max]
          'rho'             : (Npix, Npix)  normalised pupil radius in [0,1]
          'phi'             : (Npix, Npix)  azimuthal angle [rad]
          'mask'            : (Npix, Npix)  bool — True inside pupil
          'amplitude'       : (Npix, Npix)  envelope A(rho)
          'apodization'     : (Npix, Npix)  sqrt(cos theta) Abbe factor
          'phase_zernike'   : (Npix, Npix)  Zernike wavefront W(rho,phi) [rad]
          'phase_vortex'    : (Npix, Npix)  helical phase l*phi [rad]
          'phase_total'     : (Npix, Npix)  W_zernike + W_vortex [rad]
          'pupil_complex'   : (Npix, Npix)  A*sqrt(cosθ)*exp(i W_total)
        """
        ax1d = np.linspace(-1.0, 1.0, Npix)
        kx, ky = np.meshgrid(ax1d, ax1d, indexing="xy")

        rho = np.sqrt(kx ** 2 + ky ** 2)
        phi = np.arctan2(ky, kx)
        mask = rho <= 1.0

        sin_t = rho * np.sin(self.theta_max)
        cos_t = np.where(mask, np.sqrt(np.maximum(1.0 - sin_t ** 2, 0.0)), 0.0)

        amp = (self.amplitude_func(rho) if self.amplitude_func is not None
               else np.ones_like(rho))
        amp = np.where(mask, amp, 0.0)

        apod = np.where(mask, np.sqrt(np.maximum(cos_t, 1e-12)), 0.0)

        W_zernike = np.where(mask, pupil_phase_zernike(rho, phi, zernike_coeffs), 0.0)
        W_vortex = np.where(mask, vortex_charge * phi, 0.0)
        W_total = W_zernike + W_vortex

        P_complex = np.where(mask, amp * apod * np.exp(1j * W_total), 0.0 + 0j)

        return {
            "kx": ax1d,
            "ky": ax1d,
            "rho": rho,
            "phi": phi,
            "mask": mask,
            "amplitude": amp,
            "apodization": apod,
            "phase_zernike": W_zernike,
            "phase_vortex": W_vortex,
            "phase_total": W_total,
            "pupil_complex": P_complex,
        }

    # ── Convergence study ─────────────────────────────────────────────────────

    def radial_profile(
            self,
            psf: np.ndarray,
            x_out: np.ndarray,
            y_out: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Azimuthal average of the focal-plane PSF.
        Returns (r_arr [same units as x_out], intensity).
        """
        Nz, Ny, Nx = psf.shape
        iz = int(np.argmax(psf.reshape(Nz, -1).sum(axis=1)))
        xy = psf[iz]

        xx, yy = np.meshgrid(x_out, y_out)
        rr = np.sqrt(xx ** 2 + yy ** 2).ravel()
        vv = xy.ravel()

        r_max = min(np.abs(x_out).max(), np.abs(y_out).max())
        r_bins = np.linspace(0, r_max, 200)
        r_mid = 0.5 * (r_bins[:-1] + r_bins[1:])
        profile = np.zeros(len(r_mid))
        counts = np.zeros(len(r_mid))

        idx = np.digitize(rr, r_bins) - 1
        for k in range(len(r_mid)):
            sel = idx == k
            if sel.any():
                profile[k] = vv[sel].mean()
                counts[k] = sel.sum()

        return r_mid, profile


def convergence_study(
        NA: float, n: float, wavelength: float,
        zernike_coeffs: Dict[int, float],
        x_out: np.ndarray,
        y_out: np.ndarray,
        z_focus: float,
        N_theta_list: List[int] = (10, 20, 40, 80, 160),
        N_phi: int = 256,
        vortex_charge: int = 0,
) -> Dict:
    """
    Compute x-pol PSF at z=z_focus for increasing N_theta.
    Returns profiles and RMSE relative to the finest resolution.
    """
    z_arr = np.array([z_focus])
    profiles = {}

    for N_th in N_theta_list:
        sim = RichardsWolfDirect(
            NA=NA, n=n, wavelength=wavelength,
            N_theta=N_th, N_phi=N_phi, polarization="x",
        )
        psf, *_ = sim.compute(zernike_coeffs, x_out, y_out, z_arr,
                              vortex_charge=vortex_charge, verbose=False,
                              normalize=False)
        psf_focal = psf[0]
        profiles[N_th] = psf_focal

    # Normalize to finest
    ref = profiles[N_theta_list[-1]]
    ref_norm = ref / ref.max()

    rmse = {}
    for N_th in N_theta_list[:-1]:
        p = profiles[N_th] / profiles[N_th].max()
        rmse[N_th] = float(np.sqrt(np.mean((p - ref_norm) ** 2)))

    return {"profiles": profiles, "rmse": rmse, "ref_Ntheta": N_theta_list[-1]}


if __name__ == "__main__":
    NA, n, lam = 1.4, 1.515, 0.488
    N_theta = 128
    N_phi = 512
    dx = 0.016
    Nx = Ny = 128
    Nz = 81
    x_out = (np.arange(Nx) - Nx // 2) * dx
    y_out = (np.arange(Ny) - Ny // 2) * dx
    z_out = np.linspace(-2.0, 2.0, Nz)

    aber = {4: 0.3}

    pol, charge, aber = "x", 0, {}

    sim = RichardsWolfDirect(
        NA=NA, n=n, wavelength=lam,
        N_theta=N_theta, N_phi=N_phi, polarization=pol,
    )

    psf, Ex, Ey, Ez = sim.compute(
        aber, x_out, y_out, z_out,
        vortex_charge=charge, normalize=True, verbose=True,
    )

    # tf.imwrite(, psf)
