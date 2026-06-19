# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.

"""
Angular and radial Fourier-Bessel decomposition of image FFT.

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy.optimize import curve_fit
from scipy.special import jn_zeros, jv


_OTF_CACHE: dict = {}


def angular_decomposition_full(F_centred, freq_x, freq_y, nu_max, n_modes=8, n_rbins=128):
    """
    Decompose the centred FFT F(q, φ) into angular Fourier modes.

        F_m(q) = (1/2π) ∫₀²π F(q,φ) e^{-imφ} dφ

    Computation is strictly limited to q ∈ [0, nu_max].

    Parameters
    ----------
    F_centred : (H, W) complex  — FFT with DC at center (fftshift applied)
    freq_x    : (W,) float      — frequency axis [cyc/µm]
    freq_y    : (H,) float      — frequency axis [cyc/µm]
    nu_max    : float           — pupil cutoff [cyc/µm]
    n_modes   : int             — compute m = -n_modes … +n_modes
    n_rbins   : int             — radial annuli inside [0, nu_max]

    Returns
    -------
    q_grid : (n_rbins,)                   radial bin centres [cyc/µm]
    Fm     : (2*n_modes+1, n_rbins) complex
    ms     : (2*n_modes+1,) int
    meta   : dict
    """
    fx, fy  = np.meshgrid(freq_x, freq_y)
    fq      = np.sqrt(fx**2 + fy**2)
    phi     = np.arctan2(fy, fx)           # [-π, π]

    ms      = np.arange(-n_modes, n_modes + 1)
    q_edges = np.linspace(0, nu_max, n_rbins + 1)
    q_grid  = 0.5 * (q_edges[:-1] + q_edges[1:])

    Fm       = np.zeros((len(ms), n_rbins), dtype=complex)
    n_pixels = np.zeros(n_rbins, dtype=int)

    for ri in range(n_rbins):
        ring = (fq >= q_edges[ri]) & (fq < q_edges[ri + 1])
        n_pixels[ri] = ring.sum()
        if n_pixels[ri] < 2:
            continue
        phi_r = phi[ring]
        F_r   = F_centred[ring]
        # FIX: vectorise inner loop — one matrix op replaces n_ms scalar ops
        # exp_terms shape: (n_ms, n_pixels_in_ring)
        exp_terms = np.exp(-1j * ms[:, None] * phi_r[None, :])
        Fm[:, ri] = (F_r[None, :] * exp_terms).mean(axis=1)

    power_density = np.abs(Fm)**2 * q_grid[None, :]
    power_m       = 2 * np.pi * np.trapezoid(power_density, q_grid, axis=1)
    phase_m       = np.angle(Fm)
    m0_idx        = np.where(ms == 0)[0][0]
    F0_mag        = np.abs(Fm[m0_idx]) + 1e-30
    coherence_m   = np.abs(Fm) / F0_mag[None, :]
    q_peak_m      = q_grid[np.argmax(np.abs(Fm)**2, axis=1)]

    meta = dict(
        power_m=power_m, power_density=power_density,
        phase_m=phase_m, coherence_m=coherence_m,
        q_peak_m=q_peak_m, n_pixels=n_pixels,
        ms=ms, q_grid=q_grid,
    )
    return q_grid, Fm, ms, meta


def compute_aberration_indices(meta):
    """
    Scalar aberration indices derived from the angular mode power.
    All indices are 0 for a perfect system, larger for stronger aberration.
    """
    ms      = meta['ms']
    power_m = meta['power_m']
    q_grid  = meta['q_grid']

    def P(m_val):
        j = np.where(ms == m_val)[0]
        return float(power_m[j[0]]) if len(j) else 0.0

    P0 = P(0) + 1e-30
    Pt = power_m.sum() + 1e-30

    astig     = (P(2)  + P(-2)) / P0
    coma      = (P(1)  + P(-1)) / P0
    trefoil   = (P(3)  + P(-3)) / P0
    sym_ratio = 1.0 - P0 / Pt

    j2 = np.where(ms == 2)[0]
    astig_angle = None
    if len(j2) and meta['power_density'][j2[0]].max() > 0:
        pk          = np.argmax(meta['power_density'][j2[0]])
        astig_angle = np.degrees(meta['phase_m'][j2[0], pk] / 2) % 180

    j0_idx = np.where(ms == 0)[0][0]
    pd0    = meta['power_density'][j0_idx]
    q_eff  = float(np.average(q_grid, weights=pd0)) if pd0.sum() > 0 else 0.0

    return dict(
        astigmatism_index=astig, coma_index=coma,
        trefoil_index=trefoil, symmetry_ratio=sym_ratio,
        q_eff=q_eff, astigmatism_angle=astig_angle,
    )


def angular_filter_fft(F_centred, freq_x, freq_y, keep_modes):
    """
    Reconstruct the image retaining only the specified angular modes.

    keep_modes=[0]    → rotationally symmetric component
    keep_modes=[2,-2] → astigmatic component
    """
    fx, fy = np.meshgrid(freq_x, freq_y)
    phi    = np.arctan2(fy, fx)
    mask   = np.zeros_like(F_centred, dtype=complex)
    if not keep_modes:
        return mask, np.zeros(F_centred.shape)
    for m in keep_modes:
        mask += np.exp(1j * m * phi)
    mask /= (2 * np.pi)
    Ff = F_centred * mask
    return Ff, np.real(np.fft.ifft2(np.fft.ifftshift(Ff)))


def otf_ideal(q, nu_max):
    """Ideal incoherent OTF: (2/π)[arccos(u) - u√(1-u²)], u = q/nu_max."""
    u = np.clip(q / nu_max, 0, 1)
    return (2 / np.pi) * (np.arccos(u) - u * np.sqrt(1 - u**2))


def _pupil_phase(rho, a20, a40, a60=0.0):
    """Rotationally symmetric wavefront [rad] as a sum of Zernike polynomials."""
    Z20 = 2*rho**2 - 1
    Z40 = 6*rho**4 - 6*rho**2 + 1
    Z60 = 20*rho**6 - 30*rho**4 + 12*rho**2 - 1
    return a20*Z20 + a40*Z40 + a60*Z60


def compute_otf_from_wavefront(a20, a40, a60=0.0,
                                n_pupil=512, n_q=200, nu_max=1.0):
    """
    Numerically compute the radially averaged OTF from Zernike coefficients
    via 2D pupil autocorrelation.

    Returns q_grid [0, nu_max] and otf_rad (real, normalised).
    """
    rho = np.linspace(0, 1, n_pupil)
    W   = _pupil_phase(rho, a20, a40, a60)

    Np  = n_pupil
    pu  = np.linspace(-1, 1, 2 * Np)
    pux, puy = np.meshgrid(pu, pu)
    pur = np.sqrt(pux**2 + puy**2)
    P2d = np.where(pur <= 1, np.exp(1j * np.interp(pur, rho, W)), 0.0)

    FP     = np.fft.fft2(P2d)
    otf_2d = np.fft.fftshift(np.real(np.fft.ifft2(np.abs(FP)**2)))
    otf_2d /= otf_2d.max()

    sh    = otf_2d.shape
    cx, cy = sh[1] // 2, sh[0] // 2
    yy, xx = np.ogrid[:sh[0], :sh[1]]
    rr    = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    r_max_px = Np

    q_bins  = np.linspace(0, nu_max, n_q + 1)
    q_grid  = 0.5 * (q_bins[:-1] + q_bins[1:])
    otf_rad = np.zeros(n_q)
    for i in range(n_q):
        r_lo = q_bins[i]     / nu_max * r_max_px
        r_hi = q_bins[i + 1] / nu_max * r_max_px
        mask = (rr >= r_lo) & (rr < r_hi)
        if mask.sum() > 0:
            otf_rad[i] = otf_2d[mask].mean()

    return q_grid, otf_rad


def bessel_coeffs_m0(F0_q, q_grid, nu_max, n_radial=16):
    """
    Fourier-Bessel coefficients of F_0(q) for angular order m=0.

        c_{0n} = ∫₀^{ν_max} F_0(q) ψ_{0n}(q) q dq

    ψ_{0n} are orthonormal on [0, ν_max] with weight q dq.

    Returns
    -------
    c      : (n_radial,) complex   — coefficients
    Psi    : (n_radial, n_rbins)   — basis functions evaluated on q_grid
    alphas : (n_radial,)           — Bessel zeros used
    """
    alphas = jn_zeros(0, n_radial)
    Psi    = np.zeros((n_radial, len(q_grid)))
    c      = np.zeros(n_radial, dtype=complex)

    for ni in range(n_radial):
        a      = alphas[ni]
        norm   = np.sqrt(2) / (nu_max * np.abs(jv(1, a)))
        psi    = norm * jv(0, a * q_grid / nu_max)
        Psi[ni] = psi
        c[ni]  = np.trapezoid(F0_q * psi * q_grid, q_grid)

    return c, Psi, alphas


def otf_bessel_model(q_grid, nu_max):
    """
    Return (otf_id, otf_d, otf_s) interpolated onto q_grid.

    otf_id : ideal OTF
    otf_d  : OTF with unit defocus (a20=1)
    otf_s  : OTF with unit primary spherical (a40=1)

    Results are cached at module level — the 3 OTF computations are
    each ~1024×1024 FFTs and should not be repeated on every call.
    """
    # FIX: cache expensive OTF computations so repeated calls are free
    global _OTF_CACHE
    if not _OTF_CACHE:
        _OTF_CACHE['q_id'], _OTF_CACHE['otf_id'] = compute_otf_from_wavefront(0.0, 0.0)
        _OTF_CACHE['q_d'],  _OTF_CACHE['otf_d']  = compute_otf_from_wavefront(1.0, 0.0)
        _OTF_CACHE['q_s'],  _OTF_CACHE['otf_s']  = compute_otf_from_wavefront(0.0, 1.0)

    f_id = np.interp(q_grid, _OTF_CACHE['q_id'], _OTF_CACHE['otf_id'])
    f_d  = np.interp(q_grid, _OTF_CACHE['q_d'],  _OTF_CACHE['otf_d'])
    f_s  = np.interp(q_grid, _OTF_CACHE['q_s'],  _OTF_CACHE['otf_s'])
    return f_id, f_d, f_s


def fit_defocus_spherical(F0_abs, q_grid, nu_max):
    """
    Fit |F_0(q)| to:  α·OTF_ideal + β·ΔOTF_defocus + γ·ΔOTF_spherical

    Returns dict with estimated defocus/spherical amplitudes and fit quality.
    """
    otf_id, otf_d, otf_s = otf_bessel_model(q_grid, nu_max)

    d_def   = otf_d - otf_id
    d_sph   = otf_s - otf_id
    F0_norm = F0_abs / (F0_abs.max() + 1e-30)

    A      = np.column_stack([otf_id, d_def, d_sph])
    coeffs, _, _, _ = np.linalg.lstsq(A, F0_norm, rcond=None)
    fit    = A @ coeffs

    return dict(
        coeffs=coeffs, fit=fit, F0_norm=F0_norm,
        otf_id=otf_id, d_def=d_def, d_sph=d_sph,
        est_defocus=coeffs[1], est_spherical=coeffs[2],
        residual_rms=float(np.sqrt(np.mean((F0_norm - fit)**2))),
    )


def phase_polynomial_fit(F0_complex, q_grid, nu_max):
    """
    Fit unwrapped phase of F_0(q) to φ = p2·q² + p4·q⁴.

    p2 ∝ defocus,  p4 ∝ spherical aberration.
    """
    mask = q_grid <= nu_max
    q_in = q_grid[mask]
    phi  = np.unwrap(np.angle(F0_complex[mask]))

    def model(q, p2, p4):
        return p2 * q**2 + p4 * q**4

    try:
        popt, pcov = curve_fit(model, q_in, phi, p0=[0.0, 0.0])
        perr       = np.sqrt(np.diag(pcov))
        phi_fit    = model(q_in, *popt)
    except RuntimeError:
        popt    = np.array([0.0, 0.0])
        perr    = np.array([np.nan, np.nan])
        phi_fit = np.zeros_like(q_in)

    return dict(
        q_in=q_in, phi_raw=phi, phi_fit=phi_fit,
        p2=popt[0], p2_err=perr[0],
        p4=popt[1], p4_err=perr[1],
    )


def _fft_setup(image, pixel_size, NA, wavelength):
    """Shared FFT + pupil-mask setup used by all run_* functions."""
    nu_max = NA / wavelength           # cyc/µm  (incoherent cutoff = NA/λ)
    H, W   = image.shape
    F_full = np.fft.fftshift(np.fft.fft2(image.astype(np.float64)))
    freq_x = np.fft.fftshift(np.fft.fftfreq(W, d=pixel_size))
    freq_y = np.fft.fftshift(np.fft.fftfreq(H, d=pixel_size))
    fx, fy = np.meshgrid(freq_x, freq_y)
    fq     = np.sqrt(fx**2 + fy**2)
    F      = F_full * (fq <= nu_max)
    return nu_max, F_full, F, freq_x, freq_y


def _print_angular(indices, nu_max):
    print("\n  Angular mode analysis")
    print(f"  {'─'*44}")
    print(f"  Astigmatism (|m|=2): {indices['astigmatism_index']:.4f}")
    print(f"  Coma        (|m|=1): {indices['coma_index']:.4f}")
    print(f"  Trefoil     (|m|=3): {indices['trefoil_index']:.4f}")
    print(f"  Sym breaking:        {indices['symmetry_ratio']:.4f}")
    print(f"  Eff. bandwidth:      {indices['q_eff']:.4f} cyc/µm  "
          f"({indices['q_eff']/nu_max*100:.1f}%)")
    if indices['astigmatism_angle'] is not None:
        print(f"  Astig orientation:   {indices['astigmatism_angle']:.1f}°")


def run_angular_analysis(image, pixel_size, NA, wavelength, n_modes=8, n_rbins=128, verbose=True):
    """Angular-mode decomposition and aberration indices."""
    nu_max, F_full, F, freq_x, freq_y = _fft_setup(image, pixel_size, NA, wavelength)
    q_grid, Fm, ms, meta = angular_decomposition_full(
        F, freq_x, freq_y, nu_max, n_modes=n_modes, n_rbins=n_rbins)
    # FIX: compute_aberration_indices takes only (meta) — removed spurious nu_max arg
    indices = compute_aberration_indices(meta)
    if verbose:
        _print_angular(indices, nu_max)
    return dict(q_grid=q_grid, Fm=Fm, ms=ms, meta=meta, indices=indices,
                F_full=F_full, freq_x=freq_x, freq_y=freq_y,
                nu_max=nu_max, pixel_size=pixel_size, NA=NA, wavelength=wavelength)


def run_radial_mode_analysis(image, pixel_size, NA, wavelength,
                             n_radial=16, n_rbins=128, verbose=True):
    """Fourier-Bessel decomposition of F_0(q) for defocus/spherical diagnosis."""
    nu_max, F_full, F, freq_x, freq_y = _fft_setup(image, pixel_size, NA, wavelength)
    q_grid, Fm, ms, meta = angular_decomposition_full(
        F, freq_x, freq_y, nu_max, n_modes=1, n_rbins=n_rbins)

    m0_idx    = np.where(ms == 0)[0][0]
    F0        = Fm[m0_idx]
    c, Psi, alphas = bessel_coeffs_m0(F0, q_grid, nu_max, n_radial)
    power_n   = np.abs(c)**2
    fit_result = fit_defocus_spherical(np.abs(F0), q_grid, nu_max)
    phase_fit  = phase_polynomial_fit(F0, q_grid, nu_max)
    F0_recon   = (c[:, None] * Psi).sum(axis=0)

    if verbose:
        print("\n  Radial mode diagnosis (m=0)")
        print(f"  {'─'*44}")
        print(f"  Defocus  (p2·q²):  {phase_fit['p2']:+.4f} rad·µm²")
        print(f"  Spherical (p4·q⁴): {phase_fit['p4']:+.4f} rad·µm⁴")
        print(f"  OTF defocus coeff: {fit_result['est_defocus']:+.4f}")
        print(f"  OTF spherical:     {fit_result['est_spherical']:+.4f}")
        print(f"  OTF fit RMS:       {fit_result['residual_rms']:.4f}")

    return dict(c=c, Psi=Psi, alphas=alphas, power_n=power_n,
                phase_fit=phase_fit, fit_result=fit_result,
                F0=F0, F0_recon=F0_recon, q_grid=q_grid,
                nu_max=nu_max, pixel_size=pixel_size, NA=NA, wavelength=wavelength,
                image=image)


def run_angular_radial_analysis(image, pixel_size, NA, wavelength,
                                n_angular=10, n_rbins=200,
                                n_radial=20, verbose=True):
    """Combined angular + radial analysis."""
    # FIX: was 2*NA/wavelength — inconsistent with all other functions
    nu_max, F_full, F, freq_x, freq_y = _fft_setup(image, pixel_size, NA, wavelength)
    q_grid, Fm, ms, meta = angular_decomposition_full(
        F, freq_x, freq_y, nu_max, n_modes=n_angular, n_rbins=n_rbins)
    indices = compute_aberration_indices(meta)

    m0_idx    = np.where(ms == 0)[0][0]
    F0        = Fm[m0_idx]
    c, Psi, alphas = bessel_coeffs_m0(F0, q_grid, nu_max, n_radial)
    power_n   = np.abs(c)**2
    fit_result = fit_defocus_spherical(np.abs(F0), q_grid, nu_max)
    phase_fit  = phase_polynomial_fit(F0, q_grid, nu_max)

    if verbose:
        _print_angular(indices, nu_max)
        print("\n  Radial mode analysis")
        print(f"  {'─'*44}")
        print(f"  Defocus  (p2·q²):  {phase_fit['p2']:+.4f} rad·µm²")
        print(f"  Spherical (p4·q⁴): {phase_fit['p4']:+.4f} rad·µm⁴")
        print(f"  OTF defocus coeff: {fit_result['est_defocus']:+.4f}")
        print(f"  OTF spherical:     {fit_result['est_spherical']:+.4f}")
        print(f"  OTF fit RMS:       {fit_result['residual_rms']:.4f}")

    return dict(q_grid=q_grid, Fm=Fm, ms=ms, meta=meta, indices=indices,
                c=c, power_n=power_n, phase_fit=phase_fit, fit_result=fit_result,
                F_full=F_full, freq_x=freq_x, freq_y=freq_y,
                nu_max=nu_max, pixel_size=pixel_size, NA=NA, wavelength=wavelength,
                image=image)


def plot_angular_analysis(results, save_path=None):
    """
    Angular mode analysis figure.

    Parameters
    ----------
    results   : dict returned by run_angular_analysis or run_angular_radial_analysis
    save_path : str or None — if given, save PDF to this path
    """
    image    = results['image'] if 'image' in results else None
    F_full   = results['F_full']
    freq_x   = results['freq_x']
    freq_y   = results['freq_y']
    nu_max   = results['nu_max']
    q_grid   = results['q_grid']
    Fm       = results['Fm']
    ms       = results['ms']
    meta     = results['meta']
    indices  = results['indices']
    pixel_size = results['pixel_size']
    NA       = results['NA']
    wavelength = results['wavelength']

    pal = {'0': '#1D9E75', '1': '#BA7517', '2': '#534AB7',
           '3': '#D85A30', '4': '#888780'}
    m_labels = {0: 'm=0 (isotropic)', 1: 'm=1 (coma/tilt)',
                2: 'm=2 (astigmatism)', 3: 'm=3 (trefoil)',
                4: 'm=4 (quadrafoil)'}

    fig = plt.figure(figsize=(16, 13))
    gs  = GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.32)
    theta = np.linspace(0, 2 * np.pi, 400)

    # (a) image
    ax = fig.add_subplot(gs[0, 0])
    if image is not None:
        ax.imshow(image, cmap='gray')
    ax.set_title('(a) Image', fontsize=9)
    ax.axis('off')

    # (b) |FFT|²
    ax = fig.add_subplot(gs[0, 1])
    psd_log = np.log1p(np.abs(F_full)**2)
    ax.imshow(psd_log,
              extent=[freq_x[0], freq_x[-1], freq_y[0], freq_y[-1]],
              cmap='inferno', origin='lower', aspect='equal',
              vmax=np.percentile(psd_log, 99.5))
    ax.plot(nu_max * np.cos(theta), nu_max * np.sin(theta),
            '--', color='cyan', lw=0.9, label=f'ν_max={nu_max:.2f}')
    ax.set_title('(b) |FFT|²  (log)', fontsize=9)
    ax.set_xlabel('ν_x  [cyc/µm]')
    ax.set_ylabel('ν_y  [cyc/µm]')
    ax.legend(fontsize=7)

    # (c) angular power P_m
    ax = fig.add_subplot(gs[0, 2])
    power_m = meta['power_m']
    total_P = power_m.sum() + 1e-30
    colors  = ['#1D9E75' if m == 0
               else '#534AB7' if abs(m) % 2 == 0
               else '#BA7517' for m in ms]
    ax.bar(ms, power_m / total_P * 100, color=colors, width=0.7, edgecolor='none')
    top3 = np.argsort(-power_m)[:3]
    for ii in top3:
        ax.text(ms[ii], power_m[ii] / total_P * 100 + 0.3,
                f'{power_m[ii] / total_P * 100:.1f}%',
                ha='center', fontsize=7, color='#444441')
    ax.set_xlabel('Angular mode  m')
    ax.set_ylabel('% of total spectral power')
    ax.set_title('(c) Angular power  P_m', fontsize=9)
    ax.legend(handles=[
        Patch(color='#1D9E75', label='m=0: isotropic'),
        Patch(color='#534AB7', label='|m| even: astigmatism family'),
        Patch(color='#BA7517', label='|m| odd: coma family'),
    ], fontsize=7, framealpha=0.3)

    # (d) aberration index bar chart
    ax = fig.add_subplot(gs[0, 3])
    idx_names  = ['Astigmatism\n(|m|=2)', 'Coma\n(|m|=1)',
                  'Trefoil\n(|m|=3)',     'Symmetry\nbreaking']
    idx_values = [indices['astigmatism_index'], indices['coma_index'],
                  indices['trefoil_index'],     indices['symmetry_ratio']]
    idx_colors = ['#534AB7', '#BA7517', '#1D9E75', '#888780']
    bars2 = ax.barh(idx_names, idx_values, color=idx_colors,
                    height=0.5, edgecolor='none')
    ax.axvline(0.05, color='#D85A30', lw=0.8, ls='--',
               label='threshold (0.05)')
    ax.set_xlabel('Index  (0 = perfect)')
    ax.set_title('(d) Aberration indices', fontsize=9)
    ax.legend(fontsize=7)
    for bar, v in zip(bars2, idx_values):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{v:.4f}', va='center', fontsize=8)

    # (e) radial profiles |F_m(q)|²
    ax = fig.add_subplot(gs[1, :2])
    for m_val in [0, 1, 2, 3, 4]:
        j = np.where(ms == m_val)[0]
        if not len(j):
            continue
        ax.semilogy(q_grid, np.abs(Fm[j[0]])**2 + 1e-12,
                    color=pal[str(m_val)], lw=1.4,
                    label=m_labels.get(m_val, f'm={m_val}'))
    ax.axvline(nu_max, color='#D85A30', lw=1.8, ls='--', zorder=5,
               label=f'ν_max = {nu_max:.3f} cyc/µm')
    ylo, yhi = ax.get_ylim()
    ax.text(nu_max * 1.02,
            10 ** (0.2 * (np.log10(yhi) + np.log10(ylo))),
            f'ν_max\n{nu_max:.3f}', fontsize=7.5, color='#D85A30', va='center')
    ax.set_xlabel('ν  [cyc/µm]')
    ax.set_ylabel('|F_m(ν)|²  (log)')
    ax.set_title('(e) Radial profiles per angular mode', fontsize=9)
    ax.legend(fontsize=8, framealpha=0.3)

    # (f) spectral coherence
    ax = fig.add_subplot(gs[1, 2:])
    for m_val in [1, 2, 3, 4]:
        j = np.where(ms == m_val)[0]
        if not len(j):
            continue
        ax.plot(q_grid, meta['coherence_m'][j[0]],
                color=pal[str(m_val)], lw=1.4,
                label=f'γ_{m_val} = |F_{m_val}|/|F_0|')
    ax.axvline(nu_max, color='#D85A30', lw=1.8, ls='--',
               label=f'ν_max = {nu_max:.3f}')
    ax.axhline(0.10, color='gray', lw=0.4, ls=':', label='10% coherence')
    ax.set_xlabel('ν  [cyc/µm]')
    ax.set_ylabel('γ_m(ν)')
    ax.set_title('(f) Spectral coherence  γ_m = |F_m|/|F_0|', fontsize=9)
    ax.legend(fontsize=8, framealpha=0.3)
    ax.set_ylim(0, None)

    # (g) angular phase
    ax = fig.add_subplot(gs[2, :2])
    for m_val in [1, 2, 3]:
        j = np.where(ms == m_val)[0]
        if not len(j):
            continue
        j   = j[0]
        mag = np.abs(Fm[j])
        ax.scatter(q_grid, np.degrees(meta['phase_m'][j]),
                   s=mag / (mag.max() + 1e-30) * 30,
                   color=pal[str(m_val)], alpha=0.7,
                   label=f'm={m_val}  (size ∝ |F_m|)')
    ax.axvline(nu_max, color='#D85A30', lw=1.8, ls='--',
               label=f'ν_max = {nu_max:.3f}')
    ax.axhline(0, color='gray', lw=0.4, ls=':')
    ax.set_xlabel('ν  [cyc/µm]')
    ax.set_ylabel('Phase of F_m  [°]')
    ax.set_title('(g) Angular phase vs frequency', fontsize=9)
    ax.set_ylim(-185, 185)
    ax.legend(fontsize=8)

    # (h) polar heatmap
    ax = fig.add_subplot(gs[2, 2], projection='polar')
    q_2d   = np.linspace(0, nu_max, 80)
    phi_2d = np.linspace(0, 2 * np.pi, 180)
    F_recon = np.zeros((180, 80), dtype=complex)
    for m_val in range(-4, 5):
        j = np.where(ms == m_val)[0]
        if not len(j):
            continue
        Fmi = np.interp(q_2d, q_grid, np.abs(Fm[j[0]]))
        F_recon += Fmi[None, :] * np.exp(1j * m_val * phi_2d[:, None])
    ax.pcolormesh(phi_2d, q_2d / nu_max, np.abs(F_recon).T,
                  cmap='viridis', shading='auto')
    ax.set_title('(h) |F(q,φ)|  polar  (|m|≤4)', fontsize=9)
    ax.set_yticklabels([])
    ax.set_xlabel('q / ν_max', labelpad=12)

    # (i) astigmatic component
    ax = fig.add_subplot(gs[2, 3])
    _, img_astig = angular_filter_fft(F_full, freq_x, freq_y, keep_modes=[2, -2])
    im = ax.imshow(img_astig, cmap='RdBu_r')
    plt.colorbar(im, ax=ax, fraction=0.046)
    title_i = ('(i) Astigmatic component (m=±2)'
               + (f'\naxis ≈ {indices["astigmatism_angle"]:.0f}°'
                  if indices['astigmatism_angle'] is not None else ''))
    ax.set_title(title_i, fontsize=9)
    ax.axis('off')

    fig.suptitle(
        f'Angular mode analysis  |  NA={NA}, λ={wavelength} µm, '
        f'px={pixel_size} µm  |  '
        f'Astig={indices["astigmatism_index"]:.3f}  '
        f'Coma={indices["coma_index"]:.3f}  '
        f'Trefoil={indices["trefoil_index"]:.3f}',
        fontsize=9, fontweight='500',
    )
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


def plot_radial_analysis(results, save_path=None):
    """
    Radial Fourier-Bessel diagnosis figure.

    Parameters
    ----------
    results   : dict returned by run_radial_mode_analysis
    save_path : str or None
    """
    image      = results['image']
    q_grid     = results['q_grid']
    nu_max     = results['nu_max']
    pixel_size = results['pixel_size']
    NA         = results['NA']
    wavelength = results['wavelength']
    F0         = results['F0']
    F0_recon   = results['F0_recon']
    c          = results['c']
    Psi        = results['Psi']
    alphas     = results['alphas']
    power_n    = results['power_n']
    fit_result = results['fit_result']
    phase_fit  = results['phase_fit']
    n_radial   = len(c)

    fig = plt.figure(figsize=(16, 12))
    gs  = GridSpec(3, 4, figure=fig, hspace=0.44, wspace=0.34)

    # (a) image
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(image, cmap='gray', vmax=np.percentile(image, 99.5))
    ax.set_title('(a) Image', fontsize=9)
    ax.axis('off')

    # (b) |F_0(q)| vs ideal OTF vs fit
    ax = fig.add_subplot(gs[0, 1])
    F0_norm = np.abs(F0) / (np.abs(F0).max() + 1e-30)
    ax.plot(q_grid, F0_norm,
            color='#534AB7', lw=1.5, label='|F_0(q)| measured')
    ax.plot(q_grid, fit_result['fit'],
            color='#D85A30', lw=1.2, ls='--', label='Amplitude fit')
    ax.plot(q_grid, fit_result['otf_id'],
            color='#1D9E75', lw=1.0, ls=':', label='Ideal OTF')
    ax.plot(q_grid, np.abs(F0_recon) / (np.abs(F0_recon).max() + 1e-30),
            color='#BA7517', lw=0.8, ls='-.', label=f'Bessel recon (n={n_radial})')
    ax.axvline(nu_max, color='#D85A30', lw=1.2, ls='--', alpha=0.5)
    ax.set_xlabel('ν  [cyc/µm]')
    ax.set_ylabel('|F_0(ν)|  normalised')
    ax.set_title('(b) Isotropic OTF  |F_0(q)|', fontsize=9)
    ax.legend(fontsize=7, framealpha=0.3)
    ax.set_ylim(0, None)

    # (c) Phase of F_0(q)
    ax  = fig.add_subplot(gs[0, 2])
    q_p = phase_fit['q_in']
    ax.scatter(q_p, phase_fit['phi_raw'], s=6, color='#534AB7', alpha=0.6,
               label='arg(F_0)  unwrapped')
    ax.plot(q_p, phase_fit['phi_fit'], color='#D85A30', lw=1.5,
            label=f'p₂q²+p₄q⁴\np₂={phase_fit["p2"]:+.3f}\np₄={phase_fit["p4"]:+.3f}')
    ax.plot(q_p, phase_fit['p2'] * q_p**2,
            color='#1D9E75', lw=1.0, ls='--', label='p₂q²  (defocus)')
    ax.plot(q_p, phase_fit['p4'] * q_p**4,
            color='#BA7517', lw=1.0, ls=':', label='p₄q⁴  (spherical)')
    ax.axhline(0, color='gray', lw=0.4, ls=':')
    ax.axvline(nu_max, color='#D85A30', lw=1.0, ls='--', alpha=0.5)
    ax.set_xlabel('ν  [cyc/µm]')
    ax.set_ylabel('phase  [rad]')
    ax.set_title('(c) Phase of F_0: defocus vs spherical', fontsize=9)
    ax.legend(fontsize=7, framealpha=0.3)

    # (d) Bessel basis functions
    ax    = fig.add_subplot(gs[0, 3])
    cmap_n = plt.cm.plasma(np.linspace(0.1, 0.9, min(8, n_radial)))
    for ni in range(min(8, n_radial)):
        ax.plot(q_grid, Psi[ni], color=cmap_n[ni], lw=1.2,
                label=f'n={ni+1}  α={alphas[ni]:.2f}')
    ax.axhline(0, color='gray', lw=0.4, ls=':')
    ax.axvline(nu_max, color='#D85A30', lw=1.0, ls='--', alpha=0.5)
    ax.set_xlabel('ν  [cyc/µm]')
    ax.set_ylabel('ψ_{0n}(ν)')
    ax.set_title('(d) Fourier-Bessel basis ψ_{0n}(q)\n'
                 'n-1 zero crossings inside [0, ν_max]', fontsize=9)
    ax.legend(fontsize=6, framealpha=0.3, ncol=2)

    # (e) Bessel power spectrum
    ax = fig.add_subplot(gs[1, :2])
    ns  = np.arange(1, n_radial + 1)
    ax.bar(ns, power_n / power_n.sum() * 100,
           color='#534AB7', width=0.7, edgecolor='none', label='measured')
    # fingerprint curves for defocus and spherical
    c_def = np.array([np.trapezoid(fit_result['d_def'] * Psi[ni] * q_grid, q_grid)
                      for ni in range(n_radial)])
    c_sph = np.array([np.trapezoid(fit_result['d_sph'] * Psi[ni] * q_grid, q_grid)
                      for ni in range(n_radial)])
    p_def = np.abs(c_def)**2;  p_def /= p_def.sum() + 1e-30
    p_sph = np.abs(c_sph)**2;  p_sph /= p_sph.sum() + 1e-30
    ax.plot(ns, p_def * 100, 'o--', color='#1D9E75', ms=5, lw=1.0,
            label='defocus fingerprint')
    ax.plot(ns, p_sph * 100, 's--', color='#BA7517', ms=5, lw=1.0,
            label='spherical fingerprint')
    ax.set_xlabel('Radial Bessel index  n')
    ax.set_ylabel('% of m=0 Bessel power')
    ax.set_title('(e) Bessel power spectrum  |c_{0n}|²\n'
                 'Low n = defocus-like ·  High n = spherical-like', fontsize=9)
    ax.set_xticks(ns)
    ax.legend(fontsize=7.5, framealpha=0.3)

    # (f) OTF fingerprints
    ax = fig.add_subplot(gs[1, 2:])
    ax.plot(q_grid, fit_result['otf_id'],
            color='#1D9E75', lw=1.4, label='Ideal OTF')
    ax.plot(q_grid, fit_result['otf_id'] + fit_result['d_def'],
            color='#534AB7', lw=1.2, ls='--', label='Ideal + unit defocus')
    ax.plot(q_grid, fit_result['otf_id'] + fit_result['d_sph'],
            color='#BA7517', lw=1.2, ls=':', label='Ideal + unit spherical')
    ax.plot(q_grid, fit_result['F0_norm'],
            color='#D85A30', lw=1.5, label='Measured |F_0|')
    ax.axvline(nu_max, color='gray', lw=0.8, ls='--', alpha=0.6)
    ax.set_xlabel('ν  [cyc/µm]')
    ax.set_ylabel('|OTF|  normalised')
    ax.set_title('(f) OTF fingerprints', fontsize=9)
    ax.legend(fontsize=7.5, framealpha=0.3)
    ax.set_ylim(0, None)

    # (g) fit residual
    ax = fig.add_subplot(gs[2, :2])
    resid = fit_result['F0_norm'] - fit_result['fit']
    ax.plot(q_grid, resid, color='#534AB7', lw=1.3,
            label=f'residual  (RMS={fit_result["residual_rms"]:.4f})')
    ax.fill_between(q_grid, resid, alpha=0.2, color='#534AB7')
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.axvline(nu_max, color='#D85A30', lw=1.0, ls='--', alpha=0.6)
    ax.set_xlabel('ν  [cyc/µm]')
    ax.set_ylabel('|F_0| − fit')
    ax.set_title('(g) OTF fit residual', fontsize=9)
    ax.legend(fontsize=8)

    # (h) diagnosis summary table
    ax = fig.add_subplot(gs[2, 2:])
    ax.axis('off')
    p2 = phase_fit['p2']
    p4 = phase_fit['p4']
    e_d = fit_result['est_defocus']
    e_s = fit_result['est_spherical']
    dominant_n  = ns[np.argmax(power_n)]
    def_dominant = abs(p2) > abs(p4)
    hi_col = '#D85A30' if def_dominant else '#534AB7'
    lines = [
        ('Method',              'Defocus',                     'Spherical'),
        ('Phase fit',           f'p₂ = {p2:+.4f}',            f'p₄ = {p4:+.4f}'),
        ('OTF amplitude fit',   f'{e_d:+.4f}',                 f'{e_s:+.4f}'),
        ('Dominant Bessel n',
         f'n={dominant_n} (low → defocus)' if dominant_n <= 3 else f'n={dominant_n}',
         f'n={dominant_n} (high → spherical)' if dominant_n > 3 else f'n={dominant_n}'),
        ('Conclusion',
         'defocus dominant' if def_dominant else 'spherical dominant',
         f'|p₄/p₂| = {abs(p4)/(abs(p2)+1e-6):.2f}'),
    ]
    col_x = [0.02, 0.32, 0.66]
    row_y = np.linspace(0.88, 0.05, len(lines))
    ax.add_patch(plt.Rectangle((0, 0.88), 1, 0.10,
                                transform=ax.transAxes,
                                color='#E1F5EE', zorder=0))
    for ri, row in enumerate(lines):
        for ci, cell in enumerate(row):
            color  = (hi_col if ri > 0 and ci in (1, 2) else 'black')
            ax.text(col_x[ci], row_y[ri], cell,
                    transform=ax.transAxes, fontsize=8.5, va='top',
                    fontweight='bold' if ri == 0 else 'normal',
                    color=color)
    ax.set_title('(h) Defocus vs spherical diagnosis', fontsize=9)

    fig.suptitle(
        f'Radial Bessel mode diagnosis  |  '
        f'NA={NA}, λ={wavelength} µm, px={pixel_size} µm  |  '
        f'ν_max={nu_max:.3f} cyc/µm  |  n_radial={n_radial}',
        fontsize=9, fontweight='500',
    )
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    import tifffile as tf

    pixel_size = 0.054   # µm
    NA         = 1.3
    wavelength = 0.505   # µm

    filename = r"C:\Users\Ruiz\Desktop\ao_test\20260608_153723_delta710_ao_iterations_HighPass(FFT)\zernike mode #6.tiff"
    images = []
    labels = []
    with tf.TiffFile(filename) as tif:
        for page in tif.pages:
            img = page.asarray()
            label = page.description  # this is the string written with description=label
            images.append(img)
            labels.append(label)

    results = [run_angular_analysis(images[0], pixel_size, NA, wavelength, n_modes=8, n_rbins=160, verbose=True),
               run_angular_analysis(images[1], pixel_size, NA, wavelength, n_modes=8, n_rbins=160, verbose=True),
               run_angular_analysis(images[2], pixel_size, NA, wavelength, n_modes=8, n_rbins=160, verbose=True)]
    
    m = 2
    trucs = [40, 120]
    
    plt.figure()
    plt.plot(np.log(results[0]["meta"]["power_density"][2][trucs[0]:trucs[1]]), label=labels[0])
    print(np.sum(np.log(results[0]["meta"]["power_density"][2][trucs[0]:trucs[1]])))
    plt.plot(np.log(results[1]["meta"]["power_density"][2][trucs[0]:trucs[1]]), label=labels[1])
    print(np.sum(np.log(results[1]["meta"]["power_density"][2][trucs[0]:trucs[1]])))
    plt.plot(np.log(results[2]["meta"]["power_density"][2][trucs[0]:trucs[1]]), label=labels[2])
    print(np.sum(np.log(results[2]["meta"]["power_density"][2][trucs[0]:trucs[1]])))
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(np.log(results[0]["meta"]["power_density"][1][trucs[0]:trucs[1]]), label=labels[0])
    print(np.sum(np.log(results[0]["meta"]["power_density"][1][trucs[0]:trucs[1]])))
    plt.plot(np.log(results[1]["meta"]["power_density"][1][trucs[0]:trucs[1]]), label=labels[1])
    print(np.sum(np.log(results[1]["meta"]["power_density"][1][trucs[0]:trucs[1]])))
    plt.plot(np.log(results[2]["meta"]["power_density"][1][trucs[0]:trucs[1]]), label=labels[2])
    print(np.sum(np.log(results[2]["meta"]["power_density"][1][trucs[0]:trucs[1]])))
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(np.log(results[0]["meta"]["power_density"][0][trucs[0]:trucs[1]]), label=labels[0])
    print(np.sum(np.log(results[0]["meta"]["power_density"][0][trucs[0]:trucs[1]])))
    plt.plot(np.log(results[1]["meta"]["power_density"][0][trucs[0]:trucs[1]]), label=labels[1])
    print(np.sum(np.log(results[1]["meta"]["power_density"][0][trucs[0]:trucs[1]])))
    plt.plot(np.log(results[2]["meta"]["power_density"][0][trucs[0]:trucs[1]]), label=labels[2])
    print(np.sum(np.log(results[2]["meta"]["power_density"][0][trucs[0]:trucs[1]])))
    plt.legend()
    plt.show()
