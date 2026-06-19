import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


# ══════════════════════════════════════════════════════════════════
# 1. Core angular decomposition
# ══════════════════════════════════════════════════════════════════

def angular_decomposition_full(F_centred, freq_x, freq_y,
                               nu_max, n_modes=8, n_rbins=120):
    """
    Decompose the centred FFT F(q, φ) into angular Fourier modes.

    F_m(q) = (1/2π) ∫₀²π F(q,φ) e^{-imφ} dφ

    Computation is strictly limited to q ∈ [0, nu_max] — inside the pupil.

    Parameters
    ----------
    F_centred : (H, W) complex   — FFT with DC at centre (fftshift applied)
    freq_x    : (W,) float       — frequency axis [cycles/length]
    freq_y    : (H,) float
    nu_max    : float            — pupil cutoff radius (same units as freq axes)
    n_modes   : int              — compute m = -n_modes … +n_modes
    n_rbins   : int              — number of radial annuli inside pupil [0, nu_max]

    Returns
    -------
    q_grid  : (n_rbins,)                  — radial bin centres, all ≤ nu_max
    Fm      : (2*n_modes+1, n_rbins) complex
    ms      : (2*n_modes+1,)
    meta    : dict
    """
    fx, fy = np.meshgrid(freq_x, freq_y)
    fq = np.sqrt(fx ** 2 + fy ** 2)
    phi = np.arctan2(fy, fx)  # [-π, π]

    ms = np.arange(-n_modes, n_modes + 1)
    # radial grid strictly inside the pupil — no extrapolation beyond nu_max
    q_edges = np.linspace(0, nu_max, n_rbins + 1)
    q_grid = 0.5 * (q_edges[:-1] + q_edges[1:])

    Fm = np.zeros((len(ms), n_rbins), dtype=complex)
    n_pixels = np.zeros(n_rbins, dtype=int)

    for ri in range(n_rbins):
        ring = (fq >= q_edges[ri]) & (fq < q_edges[ri + 1])
        n_pixels[ri] = ring.sum()
        if n_pixels[ri] < 2:
            continue
        phi_r = phi[ring]
        F_r = F_centred[ring]
        for j, m in enumerate(ms):
            Fm[j, ri] = np.mean(F_r * np.exp(-1j * m * phi_r))

    power_density = np.abs(Fm) ** 2 * q_grid[None, :]
    power_m = 2 * np.pi * np.trapezoid(power_density, q_grid, axis=1)
    phase_m = np.angle(Fm)
    m0_idx = np.where(ms == 0)[0][0]
    F0_mag = np.abs(Fm[m0_idx]) + 1e-30
    coherence_m = np.abs(Fm) / F0_mag[None, :]
    q_peak_m = q_grid[np.argmax(np.abs(Fm) ** 2, axis=1)]

    meta = dict(
        power_m=power_m, power_density=power_density,
        phase_m=phase_m, coherence_m=coherence_m,
        q_peak_m=q_peak_m, n_pixels=n_pixels,
        ms=ms, q_grid=q_grid,
    )
    return q_grid, Fm, ms, meta


# ══════════════════════════════════════════════════════════════════
# 2. Aberration indices
# ══════════════════════════════════════════════════════════════════

def compute_aberration_indices(meta, nu_max):
    ms = meta['ms']
    power_m = meta['power_m']
    q_grid = meta['q_grid']

    def P(m_val):
        j = np.where(ms == m_val)[0]
        return float(power_m[j[0]]) if len(j) else 0.0

    P0 = P(0) + 1e-30
    Pt = power_m.sum() + 1e-30

    astig = (P(2) + P(-2)) / P0
    coma = (P(1) + P(-1)) / P0
    trefoil = (P(3) + P(-3)) / P0
    sym_ratio = 1.0 - P0 / Pt

    j2 = np.where(ms == 2)[0]
    astig_angle = None
    if len(j2) and meta['power_density'][j2[0]].max() > 0:
        pk = np.argmax(meta['power_density'][j2[0]])
        astig_angle = np.degrees(meta['phase_m'][j2[0], pk] / 2) % 180

    j0_idx = np.where(ms == 0)[0][0]
    pd0 = meta['power_density'][j0_idx]
    q_eff = float(np.average(q_grid, weights=pd0)) if pd0.sum() > 0 else 0.0

    return dict(
        astigmatism_index=astig, coma_index=coma,
        trefoil_index=trefoil, symmetry_ratio=sym_ratio,
        q_eff=q_eff, astigmatism_angle=astig_angle,
    )


# ══════════════════════════════════════════════════════════════════
# 3. Angular-mode image filter
# ══════════════════════════════════════════════════════════════════

def angular_filter_fft(F_centred, freq_x, freq_y, keep_modes):
    fx, fy = np.meshgrid(freq_x, freq_y)
    phi = np.arctan2(fy, fx)
    mask = np.zeros_like(F_centred, dtype=complex)
    if not keep_modes:
        return mask, np.zeros(F_centred.shape)
    for m in keep_modes:
        mask += np.exp(1j * m * phi)
    mask /= (2 * np.pi)
    Ff = F_centred * mask
    return Ff, np.real(np.fft.ifft2(np.fft.ifftshift(Ff)))


# ══════════════════════════════════════════════════════════════════
# 4. Main analysis and plot
# ══════════════════════════════════════════════════════════════════

def plot_angular_analysis(image, pixel_size, NA, wavelength):
    """
    Angular mode analysis.

    Frequency-domain panels show the full q_grid range [0, nu_max] — the
    computation in angular_decomposition_full is already restricted to the
    pupil, so no data exists beyond nu_max.
    """
    nu_max = NA / wavelength
    H, W = image.shape

    F_full = np.fft.fftshift(np.fft.fft2(image.astype(np.float64)))
    freq_x = np.fft.fftshift(np.fft.fftfreq(W, d=pixel_size))
    freq_y = np.fft.fftshift(np.fft.fftfreq(H, d=pixel_size))
    fx, fy = np.meshgrid(freq_x, freq_y)
    fq = np.sqrt(fx ** 2 + fy ** 2)

    # mask FFT to pupil before decomposition
    F = F_full * (fq <= nu_max)

    q_grid, Fm, ms, meta = angular_decomposition_full(
        F, freq_x, freq_y, nu_max, n_modes=8, n_rbins=128)

    indices = compute_aberration_indices(meta, nu_max)

    print("\n  Angular mode analysis results")
    print(f"  {'─' * 44}")
    print(f"  Astigmatism index  (|m|=2): {indices['astigmatism_index']:.4f}")
    print(f"  Coma index         (|m|=1): {indices['coma_index']:.4f}")
    print(f"  Trefoil index      (|m|=3): {indices['trefoil_index']:.4f}")
    print(f"  Global symmetry breaking:   {indices['symmetry_ratio']:.4f}")
    print(f"  Effective bandwidth:        {indices['q_eff']:.4f}  cyc/µm  "
          f"({indices['q_eff'] / nu_max * 100:.1f}% of pupil)")
    if indices['astigmatism_angle'] is not None:
        print(f"  Astigmatism orientation:    {indices['astigmatism_angle']:.1f}°")

    fig = plt.figure(figsize=(16, 13))
    gs = GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.32)
    theta = np.linspace(0, 2 * np.pi, 400)

    pal = {'0': '#1D9E75', '1': '#BA7517', '2': '#534AB7',
           '3': '#D85A30', '4': '#888780'}
    m_labels = {0: 'm=0 (isotropic)', 1: 'm=1 (coma/tilt)',
                2: 'm=2 (astigmatism)', 3: 'm=3 (trefoil)',
                4: 'm=4 (quadrafoil)'}

    # ── (a) image ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(image, cmap='gray')
    ax.set_title('(a) Image', fontsize=9)
    ax.axis('off')

    # ── (b) |FFT|²  — full frequency range ───────────────────
    ax = fig.add_subplot(gs[0, 1])
    psd_log = np.log1p(np.abs(F_full) ** 2)
    ax.imshow(psd_log,
              extent=[freq_x[0], freq_x[-1], freq_y[0], freq_y[-1]],
              cmap='inferno', origin='lower', aspect='equal',
              vmax=np.percentile(psd_log, 99.5))
    ax.plot(nu_max * np.cos(theta), nu_max * np.sin(theta),
            '--', color='cyan', lw=0.9, label=f'Pupil  ν_max={nu_max:.2f}')
    ax.set_title('(b) |FFT|²  (log,  full range)', fontsize=9)
    ax.set_xlabel('ν_x  [cyc/µm]')
    ax.set_ylabel('ν_y  [cyc/µm]')
    ax.legend(fontsize=7)

    # ── (c) angular power P_m ─────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    power_m = meta['power_m']
    total_P = power_m.sum() + 1e-30
    colors = ['#1D9E75' if m == 0
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
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='#1D9E75', label='m=0: isotropic'),
        Patch(color='#534AB7', label='|m| even: astigmatism family'),
        Patch(color='#BA7517', label='|m| odd: coma family'),
    ], fontsize=7, framealpha=0.3)

    # ── (d) aberration index bar chart ───────────────────────
    ax = fig.add_subplot(gs[0, 3])
    idx_names = ['Astigmatism\n(|m|=2)', 'Coma\n(|m|=1)',
                 'Trefoil\n(|m|=3)', 'Symmetry\nbreaking']
    idx_values = [indices['astigmatism_index'], indices['coma_index'],
                  indices['trefoil_index'], indices['symmetry_ratio']]
    idx_colors = ['#534AB7', '#BA7517', '#1D9E75', '#888780']
    bars2 = ax.barh(idx_names, idx_values, color=idx_colors,
                    height=0.5, edgecolor='none')
    ax.axvline(0.05, color='#D85A30', lw=0.8, ls='--',
               label='detection threshold (0.05)')
    ax.set_xlabel('Index value  (0 = perfect, 1 = severe)')
    ax.set_title('(d) Aberration indices', fontsize=9)
    ax.legend(fontsize=7)
    for bar, v in zip(bars2, idx_values):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{v:.4f}', va='center', fontsize=8)

    # ── (e) radial profiles — full q_grid range [0, nu_max] ──
    ax = fig.add_subplot(gs[1, :2])
    for m_val in [0, 1, 2, 3, 4]:
        j = np.where(ms == m_val)[0]
        if not len(j):
            continue
        ax.semilogy(q_grid, np.abs(Fm[j[0]]) ** 2 + 1e-12,
                    color=pal[str(m_val)], lw=1.4,
                    label=m_labels.get(m_val, f'm={m_val}'))

    ax.axvline(nu_max, color='#D85A30', lw=1.8, ls='--', zorder=5,
               label=f'Pupil edge  ν_max = {nu_max:.3f} cyc/µm')
    ylo, yhi = ax.get_ylim()
    ax.text(nu_max * 1.02,
            10 ** (0.2 * (np.log10(yhi) + np.log10(ylo))),
            f'ν_max\n{nu_max:.3f}', fontsize=7.5, color='#D85A30', va='center')
    ax.set_xlabel('Radial frequency  ν  [cyc/µm]')
    ax.set_ylabel('|F_m(ν)|²  (log scale)')
    ax.set_title('(e) Radial profiles per angular mode', fontsize=9)
    ax.legend(fontsize=8, framealpha=0.3)

    # ── (f) spectral coherence — full q_grid range [0, nu_max] ─
    ax = fig.add_subplot(gs[1, 2:])
    for m_val in [1, 2, 3, 4]:
        j = np.where(ms == m_val)[0]
        if not len(j):
            continue
        ax.plot(q_grid, meta['coherence_m'][j[0]],
                color=pal[str(m_val)], lw=1.4,
                label=f'γ_{m_val} = |F_{m_val}| / |F_0|')
    ax.axvline(nu_max, color='#D85A30', lw=1.8, ls='--',
               label=f'ν_max = {nu_max:.3f}')
    ax.axhline(0.10, color='gray', lw=0.4, ls=':', label='10% coherence')
    ax.set_xlabel('Radial frequency  ν  [cyc/µm]')
    ax.set_ylabel('γ_m(ν)')
    ax.set_title('(f) Spectral coherence  γ_m(ν) = |F_m| / |F_0|', fontsize=9)
    ax.legend(fontsize=8, framealpha=0.3)
    ax.set_ylim(0, None)

    # ── (g) angular phase — full q_grid range [0, nu_max] ────
    ax = fig.add_subplot(gs[2, :2])
    for m_val in [1, 2, 3]:
        j = np.where(ms == m_val)[0]
        if not len(j):
            continue
        j = j[0]
        mag = np.abs(Fm[j])
        ax.scatter(q_grid, np.degrees(meta['phase_m'][j]),
                   s=mag / (mag.max() + 1e-30) * 30,
                   color=pal[str(m_val)], alpha=0.7,
                   label=f'm={m_val}  (size ∝ |F_m|)')
    ax.axvline(nu_max, color='#D85A30', lw=1.8, ls='--',
               label=f'ν_max = {nu_max:.3f}')
    ax.axhline(0, color='gray', lw=0.4, ls=':')
    ax.set_xlabel('Radial frequency  ν  [cyc/µm]')
    ax.set_ylabel('Phase of F_m(ν)  [degrees]')
    ax.set_title('(g) Angular phase vs frequency', fontsize=9)
    ax.set_ylim(-185, 185)
    ax.legend(fontsize=8)

    # ── (h) polar heatmap — radial range = [0, nu_max] ────────
    ax = fig.add_subplot(gs[2, 2], projection='polar')
    q_2d = np.linspace(0, nu_max, 80)
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
    ax.set_title('(h) |F(q,φ)|  polar\n(modes |m|≤4,  q ∈ [0, ν_max])', fontsize=9)
    ax.set_yticklabels([])
    ax.set_xlabel('q / ν_max', labelpad=12)

    # ── (i) astigmatic component back-projected ───────────────
    ax = fig.add_subplot(gs[2, 3])
    _, img_astig = angular_filter_fft(F_full, freq_x, freq_y,
                                      keep_modes=[2, -2])
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
    plt.savefig('angular_mode_analysis.pdf', dpi=150, bbox_inches='tight')
    plt.show()
    return q_grid, Fm, ms, meta, indices


# ══════════════════════════════════════════════════════════════════
# 5. Entry point
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import tifffile as tf

    pixel_size = 0.054  # µm per pixel
    NA = 1.3
    wavelength = 0.505  # µm

    df = r"C:\Users\Ruiz\Desktop\ao_test\20241115_145719__auto_ao_iterations_SNR(FFT)\zernike mode #4.tiff"
    data_stack = []
    with tf.TiffFile(df) as tif:
        for page in tif.pages:
            data_stack.append((page.asarray(), page.description))

    plot_angular_analysis(data_stack[0][0][268:268+360,348:348+360], pixel_size, NA, wavelength)

    plot_angular_analysis(data_stack[1][0][268:268+360,348:348+360], pixel_size, NA, wavelength)

    plot_angular_analysis(data_stack[2][0][268:268+360,348:348+360], pixel_size, NA, wavelength)