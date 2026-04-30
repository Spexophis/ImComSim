import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Angular spectrum propagation
# ============================================================

def angular_spectrum(E, dx, wavelength0, n_medium, z):
    """
    Scalar angular-spectrum propagation.

    E: complex field, shape (N, N)
    dx: pixel size [m]
    wavelength0: vacuum wavelength [m]
    n_medium: background refractive index
    z: propagation distance [m]
    """
    N = E.shape[0]

    k0 = 2 * np.pi / wavelength0
    k = n_medium * k0

    fx = np.fft.fftfreq(N, d=dx)
    fy = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(fx, fy)

    kx = 2 * np.pi * FX
    ky = 2 * np.pi * FY

    # Complex sqrt allows evanescent components to decay
    kz = np.sqrt((k + 0j) ** 2 - kx ** 2 - ky ** 2)

    H = np.exp(1j * kz * z)

    E_f = np.fft.fft2(E)
    E_out = np.fft.ifft2(E_f * H)

    return E_out


# ============================================================
# Tissue generation
# ============================================================

def correlated_random_2d(N, dx, corr_len, rng):
    """
    Generate one laterally correlated random 2D field.
    """
    noise = rng.normal(size=(N, N))

    fx = np.fft.fftfreq(N, d=dx)
    fy = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(fx, fy)

    F2 = FX ** 2 + FY ** 2

    # Gaussian low-pass filter.
    # Larger corr_len gives smoother tissue.
    filt = np.exp(-0.5 * F2 * (2 * np.pi * corr_len) ** 2)

    field = np.fft.ifft2(np.fft.fft2(noise) * filt).real

    field -= field.mean()
    field /= field.std() + 1e-12

    return field


def generate_tissue_delta_n_stack(
        N,
        dx,
        dz,
        n_slices,
        dn_rms=0.01,
        corr_len_xy=3e-6,
        corr_len_z=6e-6,
        seed=1,
):
    """
    Generate a 3D tissue refractive-index fluctuation stack.

    Returns:
        delta_n_stack with shape (n_slices, N, N)

    Tissue refractive index is approximately:

        n(x, y, z) = n_background + delta_n(x, y, z)
    """
    rng = np.random.default_rng(seed)

    stack = np.zeros((n_slices, N, N), dtype=np.float32)

    # Axial correlation using AR(1)-like process
    alpha = np.exp(-dz / corr_len_z)

    previous = correlated_random_2d(N, dx, corr_len_xy, rng)

    for i in range(n_slices):
        fresh = correlated_random_2d(N, dx, corr_len_xy, rng)

        if i == 0:
            current = fresh
        else:
            current = alpha * previous + np.sqrt(1 - alpha ** 2) * fresh

        stack[i] = current.astype(np.float32)
        previous = current

    # Normalize entire 3D stack
    stack -= stack.mean()
    stack /= stack.std() + 1e-12

    # Scale to desired RMS refractive index fluctuation
    stack *= dn_rms

    return stack


def estimate_total_phase_rms(wavelength0, tissue_thickness, corr_len_z, dn_rms):
    """
    Approximate accumulated RMS phase distortion for exponentially
    correlated refractive-index fluctuations along z.
    """
    k0 = 2 * np.pi / wavelength0
    L = tissue_thickness
    lz = corr_len_z

    var_phi = 2 * k0 ** 2 * dn_rms ** 2 * (
            L * lz - lz ** 2 * (1 - np.exp(-L / lz))
    )

    return np.sqrt(var_phi)


def choose_dn_rms_from_phase_rms(
        wavelength0,
        tissue_thickness,
        corr_len_z,
        target_phase_rms,
):
    """
    Choose dn_rms to produce a desired approximate total RMS phase distortion.
    """
    k0 = 2 * np.pi / wavelength0
    L = tissue_thickness
    lz = corr_len_z

    factor = 2 * k0 ** 2 * (
            L * lz - lz ** 2 * (1 - np.exp(-L / lz))
    )

    dn_rms = target_phase_rms / np.sqrt(factor)

    return dn_rms


# ============================================================
# Input light fields
# ============================================================

def make_two_beam_field(
        X,
        Y,
        wavelength0,
        n_medium,
        fringe_period=4.0e-6,
        waist=35e-6,
):
    """
    Create two coherent tilted beams that interfere to form a sinusoidal pattern.

    fringe_period is the desired period on the focal/input plane.
    """
    k0 = 2 * np.pi / wavelength0
    k = n_medium * k0

    sin_theta = wavelength0 / (2 * n_medium * fringe_period)

    if sin_theta > 1:
        raise ValueError(
            "Requested fringe period is too small for this wavelength and medium."
        )

    theta = np.arcsin(sin_theta)

    E1 = np.exp(1j * k * np.sin(theta) * X)
    E2 = np.exp(-1j * k * np.sin(theta) * X)

    envelope = np.exp(-(X ** 2 + Y ** 2) / waist ** 2)

    E = envelope * (E1 + E2)

    # Normalize
    E /= np.sqrt(np.mean(np.abs(E) ** 2) + 1e-12)

    return E, theta


def make_high_na_focusing_field(
        X,
        Y,
        wavelength0,
        n_medium,
        NA=0.8,
        focal_length=60e-6,
        phase_type="spherical",
):
    """
    Create a converging wave from a high-NA pupil.

    phase_type:
        "parabolic"  - paraxial lens phase
        "spherical"  - better scalar approximation for higher NA
    """
    if NA > n_medium:
        raise ValueError("NA cannot be larger than the refractive index of the medium.")

    k0 = 2 * np.pi / wavelength0
    k = n_medium * k0

    alpha = np.arcsin(NA / n_medium)
    pupil_radius = focal_length * np.tan(alpha)

    R = np.sqrt(X ** 2 + Y ** 2)
    pupil = (R <= pupil_radius).astype(float)

    if phase_type == "parabolic":
        phase = -k * (X ** 2 + Y ** 2) / (2 * focal_length)

    elif phase_type == "spherical":
        phase = -k * (np.sqrt(X ** 2 + Y ** 2 + focal_length ** 2) - focal_length)

    else:
        raise ValueError("phase_type must be 'parabolic' or 'spherical'.")

    E = pupil * np.exp(1j * phase)

    # Normalize only over nonzero pupil area
    mask = pupil > 0
    E /= np.sqrt(np.mean(np.abs(E[mask]) ** 2) + 1e-12)

    return E, pupil_radius


def make_focal_array_field_exact_spherical(
        X,
        Y,
        wavelength0,
        n_medium,
        NA=0.8,
        focal_length=60e-6,
        n_foci=7,
        spacing=3.0e-6,
        weights=None,
        global_phases=None,
):
    """
    High-NA safer focal-array field.

    Each focus is generated by an exact spherical converging wave
    aimed at its own target point:

        target_m = (x_m, 0, focal_length)

    This avoids the coma caused by using:
        axial spherical lens phase + linear tilt phase

    Parameters
    ----------
    X, Y : 2D arrays
        Coordinate grids at the initial plane.
    wavelength0 : float
        Vacuum wavelength [m].
    n_medium : float
        Refractive index of medium.
    NA : float
        Numerical aperture.
    focal_length : float
        Propagation distance to the focal plane [m].
    n_foci : int
        Number of focal spots.
    spacing : float
        Spacing between neighboring foci in the focal plane [m].
    weights : None or array
        Optional amplitude weight for each focus.
    global_phases : None or array
        Optional global phase offset for each focus.

    Returns
    -------
    E : complex 2D array
        Initial field.
    x_positions : 1D array
        Target x positions of foci [m].
    pupil_radius : float
        Radius of the circular pupil [m].
    chief_ray_angles : 1D array
        Approximate center-ray tilt angles [rad].
    """
    if NA > n_medium:
        raise ValueError("NA cannot be larger than n_medium.")

    k0 = 2 * np.pi / wavelength0
    k = n_medium * k0

    alpha = np.arcsin(NA / n_medium)
    pupil_radius = focal_length * np.tan(alpha)

    R = np.sqrt(X ** 2 + Y ** 2)
    pupil = R <= pupil_radius

    # Check whether the pupil fits inside the simulation grid
    half_grid = np.max(np.abs(X))
    if pupil_radius > 0.9 * half_grid:
        print(
            "Warning: pupil is close to or larger than the simulation window. "
            "This can cause clipping artifacts that look like aberration. "
            "Increase N, increase dx, reduce NA, or reduce focal_length."
        )

    # Evenly spaced target positions along x
    x_positions = (np.arange(n_foci) - (n_foci - 1) / 2) * spacing

    if weights is None:
        weights = np.ones(n_foci)

    if global_phases is None:
        global_phases = np.zeros(n_foci)

    weights = np.asarray(weights, dtype=float)
    global_phases = np.asarray(global_phases, dtype=float)

    E = np.zeros_like(X, dtype=np.complex128)

    for m, xm in enumerate(x_positions):
        # Exact distance from every pupil point to the target focus
        path = np.sqrt((X - xm) ** 2 + Y ** 2 + focal_length ** 2)

        # Remove a constant phase for numerical cleanliness.
        # This does not change the physics.
        phase = -k * (path - focal_length)

        component = np.zeros_like(E)
        component[pupil] = np.exp(1j * (phase[pupil] + global_phases[m]))

        # Normalize each component so each focus receives similar input power
        component /= np.sqrt(np.mean(np.abs(component[pupil]) ** 2) + 1e-12)

        E += weights[m] * component

    # Normalize total field
    E /= np.sqrt(np.mean(np.abs(E[pupil]) ** 2) + 1e-12)

    chief_ray_angles = np.arctan2(x_positions, focal_length)

    return E, x_positions, pupil_radius, chief_ray_angles


# ============================================================
# Propagation through tissue
# ============================================================

def propagate_through_tissue_and_record(
        E0,
        delta_n_stack,
        dx,
        dz,
        wavelength0,
        n_background,
        absorption_coeff=0.0,
        record_every=1,
):
    """
    Propagate a complex light field through tissue.

    Each tissue slice applies:

        phase shift = k0 * delta_n(x,y,z) * dz

    Then the field is propagated through dz of background medium.
    """
    k0 = 2 * np.pi / wavelength0

    E = E0.copy().astype(np.complex128)

    intensity_stack = [np.abs(E) ** 2]
    z_positions = [0.0]

    for i in range(delta_n_stack.shape[0]):

        delta_n = delta_n_stack[i]

        # Tissue-induced phase delay
        phase = k0 * delta_n * dz
        E *= np.exp(1j * phase)

        # Optional absorption
        E *= np.exp(-0.5 * absorption_coeff * dz)

        # Free propagation through this slice
        E = angular_spectrum(
            E,
            dx=dx,
            wavelength0=wavelength0,
            n_medium=n_background,
            z=dz,
        )

        if (i + 1) % record_every == 0:
            intensity_stack.append(np.abs(E) ** 2)
            z_positions.append((i + 1) * dz)

    return E, np.array(intensity_stack), np.array(z_positions)


# ============================================================
# Visualization helpers
# ============================================================

def show_complex_field(E, title, extent_um):
    """
    Show intensity and phase of a complex optical field.
    """
    intensity = np.abs(E) ** 2
    phase = np.angle(E)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(intensity, cmap="gray", extent=extent_um)
    plt.title(title + "\nIntensity")
    plt.xlabel("x [um]")
    plt.ylabel("y [um]")
    plt.colorbar(label="Intensity")

    plt.subplot(1, 2, 2)
    plt.imshow(phase, cmap="twilight", extent=extent_um)
    plt.title(title + "\nPhase")
    plt.xlabel("x [um]")
    plt.ylabel("y [um]")
    plt.colorbar(label="Phase [rad]")

    plt.tight_layout()
    plt.show()


def show_tissue_slice(delta_n_stack, slice_id, dx, dz):
    """
    Show one x-y tissue slice as refractive index fluctuation.
    """
    n_slices, N, _ = delta_n_stack.shape

    if slice_id < 0 or slice_id >= n_slices:
        raise ValueError("slice_id out of range.")

    x = (np.arange(N) - N // 2) * dx * 1e6
    y = (np.arange(N) - N // 2) * dx * 1e6

    extent = [x.min(), x.max(), y.min(), y.max()]
    z_um = slice_id * dz * 1e6

    plt.figure(figsize=(5, 4))
    plt.imshow(delta_n_stack[slice_id], cmap="RdBu_r", extent=extent)
    plt.colorbar(label="Delta n")
    plt.title(f"Tissue slice at z = {z_um:.1f} um")
    plt.xlabel("x [um]")
    plt.ylabel("y [um]")
    plt.tight_layout()
    plt.show()


def show_tissue_xz_section(delta_n_stack, dx, dz):
    """
    Show x-z section through the center of the tissue.
    """
    n_slices, N, _ = delta_n_stack.shape

    y_center = N // 2
    xz = delta_n_stack[:, y_center, :]

    x_um = (np.arange(N) - N // 2) * dx * 1e6
    z_um = np.arange(n_slices) * dz * 1e6

    extent = [
        x_um.min(),
        x_um.max(),
        z_um.max(),
        z_um.min(),
    ]

    plt.figure(figsize=(8, 4))
    plt.imshow(xz, cmap="RdBu_r", aspect="auto", extent=extent)
    plt.colorbar(label="Delta n")
    plt.title("Tissue x-z refractive-index section")
    plt.xlabel("x [um]")
    plt.ylabel("z [um]")
    plt.tight_layout()
    plt.show()


def show_cumulative_tissue_phase(delta_n_stack, dz, wavelength0, extent_um):
    """
    Show accumulated tissue phase delay through the whole tissue.
    """
    k0 = 2 * np.pi / wavelength0

    cumulative_phase = np.sum(k0 * delta_n_stack * dz, axis=0)

    plt.figure(figsize=(5, 4))
    plt.imshow(cumulative_phase, cmap="twilight", extent=extent_um)
    plt.colorbar(label="Accumulated phase [rad]")
    plt.title("Cumulative tissue phase distortion")
    plt.xlabel("x [um]")
    plt.ylabel("y [um]")
    plt.tight_layout()
    plt.show()


def show_intensity_xz(intensity_stack, z_positions, dx, title, log_scale=True):
    """
    Show x-z intensity section through the center of the propagated beam.
    """
    n_frames, N, _ = intensity_stack.shape

    y_center = N // 2
    xz = intensity_stack[:, y_center, :]

    if log_scale:
        xz_plot = np.log10(xz / (xz.max() + 1e-12) + 1e-6)
        label = "log10 normalized intensity"
    else:
        xz_plot = xz / (xz.max() + 1e-12)
        label = "normalized intensity"

    x_um = (np.arange(N) - N // 2) * dx * 1e6
    z_um = z_positions * 1e6

    extent = [
        x_um.min(),
        x_um.max(),
        z_um.max(),
        z_um.min(),
    ]

    plt.figure(figsize=(8, 4))
    plt.imshow(xz_plot, cmap="inferno", aspect="auto", extent=extent)
    plt.colorbar(label=label)
    plt.title(title)
    plt.xlabel("x [um]")
    plt.ylabel("z [um]")
    plt.tight_layout()
    plt.show()


def compare_center_line(E_before, E_after, dx, title):
    """
    Compare central intensity line before and after tissue.
    """
    N = E_before.shape[0]
    x_um = (np.arange(N) - N // 2) * dx * 1e6

    I0 = np.abs(E_before) ** 2
    I1 = np.abs(E_after) ** 2

    plt.figure(figsize=(7, 4))
    plt.plot(x_um, I0[N // 2, :] / (I0.max() + 1e-12), label="Before tissue")
    plt.plot(x_um, I1[N // 2, :] / (I1.max() + 1e-12), label="After tissue")
    plt.xlabel("x [um]")
    plt.ylabel("Normalized intensity")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Simulation cases
# ============================================================

def run_two_beam_case(delta_n_stack):
    print("\nRunning two-beam interference case...")

    fringe_period = 0.32e-6
    waist = 2e-3

    E0, theta = make_two_beam_field(
        X,
        Y,
        wavelength0=wavelength0,
        n_medium=n_background,
        fringe_period=fringe_period,
        waist=waist,
    )

    print(f"Two-beam half angle theta = {np.rad2deg(theta):.3f} degrees")
    print(f"Fringe period = {fringe_period * 1e6:.2f} um")

    show_complex_field(E0, "Initial two-beam field", extent_um)

    E_out, intensity_stack, z_positions = propagate_through_tissue_and_record(
        E0,
        delta_n_stack,
        dx=dx,
        dz=dz,
        wavelength0=wavelength0,
        n_background=n_background,
        absorption_coeff=absorption_coeff,
        record_every=record_every,
    )

    show_complex_field(E_out, "Two-beam field after tissue", extent_um)

    show_intensity_xz(
        intensity_stack,
        z_positions,
        dx,
        title="Two-beam propagation through tissue, x-z section",
        log_scale=False,
    )

    compare_center_line(
        E0,
        E_out,
        dx,
        title="Two-beam fringe degradation after tissue",
    )


def run_high_na_case(delta_n_stack):
    print("\nRunning high-NA focusing case...")

    NA = 1.3
    focal_length = tissue_thickness

    E0, pupil_radius = make_high_na_focusing_field(
        X,
        Y,
        wavelength0=wavelength0,
        n_medium=n_background,
        NA=NA,
        focal_length=focal_length,
        phase_type="spherical",
    )

    print(f"NA = {NA}")
    print(f"Focal length = {focal_length * 1e6:.1f} um")
    print(f"Pupil radius = {pupil_radius * 1e6:.1f} um")

    show_complex_field(E0, "Initial high-NA pupil field", extent_um)

    # Clean propagation to focus without tissue
    E_clean_focus = angular_spectrum(
        E0,
        dx=dx,
        wavelength0=wavelength0,
        n_medium=n_background,
        z=focal_length,
    )

    show_complex_field(E_clean_focus, "Clean focus without tissue", extent_um)

    # Propagation through tissue
    E_out, intensity_stack, z_positions = propagate_through_tissue_and_record(
        E0,
        delta_n_stack,
        dx=dx,
        dz=dz,
        wavelength0=wavelength0,
        n_background=n_background,
        absorption_coeff=absorption_coeff,
        record_every=record_every,
    )

    show_complex_field(E_out, "High-NA focus after tissue", extent_um)

    show_intensity_xz(
        intensity_stack,
        z_positions,
        dx,
        title="High-NA focusing through tissue, x-z section",
        log_scale=True,
    )

    compare_center_line(
        E_clean_focus,
        E_out,
        dx,
        title="Clean focus versus tissue-distorted focus",
    )


def run_focal_array_case(delta_n_stack=None, use_tissue=False):
    print("\nRunning corrected focal-array case...")

    NA = 1.2
    focal_length = tissue_thickness

    n_foci = 9
    spacing = 0.96e-6

    E0, x_positions, pupil_radius, chief_ray_angles = make_focal_array_field_exact_spherical(
        X,
        Y,
        wavelength0=wavelength0,
        n_medium=n_background,
        NA=NA,
        focal_length=focal_length,
        n_foci=n_foci,
        spacing=spacing,
    )

    print(f"NA = {NA}")
    print(f"Focal length = {focal_length * 1e6:.1f} um")
    print(f"Pupil radius = {pupil_radius * 1e6:.1f} um")
    print(f"Number of foci = {n_foci}")
    print(f"Focus spacing = {spacing * 1e6:.2f} um")
    print("Target focal x-positions [um]:")
    print(x_positions * 1e6)
    print("Approximate chief-ray angles [deg]:")
    print(np.rad2deg(chief_ray_angles))

    show_complex_field(
        E0,
        "Corrected initial field for focal array",
        extent_um,
    )

    # Propagate without tissue to focal plane
    E_clean_focus = angular_spectrum(
        E0,
        dx=dx,
        wavelength0=wavelength0,
        n_medium=n_background,
        z=focal_length,
    )

    show_complex_field(
        E_clean_focus,
        "Corrected focal array without tissue",
        extent_um,
    )

    # Show focal-plane x profile
    I = np.abs(E_clean_focus) ** 2
    I_norm = I / (I.max() + 1e-12)

    plt.figure(figsize=(8, 4))
    plt.plot(x * 1e6, I_norm[N // 2, :])
    for xm in x_positions:
        plt.axvline(xm * 1e6, linestyle="--", alpha=0.4)
    plt.xlabel("x [um]")
    plt.ylabel("Normalized intensity")
    plt.title("Corrected focal-array intensity profile")
    plt.tight_layout()
    plt.show()

    # Show x-z propagation without tissue
    z_list = np.linspace(0, focal_length, 100)
    intensity_stack = []

    for z in z_list:
        Ez = angular_spectrum(
            E0,
            dx=dx,
            wavelength0=wavelength0,
            n_medium=n_background,
            z=z,
        )
        intensity_stack.append(np.abs(Ez) ** 2)

    intensity_stack = np.array(intensity_stack)

    show_intensity_xz(
        intensity_stack,
        z_list,
        dx,
        title="Corrected focal-array propagation without tissue, x-z section",
        log_scale=True,
    )

    # Optional tissue case
    if use_tissue and delta_n_stack is not None:
        E_out, intensity_stack_tissue, z_positions = propagate_through_tissue_and_record(
            E0,
            delta_n_stack,
            dx=dx,
            dz=dz,
            wavelength0=wavelength0,
            n_background=n_background,
            absorption_coeff=absorption_coeff,
            record_every=record_every,
        )

        show_complex_field(
            E_out,
            "Corrected focal array after tissue",
            extent_um,
        )

        show_intensity_xz(
            intensity_stack_tissue,
            z_positions,
            dx,
            title="Corrected focal-array propagation through tissue, x-z section",
            log_scale=True,
        )

        compare_center_line(
            E_clean_focus,
            E_out,
            dx,
            title="Corrected focal array: without tissue vs after tissue",
        )


# ============================================================
# Main
# ============================================================

def main():
    print("Generating tissue layer...")
    print(f"N = {N}")
    print(f"dx = {dx * 1e6:.3f} um")
    print(f"tissue thickness = {tissue_thickness * 1e6:.1f} um")
    print(f"dz = {dz * 1e6:.1f} um")
    print(f"number of slices = {n_slices}")
    print(f"dn_rms = {dn_rms}")

    delta_n_stack = generate_tissue_delta_n_stack(
        N=N,
        dx=dx,
        dz=dz,
        n_slices=n_slices,
        dn_rms=dn_rms,
        corr_len_xy=corr_len_xy,
        corr_len_z=corr_len_z,
        seed=random_seed,
    )

    # Visualize tissue
    show_tissue_slice(delta_n_stack, slice_id=0, dx=dx, dz=dz)
    show_tissue_slice(delta_n_stack, slice_id=n_slices // 2, dx=dx, dz=dz)
    show_tissue_xz_section(delta_n_stack, dx=dx, dz=dz)
    show_cumulative_tissue_phase(
        delta_n_stack,
        dz=dz,
        wavelength0=wavelength0,
        extent_um=extent_um,
    )

    if RUN_CASE == "two_beam":
        run_two_beam_case(delta_n_stack)

    elif RUN_CASE == "high_na":
        run_high_na_case(delta_n_stack)

    elif RUN_CASE == "focal_array":
        run_focal_array_case(delta_n_stack, use_tissue=True)

    elif RUN_CASE == "all":
        run_two_beam_case(delta_n_stack)
        run_high_na_case(delta_n_stack)
        run_focal_array_case(delta_n_stack, use_tissue=True)

    else:
        raise ValueError("RUN_CASE must be 'two_beam', 'high_na', 'focal_array', or 'all'.")


if __name__ == "__main__":
    # ============================================================
    # Configuration
    # ============================================================

    N = 2500
    dx = 0.08e-6  # pixel size [m]
    wavelength0 = 488e-9  # vacuum wavelength [m]
    n_background = 1.37  # average tissue refractive index

    tissue_thickness = 20e-6  # tissue thickness [m]
    dz = 0.5e-6  # slice thickness [m]
    n_slices = int(round(tissue_thickness / dz))

    # Tissue structure parameters
    dn_rms = 0.008  # RMS refractive index fluctuation
    corr_len_xy = 6.0e-6  # lateral correlation length [m]
    corr_len_z = 8.0e-6  # axial correlation length [m]

    # Absorption coefficient, intensity attenuation [1/m]
    # 0 means no absorption
    absorption_coeff = 0.0

    # Which input field to simulate:
    # "two_beam", "high_na", "focal_array", or "all"
    RUN_CASE = "two_beam"

    # Recording interval for propagation movie / x-z section
    record_every = 1

    # Random seed for reproducible tissue
    random_seed = 1

    # ============================================================
    # Grid
    # ============================================================

    x = (np.arange(N) - N // 2) * dx
    y = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X ** 2 + Y ** 2)

    extent_um = [
        x.min() * 1e6,
        x.max() * 1e6,
        y.min() * 1e6,
        y.max() * 1e6,
    ]

    print("Generating tissue layer...")
    print(f"N = {N}")
    print(f"dx = {dx * 1e6:.3f} um")
    print(f"tissue thickness = {tissue_thickness * 1e6:.1f} um")
    print(f"dz = {dz * 1e6:.1f} um")
    print(f"number of slices = {n_slices}")
    print(f"dn_rms = {dn_rms}")

    delta_n_stack = generate_tissue_delta_n_stack(
        N=N,
        dx=dx,
        dz=dz,
        n_slices=n_slices,
        dn_rms=dn_rms,
        corr_len_xy=corr_len_xy,
        corr_len_z=corr_len_z,
        seed=random_seed,
    )

    print("\nRunning two-beam interference case...")
    fringe_period = 0.32e-6
    waist = 2e-3

    E0_sw, theta = make_two_beam_field(
        X,
        Y,
        wavelength0=wavelength0,
        n_medium=n_background,
        fringe_period=fringe_period,
        waist=waist,
    )

    print(f"Two-beam half angle theta = {np.rad2deg(theta):.3f} degrees")
    print(f"Fringe period = {fringe_period * 1e6:.2f} um")

    E_out_sw, intensity_stack_sw, z_positions_sw = propagate_through_tissue_and_record(
        E0_sw,
        delta_n_stack,
        dx=dx,
        dz=dz,
        wavelength0=wavelength0,
        n_background=n_background,
        absorption_coeff=absorption_coeff,
        record_every=record_every,
    )

    print("\nRunning high-NA focusing case...")

    NA = 1.3
    focal_length = tissue_thickness

    E0_foc, pupil_radius = make_high_na_focusing_field(
        X,
        Y,
        wavelength0=wavelength0,
        n_medium=n_background,
        NA=NA,
        focal_length=focal_length,
        phase_type="spherical",
    )

    print(f"NA = {NA}")
    print(f"Focal length = {focal_length * 1e6:.1f} um")
    print(f"Pupil radius = {pupil_radius * 1e6:.1f} um")

    E_foc = angular_spectrum(
        E0_foc,
        dx=dx,
        wavelength0=wavelength0,
        n_medium=n_background,
        z=focal_length,
    )

    E_out_foc, intensity_stack_foc, z_positions_foc = propagate_through_tissue_and_record(
        E0_foc,
        delta_n_stack,
        dx=dx,
        dz=dz,
        wavelength0=wavelength0,
        n_background=n_background,
        absorption_coeff=absorption_coeff,
        record_every=record_every,
    )

    print("\nRunning corrected focal-array case...")

    n_foci = 10
    spacing = 0.96e-6

    E0_fa, x_positions_fa, pupil_radius, chief_ray_angles = make_focal_array_field_exact_spherical(
        X,
        Y,
        wavelength0=wavelength0,
        n_medium=n_background,
        NA=NA,
        focal_length=focal_length,
        n_foci=n_foci,
        spacing=spacing,
    )

    print(f"NA = {NA}")
    print(f"Focal length = {focal_length * 1e6:.1f} um")
    print(f"Pupil radius = {pupil_radius * 1e6:.1f} um")
    print(f"Number of foci = {n_foci}")
    print(f"Focus spacing = {spacing * 1e6:.2f} um")
    print("Target focal x-positions [um]:")
    print(x_positions_fa * 1e6)
    print("Approximate chief-ray angles [deg]:")
    print(np.rad2deg(chief_ray_angles))

    # Propagate without tissue to focal plane
    E_fa = angular_spectrum(
        E0_fa,
        dx=dx,
        wavelength0=wavelength0,
        n_medium=n_background,
        z=focal_length,
    )

    # Show focal-plane x profile
    I = np.abs(E_fa) ** 2
    I_norm = I / (I.max() + 1e-12)

    z_list = np.linspace(0, focal_length, 100)
    intensity_stack = []

    for z in z_list:
        Ez = angular_spectrum(
            E0_fa,
            dx=dx,
            wavelength0=wavelength0,
            n_medium=n_background,
            z=z,
        )
        intensity_stack.append(np.abs(Ez) ** 2)

    intensity_stack = np.array(intensity_stack)

    E_out_fa, intensity_stack_tissue, z_positions = propagate_through_tissue_and_record(
        E0_fa,
        delta_n_stack,
        dx=dx,
        dz=dz,
        wavelength0=wavelength0,
        n_background=n_background,
        absorption_coeff=absorption_coeff,
        record_every=record_every,
    )
