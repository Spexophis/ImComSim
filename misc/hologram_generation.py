import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# -------- Camera / microscope --------
OBJECTIVE_MAG = 25
CAMERA_PIXEL_UM = 6.5 / OBJECTIVE_MAG  # Kinetix22: 6.5 µm pixel / 25X = 0.26 µm/pixel

# If your camera adapter is not 1.0X, modify this:
CAMERA_ADAPTER_MAG = 1.0
CAMERA_PIXEL_UM = CAMERA_PIXEL_UM / CAMERA_ADAPTER_MAG

# -------- Objective / pupil hologram --------
WAVELENGTH_UM = 0.488  # 488 nm
OBJECTIVE_F_UM = 200_000 / OBJECTIVE_MAG  # 200 mm tube lens / 25X = 8000 µm

# -------- SLM --------
SLM_WIDTH = 1272
SLM_HEIGHT = 1024
SLM_PIXEL_UM = 12.5

# Your projected beam diameter on the SLM
BEAM_DIAMETER_UM = 8250.0  # 8.25 mm, adjust 8000–8500

# Hamamatsu SLM phase image range
PHASE_LEVELS = 256  # 8-bit image, 0–255

# Optional global correction terms
PHASE_OFFSET_RAD = 0.0

# Optional carrier/blaze to move the whole pattern.
# Keep as (0, 0) first. Later you can use this to shift all spots.
GLOBAL_SPOT_OFFSET_UM = (0.0, 0.0)

# Weighted Gerchberg-Saxton style target equalization
N_WEIGHT_ITERATIONS = 10


def load_camera_image(path: str) -> np.ndarray:
    """
    Load a camera image from file.
    Replace this function with Teledyne/PVCAM acquisition later.
    """
    path = Path(path)

    if not path.exists():
        print(f"Image file not found: {path}")
        print("Generating a dummy image for testing.")
        img = np.zeros((2400, 2400), dtype=np.float32)
        yy, xx = np.indices(img.shape)
        img += np.exp(-((xx - 1200) ** 2 + (yy - 1200) ** 2) / (2 * 200 ** 2))
        img += 0.2 * np.random.random(img.shape)
        return img

    if iio is None:
        raise ImportError("Please install imageio: pip install imageio tifffile")

    img = iio.imread(path)
    if img.ndim > 2:
        img = img[..., 0]

    return img.astype(np.float32)


def pick_points_on_image(img: np.ndarray):
    """
    Display image and let user click target points.
    Left click: add point.
    Press Enter when finished.
    """
    points = []

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap="gray")
    ax.set_title("Click target positions, then press Enter")
    ax.set_xlabel("camera x pixel")
    ax.set_ylabel("camera y pixel")

    clicked_plot, = ax.plot([], [], "ro", markersize=6)

    def onclick(event):
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = float(event.xdata)
        y = float(event.ydata)
        points.append((x, y))

        xs, ys = zip(*points)
        clicked_plot.set_data(xs, ys)
        fig.canvas.draw_idle()

        print(f"Picked camera pixel: x={x:.1f}, y={y:.1f}")

    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    fig.canvas.mpl_disconnect(cid)
    return np.array(points, dtype=float)


def camera_pixels_to_sample_um(
    camera_xy_px: np.ndarray,
    image_shape,
    camera_pixel_um: float = CAMERA_PIXEL_UM,
    center_px=None,
    flip_x: bool = False,
    flip_y: bool = True,
    rotation_deg: float = 0.0,
):
    """
    Convert clicked camera pixels to sample coordinates in µm.

    Output coordinates are relative to the center of the camera image.

    Important:
    - flip_y=True is often useful because image y increases downward.
    - flip_x, flip_y, and rotation_deg need experimental calibration.
    """

    h, w = image_shape[:2]

    if center_px is None:
        cx = (w - 1) / 2
        cy = (h - 1) / 2
    else:
        cx, cy = center_px

    x_px = camera_xy_px[:, 0] - cx
    y_px = camera_xy_px[:, 1] - cy

    if flip_x:
        x_px = -x_px
    if flip_y:
        y_px = -y_px

    x_um = x_px * camera_pixel_um
    y_um = y_px * camera_pixel_um

    theta = np.deg2rad(rotation_deg)
    xr = np.cos(theta) * x_um - np.sin(theta) * y_um
    yr = np.sin(theta) * x_um + np.cos(theta) * y_um

    return np.column_stack([xr, yr])


def make_slm_grid():
    """
    Return SLM physical coordinate arrays X, Y in µm.
    X and Y are centered on the SLM.
    """
    x = (np.arange(SLM_WIDTH) - (SLM_WIDTH - 1) / 2) * SLM_PIXEL_UM
    y = (np.arange(SLM_HEIGHT) - (SLM_HEIGHT - 1) / 2) * SLM_PIXEL_UM
    X, Y = np.meshgrid(x, y)
    return X, Y


def make_circular_pupil_mask(X, Y, beam_diameter_um=BEAM_DIAMETER_UM):
    """
    Circular illuminated beam mask on the SLM.
    """
    R = np.sqrt(X**2 + Y**2)
    return R <= beam_diameter_um / 2


def compute_target_fields(X, Y, target_xy_um, wavelength_um, objective_f_um):
    """
    For each target sample coordinate, compute the pupil phase ramp.

    A focus at sample position (x0, y0) corresponds to:

        phase = 2π / λ * (x0 X + y0 Y) / f_obj

    where X,Y are pupil coordinates.
    """
    n_targets = len(target_xy_um)
    fields = []

    for j in range(n_targets):
        x0, y0 = target_xy_um[j]
        phase = (2 * np.pi / wavelength_um) * (x0 * X + y0 * Y) / objective_f_um
        fields.append(np.exp(1j * phase))

    return np.array(fields)


def estimate_target_intensities(phase, mask, target_fields):
    """
    Estimate relative intensity at each target using overlap integral.
    """
    pupil = mask * np.exp(1j * phase)
    intensities = []

    for field_j in target_fields:
        amp = np.sum(pupil * np.conj(field_j))
        intensities.append(np.abs(amp) ** 2)

    intensities = np.array(intensities, dtype=float)
    intensities /= np.mean(intensities) + 1e-30
    return intensities


def generate_multifocal_hologram(
    target_xy_um: np.ndarray,
    wavelength_um: float = WAVELENGTH_UM,
    objective_f_um: float = OBJECTIVE_F_UM,
    beam_diameter_um: float = BEAM_DIAMETER_UM,
    n_weight_iterations: int = N_WEIGHT_ITERATIONS,
    random_seed: int = 1,
):
    """
    Generate a phase-only hologram for multiple focal spots.

    Uses random-superposition plus simple weighted target equalization.

    Returns:
        phase_rad: float array, 0–2π
        phase_u8: uint8 phase image, 0–255
        mask: illuminated pupil mask
    """

    if len(target_xy_um) == 0:
        raise ValueError("No target points were selected.")

    X, Y = make_slm_grid()
    mask = make_circular_pupil_mask(X, Y, beam_diameter_um)

    # Add optional global spot offset
    target_xy_um = np.array(target_xy_um, dtype=float)
    target_xy_um[:, 0] += GLOBAL_SPOT_OFFSET_UM[0]
    target_xy_um[:, 1] += GLOBAL_SPOT_OFFSET_UM[1]

    target_fields = compute_target_fields(
        X, Y, target_xy_um, wavelength_um, objective_f_um
    )

    rng = np.random.default_rng(random_seed)
    n_targets = len(target_xy_um)

    weights = np.ones(n_targets, dtype=float)
    random_phases = rng.uniform(0, 2 * np.pi, size=n_targets)

    phase = np.zeros((SLM_HEIGHT, SLM_WIDTH), dtype=float)

    for it in range(n_weight_iterations + 1):
        complex_sum = np.zeros((SLM_HEIGHT, SLM_WIDTH), dtype=np.complex128)

        for j in range(n_targets):
            complex_sum += weights[j] * target_fields[j] * np.exp(1j * random_phases[j])

        phase = np.angle(complex_sum)
        phase = np.mod(phase + PHASE_OFFSET_RAD, 2 * np.pi)

        intensities = estimate_target_intensities(phase, mask, target_fields)

        if it < n_weight_iterations:
            # Increase weak targets, decrease strong targets
            weights *= 1.0 / np.sqrt(intensities + 1e-12)
            weights /= np.mean(weights)

        print(
            f"Iteration {it:02d}: "
            f"target intensity min={intensities.min():.3f}, "
            f"max={intensities.max():.3f}, "
            f"std={intensities.std():.3f}"
        )

    # Outside illuminated pupil, set phase to 0
    phase = np.where(mask, phase, 0.0)

    phase_u8 = np.round(phase / (2 * np.pi) * (PHASE_LEVELS - 1)).astype(np.uint8)

    return phase, phase_u8, mask


if __name__ == "__main__":

    img = load_camera_image(CAMERA_IMAGE_PATH)

    points_px = pick_points_on_image(img)

    print("\nPicked points in camera pixels:")
    print(points_px)

    target_xy_um = camera_pixels_to_sample_um(
        points_px,
        img.shape,
        camera_pixel_um=CAMERA_PIXEL_UM,
        center_px=None,      # default: image center
        flip_x=False,        # calibrate this
        flip_y=True,         # often correct for image coordinates
        rotation_deg=0.0,    # calibrate this
    )

    print("\nTarget points in sample coordinates, µm:")
    print(target_xy_um)

    phase_rad, phase_u8, mask = generate_multifocal_hologram(
        target_xy_um,
        wavelength_um=WAVELENGTH_UM,
        objective_f_um=OBJECTIVE_F_UM,
        beam_diameter_um=BEAM_DIAMETER_UM,
        n_weight_iterations=N_WEIGHT_ITERATIONS,
    )
