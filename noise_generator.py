import numpy as np


def add_illumination_noise(field_intensity, n_photons_total):
    """
    field_intensity  : float array, normalized illumination intensity (0–1)
                       e.g. your computed |E|² from the Richards-Wolf model
    n_photons_total  : total expected photon count across the FOV per frame

    Returns noisy intensity in photons (float, same shape as input)
    """
    # Scale to expected photon counts
    mean_photons = field_intensity / field_intensity.sum() * n_photons_total

    # Poisson sample
    return np.random.poisson(mean_photons).astype(float)

def add_readout_noise(image_photons, qe=0.9, gain=1.0, read_noise_e=3.0, offset=100):
    """
    image_photons : float array, expected photon counts per pixel
    qe            : quantum efficiency (photons → photoelectrons)
    gain          : e⁻/ADU
    read_noise_e  : read noise in electrons (RMS)
    offset        : ADC offset / bias in ADU
    """
    # 1. QE: Poisson-sample the detected photoelectrons
    electrons = np.random.poisson(image_photons * qe).astype(float)

    # 2. Read noise (Gaussian, in electrons)
    electrons += np.random.normal(0, read_noise_e, electrons.shape)

    # 3. Convert to ADU
    adu = electrons / gain + offset

    return np.clip(np.round(adu), 0, 65535).astype(np.uint16)
