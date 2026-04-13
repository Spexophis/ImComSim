import numpy as np
import psf_generator
import pupil_wavefront_modulator
import vectorical_focusing

exc_wl = 0.488
emi_wl = 0.505

na = 1.4

m = pupil_wavefront_modulator.PupilWavefrontModulator(zernike_coeffs={},
                                                      amplitude='gaussian', amplitude_params={'sigma': 0.9})

sc_nx = 128
sc_ny = 128
sc_nz = 64
sc_stp = 0.04  # um
x_out = (np.arange(sc_nx) - sc_nx // 2) * sc_stp
y_out = (np.arange(sc_ny) - sc_ny // 2) * sc_stp
z_out = np.linspace(-1.6, 1.6, sc_nz)

N_theta = 128
N_phi = 512
polr = "left_circular"
vch = 1

v = vectorical_focusing.RichardsWolfDirect(NA=na, n=1.515, wavelength=exc_wl,
                                           N_theta=N_theta, N_phi=N_phi, polarization=polr,
                                           amplitude_func=m.as_amplitude_func())

nxy = 128
dxy = 0.04  # um

p = psf_generator.PSF(wl=emi_wl, na=na, dx=dxy, nx=nxy)
p.bpp = m.to_psf_wavefront(nx=p.nx, radius=p.radius)

psf3d = p.get_3d_psf((0, 0, 0), -1.6, 1.6, 0.16)


psf, _, _, _ = v.compute(m.zernike_coeffs, x_out, y_out, z_out,
                         vortex_charge=vch, normalize=True, verbose=False)
