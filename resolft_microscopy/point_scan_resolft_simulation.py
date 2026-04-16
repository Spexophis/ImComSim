import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf
from scipy.optimize import curve_fit
import photophysics_simulator
import psf_generator
import pupil_wavefront_modulator
import vectorical_focusing
from noise_generator import add_readout_noise

rsEGFP2_off_state = photophysics_simulator.NegativeSwitchers(extincion_coeff_on=[5260, 51560],
                                                             extincion_coeff_off=[22000, 60],
                                                             wavelength=[405, 488],
                                                             lifetime_on=1.6E-6,
                                                             lifetime_off=20E-9,
                                                             qy_cis_to_trans_anionic=1.65E-2,
                                                             qy_trans_to_cis_neutral=0.33,
                                                             qy_fluorescence_on=0.35,
                                                             initial_populations=[1, 0, 0, 0])

rsEGFP2_on_state = photophysics_simulator.NegativeSwitchers(extincion_coeff_on=[5260, 51560],
                                                            extincion_coeff_off=[22000, 60],
                                                            wavelength=[405, 488],
                                                            lifetime_on=1.6E-6,
                                                            lifetime_off=20E-9,
                                                            qy_cis_to_trans_anionic=1.65E-2,
                                                            qy_trans_to_cis_neutral=0.33,
                                                            qy_fluorescence_on=0.35,
                                                            initial_populations=[0, 0, 1, 0])

act_wl = 0.405
exc_wl = 0.488
emi_wl = 0.505

na = 1.4

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

nxy = 128
dxy = 0.04  # um

p = psf_generator.PSF(wl=emi_wl, na=na, dx=dxy, nx=nxy)
pn_rd = (emi_wl / (2 * na)) / dxy
ph_msk = p._disc(radius=pn_rd) * 1

v_405 = vectorical_focusing.RichardsWolfDirect(NA=na, n=1.515, wavelength=act_wl,
                                               N_theta=N_theta, N_phi=N_phi, polarization=polr)

v_488 = vectorical_focusing.RichardsWolfDirect(NA=na, n=1.515, wavelength=exc_wl,
                                               N_theta=N_theta, N_phi=N_phi, polarization=polr)

abbr = [{1: 0.0},
        {4: 0.2}, {4: 0.4}, {4: 0.6}, {4: 0.8},
        {5: 0.2}, {5: 0.4}, {5: 0.6}, {5: 0.8},
        {7: 0.2}, {7: 0.4}, {7: 0.6}, {7: 0.8},
        {11: 0.2}, {11: 0.4}, {11: 0.6}, {11: 0.8}]

for aber in abbr:
    m = pupil_wavefront_modulator.PupilWavefrontModulator(zernike_coeffs=aber,
                                                          amplitude='gaussian', amplitude_params={'sigma': 0.9})

    foci, _, _, _ = v_405.compute(m.zernike_coeffs, x_out, y_out, z_out,
                                  vortex_charge=0, normalize=False, verbose=False)
    # on_switch = add_illumination_noise(foci[sc_nz // 2], 2 ** 16)
    # on_switch = on_switch / on_switch.max()
    on_switch = 40 * foci[sc_nz // 2] / foci[sc_nz // 2].sum()

    donut, _, _, _ = v_488.compute(m.zernike_coeffs, x_out, y_out, z_out,
                                   vortex_charge=vch, normalize=False, verbose=False)
    # off_switch = add_illumination_noise(donut[sc_nz // 2], 2 ** 16)
    # off_switch = off_switch / off_switch.max()
    off_switch = 80 * donut[sc_nz // 2] / donut[sc_nz // 2].sum()

    # focus, _, _, _ = v_488.compute(m.zernike_coeffs, x_out, y_out, z_out,
    #                                vortex_charge=0, normalize=False, verbose=False)
    # read_out = add_readout_noise(focus[sc_nz // 2])
    # read_out = 20 * focus[sc_nz // 2] / focus[sc_nz // 2].sum()

    p.bpp = m.to_psf_wavefront(nx=p.nx, radius=p.radius)

    kcs = np.zeros((sc_ny, sc_nx, 45, 4))
    psfs = np.zeros((sc_ny, sc_nx, nxy, nxy))
    kcs_m = np.zeros((sc_ny, sc_nx, 45))
    psfs_m = np.zeros((sc_ny, sc_nx, nxy, nxy))
    ratio_map = np.zeros((sc_ny, sc_nx))

    for i in range(sc_nx):
        for j in range(sc_ny):
            print(j, i)
            lon = on_switch[j, i]
            lof = off_switch[j, i]
            switching_pulse = photophysics_simulator.ModulatedLasers(wavelengths=[405, 488],
                                                                     power_densities=[lon, lof],
                                                                     pulse_widths=[50e-3, 250e-3],
                                                                     t_start=[0., 100e-3],
                                                                     dwell_time=450e-3)
            switching_experiment = photophysics_simulator.Experiment(illumination=switching_pulse,
                                                                     fluorophore=rsEGFP2_off_state)
            fluo_populations = switching_experiment.solve_kinetics(0.01)
            emin = np.trapezoid(fluo_populations[11:35, 3], dx=0.01)
            psf_full = np.fft.fftshift(p.get_2d_psf((x_out[i], y_out[j], 0)))
            psf_filt = ph_msk * psf_full
            ratio = psf_filt.sum() / psf_full.sum()
            fluo_filt = fluo_populations * ratio
            ratio_map[j, i] = ratio
            kcs[j, i, :, :] = fluo_populations
            psfs[j, i, :, :] = psf_full
            kcs_m[j, i, :] = fluo_filt[:, 3]
            psfs_m[j, i, :, :] = psf_filt

    # curve = np.sum(kcs_m, axis=(0, 1))
    # plt.figure()
    # plt.plot(curve[11:35])
    # plt.show()

    zm = list(aber.keys())[0]
    za = aber[zm]
    tf.imwrite(f"C:\\Users\\ruizhe.lin\\Desktop\\swkinetics\\ratio_z{zm}_{za}.tif", on_switch)
    tf.imwrite(f"C:\\Users\\ruizhe.lin\\Desktop\\swkinetics\\on_switch_z{zm}_{za}.tif", on_switch)
    tf.imwrite(f"C:\\Users\\ruizhe.lin\\Desktop\\swkinetics\\off_switch_z{zm}_{za}.tif", off_switch)
    tf.imwrite(f"C:\\Users\\ruizhe.lin\\Desktop\\swkinetics\\psfs_z{zm}_{za}.tif", psfs)
    tf.imwrite(f"C:\\Users\\ruizhe.lin\\Desktop\\swkinetics\\kcs_z{zm}_{za}.tif", kcs)
    tf.imwrite(f"C:\\Users\\ruizhe.lin\\Desktop\\swkinetics\\psfs_m_z{zm}_{za}.tif", psfs_m)
    tf.imwrite(f"C:\\Users\\ruizhe.lin\\Desktop\\swkinetics\\kcs_m_z{zm}_{za}.tif", kcs_m)

def exp_func(x_, a_, b_, c_):
    return a_ * np.exp(b_ * x_) + c_

img_0 = tf.imread(r"C:\Users\ruizhe.lin\Desktop\swkinetics_high405\kcs_m_z1_0.0.tif")
img_1 = tf.imread(r"C:\Users\ruizhe.lin\Desktop\swkinetics_high405\kcs_m_z11_0.2.tif")
img_2 = tf.imread(r"C:\Users\ruizhe.lin\Desktop\swkinetics_high405\kcs_m_z11_0.4.tif")
img_3 = tf.imread(r"C:\Users\ruizhe.lin\Desktop\swkinetics_high405\kcs_m_z11_0.6.tif")
img_4 = tf.imread(r"C:\Users\ruizhe.lin\Desktop\swkinetics_high405\kcs_m_z11_0.8.tif")

x = 0.01 * np.arange(24)

plt.figure()
curve = img_0[ph_msk == 1].sum(axis=0)
plt.plot(curve[11:35], label="0")
print((curve[35]  - curve[11]))
# --- Fit ---
y = curve[11:35]
p0 = [curve[11], -1, curve[35]]  # initial guess for [a, b, c]
popt, pcov = curve_fit(exp_func, x, y, p0=p0, maxfev=10000)
a, b, c = popt
perr = np.sqrt(np.diag(pcov))  # 1-sigma uncertainties

print(f"a = {a:.4f} ± {perr[0]:.4f}")
print(f"b = {b:.4f} ± {perr[1]:.4f}")
print(f"c = {c:.4f} ± {perr[2]:.4f}")

# --- Evaluate fit ---
x_fit = np.linspace(x.min(), x.max(), 320)
y_fit = exp_func(x_fit, *popt)

residuals = y - exp_func(x, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1 - ss_res / ss_tot
print(f"R² = {r_squared:.6f}")

k      = -b
tau    = -1.0 / b
t_half = np.log(2) / k
print(k, tau, t_half)

curve = img_1[ph_msk == 1].sum(axis=0)
plt.plot(curve[11:35], label="1")
print(curve[35]  - curve[11])

# --- Fit ---
y = curve[11:35]
p0 = [curve[11], -1, curve[35]]  # initial guess for [a, b, c]
popt, pcov = curve_fit(exp_func, x, y, p0=p0, maxfev=10000)
a, b, c = popt
perr = np.sqrt(np.diag(pcov))  # 1-sigma uncertainties

print(f"a = {a:.4f} ± {perr[0]:.4f}")
print(f"b = {b:.4f} ± {perr[1]:.4f}")
print(f"c = {c:.4f} ± {perr[2]:.4f}")

# --- Evaluate fit ---
x_fit = np.linspace(x.min(), x.max(), 320)
y_fit = exp_func(x_fit, *popt)

residuals = y - exp_func(x, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1 - ss_res / ss_tot
print(f"R² = {r_squared:.6f}")

k      = -b
tau    = -1.0 / b
t_half = np.log(2) / k
print(k, tau, t_half)

curve = img_2[ph_msk == 1].sum(axis=0)
plt.plot(curve[11:35], label="2")
print(curve[35]  - curve[11])

# --- Fit ---
y = curve[11:35]
p0 = [curve[11], -1, curve[35]]  # initial guess for [a, b, c]
popt, pcov = curve_fit(exp_func, x, y, p0=p0, maxfev=10000)
a, b, c = popt
perr = np.sqrt(np.diag(pcov))  # 1-sigma uncertainties

print(f"a = {a:.4f} ± {perr[0]:.4f}")
print(f"b = {b:.4f} ± {perr[1]:.4f}")
print(f"c = {c:.4f} ± {perr[2]:.4f}")

# --- Evaluate fit ---
x_fit = np.linspace(x.min(), x.max(), 320)
y_fit = exp_func(x_fit, *popt)

residuals = y - exp_func(x, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1 - ss_res / ss_tot
print(f"R² = {r_squared:.6f}")

k      = -b
tau    = -1.0 / b
t_half = np.log(2) / k
print(k, tau, t_half)

curve = img_3[ph_msk == 1].sum(axis=0)
plt.plot(curve[11:35], label="3")
print(curve[35]  - curve[11])

# --- Fit ---
y = curve[11:35]
p0 = [curve[11], -1, curve[35]]  # initial guess for [a, b, c]
popt, pcov = curve_fit(exp_func, x, y, p0=p0, maxfev=10000)
a, b, c = popt
perr = np.sqrt(np.diag(pcov))  # 1-sigma uncertainties

print(f"a = {a:.4f} ± {perr[0]:.4f}")
print(f"b = {b:.4f} ± {perr[1]:.4f}")
print(f"c = {c:.4f} ± {perr[2]:.4f}")

# --- Evaluate fit ---
x_fit = np.linspace(x.min(), x.max(), 320)
y_fit = exp_func(x_fit, *popt)

residuals = y - exp_func(x, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1 - ss_res / ss_tot
print(f"R² = {r_squared:.6f}")

k      = -b
tau    = -1.0 / b
t_half = np.log(2) / k
print(k, tau, t_half)

curve = img_4[ph_msk == 1].sum(axis=0)
plt.plot(curve[11:35], label="4")

# --- Fit ---
y = curve[11:35]
p0 = [curve[11], -1, curve[35]]  # initial guess for [a, b, c]
popt, pcov = curve_fit(exp_func, x, y, p0=p0, maxfev=10000)
a, b, c = popt
perr = np.sqrt(np.diag(pcov))  # 1-sigma uncertainties

print(f"a = {a:.4f} ± {perr[0]:.4f}")
print(f"b = {b:.4f} ± {perr[1]:.4f}")
print(f"c = {c:.4f} ± {perr[2]:.4f}")

# --- Evaluate fit ---
x_fit = np.linspace(x.min(), x.max(), 320)
y_fit = exp_func(x_fit, *popt)

residuals = y - exp_func(x, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1 - ss_res / ss_tot
print(f"R² = {r_squared:.6f}")

k      = -b
tau    = -1.0 / b
t_half = np.log(2) / k
print(k, tau, t_half)

plt.show()
plt.legend()
