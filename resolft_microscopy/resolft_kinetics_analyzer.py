import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf
from scipy.optimize import curve_fit

import psf_generator


# --- Define the model ---
def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c  # c handles vertical offset


nxy = 128
dxy = 0.04  # um
act_wl = 0.405
exc_wl = 0.488
emi_wl = 0.505

na = 1.4
p = psf_generator.PSF(wl=emi_wl, na=na, dx=dxy, nx=nxy)

pn_rd = 1 * (emi_wl / (2 * na)) / dxy
ph_msk = p._disc(radius=pn_rd) * 1

zn = 11

img_0 = tf.imread(f"C:/Users/ruizhe.lin/Desktop/abresolft_high488/kcs_m_z0_0.0.tif")
img_1 = tf.imread(f"C:/Users/ruizhe.lin/Desktop/abresolft_high488/kcs_m_z{zn}_0.2.tif")
img_2 = tf.imread(f"C:/Users/ruizhe.lin/Desktop/abresolft_high488/kcs_m_z{zn}_0.4.tif")
img_3 = tf.imread(f"C:/Users/ruizhe.lin/Desktop/abresolft_high488/kcs_m_z{zn}_0.6.tif")
img_4 = tf.imread(f"C:/Users/ruizhe.lin/Desktop/abresolft_high488/kcs_m_z{zn}_0.8.tif")

x = 0.01 * np.arange(24)

xt = 0.01 * np.arange(45)

fig, ax = plt.subplots(figsize=(3.5, 2))

curve = img_0[ph_msk == 1].sum(axis=0)

ax.plot(xt,
        curve,
        linewidth=1.0,
        label="aberration free")

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
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - ss_res / ss_tot
print(f"R² = {r_squared:.6f}")

curve = img_1[ph_msk == 1].sum(axis=0)
ax.plot(xt,
        curve,
        linewidth=1.0,
        label="aberration - 0.2")

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
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - ss_res / ss_tot
print(f"R² = {r_squared:.6f}")

curve = img_2[ph_msk == 1].sum(axis=0)
ax.plot(xt,
        curve,
        linewidth=1.0,
        label="aberration - 0.4")

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
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - ss_res / ss_tot
print(f"R² = {r_squared:.6f}")

curve = img_3[ph_msk == 1].sum(axis=0)
ax.plot(xt,
        curve,
        linewidth=1.0,
        label="aberration - 0.6")

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
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - ss_res / ss_tot
print(f"R² = {r_squared:.6f}")

curve = img_4[ph_msk == 1].sum(axis=0)
ax.plot(xt,
        curve,
        linewidth=1.0,
        label="aberration - 0.8")

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
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - ss_res / ss_tot
print(f"R² = {r_squared:.6f}")


ax.legend(fontsize=8, frameon=True)

fig.tight_layout()

fig.savefig(f"C:/Users/ruizhe.lin/Desktop/high488_z{zn}_comparison.png", dpi=400)
plt.close(fig)

