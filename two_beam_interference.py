import random

import numpy as np
import tifffile as tf
from numpy.fft import fft, ifft, fftshift
from scipy.ndimage import gaussian_filter


def plane_wave(zb, yb, z0, y0, width, angle):
    za = (zb - z0) * np.cos(angle) + (yb - y0) * np.sin(angle)
    ya = (yb - y0) * np.cos(angle) - (zb - z0) * np.sin(angle)
    zr = np.pi * width ** 2
    q = za - 1.j * zr
    return -1.j * zr * np.exp(2 * np.pi * 1.j * (za + ya * ya / (2 * q))) / q


def spherical_wave(zb, yb, z0, y0, amp, angle):
    c = np.sqrt((zb - z0) ** 2 + (yb - y0) ** 2)
    mag = np.divide(amp, c)
    za = (zb - z0) * np.cos(angle) + (yb - y0) * np.sin(angle)
    ya = (yb - y0) * np.cos(angle) - (zb - z0) * np.sin(angle)
    sph = np.exp(2.0j * np.pi * np.sqrt(ya ** 2 + za ** 2))
    return mag * sph


def circle(x, y, x0, y0, r):
    xa = x - x0
    ya = y - y0
    return xa * xa + ya * ya < (r * r)


def generate_lens(axs, r1=40, r2=320, zpm=20, thk=10, yp=0, wth=80, ri=1.5168):
    zz, yy = axs
    d = 2 * thk
    zp1 = zpm - thk + r1
    zp2 = zpm + thk - r2
    lens = circle(zz, yy, zp1, yp, r1) * circle(zz, yy, zp2, yp, r2) * (yy < 0.5 * wth) * (yy > - 0.5 * wth) * (ri - 1)
    foc = 1 / ((ri - 1) * (1 / r1 - 1 / r2 + ((ri - 1) * d) / (ri * r1 * r2)))
    return lens, foc


def generate_ri_spheres(axs, space, num=20, pzr=(65, 75), pyr=(-40, 40), rr=(0, 8), axr=(60, 80), sig=32):
    spheres = np.zeros(space)
    zz, yy = axs
    for i in range(num):
        zps = random.uniform(pzr[0], pzr[1])
        yps = random.uniform(pyr[0], pyr[1])
        radius = random.uniform(rr[0], rr[1])
        rind = random.uniform(1.0, 1.7) - 1.33
        spheres += rind * circle(zz, yy, zps, yps, radius) * (zz < axr[1]) * (zz > axr[0])
    if sig:
        return gaussian_filter(spheres, sigma=sig)
    else:
        return spheres


def generate_layer(axs, space, axr=(60, 80), sig=20):
    zz, yy = axs
    lyr = (np.random.random(space) - 0.33) * (zz < axr[1]) * (zz > axr[0])
    if sig:
        return gaussian_filter(lyr, sigma=sig)
    else:
        return lyr


def propagation(space, einput, idm, dyz, wavelength=0.488):
    _dy, _dz = dyz
    z_pxs, y_pxs = space
    phs = np.zeros(space)
    itn = np.zeros(space)
    b = fftshift(fft(einput))
    k = 2.0 * np.pi / wavelength
    k2 = k * k
    kym = 1.0 * np.pi / _dy
    dky = 2 * kym / y_pxs
    ky = np.arange(-kym, kym, dky)
    ky2 = ky * ky
    ky2c = ky2.astype('complex')
    kz = np.sqrt(k2 - ky2c)
    ph = 1.0j * kz * _dz
    for jj in range(0, z_pxs):
        c = ifft(fftshift(b)) * np.exp(2.0j * np.pi * idm[jj, :] * _dz)
        b = fftshift(fft(c)) * np.exp(1.0j * kz * _dz)
        phs[jj, :] += np.angle(c)
        itn[jj, :] += np.abs(c) ** 2
    return phs, itn


zmin = 0
zmax = 320
ymin = -160
ymax = 160
dz = 0.02
dy = 0.02
zoom = 1
Z, Y = np.mgrid[zmin / zoom:zmax / zoom:dz / zoom, ymin / zoom:ymax / zoom:dy / zoom]
z_pts, y_pts = np.shape(Z)

nr = np.zeros((z_pts, y_pts))
le = generate_lens(axs=(Z, Y))
nr += le[0]
# nr += generate_ri_spheres(axs=(Z, Y), space=(z_pts, y_pts))
nr += generate_layer(axs=(Z, Y), space=(z_pts, y_pts))

BeamSize = 30
BAngle = 0 * np.pi / 180
BeamOffset = 0
# e0 = plane_wave(Z[0, :], Y[0, :], 0, -BeamOffset, BeamSize, BAngle)
# e1 = gaussian_beam(Z[0, :], Y[0, :], 0, BeamOffset, BeamSize, -BAngle)
zo = le[1] - 20
E = spherical_wave(Z[0, :], Y[0, :], -zo, 25, 1.0, 0 * np.pi / 180) + spherical_wave(Z[0, :], Y[0, :], -zo, -25, 1.0, 0 * np.pi / 180)

phase, intensity = propagation(space=(z_pts, y_pts), einput=E, idm=nr, dyz=(dy, dz))

# plt.figure(figsize=(16, 10))
# plt.plot(Y[1, 400:1200], intensity[0, 400:1200], label="initial", linewidth=1.0)
# plt.plot(Y[1, 400:1200], intensity[280, 400:1200], label="midway", linewidth=1.2)
# plt.plot(Y[1, 400:1200], intensity[500, 400:1200], label="exit", linewidth=1.2)
# plt.plot(Y[1, 400:1200], intensity[1211, 400:1200], label="distant", linewidth=1.25)
# plt.xlabel(r'x / $\mu$m', fontsize=12)
# plt.ylabel('Intensity', fontsize=12)
# plt.grid(True, linestyle='--', linewidth=0.4, alpha=0.7)
# plt.xticks(ha='center', fontsize=10)
# plt.legend(fontsize=12, loc='best')
# plt.tight_layout()
# plt.show()

tf.imwrite(r"C:\Users\ruizhe.lin\Desktop\refractive_index_distribution.tif", nr)
tf.imwrite(r"C:\Users\ruizhe.lin\Desktop\one_beam_focusing_intensity.tif", intensity)
tf.imwrite(r"C:\Users\ruizhe.lin\Desktop\one_beam_focusing_phase.tif", phase)
# tf.imwrite(r"C:\Users\ruizhe.lin\Desktop\two_beam_interference_intensity_balls_g32.tif", intensity)
# tf.imwrite(r"C:\Users\ruizhe.lin\Desktop\two_beam_interference_phase_balls_g32.tif", phase)
