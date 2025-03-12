import os
import random
import time
import math
import numpy as np
import tifffile as tf
from numpy.fft import fft, ifft, fftshift
from scipy.ndimage import gaussian_filter


def plane_wave(zb, yb, z0, y0, width, angle):
    za = (zb - z0) * np.cos(angle) + (yb - y0) * np.sin(angle)
    ya = (yb - y0) * np.cos(angle) - (zb - z0) * np.sin(angle)
    zr = np.pi * width ** 2
    q = za - 1.j * zr
    m = np.abs(yb - y0) < width
    return -1.j * m * zr * np.exp(2 * m * np.pi * 1.j * (za + ya * ya / (2 * q))) / q


def spherical_wave(zb, yb, z0, y0, amp, width):
    m = np.abs(yb - y0) < width
    sph = m * np.exp(-2.0j * m * np.pi * np.sqrt((zb - z0) ** 2 + (yb - y0) ** 2))
    return amp * sph


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


def generate_non_overlapping_circles(axs, space, pyr=(-80, 80), pzr=(50, 80), rr=(0.1, 1), ris=(1.36, 1.45), coverage_fraction=0.70, max_attempts=100000):
    zz, yy = axs
    min_diameter, max_diameter = rr
    rmin, rmax = ris
    y_min, y_max = pyr
    z_min, z_max = pzr
    spheres = np.zeros(space)
    masks = np.zeros(space)
    width = y_max - y_min
    height = z_max - z_min
    area = width * height
    target_area = coverage_fraction * area
    circles = []
    current_area = 0.0
    attempts = 0
    while current_area < target_area and attempts < max_attempts:
        attempts += 1
        d = random.uniform(min_diameter, max_diameter)
        r = d / 2.0
        y_c = random.uniform(y_min + r, y_max - r)
        z_c = random.uniform(z_min + r, z_max - r)
        overlap = False
        for (y_i, z_i, r_i, _) in circles:
            dist = math.hypot(y_c - y_i, z_c - z_i)
            if dist <= (r + r_i):
                overlap = True
                break
        if not overlap:
            rind = random.uniform(rmin, rmax)
            circles.append((y_c, z_c, r, rind))
            current_area += np.pi * r * r
            c = circle(zz, yy, z_c, y_c, r)
            masks += c
            spheres += (rind - 1.33) * c
    return spheres, masks, circles


def generate_small_circles_in_circle(axs, space, cs, rr=(0.1, 1), ris=(1.36, 1.45), coverage_fraction=0.15, max_attempts=5000):
    zz, yy = axs
    rmin, rmax = ris
    small_spheres = np.zeros(space)
    small_masks = np.zeros(space)
    for circ in cs:
        x_center, y_center, R, RI = circ
        min_diameter, max_diameter = rr
        target_area_small = coverage_fraction * np.pi * (R**2)
        small_circles = []
        current_area = 0.0
        attempts = 0
        while current_area < target_area_small and attempts < max_attempts:
            attempts += 1
            d = random.uniform(min_diameter, max_diameter)
            r_s = d / 2.0
            y_s = random.uniform(x_center - (R - r_s), x_center + (R - r_s))
            z_s = random.uniform(y_center - (R - r_s), y_center + (R - r_s))
            dist_big = math.hypot(y_s - x_center, z_s - y_center)
            if dist_big > (R - r_s):
                continue
            overlap = False
            for (y_i, z_i, r_i) in small_circles:
                dist_small = math.hypot(y_s - y_i, z_s - z_i)
                if dist_small <= (r_s + r_i):
                    overlap = True
                    break
            if not overlap:
                small_circles.append((y_s, z_s, r_s))
                current_area += np.pi * (r_s**2)
                rind = random.uniform(rmin, rmax)
                c = circle(zz, yy, z_s, y_s, r_s)
                small_masks += c
                small_spheres += (rind - RI) * c
    return small_spheres, small_masks


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
    for jj in range(0, z_pxs):
        c = ifft(fftshift(b)) * np.exp(2.0j * np.pi * idm[jj, :] * _dz)
        b = fftshift(fft(c)) * np.exp(1.0j * kz * _dz)
        phs[jj, :] += np.angle(c)
        itn[jj, :] += np.abs(c) ** 2
    return phs, itn


if __name__ == '__main__':
    t = time.strftime("%Y%m%d%H%M")
    fd = r"C:\Users\ruizhe.lin\Desktop"

    zmin = 0
    zmax = 100
    ymin = -100
    ymax = 100
    dz = 0.02
    dy = 0.02
    zoom = 1
    Z, Y = np.mgrid[zmin / zoom:zmax / zoom:dz / zoom, ymin / zoom:ymax / zoom:dy / zoom]
    z_pts, y_pts = np.shape(Z)

    # lay = generate_layer(axs=(Z, Y), space=(z_pts, y_pts), axr=(40, 90), sig=64)
    # bls = generate_ri_spheres(axs=(Z, Y), space=(z_pts, y_pts), num=400, pzr=(50, 80), pyr=(-80, 80), rr=(0, 1), axr=(50, 90), sig=16)
    nr = np.zeros((z_pts, y_pts))
    nr[:2400, :] = 1.52 - 1.33
    # nr += lay
    # nr += bls

    spheres, masks, circles = generate_non_overlapping_circles(axs=(Z, Y), space=(z_pts, y_pts), pyr=(-100, 100),
                                                                   pzr=(50, 90), rr=(0.1, 1), ris=(1.35, 1.45),
                                                                   coverage_fraction=0.15, max_attempts=100000)
    # small_spheres, small_masks = generate_small_circles_in_circle(axs=(Z, Y), space=(z_pts, y_pts), cs=circles,
    #                                                               rr=(0.1, 1), ris=(1.38, 1.45), coverage_fraction=0.15,
    #                                                               max_attempts=10000)
    nr += spheres

    BeamSize = 40
    BAngle = 28 * np.pi / 180
    BeamOffset = 20
    E = plane_wave(Z[0, :], Y[0, :], 0, -BeamOffset, BeamSize, BAngle) + plane_wave(Z[0, :], Y[0, :], 0, BeamOffset,
                                                                                    BeamSize, -BAngle)

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

    tf.imwrite(str(os.path.join(fd, t + r"_refractive_index_distribution.tif")), nr)
    tf.imwrite(str(os.path.join(fd, t + r"_two_beam_interference_intensity.tif")), intensity)
    tf.imwrite(str(os.path.join(fd, t + r"_two_beam_interference_phase.tif")), phase)

    zo = 40
    E = spherical_wave(Z[0, :], Y[0, :], zo, 0, 1.0, 150)

    phase, intensity = propagation(space=(z_pts, y_pts), einput=E, idm=nr, dyz=(dy, dz))

    tf.imwrite(str(os.path.join(fd, t + r"_refractive_index_distribution.tif")), nr)
    tf.imwrite(str(os.path.join(fd, t + r"_one_beam_focus_intensity.tif")), intensity)
    tf.imwrite(str(os.path.join(fd, t + r"_one_beam_focus_phase.tif")), phase)
