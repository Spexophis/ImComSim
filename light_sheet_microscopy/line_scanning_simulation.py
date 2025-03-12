import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf
from numpy.fft import fftshift
from scipy.special import factorial
import photophysics_simulator

try:
    import cupy as cp

    cupy_available = True


    def fft2(data):
        data_gpu = cp.asarray(data)
        fft2_gpu = cp.fft.fft2(data_gpu)
        return fft2_gpu.get()


    def ifft2(data):
        data_gpu = cp.asarray(data)
        ifft2_gpu = cp.fft.ifft2(data_gpu)
        return ifft2_gpu.get()

except ImportError:
    cp = None
    cupy_available = False


    def fft2(data):
        return np.fft.fft2(data)


    def ifft2(data):
        return np.fft.ifft2(data)

matplotlib.use('Qt5Agg')
plt.ion()


class LSS:

    def __init__(self):
        self.wl = 0.488
        self.na = 1.4
        self.n2 = 1.512
        self.dx = 0.01
        self.nx = 512
        self.dp = 1 / (self.nx * self.dx)
        self.radius = (self.na / self.wl) / self.dp
        self.xv, self.yv = None, None
        self.rho = None
        self.phi = None
        self.bpp = None
        self.dsk = None
        self.msk = None
        self.stack = None

    def set_parameters(self, wl=None, na=None, dx=None, nx=None):
        if wl is not None:
            self.wl = wl
        if na is not None:
            self.na = na
        if dx is not None:
            self.dx = dx
        if nx is not None:
            self.nx = nx
        self.dp = 1 / (self.nx * self.dx)
        self.radius = (self.na / self.wl) / self.dp

    def mesh_grid(self):
        ox = self.nx / 2
        x = np.linspace(-ox, ox - 1, self.nx)
        self.xv, self.yv = np.meshgrid(x, x)
        self.rho = np.sqrt(self.xv ** 2 + self.yv ** 2)
        self.phi = np.arctan2(self.yv, self.xv)

    def flat_pupil(self, verbose=False):
        self.dsk = self.disc_array(radius=self.radius, origin=(0, 0))
        self.msk = np.zeros(self.dsk.shape)
        self.bpp = self.dsk * np.exp(2j * np.pi * self.dsk * np.zeros((self.nx, self.nx)))
        if verbose:
            plt.figure()
            plt.imshow(np.abs(self.bpp))
            plt.show()

    def round_masks(self, rds, origins, verbose=False):
        s = np.zeros(self.dsk.shape)
        for r, o in zip(rds, origins):
            m = self.disc_array(r, origin=o) * self.dsk
            s = np.logical_or(s != 0, m != 0).astype(int)
        self.msk = np.logical_or(s != 0, self.msk != 0).astype(int)
        if verbose:
            plt.figure()
            plt.imshow(self.msk)
            plt.show()

    def square_masks(self, mask_sizes, origins, verbose=False):
        s = np.zeros(self.dsk.shape)
        for sz, og in zip(mask_sizes, origins):
            y_origin, x_origin = og
            mask_height, mask_width = sz
            m = (np.abs(self.xv - x_origin) <= mask_width) * (np.abs(self.yv - y_origin) <= mask_height) * self.dsk
            s = np.logical_or(s != 0, m != 0).astype(int)
        self.msk = np.logical_or(s != 0, self.msk != 0).astype(int)
        if verbose:
            plt.figure()
            plt.imshow(self.msk)
            plt.show()

    def add_zernike(self, zas, verbose=False):
        zks = np.zeros((self.nx, self.nx))
        for zn, za in enumerate(zas):
            zks += za * self.zernike(zn)
        self.bpp *= self.msk * np.exp(2j * np.pi * self.msk * zks)
        if verbose:
            plt.figure()
            plt.imshow(np.angle(self.bpp))
            plt.show()

    def add_half(self, orient="x", verbose=False):
        if orient == "x":
            phase_map = 0.5 * (self.yv >= 0)
        else:
            phase_map = 0.5 * (self.xv >= 0)
        self.bpp *= self.msk * np.exp(2j * np.pi * self.msk * phase_map)
        if verbose:
            plt.figure()
            plt.imshow(np.angle(self.bpp))
            plt.show()

    def add_gradients(self, gd, verbose=False):
        gdy, gdx = gd
        dx = self.yv + self.radius
        dx = dx * (dx >= 0) * (dx <= 2 * self.radius)
        dx = gdx * self.wl * dx / dx.max()
        dy = self.xv + self.radius
        dy = dy * (dy >= 0) * (dy <= 2 * self.radius)
        dy = gdy * self.wl * dy / dy.max()
        phase_map = dx + dy
        self.bpp *= self.msk * np.exp(2j * np.pi * self.msk * phase_map)
        if verbose:
            plt.figure()
            plt.imshow(np.angle(self.bpp))
            plt.show()

    def add_roof(self, gd, verbose=False):
        gdy, gdx = gd
        dx = np.abs(self.yv)
        dx = dx * (dx >= 0) * (dx <= self.radius)
        dx = gdx * self.wl * dx / dx.max()
        dy = np.abs(self.xv)
        dy = dy * (dy >= 0) * (dy <= 2 * self.radius)
        dy = gdy * self.wl * dy / dy.max()
        phase_map = dx + dy
        self.bpp *= self.msk * np.exp(2j * np.pi * self.msk * phase_map)
        if verbose:
            plt.figure()
            plt.imshow(np.angle(self.bpp))
            plt.show()

    def add_cylindrical(self, focal_length, direction="x", verbose=False):
        k = 2 * np.pi / self.wl
        if direction == "x":
            phase_map = k * (self.yv ** 2) / (2 * focal_length)
        else:
            phase_map = k * (self.xv ** 2) / (2 * focal_length)
        if verbose:
            plt.figure()
            plt.imshow(phase_map)
            plt.show()
        self.bpp *= self.msk * np.exp(2j * np.pi * phase_map)

    def add_spherical(self, focal_length, origin=(0, 0), verbose=False):
        k = 2 * np.pi / self.wl
        phase_map = k * np.sqrt((self.xv - origin[0]) ** 2 + (self.yv - origin[1]) ** 2) / (2 * focal_length)
        if verbose:
            plt.figure()
            plt.imshow(phase_map)
            plt.show()
        self.bpp *= self.msk * np.exp(2j * np.pi * phase_map)

    def lateral_mode(self, pos):
        x, y = pos
        alpha = 2 * np.pi / self.nx / self.dx
        pwf = np.exp(1j * self.msk * alpha * (self.xv * x + self.yv * y))
        return pwf

    def axial_mode(self, d):
        ro = self.rho / self.radius
        mk = (ro <= 1.0).astype(np.float64)
        ph = mk * (self.n2 * d / self.wl) * np.sqrt(1 - (self.na * mk * ro / self.n2) ** 2)
        wf = np.exp(2j * np.pi * self.msk * ph)
        return wf

    def get_3d_psf(self, pos, start, stop, step):
        n_steps = int((stop - start) / step + 1)
        zarr = np.linspace(start / step, stop / step, n_steps).astype(np.int64)
        zarr = zarr[0:n_steps - 1]
        zarr = np.roll(zarr, int((n_steps - 1) / 2))
        self.stack = np.zeros((n_steps - 1, self.nx, self.nx))
        for m, z in enumerate(zarr):
            ph_z = self.axial_mode(z * step)
            ph_x = self.lateral_mode(pos)
            wf = self.msk * self.bpp * ph_z * ph_x
            self.stack[m] = np.abs(fft2(wf)) ** 2

    def generate_focal_array(self, xps, yps, start, stop, step, coh=True):
        n_steps = int((stop - start) / step + 1)
        zarr = np.linspace(start / step, stop / step, n_steps).astype(np.int64)
        zarr = zarr[0:n_steps - 1]
        zarr = np.roll(zarr, int((n_steps - 1) / 2))
        img = np.zeros((n_steps - 1, self.nx, self.nx))
        if coh:
            for m, z in enumerate(zarr):
                wft = np.zeros((self.nx, self.nx), dtype=np.complex64)
                ph_z = self.axial_mode(z * step)
                for xp in xps:
                    for yp in yps:
                        ph_x = self.lateral_mode((xp, yp))
                        wft += self.msk * self.bpp * ph_z * ph_x
                img[m] = np.abs(fft2(wft)) ** 2
        else:
            for xp in xps:
                for yp in yps:
                    self.get_3d_psf((xp, yp), start, stop, step)
                    img += self.stack
        return img

    def generate_dammann_grating(self, target_pattern, iterations=100):
        phase_pattern = np.random.uniform(0, 2 * np.pi, target_pattern.shape)  # Random continuous phase
        for _ in range(iterations):
            field = self.dsk * np.exp(1j * phase_pattern)
            far_field = np.fft.fftshift(fft2(np.fft.fftshift(field)))
            far_field_amplitude = np.sqrt(target_pattern)
            far_field = far_field_amplitude * np.exp(1j * np.angle(far_field))
            updated_field = np.fft.ifftshift(ifft2(np.fft.ifftshift(far_field)))
            phase_pattern = np.angle(updated_field)
        binary_phase_pattern = np.where(phase_pattern >= 0, 0, np.pi)
        if np.all(binary_phase_pattern == 0):
            raise ValueError("Output phase pattern is all zeros. Error in the optimization process.")
        else:
            return binary_phase_pattern

    def disc_array(self, radius=32.0, origin=(0, 0), dtype=int):
        y_origin, x_origin = origin
        rh = np.sqrt((self.xv - x_origin) ** 2 + (self.yv - y_origin) ** 2)
        disc = (rh < radius).astype(dtype)
        return disc

    @staticmethod
    def zernike_j_nm(j):
        n = int((-1. + np.sqrt(8 * (j - 1) + 1)) / 2.)
        p = (j - (n * (n + 1)) / 2.)
        k = n % 2
        m = int((p + k) / 2.) * 2 - k
        if m != 0:
            if j % 2 == 0:
                s = 1
            else:
                s = -1
            m *= s
        return n, m

    def zernike(self, jnm):
        n, m = self.zernike_j_nm(jnm + 1)
        if (n < 0) or (n < abs(m)) or (n % 2 != abs(m) % 2):
            raise ValueError("n and m are not valid Zernike indices")
        kmax = int((n - abs(m)) / 2)
        ro = self.rho / self.radius
        _R = 0
        _O = 0
        _C = 0
        if m == 0:
            _C = np.sqrt(n + 1)
        else:
            _C = np.sqrt(2 * n + 1)
        for k in range(kmax + 1):
            _R += (-1) ** k * factorial(n - k) / (
                    factorial(k) * factorial(0.5 * (n + abs(m)) - k) * factorial(0.5 * (n - abs(m)) - k)) * ro ** (
                          n - 2 * k)
        if m >= 0:
            _O = np.cos(m * self.phi)
        if m < 0:
            _O = - np.sin(m * self.phi)
        return _C * _R * _O * self.dsk

    def save_result(self, fd):
        t = time.strftime("%Y%m%d%H%M")
        tf.imwrite(str(os.path.join(fd, t + '_simulated_pupil.tif')),
                   np.array([self.msk, np.abs(self.bpp), np.angle(self.bpp)]))
        tf.imwrite(str(os.path.join(fd, t + '_simulated_image.tif')), fftshift(self.stack))


if __name__ == '__main__':
    p = LSS()
    p.mesh_grid()
    p.flat_pupil()
    p.msk = p.dsk
    # p.square_masks(mask_sizes=[(2, 10), (2, 10)], origins=[(0, -20), (0, 20)], verbose=True)
    # p.round_masks([1, 1], [(20, 0), (-20, 0)], verbose=True)
    p.get_3d_psf((0., 0.), -2, 2, 0.02)
    p.save_result(r"C:\Users\ruizhe.lin\Documents\Data")
