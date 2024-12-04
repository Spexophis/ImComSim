import os
import time
import numpy as np
from numpy.fft import fft2, fftshift
import tifffile as tf
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')
plt.ion()

class LSS:

    def __init__(self):
        self.wl = 0.488
        self.na = 1.4
        self.n2 = 1.512
        self.dx = 0.01
        self.nx = 1024
        self.dp = 1 / (self.nx * self.dx)
        self.radius = (self.na / self.wl) / self.dp
        self.xv, self.yv = None, None
        self.rho = None
        self.bpp = None
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

    def flat_pupil(self, origin=(0, 0), verbose=False):
        self.msk = self.disc_array(self.radius, origin=origin)
        self.bpp = self.msk * np.exp(2j * np.pi * np.zeros((self.nx, self.nx)))
        if verbose:
            plt.figure()
            plt.imshow(np.abs(self.bpp))
            plt.show()

    def round_masks(self, rds, origins, verbose=False):
        s = np.zeros(self.msk.shape)
        for r, o in zip(rds, origins):
            m = self.disc_array(r, origin=o)
            s = np.logical_or(s != 0, m != 0).astype(int)
        self.msk *= s
        self.bpp *= self.msk
        if verbose:
            plt.figure()
            plt.imshow(np.abs(self.bpp))
            plt.show()

    def square_masks(self, mask_sizes, origins, verbose=False):
        s = np.zeros(self.msk.shape)
        for sz, og in zip(mask_sizes, origins):
            y_origin, x_origin = og
            mask_height, mask_width = sz
            m = (np.abs(self.xv - x_origin) <= mask_width) * (np.abs(self.yv - y_origin) <= mask_height)
            s = np.logical_or(s != 0, m != 0).astype(int)
        self.msk *= s
        self.bpp *= self.msk
        if verbose:
            plt.figure()
            plt.imshow(np.abs(self.bpp))
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

    def get_2d_psf(self, x, y, coh=True):
        alpha = 2 * np.pi / self.nx / self.dx
        ph = self.msk * np.exp(1j * alpha * (self.xv * x + self.yv * y))
        wfp = ph * self.bpp
        if coh:
            psf_dist = np.fft.fft2(wfp)
            return psf_dist
        else:
            psf_dist = np.abs(np.fft.fft2(wfp)) ** 2
            psf_dist /= psf_dist.sum()
            return psf_dist

    def focus_mode(self, d):
        ro = self.rho / self.radius
        mk = (ro <= 1.0).astype(np.float64)
        wf = mk * (self.n2 * d / self.wl) * np.sqrt(1 - (self.na * mk * ro / self.n2) ** 2)
        return wf

    def get_3d_psf(self, start, stop, step, coh=True):
        n_steps = int((stop - start) / step + 1)
        zarr = np.linspace(start / step, stop / step, n_steps).astype(np.int64)
        zarr = zarr[0:n_steps - 1]
        zarr = np.roll(zarr, int((n_steps - 1) / 2))
        self.stack = np.zeros((n_steps - 1, self.nx, self.nx))
        for m, z in enumerate(zarr):
            ph = self.focus_mode(z * step)
            wf = self.bpp * np.exp(2j * np.pi * ph)
            if coh:
                self.stack[m] = fft2(wf)
            else:
                self.stack[m] = np.abs(fft2(wf)) ** 2

    def disc_array(self, radius=32.0, origin=(0, 0), dtype=np.float64):
        y_origin, x_origin = origin
        rh = np.sqrt((self.xv - x_origin) ** 2 + (self.yv - y_origin) ** 2)
        disc = (rh < radius).astype(dtype)
        return disc

    def save_result(self, fd):
        t = time.strftime("%Y%m%d%H%M")
        tf.imwrite(str(os.path.join(fd, t + '_simulated_pupil.tif')), np.array([np.abs(self.bpp), np.angle(self.bpp)]))
        tf.imwrite(str(os.path.join(fd, t + '_simulated_image.tif')), fftshift(self.stack))

if __name__ == '__main__':
    p = LSS()
    p.mesh_grid()
    p.flat_pupil()
    p.get_3d_psf(-2, 2, 0.02)
    p.save_result(r"C:\Users\ruizhe.lin\Documents\Data")
