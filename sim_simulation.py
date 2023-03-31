"""
This is the three-dimensional structured illumination microscopy simulation.
First created on Fri May 24 10:15:59 2013 @author: Kner
"""

import numpy as np
import numpy.random as rd
from scipy.special import factorial
import tifffile as tf
import time
import concurrent.futures


class sim():

    def __init__(self):

        self.nxh = 64
        self.nyh = 64
        self.nzh = 16
        self.dx = 0.08  # um
        self.dy = 0.08  # um
        self.dz = 0.16  # um
        self.wl = 0.505  # um
        self.na = 1.4
        self.n2 = 1.512
        self.sp = 0.5  # um
        self.number_of_angles = 3
        self.number_of_phases = 5
        self.number_of_fluorophores = None
        self.focal_plane = self.nzh
        self.I = 512
        self.image_stack = None
        self.out = None
        self.cam_offset = 20.0

    def _get_point_objects(self, number_of_fluorophores, singleplane=True):
        self.number_of_fluorophores = number_of_fluorophores
        n = self.number_of_fluorophores
        self.xps = (self.dx * self.nxh * 2) * (0.8 * rd.rand(n) + 0.1)
        self.yps = (self.dx * self.nyh * 2) * (0.8 * rd.rand(n) + 0.1)
        if singleplane:
            self.zps = np.zeros(n)
        else:
            self.zps = (self.dz * self.nzh * 2) * (0.8 * rd.rand(n) - 0.4)

    def _get_line_objects(self, number_of_lines, singleplane=True):
        number_of_fluorophores_per_line = np.random.randint(128, 512, number_of_lines)
        x_start = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_lines) + 0.1)
        y_start = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_lines) + 0.1)
        x_end = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_lines) + 0.1)
        y_end = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_lines) + 0.1)
        xps = np.zeros(1)
        yps = np.zeros(1)
        if singleplane:
            z_start = np.zeros(number_of_lines)
            z_end = np.zeros(number_of_lines)
            zps = np.zeros(1)
        else:
            z_start = (self.dz * self.nzh * 2) * (0.4 * np.random.rand(number_of_lines) - 0.4)
            z_end = (self.dz * self.nzh * 2) * (0.4 * np.random.rand(number_of_lines))
            zps = np.zeros(1)
        for i in range(number_of_lines):
            xps = np.concatenate((xps, np.linspace(x_start[i], x_end[i], number_of_fluorophores_per_line[i])), axis=0)
            yps = np.concatenate((yps, np.linspace(y_start[i], y_end[i], number_of_fluorophores_per_line[i])), axis=0)
            zps = np.concatenate((zps, np.linspace(z_start[i], z_end[i], number_of_fluorophores_per_line[i])), axis=0)
        self.xps = np.delete(xps, 0)
        self.yps = np.delete(yps, 0)
        self.zps = np.delete(zps, 0)
        self.number_of_fluorophores = number_of_fluorophores_per_line.sum()

    def _get_both_objects(self, number_of_lines, number_of_fluorophores):
        n = number_of_fluorophores
        xps = (self.dx * self.nxh * 2) * (0.8 * rd.rand(n) + 0.1)
        yps = (self.dx * self.nyh * 2) * (0.8 * rd.rand(n) + 0.1)
        zps = (self.dz * self.nzh * 2) * (0.8 * rd.rand(n) - 0.4)
        number_of_fluorophores_per_line = np.random.randint(128, 512, number_of_lines)
        x_start = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_lines) + 0.1)
        y_start = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_lines) + 0.1)
        x_end = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_lines) + 0.1)
        y_end = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_lines) + 0.1)
        z_start = (self.dz * self.nzh * 2) * (0.4 * np.random.rand(number_of_lines) - 0.4)
        z_end = (self.dz * self.nzh * 2) * (0.4 * np.random.rand(number_of_lines))
        for i in range(number_of_lines):
            xps = np.concatenate((xps, np.linspace(x_start[i], x_end[i], number_of_fluorophores_per_line[i])), axis=0)
            yps = np.concatenate((yps, np.linspace(y_start[i], y_end[i], number_of_fluorophores_per_line[i])), axis=0)
            zps = np.concatenate((zps, np.linspace(z_start[i], z_end[i], number_of_fluorophores_per_line[i])), axis=0)
        self.xps = xps
        self.yps = yps
        self.zps = zps
        self.number_of_fluorophores = number_of_fluorophores_per_line.sum() + number_of_fluorophores

    def _get_pupil(self, zarr=None):
        dp = 1 / (self.nxh * 2 * self.dx)
        radius = (self.na / self.wl) / dp
        msk = self._shift(self._disc_array(shape=(self.nxh * 2, self.nyh * 2), radius=radius)) / np.sqrt(np.pi * radius**2) / (self.nxh*2)
        phi = np.zeros((self.nxh * 2, self.nyh * 2))
        if zarr is not None:
            for z in range(len(zarr)):
                n, m = self._zernike_nm(z + 1)
                phi += zarr[z] * self._zernike(n, m, self.rho, self.phi)
            self.wf = self.wf * np.exp(1j * phi).astype(np.complex64)
        else:
            self.wf = msk * np.exp(1j * phi).astype(np.complex64)

    def _add_psf_2d(self, x, y, I):
        nx = self.nxh * 2
        ny = self.nyh * 2
        alpha = 2 * np.pi / nx / self.dx
        gxy = lambda m, n: np.exp(1j * alpha * (m * x + n * y)).astype(np.complex64)
        ph = np.fft.fftshift(np.fromfunction(gxy, (nx, ny), dtype=np.float32))-
        wfp = np.sqrt(I) * ph * self.wf
        return np.abs(np.fft.fft2(wfp)) ** 2

    def _add_psf_3d(self, x, y, z, I):
        nx = self.nxh * 2
        ny = self.nyh * 2
        ph = np.ones((1, nx, ny), dtype=np.complex64)
        alpha = 2 * np.pi / nx / self.dx
        gxy = lambda m, n: np.exp(1j * alpha * (m * x + n * y)).astype(np.complex64)
        ph[0, :, :] = np.fft.fftshift(np.fromfunction(gxy, (nx, ny)))
        wfp = np.sqrt(I) * ph * np.exp(1j * self._focus_mode(z)) * self.wf
        return np.abs(np.fft.fft2(wfp)) ** 2

    def _focus_mode(self, w=0):  # wavefront phase for focusing
        """ focus mode, d is depth in microns, nap is num. ap.
        focuses, with depth correction """
        nap = self.na
        n2 = self.n2
        wl = self.wl
        if nap > n2:
            raise "Numerical aperture cannot be greater than n2!"
        dp = 1 / (self.nxh * 2 * self.dx)
        radius = self.dx * (self.na / self.wl) / dp
        msk = self._disc_array([0, 0], radius)
        sinphim = (nap / n2)
        rho = msk * self.rho / radius
        return np.fft.fftshift(2 * np.pi * msk * n2 * w * np.sqrt(1 - (sinphim * rho) ** 2) / wl)

    def _get_one_img_2d(self, indices):
        angle = indices[0] * (2 * np.pi / self.number_of_angles)
        phase = indices[1] * (2 * np.pi / self.number_of_phases)
        nx = self.nxh * 2
        ny = self.nyh * 2
        kx = 2 * np.pi * np.cos(angle) / self.sp
        ky = 2 * np.pi * np.sin(angle) / self.sp
        img = self.cam_offset * np.ones((nx, ny))
        for m in range(self.number_of_fluorophores):
            Ip = self.I * 0.5 * (1 + np.cos(kx * self.xps[m] + ky * self.yps[m] + phase))
            img += self._add_psf_2d(self.xps[m], self.yps[m], Ip)
        self.out[indices[0] * self.number_of_phases + indices[1], :, :] = rd.poisson(img)
        return 'done', 'angle', indices[0], 'phase', indices[1]

    def _get_one_img_3d(self, indices):
        angle = indices[0] * (2 * np.pi / self.number_of_angles)
        phase = indices[1] * (2 * np.pi / self.number_of_phases)
        nx = self.nxh * 2
        ny = self.nyh * 2
        nz = self.nzh * 2
        phim = self.na / self.n2
        kx = 2 * np.pi * np.cos(angle) / self.sp
        ky = 2 * np.pi * np.sin(angle) / self.sp
        dkz = (np.pi / self.sp) * (1 - np.sqrt(1 - phim ** 2))
        imgstack = np.zeros((nz, nx, ny))
        for zp in range(nz):
            zplane = self.dz * (zp - self.focal_plane)
            img = self.cam_offset * np.ones((1, nx, ny))
            for m in range(self.number_of_fluorophores):
                # Ip = Iph*0.5*(1+N.cos(kx*self.xps[m]+ky*self.yps[m]+phase))
                cs2xy = np.cos(2 * kx * self.xps[m] + 2 * ky * self.yps[m] + 2 * phase)
                csxy = np.cos(kx * self.xps[m] + ky * self.yps[m] + phase)
                csz = np.cos(dkz * (self.zps[m] - zplane))
                Ip = self.I * (3 + 2 * cs2xy + 4 * csz * csxy)
                img += self._add_psf_3d(self.xps[m], self.yps[m], self.zps[m] - zplane, Ip)
            imgstack[zp, :, :] = img[0]  # fft2(self.imgf).real
        self.out[indices[0], indices[1], :, :, :] = rd.poisson(imgstack)
        return 'done', 'angle', indices[0], 'phase', indices[1]

    def _run_all_angles_2d(self, cocurrent_method='threadpool'):
        nx = self.nxh * 2
        ny = self.nxh * 2
        self.number_of_angles = 3
        self.number_of_phases = 3
        sz = self.number_of_angles * self.number_of_phases
        self.out = np.zeros((sz, nx, ny), dtype=np.float32)
        indices_list = [(n, m) for n in range(self.number_of_angles) for m in range(self.number_of_phases)]
        if cocurrent_method == 'threadpool':
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._get_one_img_2d, indices) for indices in indices_list]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                print(future.result())
        if cocurrent_method == 'processpool':
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(self._get_one_img_2d, indices) for indices in indices_list]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                print(future.result())

    def _run_all_angles_3d(self, cocurrent_method='threadpool'):
        nx = self.nxh * 2
        ny = self.nxh * 2
        nz = self.nzh * 2
        self.number_of_angles = 3
        self.number_of_phases = 5
        sz = self.number_of_angles * self.number_of_phases * nz
        self.out = np.zeros((self.number_of_angles, self.number_of_phases, nz, nx, ny))
        indices_list = [(n, m) for n in range(self.number_of_angles) for m in range(self.number_of_phases)]
        if cocurrent_method == 'threadpool':
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._get_one_img_3d, indices) for indices in indices_list]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                print(future.result())
        if cocurrent_method == 'processpool':
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(self._get_one_img_3d, indices) for indices in indices_list]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                print(future.result())
        self.image_stack = self.out.swapaxes(1, 2).swapaxes(0, 1).reshape(sz, nx, ny)

    def _save_result_2d(self):
        t = time.strftime("%Y%m%d%H%M")
        path = t + '_'
        tf.imwrite(path + 'si2d_simulation_image_stack.tif', self.out.astype(np.uint16),
                   photometric='minisblack',
                   metadata={'number of phases': self.number_of_phases,
                             'number of angles': self.number_of_angles,
                             'pixel size (xy)': self.dx,
                             'wavelength': self.wl, 'numerical aperture': self.na,
                             'pattern spacing': self.sp})

    def _save_result_3d(self):
        t = time.strftime("%Y%m%d%H%M")
        path = t + '_'
        tf.imwrite(path + 'si3d_simulation_image_stack.tif', self.image_stack.astype(np.uint16),
                   photometric='minisblack',
                   metadata={'number of phases': self.number_of_phases,
                             'number of angles': self.number_of_angles,
                             'pixel size (xy)': self.dx, 'pixel size (z)': self.dz,
                             'wavelength': self.wl, 'numerical aperture': self.na,
                             'pattern spacing': self.sp})

    def _image_grid_polar(self, x, y):
        return np.sqrt(x ** 2 + y ** 2), np.arctan2(y, x)

    def _disc_array(self, shape=(128, 128), radius=64):
        nx, ny = shape
        ox = nx / 2
        oy = ny / 2
        x = np.linspace(-ox, ox - 1, nx)
        y = np.linspace(-oy, oy - 1, ny)
        X, Y = np.meshgrid(x, y)
        rho = np.sqrt(X ** 2 + Y ** 2)
        return rho <= radius

    def _radial_Array(self, shape=(128, 128), f=None, origin=None):
        nx = shape[0]
        ny = shape[1]
        ox = nx / 2
        oy = ny / 2
        x = np.linspace(-ox, ox - 1, nx)
        y = np.linspace(-oy, oy - 1, ny)
        X, Y = np.meshgrid(x, y)
        rho = np.sqrt(X ** 2 + Y ** 2)
        rarr = f(rho)
        if not origin == None:
            s0 = origin[0] - nx / 2
            s1 = origin[1] - ny / 2
            rarr = np.roll(np.roll(rarr, int(s0), 0), int(s1), 1)
        return rarr

    def _shift(self, arr, shifts=None):
        if shifts is None:
            shifts = np.array(arr.shape) / 2
        if len(arr.shape) == len(shifts):
            for m, p in enumerate(shifts):
                arr = np.roll(arr, int(p), m)
        return arr

    def _zernike_j_nm(self, j):
        if j < 1:
            raise ValueError("j must be a positive integer")
        n = 0
        while j > n:
            n += 1
            j -= n
        m = -2 * j + n
        if n % 2 == 0:
            m = -m
        return n, m

    def _zernike(self, n, m, radius=64, shape=(128, 128), origin=None):
        if (n < 0) or (n < abs(m)) or (n % 2 != abs(m) % 2):
            raise ValueError("n and m are not valid Zernike indices")
        if m < 0:
            return ((-1) ** ((n - abs(m)) / 2)) * self._zernike(n, -m, radius, shape, origin)
        # Compute the polynomial.
        
        rho, phi = self._image_grid_polar(xv, yv)
        kmax = int((n - abs(m)) / 2)
        summation = 0
        for k in range(kmax + 1):
            summation += ((-1) ** k * factorial(n - k) /
                          (factorial(k) * factorial(0.5 * (n + abs(m)) - k) *
                           factorial(0.5 * (n - abs(m)) - k)) *
                          rho ** (n - 2 * k))
        return summation * np.cos(m * phi)


if __name__ == '__main__':
    s = sim()
    # s._get_line_objects(4, False)
    # s._get_point_objects(512, True)
    # s._get_both_objects(8, 512)
    s._get_pupil()
    s._get_point_objects(512, True)
    s._run_all_angles_2d()
    s._save_result_2d()
