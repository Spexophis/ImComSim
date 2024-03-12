"""
This script generates simulated data of the 2-dimensional nonlinear structured illumination microscopy.
Ruizhe Lin
2024-01-12
"""

import concurrent.futures
import time

import numpy as np
import numpy.random as rd
import tifffile as tf
from scipy.special import factorial

import photophysics_simulator as phs


class NLSIM:

    def __init__(self):

        self.nxh = 128
        self.nyh = 128
        self.nzh = 16
        self.dx = 0.08  # um
        self.dy = 0.08  # um
        self.dz = 0.16  # um
        self.wl = 0.505  # um
        self.na = 1.4
        self.n2 = 1.512
        self.sp = 0.24  # um
        self.number_of_angles = 0
        self.number_of_phases = 0
        self.number_of_fluorophores = 0
        self.xps = np.array([])
        self.yps = np.array([])
        self.zps = np.array([])
        self.sw = np.array([])
        self.focal_plane = self.nzh
        self.I = None
        self.out = None
        self.cam_offset = 80.0
        self.pw_act = 0.2  # kW/cm2
        self.pw_off = 5.0
        self.pw_read = 1.0
        self.expo_act = 0.1  # ms
        self.expo_off = 2.0
        self.expo_read = 1.0
        self.rsEGFP2_on_state = phs.NegativeSwitchers(extincion_coeff_on=[5260, 51560],
                                                      extincion_coeff_off=[22000, 60],
                                                      wavelength=[405, 488],
                                                      lifetime_on=1.6E-6,
                                                      lifetime_off=20E-9,
                                                      qy_cis_to_trans_anionic=1.65E-2,
                                                      qy_trans_to_cis_neutral=0.33,
                                                      qy_fluorescence_on=0.35,
                                                      initial_populations=[0, 0, 1, 0])
        self.rsEGFP2_off_state = phs.NegativeSwitchers(extincion_coeff_on=[5260, 51560],
                                                       extincion_coeff_off=[22000, 60],
                                                       wavelength=[405, 488],
                                                       lifetime_on=1.6E-6,
                                                       lifetime_off=20E-9,
                                                       qy_cis_to_trans_anionic=1.65E-2,
                                                       qy_trans_to_cis_neutral=0.33,
                                                       qy_fluorescence_on=0.35,
                                                       initial_populations=[0, 0, 1, 0])

    def get_point_objects(self, number_of_dots):
        coords_x = (self.dx * self.nxh * 2) * (0.8 * rd.rand(number_of_dots) + 0.1)
        coords_y = (self.dy * self.nyh * 2) * (0.8 * rd.rand(number_of_dots) + 0.1)
        coords_z = (self.dz * self.nzh * 2) * (0.8 * rd.rand(number_of_dots) - 0.4)
        return number_of_dots, coords_x, coords_y, coords_z

    def get_line_objects(self, number_of_lines):
        number_of_fluorophores_per_line = np.random.randint(128, 512, number_of_lines)
        x_start = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_lines) + 0.1)
        y_start = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_lines) + 0.1)
        x_end = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_lines) + 0.1)
        y_end = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_lines) + 0.1)
        z_start = (self.dz * self.nzh * 2) * (0.4 * np.random.rand(number_of_lines) - 0.4)
        z_end = (self.dz * self.nzh * 2) * (0.4 * np.random.rand(number_of_lines))
        coords_x = np.array([])
        coords_y = np.array([])
        coords_z = np.array([])
        for i in range(number_of_lines):
            coords_x = np.concatenate((coords_x, np.linspace(x_start[i], x_end[i], number_of_fluorophores_per_line[i])))
            coords_y = np.concatenate((coords_y, np.linspace(y_start[i], y_end[i], number_of_fluorophores_per_line[i])))
            coords_z = np.concatenate((coords_z, np.linspace(z_start[i], z_end[i], number_of_fluorophores_per_line[i])))
        return number_of_fluorophores_per_line.sum(), coords_x, coords_y, coords_z

    def get_polynomial_objects(self, number_of_polynomials):
        number_of_fluorophores_per_polynomial = np.random.randint(128, 512, number_of_polynomials)
        coords_x = np.array([])
        coords_y = np.array([])
        coords_z = np.array([])
        x_start = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_polynomials) + 0.1)
        y_start = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_polynomials) + 0.1)
        x_end = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_polynomials) + 0.1)
        y_end = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_polynomials) + 0.1)
        z_start = (self.dz * self.nzh * 2) * (0.4 * np.random.rand(number_of_polynomials) - 0.4)
        z_end = (self.dz * self.nzh * 2) * (0.4 * np.random.rand(number_of_polynomials))
        for i in range(number_of_polynomials):
            degrees = np.random.randint(2, 8, 3)
            poly_x = np.poly1d(np.random.uniform(-1, 1, size=(degrees[0] + 1)))
            poly_y = np.poly1d(np.random.uniform(-1, 1, size=(degrees[1] + 1)))
            poly_z = np.poly1d(np.random.uniform(-1, 1, size=(degrees[2] + 1)))
            t = np.linspace(-1, 1, number_of_fluorophores_per_polynomial[i])
            x = poly_x(t)
            y = poly_y(t)
            z = poly_z(t)
            x = self.normalize(x, (x_start[i], x_end[i]))
            y = self.normalize(y, (y_start[i], y_end[i]))
            z = self.normalize(z, (z_start[i], z_end[i]))
            coords_x = np.concatenate((coords_x, x))
            coords_y = np.concatenate((coords_y, y))
            coords_z = np.concatenate((coords_z, z))
        return number_of_fluorophores_per_polynomial.sum(), coords_x, coords_y, coords_z

    def get_curve_objects(self, number_of_curves):
        number_of_fluorophores_per_curve = np.random.randint(128, 512, number_of_curves)
        coords_x = np.array([])
        coords_y = np.array([])
        coords_z = np.array([])
        x_start = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_curves) + 0.1)
        y_start = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_curves) + 0.1)
        x_end = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_curves) + 0.1)
        y_end = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_curves) + 0.1)
        z_start = (self.dz * self.nzh * 2) * (0.4 * np.random.rand(number_of_curves) - 0.4)
        z_end = (self.dz * self.nzh * 2) * (0.4 * np.random.rand(number_of_curves))
        for c in range(number_of_curves):
            num_coeffs = np.random.randint(2, 8)
            coeffs = np.random.uniform(-1, 1, (3, num_coeffs))
            phases = np.random.uniform(0, 2 * np.pi, (3, num_coeffs))
            t = np.linspace(0, 2 * np.pi, number_of_fluorophores_per_curve[c])
            x = self.fc(t, coeffs[0], phases[0])
            y = self.fc(t, coeffs[1], phases[1])
            z = self.fc(t, coeffs[2], phases[2])
            x = self.normalize(x, (x_start[c], x_end[c]))
            y = self.normalize(y, (y_start[c], y_end[c]))
            z = self.normalize(z, (z_start[c], z_end[c]))
            coords_x = np.concatenate((coords_x, x))
            coords_y = np.concatenate((coords_y, y))
            coords_z = np.concatenate((coords_z, z))
        return number_of_fluorophores_per_curve.sum(), coords_x, coords_y, coords_z

    @staticmethod
    def fc(t, cs, phase):
        return sum(a * np.sin(i * t + p) for i, (a, p) in enumerate(zip(cs, phase), 1))

    @staticmethod
    def normalize(coord, range_):
        r = np.max(coord) - np.min(coord)
        if r == 0:
            return np.full(coord.shape, (range_[1] + range_[0]) / 2)
        else:
            return (coord - np.min(coord)) / (np.max(coord) - np.min(coord)) * np.abs(range_[1] - range_[0]) + np.min(
                range_)

    def get_objects(self, number_of_dots=None, number_of_lines=None, number_of_polynomials=None, number_of_curves=None):
        if number_of_dots is not None:
            _n, _x, _y, _z = self.get_point_objects(number_of_dots)
            self.number_of_fluorophores = self.number_of_fluorophores + _n
            self.xps = np.concatenate((self.xps, _x))
            self.yps = np.concatenate((self.yps, _y))
            self.zps = np.concatenate((self.zps, _z))
        if number_of_lines is not None:
            _n, _x, _y, _z = self.get_line_objects(number_of_lines)
            self.number_of_fluorophores = self.number_of_fluorophores + _n
            self.xps = np.concatenate((self.xps, _x))
            self.yps = np.concatenate((self.yps, _y))
            self.zps = np.concatenate((self.zps, _z))
        if number_of_polynomials is not None:
            _n, _x, _y, _z = self.get_polynomial_objects(number_of_polynomials)
            self.number_of_fluorophores = self.number_of_fluorophores + _n
            self.xps = np.concatenate((self.xps, _x))
            self.yps = np.concatenate((self.yps, _y))
            self.zps = np.concatenate((self.zps, _z))
        if number_of_curves is not None:
            _n, _x, _y, _z = self.get_curve_objects(number_of_curves)
            self.number_of_fluorophores = self.number_of_fluorophores + _n
            self.xps = np.concatenate((self.xps, _x))
            self.yps = np.concatenate((self.yps, _y))
            self.zps = np.concatenate((self.zps, _z))
        self.sw = np.ones(self.number_of_fluorophores)

    def get_pupil(self, zarr=None):
        dp = 1 / (self.nxh * 2 * self.dx)
        radius = (self.na / self.wl) / dp
        msk = self._shift(self._disc_array(shape=(self.nxh * 2, self.nyh * 2), radius=radius)) / np.sqrt(
            np.pi * radius ** 2) / (self.nxh * 2)
        phi = np.zeros((self.nxh * 2, self.nyh * 2))
        self.wf = msk * np.exp(1j * phi).astype(np.complex64)
        if zarr is not None:
            for z in range(len(zarr)):
                n, m = self._zernike_j_nm(z + 1)
                phi += zarr[z] * self._zernike(n, m, radius=radius, shape=(self.nxh * 2, self.nyh * 2))
            self.wf *= np.exp(1j * phi).astype(np.complex64)

    def _on_probability(self, pw=0.5, expo=1.0):
        on_switching_pulse = phs.ModulatedLasers(wavelengths=[405, 488], power_densities=[pw, 0.0],
                                                 pulse_widths=[expo, 0.0], t_start=[1, 2.5], dwell_time=30)
        on_switching_experiment = phs.Experiment(illumination=on_switching_pulse, fluorophore=self.rsEGFP2_off_state)
        populations = on_switching_experiment.solve_kinetics(0.01)
        p_on = populations[-1, 2]
        return 0 if rd.random() < p_on else 1

    def _off_probability(self, pw=0.5, expo=1.0):
        off_switching_pulse = phs.ModulatedLasers(wavelengths=[405, 488], power_densities=[0.0, pw],
                                                  pulse_widths=[0.0, expo], t_start=[1, 2.5], dwell_time=30)
        off_switching_experiment = phs.Experiment(illumination=off_switching_pulse, fluorophore=self.rsEGFP2_on_state)
        populations = off_switching_experiment.solve_kinetics(0.01)
        p_off = populations[-1, 0]
        return 0 if rd.random() > p_off else 1

    def _add_psf_2d(self, x, y, I):
        nx = self.nxh * 2
        ny = self.nyh * 2
        alpha = 2 * np.pi / nx / self.dx
        gxy = lambda m, n: np.exp(1j * alpha * (m * x + n * y)).astype(np.complex64)
        ph = self._shift(np.fromfunction(gxy, (nx, ny), dtype=np.float32))
        wfp = np.sqrt(I) * ph * self.wf
        return np.abs(np.fft.fft2(wfp)) ** 2

    def _get_one_img_2d(self, indices):
        nx = self.nxh * 2
        ny = self.nyh * 2
        self.out[indices[0] * self.number_of_phases + indices[1], :, :] = self.cam_offset + np.zeros((nx, ny))
        for m in range(self.number_of_fluorophores):
            if self.sw[m]:
                I_off = 1 + np.cos(
                    self.kx[indices[0]] * self.xps[m] + self.ky[indices[0]] * self.yps[m] + np.pi + self.phase[
                        indices[1]])
                self.sw[m] = self.sw[m] * self._off_probability(I_off * self.pw_off, self.expo_off)
                if self.sw[m]:
                    I_read = self.I * 0.5 * (1 + np.cos(
                        self.kx[indices[0]] * self.xps[m] + self.ky[indices[0]] * self.yps[m] + self.phase[indices[1]]))
                    self.sw[m] = self.sw[m] * self._off_probability(I_read * self.pw_read, self.expo_read)
                    self.out[indices[0] * self.number_of_phases + indices[1], :, :] += self._add_psf_2d(self.xps[m],
                                                                                                        self.yps[m],
                                                                                                        I_read)
            if self.sw[m] == 0:
                self.sw[m] = self._on_probability(self.pw_act, self.expo_act)
        self.out[indices[0] * self.number_of_phases + indices[1], :, :] = rd.poisson(
            self.out[indices[0] * self.number_of_phases + indices[1], :, :])
        return 'done', 'angle', indices[0], 'phase', indices[1]

    def nlsim_2d(self, nang=7, nph=7, I=1600, parallel=True):
        nx = self.nxh * 2
        ny = self.nxh * 2
        self.number_of_angles = nang
        self.number_of_phases = nph
        self.I = I
        self.angle = [n * (2 * np.pi / self.number_of_angles) for n in range(self.number_of_angles)]
        self.phase = [n * (2 * np.pi / self.number_of_phases) for n in range(self.number_of_phases)]
        self.kx = 2 * np.pi * np.cos(self.angle) / self.sp
        self.ky = 2 * np.pi * np.sin(self.angle) / self.sp
        self.out = np.zeros((self.number_of_angles * self.number_of_phases, nx, ny))
        indices_list = [(n, m) for n in range(self.number_of_angles) for m in range(self.number_of_phases)]
        if parallel:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._get_one_img_2d, indices) for indices in indices_list]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                print(future.result())
        else:
            for indices in indices_list:
                results = self._get_one_img_2d(indices)
                print(results)

    def save_result_2d(self):
        t = time.strftime("%Y%m%d%H%M")
        path = t + '_'
        tf.imwrite(path + 'nlsim2d_simulation_image_stack.tif', self.out, photometric='minisblack',
                   metadata={'number of phases': self.number_of_phases,
                             'number of angles': self.number_of_angles,
                             'pixel size (xy)': self.dx,
                             'wavelength': self.wl, 'numerical aperture': self.na,
                             'pattern spacing': self.sp})

    def _add_psf_3d(self, x, y, z, I):
        nx = self.nxh * 2
        ny = self.nyh * 2
        alpha = 2 * np.pi / nx / self.dx
        gxy = lambda m, n: np.exp(1j * alpha * (m * x + n * y)).astype(np.complex64)
        ph = self._shift(np.fromfunction(gxy, (nx, ny), dtype=np.float32))
        df = np.exp(1j * self._focus_mode(z)).astype(np.complex64)
        wfp = np.sqrt(I) * ph * df * self.wf
        return np.abs(np.fft.fft2(wfp)) ** 2

    def _focus_mode(self, w=0):  # wavefront phase for focusing
        """ focus mode, d is depth in microns, nap is num. ap.
        focuses, with depth correction """
        na = self.na
        n2 = self.n2
        wl = self.wl
        if na > n2:
            raise "Numerical aperture cannot be greater than n2!"
        dp = 1 / (self.nxh * 2 * self.dx)
        radius = (self.na / self.wl) / dp
        sinphim = na / n2
        msk = self._disc_array(shape=(self.nxh * 2, self.nyh * 2), radius=radius, origin=(0, 0))
        rho = msk * self._radial_Array(shape=(self.nxh * 2, self.nyh * 2), f=lambda x: x, origin=(0, 0)) / radius
        return 2 * np.pi * msk * n2 * w * np.sqrt(1 - (sinphim * rho) ** 2) / wl

    def _get_one_img_3d(self, indices):
        nx = self.nxh * 2
        ny = self.nyh * 2
        nz = self.nzh * 2
        self.out[indices[0], indices[1], :, :, :] = self.cam_offset + np.zeros((nz, nx, ny))
        for zp in range(nz):
            zplane = self.dz * (zp - self.focal_plane)
            for m in range(self.number_of_fluorophores):
                cs2xy = np.cos(
                    self.kx[indices[0]] * self.xps[m] + self.ky[indices[0]] * self.yps[m] + 2 * self.phase[
                        indices[1]])
                csxy = np.cos(
                    0.5 * self.kx[indices[0]] * self.xps[m] + 0.5 * self.ky[indices[0]] * self.yps[m] + self.phase[
                        indices[1]])
                csz = np.cos(self.kz * (self.zps[m] - zplane))
                I_off = self.I * (3 + 2 * cs2xy + 4 * csz * csxy)
                sw = self._off_probability(I_off)
                cs2xy = np.cos(
                    self.kx[indices[0]] * self.xps[m] + self.ky[indices[0]] * self.yps[m] + 2 * self.phase[
                        indices[1]])
                csxy = np.cos(
                    0.5 * self.kx[indices[0]] * self.xps[m] + 0.5 * self.ky[indices[0]] * self.yps[m] + self.phase[
                        indices[1]])
                csz = np.cos(self.kz * (self.zps[m] - zplane))
                I_read = self.I * (3 + 2 * cs2xy + 4 * csz * csxy)
                self.out[indices[0], indices[1], zp, :, :] += sw * self._add_psf_3d(self.xps[m], self.yps[m],
                                                                                    self.zps[m] - zplane, I_read)
        self.out[indices[0], indices[1], :, :, :] = rd.poisson(self.out[indices[0], indices[1], :, :, :])
        return 'done', 'angle', indices[0], 'phase', indices[1]

    def nlsim_3d(self, nang=3, nph=5, I=1000, parallel=True):
        nx = self.nxh * 2
        ny = self.nxh * 2
        nz = self.nzh * 2
        self.number_of_angles = nang
        self.number_of_phases = nph
        self.I = I
        self.angle = [n * (2 * np.pi / self.number_of_angles) for n in range(self.number_of_angles)]
        self.phase = [n * (2 * np.pi / self.number_of_phases) for n in range(self.number_of_phases)]
        self.kx = 2 * np.pi * np.cos(self.angle) / self.sp
        self.ky = 2 * np.pi * np.sin(self.angle) / self.sp
        phim = self.na / self.n2
        self.kz = (2 * np.pi / self.sp) * (1 - np.sqrt(1 - phim ** 2))
        self.out = np.zeros((self.number_of_angles, self.number_of_phases, nz, nx, ny))
        indices_list = [(n, m) for n in range(self.number_of_angles) for m in range(self.number_of_phases)]
        if parallel:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._get_one_img_3d, indices) for indices in indices_list]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                print(future.result())
        else:
            for indices in indices_list:
                results = self._get_one_img_3d(indices)
                print(results)

    def save_result_3d(self):
        t = time.strftime("%Y%m%d%H%M")
        path = t + '_'
        tf.imwrite(path + 'nlsim3d_simulation_image_stack.tif', self.out, photometric='minisblack',
                   metadata={'number of phases': self.number_of_phases,
                             'number of angles': self.number_of_angles,
                             'pixel size (xy)': self.dx, 'pixel size (z)': self.dz,
                             'wavelength': self.wl, 'numerical aperture': self.na,
                             'pattern spacing': self.sp})

    @staticmethod
    def _image_grid_polar(x, y):
        return np.sqrt(x ** 2 + y ** 2), np.arctan2(y, x)

    @staticmethod
    def _disc_array(shape=(128, 128), radius=64, origin=None):
        nx, ny = shape
        ox = nx / 2
        oy = ny / 2
        x = np.linspace(-ox, ox - 1, nx)
        y = np.linspace(-oy, oy - 1, ny)
        X, Y = np.meshgrid(x, y)
        rho = np.sqrt(X ** 2 + Y ** 2)
        disc = (rho < radius)
        if not origin is None:
            s0 = origin[0] - int(nx / 2)
            s1 = origin[1] - int(ny / 2)
            disc = np.roll(np.roll(disc, int(s0), 0), int(s1), 1)
        return disc

    @staticmethod
    def _radial_array(shape=(128, 128), f=None, origin=None):
        nx = shape[0]
        ny = shape[1]
        ox = nx / 2
        oy = ny / 2
        x = np.linspace(-ox, ox - 1, nx)
        y = np.linspace(-oy, oy - 1, ny)
        X, Y = np.meshgrid(x, y)
        rho = np.sqrt(X ** 2 + Y ** 2)
        rarr = f(rho)
        if not origin is None:
            s0 = origin[0] - nx / 2
            s1 = origin[1] - ny / 2
            rarr = np.roll(np.roll(rarr, int(s0), 0), int(s1), 1)
        return rarr

    @staticmethod
    def _shift(arr, shifts=None):
        if shifts is None:
            shifts = np.array(arr.shape) / 2
        if len(arr.shape) == len(shifts):
            for m, p in enumerate(shifts):
                arr = np.roll(arr, int(p), m)
        return arr

    @staticmethod
    def _zernike_j_nm(j):
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
        nx, ny = shape
        ox = nx / 2
        oy = ny / 2
        x = np.linspace(-ox, ox - 1, nx) / radius
        y = np.linspace(-oy, oy - 1, ny) / radius
        xv, yv = np.meshgrid(x, y)
        rho, phi = self._image_grid_polar(xv, yv)
        kmax = int((n - abs(m)) / 2)
        summation = 0
        for k in range(kmax + 1):
            summation += ((-1) ** k * factorial(n - k) /
                          (factorial(k) * factorial(0.5 * (n + abs(m)) - k) *
                           factorial(0.5 * (n - abs(m)) - k)) *
                          rho ** (n - 2 * k))
        return summation * np.cos(m * phi) * self._disc_array(shape, radius)


if __name__ == '__main__':
    s = NLSIM()
    s.get_objects(1024, 8, 8, 8)
    s.get_pupil()
    # s._get_pupil(zarr=[0., 0., 0, 1.])
    s.nlsim_2d()
    s.save_result_2d()
    # s.nlsim_3d()
    # s.save_result_3d()
