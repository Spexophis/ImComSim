"""
This script generates simulated data of the 2-dimensional nonlinear structured illumination microscopy.
Ruizhe Lin
2024-01-12
"""

import sys
import time
import concurrent.futures
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import numpy.random as rd
import tifffile as tf

import photophysics_simulator as phs
import psf_generator as pg


class NLSIM:

    def __init__(self):

        self.nxh = 80
        self.nyh = 80
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
        self.pw_act = 0.5  # kW/cm2
        self.pw_off = 1.0
        self.pw_read = 1.0
        self.expo_act = 0.6  # ms
        self.expo_off = 0.8
        self.expo_read = 1.0
        self.qy = 0.65
        self._psf_obj = None
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
                                                       initial_populations=[1, 0, 0, 0])

    def get_point_objects(self, number_of_dots):
        coords_x = (self.dx * self.nxh * 2) * (0.8 * rd.rand(number_of_dots) + 0.1)
        coords_y = (self.dy * self.nyh * 2) * (0.8 * rd.rand(number_of_dots) + 0.1)
        coords_z = (self.dz * self.nzh * 2) * (0.8 * rd.rand(number_of_dots) - 0.4)
        return number_of_dots, coords_x, coords_y, coords_z

    def get_line_objects(self, number_of_lines):
        counts = np.random.randint(256, 1024, number_of_lines)
        x_start = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_lines) + 0.1)
        y_start = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_lines) + 0.1)
        x_end = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_lines) + 0.1)
        y_end = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_lines) + 0.1)
        z_start = (self.dz * self.nzh * 2) * (0.4 * np.random.rand(number_of_lines) - 0.4)
        z_end = (self.dz * self.nzh * 2) * (0.4 * np.random.rand(number_of_lines))
        coords_x = np.concatenate([np.linspace(x_start[i], x_end[i], counts[i]) for i in range(number_of_lines)])
        coords_y = np.concatenate([np.linspace(y_start[i], y_end[i], counts[i]) for i in range(number_of_lines)])
        coords_z = np.concatenate([np.linspace(z_start[i], z_end[i], counts[i]) for i in range(number_of_lines)])
        return counts.sum(), coords_x, coords_y, coords_z

    def get_polynomial_objects(self, number_of_polynomials):
        counts = np.random.randint(256, 1024, number_of_polynomials)
        x_start = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_polynomials) + 0.1)
        y_start = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_polynomials) + 0.1)
        x_end = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_polynomials) + 0.1)
        y_end = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_polynomials) + 0.1)
        z_start = (self.dz * self.nzh * 2) * (0.4 * np.random.rand(number_of_polynomials) - 0.4)
        z_end = (self.dz * self.nzh * 2) * (0.4 * np.random.rand(number_of_polynomials))
        xs, ys, zs = [], [], []
        for i in range(number_of_polynomials):
            degrees = np.random.randint(2, 8, 3)
            poly_x = np.poly1d(np.random.uniform(-1, 1, size=(degrees[0] + 1)))
            poly_y = np.poly1d(np.random.uniform(-1, 1, size=(degrees[1] + 1)))
            poly_z = np.poly1d(np.random.uniform(-1, 1, size=(degrees[2] + 1)))
            t = np.linspace(-1, 1, counts[i])
            xs.append(self.normalize(poly_x(t), (x_start[i], x_end[i])))
            ys.append(self.normalize(poly_y(t), (y_start[i], y_end[i])))
            zs.append(self.normalize(poly_z(t), (z_start[i], z_end[i])))
        return counts.sum(), np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)

    def get_curve_objects(self, number_of_curves):
        counts = np.random.randint(128, 512, number_of_curves)
        x_start = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_curves) + 0.1)
        y_start = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_curves) + 0.1)
        x_end = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_curves) + 0.1)
        y_end = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_curves) + 0.1)
        z_start = (self.dz * self.nzh * 2) * (0.4 * np.random.rand(number_of_curves) - 0.4)
        z_end = (self.dz * self.nzh * 2) * (0.4 * np.random.rand(number_of_curves))
        xs, ys, zs = [], [], []
        for c in range(number_of_curves):
            num_coeffs = np.random.randint(2, 8)
            coeffs = np.random.uniform(-1, 1, (3, num_coeffs))
            phases = np.random.uniform(0, 2 * np.pi, (3, num_coeffs))
            t = np.linspace(0, 2 * np.pi, counts[c])
            xs.append(self.normalize(self.fc(t, coeffs[0], phases[0]), (x_start[c], x_end[c])))
            ys.append(self.normalize(self.fc(t, coeffs[1], phases[1]), (y_start[c], y_end[c])))
            zs.append(self.normalize(self.fc(t, coeffs[2], phases[2]), (z_start[c], z_end[c])))
        return counts.sum(), np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)

    @staticmethod
    def fc(t, cs, phase):
        return sum(a * np.sin(i * t + p) for i, (a, p) in enumerate(zip(cs, phase), 1))

    @staticmethod
    def normalize(coord, range_):
        lo, hi = np.min(coord), np.max(coord)
        r = hi - lo
        if r == 0:
            return np.full(coord.shape, (range_[1] + range_[0]) / 2)
        return (coord - lo) / r * np.abs(range_[1] - range_[0]) + np.min(range_)

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
        nx = self.nxh * 2
        self._psf_obj = pg.PSF(wl=self.wl, na=self.na, n2=self.n2, dx=self.dx, nx=nx)
        if zarr is None:
            self._psf_obj.flat_wavefront()
        else:
            self._psf_obj.aberration_wavefront(zarr)

    def _on_probability(self, pw=0.5, expo=1.0):
        on_switching_pulse = phs.ModulatedLasers(wavelengths=[405, 488], power_densities=[pw, 0.0],
                                                 pulse_widths=[expo, 0.0], t_start=[1, 2.5], dwell_time=30)
        on_switching_experiment = phs.Experiment(illumination=on_switching_pulse, fluorophore=self.rsEGFP2_off_state)
        populations = on_switching_experiment.solve_kinetics(0.01)
        p_on = populations[-1, 2]
        return 1 if rd.random() < p_on else 0

    def _off_probability(self, pw=0.5, expo=1.0):
        off_switching_pulse = phs.ModulatedLasers(wavelengths=[405, 488], power_densities=[0.0, pw],
                                                  pulse_widths=[0.0, expo], t_start=[1, 2.5], dwell_time=30)
        off_switching_experiment = phs.Experiment(illumination=off_switching_pulse, fluorophore=self.rsEGFP2_on_state)
        populations = off_switching_experiment.solve_kinetics(0.01)
        p_off = populations[-1, 0]
        return 0 if rd.random() < p_off else 1

    def _add_psf_2d(self, x, y, n_photons):
        psf_dist = self._psf_obj.get_2d_psf((x, y, 0))
        nx = self.nxh * 2
        return np.random.multinomial(int(n_photons), psf_dist.ravel()).reshape(nx, nx)

    def _add_psf_3d(self, x, y, z, I):
        return I * self._psf_obj.get_2d_psf((x, y, z))

    def _get_one_img_2d(self, indices):
        ang, ph_idx = indices
        out_idx = ang * self.number_of_phases + ph_idx
        self.out[out_idx] = self.cam_offset
        # Precompute illumination for all fluorophores before the sequential sw-update loop
        phi_m = self.kx[ang] * self.xps + self.ky[ang] * self.yps
        phase_val = self.phase[ph_idx]
        I_off_arr = 0.5 * (1 + np.cos(phi_m + np.pi + phase_val))
        I_read_arr = 0.5 * (1 + np.cos(phi_m + phase_val))
        for m in range(self.number_of_fluorophores):
            if self.sw[m]:
                self.sw[m] *= self._off_probability(I_off_arr[m] * self.pw_off, self.expo_off)
                if self.sw[m]:
                    self.sw[m] *= self._off_probability(I_read_arr[m] * self.pw_read, self.expo_read)
                    if rd.random() < self.qy:
                        self.out[out_idx] += self._add_psf_2d(self.xps[m], self.yps[m],
                                                               self.I * I_read_arr[m])
            if self.sw[m] == 0:
                self.sw[m] = self._on_probability(self.pw_act, self.expo_act)
        self.out[out_idx] = rd.poisson(self.out[out_idx])
        return 'done', 'angle', ang, 'phase', ph_idx

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

    def _get_one_img_3d(self, indices):
        nz = self.nzh * 2
        ang, ph_idx = indices
        self.out[ang, ph_idx] = self.cam_offset
        # Precompute xy-phase terms (independent of z-plane) for all fluorophores
        phi_m = self.kx[ang] * self.xps + self.ky[ang] * self.yps
        phase_val = self.phase[ph_idx]
        cs2xy = np.cos(phi_m + 2 * phase_val)          # (n_fluor,)
        csxy = np.cos(0.5 * phi_m + phase_val)         # (n_fluor,)
        for zp in range(nz):
            zplane = self.dz * (zp - self.focal_plane)
            csz = np.cos(self.kz * (self.zps - zplane))  # (n_fluor,) — only z-term changes
            illumination = self.I * (3 + 2 * cs2xy + 4 * csz * csxy)
            for m in range(self.number_of_fluorophores):
                sw = self._off_probability(illumination[m])
                self.out[ang, ph_idx, zp] += sw * self._add_psf_3d(
                    self.xps[m], self.yps[m], self.zps[m] - zplane, illumination[m])
        self.out[ang, ph_idx] = rd.poisson(self.out[ang, ph_idx])
        return 'done', 'angle', ang, 'phase', ph_idx

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


if __name__ == '__main__':
    s = NLSIM()
    s.get_objects(1024, 6, 6, 6)
    s.get_pupil()
    # s.get_pupil(zarr=[0., 0., 0, 1.])
    s.nlsim_2d(parallel=True)
    s.save_result_2d()
    # s.nlsim_3d()
    # s.save_result_3d()
