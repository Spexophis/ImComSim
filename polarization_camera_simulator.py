import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf
import photophysics_simulator as phs

matplotlib.use('Qt5Agg')
plt.ion()


class POLAR:

    def __init__(self):

        self.nxh = 128
        self.nyh = 128
        self.nzh = 16
        self.dx = 0.08  # um
        self.dy = 0.08  # um
        self.dz = 0.16  # um
        self.wl = 0.505  # um
        self.na = 0.75
        self.n2 = 1.512
        self.dp = 1 / (self.nxh * 2 * self.dx)
        self.radius = (self.na / self.wl) / self.dp
        self.xv, self.yv, self.zr, self.rho, self.phi = self.mesh_grid()
        self.msk = None
        self.bpp = None
        self.num_of_fluo = 0
        self.xps = np.array([])
        self.yps = np.array([])
        self.zps = np.array([])
        self.d_x = np.array([])
        self.d_y = np.array([])
        self.d_z = np.array([])
        self.sz = np.array([])
        self.d_r = np.array([])
        self.i_x = None
        self.i_y = None
        self.illu = 1
        self.out = None
        self.cam_offset = 80.0
        self.lft = 2.5e-9  # s
        self.sampling = 20
        self.kb = 1.38e-23
        self.ita = 0.001  # 1.2 - glycerol at 25 degree temperature
        self.temperature = 298  # K
        self.t = None
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

    def mesh_grid(self):
        x = np.linspace(-self.nxh, self.nxh - 1, self.nxh * 2)
        xv, yv = np.meshgrid(x, x)
        rho = np.sqrt(xv ** 2 + yv ** 2)
        phi = np.arctan2(yv, xv)
        zr = self.dz * np.linspace(-self.nzh, self.nzh - 1, self.nzh * 2)
        return xv, yv, zr, rho, phi

    def flat_pupil(self):
        nx = self.nxh * 2
        self.msk = self.disc_array(radius=self.radius, origin=(0, 0))
        self.bpp = self.msk * np.exp(2j * np.pi * np.zeros((nx, nx)))

    def get_illumination(self, intensity=1000, pol=None):
        if pol is not None:
            ang = pol * np.pi / 180
            self.i_x, self.i_y = np.cos(ang), np.sin(ang)
        else:
            self.i_x, self.i_y = None, None
        self.illu = intensity

    def get_polynomial_objects(self, number_of_polynomials):
        number_of_fluo_per_polynomial = np.random.randint(256, 1024, number_of_polynomials)
        ms = np.array([])
        mdr = np.array([])
        cords_x = np.array([])
        cords_y = np.array([])
        dev_x = np.array([])
        dev_y = np.array([])
        x_start = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_polynomials) - 0.4)
        y_start = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_polynomials) - 0.4)
        x_end = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_polynomials) - 0.4)
        y_end = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_polynomials) - 0.4)
        for i in range(number_of_polynomials):
            r = np.random.randint(2, 200) * 1e-9 * np.ones(number_of_fluo_per_polynomial[i])
            ms = np.concatenate((ms, r))
            dr = (3 * self.kb * self.temperature) / (4 * np.pi * self.ita * (r ** 3))
            mdr = np.concatenate((mdr, np.sqrt(2 * dr * self.lft)))
            degrees = np.random.randint(2, 8, 3)
            poly_x = np.poly1d(np.random.uniform(-1, 1, size=(degrees[0] + 1)))
            poly_y = np.poly1d(np.random.uniform(-1, 1, size=(degrees[1] + 1)))
            t = np.linspace(-1, 1, number_of_fluo_per_polynomial[i])
            x = poly_x(t)
            y = poly_y(t)
            x = self._normalize(x, (x_start[i], x_end[i]))
            y = self._normalize(y, (y_start[i], y_end[i]))
            cords_x = np.concatenate((cords_x, x))
            cords_y = np.concatenate((cords_y, y))
            poly_dx = poly_x.deriv()
            poly_dy = poly_y.deriv()
            dx = poly_dx(t)
            dy = poly_dy(t)
            dev_x = np.concatenate((dev_x, dx))
            dev_y = np.concatenate((dev_y, dy))
        return number_of_fluo_per_polynomial.sum(), cords_x, cords_y, dev_x, dev_y, ms, mdr

    def get_objects(self, number_of_polynomials=8):
        _n, _x, _y, _dx, _dy, _r, _dr = self.get_polynomial_objects(number_of_polynomials)
        self.num_of_fluo = _n
        self.xps = _x
        self.yps = _y
        magnitude = np.sqrt(_dx ** 2 + _dy ** 2)
        unit_dx = _dx / magnitude
        unit_dy = _dy / magnitude
        self.d_x = unit_dx
        self.d_y = unit_dy
        self.sz = _r
        self.d_r = _dr
        self.t = time.strftime("%Y%m%d%H%M")

    def lateral_mode(self, pos):
        x, y = pos
        nx = self.nxh * 2
        alpha = 2 * np.pi / nx / self.dx
        pwf = np.exp(1j * self.msk * alpha * (self.xv * x + self.yv * y))
        return pwf

    def axial_mode(self, d):
        ro = self.rho / self.radius
        mk = (ro <= 1.0).astype(np.float64)
        ph = mk * (self.n2 * d / self.wl) * np.sqrt(1 - (self.na * mk * ro / self.n2) ** 2)
        wf = np.exp(2j * np.pi * self.msk * ph)
        return wf

    def generate_photon_distributions(self, n, distribution_map):
        flat_distribution = distribution_map.flatten()
        indices = np.random.choice(np.arange(len(flat_distribution)), size=n, p=flat_distribution)
        distributions = np.column_stack(np.unravel_index(indices, distribution_map.shape))
        counts = self.count_photons(distributions)
        return counts

    def count_photons(self, points):
        counts = np.zeros((self.nxh * 2, self.nyh * 2), dtype=int)
        for point in points:
            x, y = point
            counts[x, y] += 1
        return counts

    def on_probability(self, pw=0.5, expo=1.0):
        on_switching_pulse = phs.ModulatedLasers(wavelengths=[405, 488], power_densities=[pw, 0.0],
                                                 pulse_widths=[expo, 0.0], t_start=[1, 2.5], dwell_time=30)
        on_switching_experiment = phs.Experiment(illumination=on_switching_pulse, fluorophore=self.rsEGFP2_off_state)
        populations = on_switching_experiment.solve_kinetics(0.01)
        p_on = populations[-1, 2]
        return 1 if np.random.random() < p_on else 0

    def get_2d_psf(self, x, y, n_photons):
        ph_x = self.lateral_mode(pos=(x, y))
        wfp = self.msk * self.bpp * ph_x
        psf_dist = np.abs(np.fft.fft2(wfp)) ** 2
        psf_dist /= psf_dist.sum()
        psf_img = self.generate_photon_distributions(int(n_photons), psf_dist)
        return np.fft.fftshift(psf_img)

    def get_3d_psf(self, x, y, z, n_photons):
        ph_x = self.lateral_mode(pos=(x, y))
        ph_z = self.axial_mode(z)
        wfp = self.msk * self.bpp * ph_z * ph_x
        psf_dist = np.abs(np.fft.fft2(wfp)) ** 2
        psf_dist /= psf_dist.sum()
        psf_img = self.generate_photon_distributions(int(n_photons), psf_dist)
        return np.fft.fftshift(psf_img)

    def get_one_img_2d(self, idx):
        for i in range(self.sampling):
            rta = i * self.d_r[idx] / self.sampling
            rdx, rdy = self._rotate_vector(self.d_x[idx], self.d_y[idx], rta)
            pix_map = self._vector_to_pixel_map(rdx, rdy)
            pol_msk = np.tile(pix_map, (self.nxh, self.nyh))
            if self.i_x is not None or self.i_y is not None:
                swo = 0.5 * self._vector_projection(self.i_x, self.i_y, self.d_x[idx], self.d_y[idx])
                prb = self.on_probability(pw=swo, expo=1.0)
                if prb:
                    pol_cam = pol_msk * self.get_2d_psf(self.xps[idx], self.yps[idx], self.illu)
                    self.out += pol_cam
            else:
                pol_cam = pol_msk * self.get_2d_psf(self.xps[idx], self.yps[idx], self.illu)
                self.out += pol_cam
        return idx, 'done'

    def generate_data_2d(self):
        nx = self.nxh * 2
        self.out = np.zeros((nx, nx)) + self.cam_offset
        for i in range(self.num_of_fluo):
            self.get_one_img_2d(i)
        self.out = np.random.poisson(self.out)

    def get_one_img_3d(self, idx):
        nx = self.nxh * 2
        nz = self.nzh * 2
        for i in range(self.sampling):
            rta = i * self.d_r[idx] / self.sampling
            rdx, rdy = self._rotate_vector(self.d_x[idx], self.d_y[idx], rta)
            pix_map = self._vector_to_pixel_map(rdx, rdy)
            pol_msk = np.tile(pix_map, (self.nxh, self.nyh))
            if self.i_x is not None or self.i_y is not None:
                exc = self.illu * self._vector_projection(self.i_x, self.i_y, self.d_x[idx], self.d_y[idx])
            else:
                exc = self.illu
            temp = np.zeros((nz, nx, nx))
            for m, pz in enumerate(self.zr):
                temp[m] = pol_msk * self.get_3d_psf(self.xps[idx], self.yps[idx], pz, exc)
            self.out += temp
        return idx, 'done'

    def generate_data_3d(self):
        nx = self.nxh * 2
        nz = self.nzh * 2
        self.out = np.zeros((nz, nx, nx)) + self.cam_offset
        for i in range(self.num_of_fluo):
            self.get_one_img_3d(i)
        self.out = np.random.poisson(self.out)

    def disc_array(self, radius=32.0, origin=(0, 0), dtype=int):
        y_origin, x_origin = origin
        rh = np.sqrt((self.xv - x_origin) ** 2 + (self.yv - y_origin) ** 2)
        disc = (rh < radius).astype(dtype)
        return disc

    @staticmethod
    def _normalize(coord, range_):
        r = np.max(coord) - np.min(coord)
        if r == 0:
            return np.full(coord.shape, (range_[1] + range_[0]) / 2)
        else:
            return (coord - np.min(coord)) / (np.max(coord) - np.min(coord)) * np.abs(range_[1] - range_[0]) + np.min(
                range_)

    @staticmethod
    def _rotate_vector(xv, yv, theta_rad):
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)
        x_rot = xv * cos_theta - yv * sin_theta
        y_rot = xv * sin_theta + yv * cos_theta
        return x_rot, y_rot

    @staticmethod
    def _vector_projection(ix, iy, xx, yy):
        v = np.array([ix, iy])
        d = np.array([xx, yy])
        if not np.isclose(np.linalg.norm(v), 1.0) or not np.isclose(np.linalg.norm(d), 1.0):
            raise ValueError("Both v and d must be unit vectors.")
        proj_v = np.dot(v, d) * d
        projection = np.sqrt(np.dot(proj_v, proj_v)) / np.sqrt(np.dot(v, v))
        return projection

    @staticmethod
    def _vector_to_pixel_map(_dx, _dy):
        unit_vector = np.array([_dx, _dy])
        rotated_vector = -unit_vector
        directions = {
            "0_deg": np.array([1, 0]),
            "45_deg": np.array([np.sqrt(2) / 2, np.sqrt(2) / 2]),
            "90_deg": np.array([0, 1]),
            "135_deg": np.array([-np.sqrt(2) / 2, np.sqrt(2) / 2])
        }
        pixel_map = np.zeros((2, 2))
        for angle, vector in directions.items():
            combined_projection = abs(np.dot(unit_vector, vector)) + abs(np.dot(rotated_vector, vector))
            if angle == "0_deg":
                pixel_map[1, 1] = combined_projection  # Bottom-right
            elif angle == "45_deg":
                pixel_map[0, 1] = combined_projection  # Top-right
            elif angle == "90_deg":
                pixel_map[0, 0] = combined_projection  # Top-left
            elif angle == "135_deg":
                pixel_map[1, 0] = combined_projection  # Bottom-left
        pixel_map /= pixel_map.max() if pixel_map.max() > 0 else 1
        return pixel_map


class RECON(POLAR):
    def __init__(self):
        super().__init__()
        self.data = None
        self.img_0 = None
        self.img_45 = None
        self.img_90 = None
        self.img_135 = None
        self.res = None

    def load_data(self, fd=None):
        if fd is not None:
            self.data = tf.imread(fd)
        else:
            self.data = self.out

    def sub_bg(self, bg):
        self.data[self.data > bg] = self.data[self.data > bg] - bg
        self.data[self.data <= bg] = 0

    def split_channels(self):
        self.img_0 = self._interp(self.data[1::2, 1::2], 2)
        self.img_45 = self._interp(self.data[::2, 1::2], 2)
        self.img_90 = self._interp(self.data[::2, ::2], 2)
        self.img_135 = self._interp(self.data[1::2, ::2], 2)

    def compute_anisotropy(self):
        horizontal = self.img_0
        vertical = self.img_90
        self.res = (horizontal - vertical) / (horizontal + 2 * vertical)

    def save_results(self, fd=None, verbose=False):
        if verbose:
            sizes = 4 + 80 * p.sz / p.sz.max()
            plt.figure(figsize=(10, 10))
            plt.scatter(self.xps, self.yps, s=sizes, label="Points")
            plt.quiver(self.xps, self.yps, self.d_x, self.d_y, angles="xy", scale_units="xy", scale=1,
                       color="red", alpha=0.5, label="Vectors")
            plt.title("Points and Vectors")
            plt.xlabel("X Coordinate")
            plt.ylabel("Y Coordinate")
            plt.legend()
            plt.axis("equal")
        if self.i_x is not None or self.i_y is not None:
            self.t = self.t + r"_lpi"
        else:
            self.t = self.t + r"_cpi"
        if fd is None:
            plt.savefig(fname=self.t + '_pol_cam_simulation_vmap.png', dpi=600)
            tf.imwrite(self.t + '_pol_cam_simulation_image.tif', self.out)
            tf.imwrite(self.t + '_pol_cam_simulation_anisotropy.tif', self.res)
        else:
            plt.savefig(fname=os.path.join(fd, self.t + '_pol_cam_simulation_vmap.png'), dpi=600)
            tf.imwrite(str(os.path.join(fd, self.t + '_pol_cam_simulation_image.tif')), self.out)
            tf.imwrite(str(os.path.join(fd, self.t + '_pol_cam_simulation_anisotropy.tif')), self.res)

    @staticmethod
    def _interp(arr, ratio):
        nx, ny = arr.shape
        px = int((nx * (ratio - 1)) / 2)
        py = int((ny * (ratio - 1)) / 2)
        arf = np.fft.fft2(arr)
        aro = np.pad(np.fft.fftshift(arf), ((px, px), (py, py)), 'constant', constant_values=(0, 0))
        return np.abs(np.fft.ifft2(np.fft.fftshift(aro)))


if __name__ == '__main__':
    p = RECON()
    p.get_objects(number_of_polynomials=8)
    p.flat_pupil()
    p.get_illumination(intensity=200)
    p.generate_data_2d()
    p.load_data()
    p.sub_bg(80)
    p.split_channels()
    p.compute_anisotropy()
    p.save_results(r"C:\Users\ruizhe.lin\Desktop\polcam_an", True)
    p.get_illumination(intensity=20, pol=45)
    p.generate_data_2d()
    p.sub_bg(80)
    p.split_channels()
    p.compute_anisotropy()
    p.save_results(r"C:\Users\ruizhe.lin\Desktop\polcam_an", True)
