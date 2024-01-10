import numpy as np
import numpy.random as rd
from scipy.special import factorial
import tifffile as tf
import time
import concurrent.futures


class SIM:

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
        self.sp = 0.24  # um
        self.number_of_angles = 0
        self.number_of_phases = 0
        self.number_of_fluorophores = 0
        self.xps = np.array([])
        self.yps = np.array([])
        self.zps = np.array([])
        self.focal_plane = self.nzh
        self.I = None
        self.out = None
        self.cam_offset = 80.0

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

    def get_curve_objects(self, number_of_curves):
        number_of_fluorophores_per_curve = np.random.randint(128, 512, number_of_curves)
        coords_x = np.array([])
        coords_y = np.array([])
        coords_z = np.array([])
        for i in range(number_of_curves):
            degrees = np.random.randint(2, 8, 3)
            fx = np.poly1d(np.random.randint(-4, 4, size=(degrees[0] + 1)))
            fy = np.poly1d(np.random.randint(-4, 4, size=(degrees[1] + 1)))
            fz = np.poly1d(np.random.randint(-4, 4, size=(degrees[2] + 1)))
            t = np.linspace(np.random.randint(-16, -2), np.random.randint(2, 16), number_of_fluorophores_per_curve[i])
            x = fx(t)
            y = fy(t)
            z = fz(t)
            x = (self.dx * self.nxh * 2) * (0.8 * (np.abs(x) / np.abs(x).max()) + 0.1)
            y = (self.dx * self.nxh * 2) * (0.8 * (np.abs(y) / np.abs(y).max()) + 0.1)
            z = (self.dz * self.nzh * 2) * (0.8 * (z / np.abs(z).max()) + 0.1)
            coords_x = np.concatenate((coords_x, x))
            coords_y = np.concatenate((coords_y, y))
            coords_z = np.concatenate((coords_z, z))
        return number_of_fluorophores_per_curve.sum(), coords_x, coords_y, coords_z

    def get_objects(self, number_of_dots=None, number_of_lines=None, number_of_curves=None):
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
        if number_of_curves is not None:
            _n, _x, _y, _z = self.get_curve_objects(number_of_curves)
            self.number_of_fluorophores = self.number_of_fluorophores + _n
            self.xps = np.concatenate((self.xps, _x))
            self.yps = np.concatenate((self.yps, _y))
            self.zps = np.concatenate((self.zps, _z))

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

    def _add_psf_2d(self, x, y, I):
        nx = self.nxh * 2
        ny = self.nyh * 2
        alpha = 2 * np.pi / nx / self.dx
        gxy = lambda m, n: np.exp(1j * alpha * (m * x + n * y)).astype(np.complex64)
        ph = self._shift(np.fromfunction(gxy, (nx, ny), dtype=np.float32))
        wfp = np.sqrt(I) * ph * self.wf
        return np.abs(np.fft.fft2(wfp)) ** 2

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

    def _get_one_img_2d(self, indices):
        nx = self.nxh * 2
        ny = self.nyh * 2
        self.out[indices[0] * self.number_of_phases + indices[1], :, :] = self.cam_offset + np.zeros((nx, ny))
        for m in range(self.number_of_fluorophores):
            Ip = self.I * 0.5 * (1 + np.cos(
                self.kx[indices[0]] * self.xps[m] + self.ky[indices[0]] * self.yps[m] + self.phase[indices[1]]))
            self.out[indices[0] * self.number_of_phases + indices[1], :, :] += self._add_psf_2d(self.xps[m],
                                                                                                self.yps[m], Ip)
        self.out[indices[0] * self.number_of_phases + indices[1], :, :] = rd.poisson(
            self.out[indices[0] * self.number_of_phases + indices[1], :, :])
        return 'done', 'angle', indices[0], 'phase', indices[1]

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
                Ip = self.I * (3 + 2 * cs2xy + 4 * csz * csxy)
                self.out[indices[0], indices[1], zp, :, :] += self._add_psf_3d(self.xps[m], self.yps[m],
                                                                               self.zps[m] - zplane, Ip)
        self.out[indices[0], indices[1], :, :, :] = rd.poisson(self.out[indices[0], indices[1], :, :, :])
        return 'done', 'angle', indices[0], 'phase', indices[1]

    def sim_2d(self, nang=3, nph=3, I=1000):
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
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._get_one_img_2d, indices) for indices in indices_list]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            print(future.result())

    def sim_3d(self, nang=3, nph=5, I=1000):
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
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._get_one_img_3d, indices) for indices in indices_list]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            print(future.result())

    def save_result_2d(self):
        t = time.strftime("%Y%m%d%H%M")
        path = t + '_'
        tf.imwrite(path + 'si2d_simulation_image_stack.tif', self.out, photometric='minisblack',
                   metadata={'number of phases': self.number_of_phases,
                             'number of angles': self.number_of_angles,
                             'pixel size (xy)': self.dx,
                             'wavelength': self.wl, 'numerical aperture': self.na,
                             'pattern spacing': self.sp})

    def save_result_3d(self):
        t = time.strftime("%Y%m%d%H%M")
        path = t + '_'
        tf.imwrite(path + 'si3d_simulation_image_stack.tif', self.out, photometric='minisblack',
                   metadata={'number of phases': self.number_of_phases,
                             'number of angles': self.number_of_angles,
                             'pixel size (xy)': self.dx, 'pixel size (z)': self.dz,
                             'wavelength': self.wl, 'numerical aperture': self.na,
                             'pattern spacing': self.sp})

    @staticmethod
    def _on_probability(I_on):
        p_on = np.exp(-I_on * 5)
        return 1 if rd.random() < p_on else 0

    @staticmethod
    def _off_probability(I_off):
        p_off = np.exp(-I_off * 8)
        return 0 if rd.random() > p_off else 1

    def _get_one_img_nl_2d(self, indices):
        nx = self.nxh * 2
        ny = self.nyh * 2
        self.out[indices[0] * self.number_of_phases + indices[1], :, :] = self.cam_offset + np.zeros((nx, ny))
        for m in range(self.number_of_fluorophores):
            I_off = 1 + np.cos(
                self.kx[indices[0]] * self.xps[m] + self.ky[indices[0]] * self.yps[m] + np.pi + self.phase[indices[1]])
            sw = self._off_probability(I_off)
            I_read = sw * self.I * 0.5 * (1 + np.cos(
                self.kx[indices[0]] * self.xps[m] + self.ky[indices[0]] * self.yps[m] + self.phase[indices[1]]))
            self.out[indices[0] * self.number_of_phases + indices[1], :, :] += self._add_psf_2d(self.xps[m],
                                                                                                self.yps[m], I_read)
        self.out[indices[0] * self.number_of_phases + indices[1], :, :] = rd.poisson(
            self.out[indices[0] * self.number_of_phases + indices[1], :, :])
        return 'done', 'angle', indices[0], 'phase', indices[1]

    def nlsim_2d(self, nang=7, nph=7, I=1600):
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
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._get_one_img_nl_2d, indices) for indices in indices_list]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            print(future.result())

    def save_result_nl2d(self):
        t = time.strftime("%Y%m%d%H%M")
        path = t + '_'
        tf.imwrite(path + 'nlsi2d_simulation_image_stack.tif', self.out, photometric='minisblack',
                   metadata={'number of phases': self.number_of_phases,
                             'number of angles': self.number_of_angles,
                             'pixel size (xy)': self.dx,
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
    def _radial_Array(shape=(128, 128), f=None, origin=None):
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
    s = SIM()
    # s._get_line_objects(4, False)
    # s._get_point_objects(256, True)
    s.get_objects(1024, 8, 8)
    s.get_pupil()
    # s._get_pupil(zarr=[0., 0., 0, 1.])
    # s.sim_2d()
    # s.save_result_2d()
    # s.sim_3d()
    # s.save_result_3d()
    s.nlsim_2d()
    s.save_result_nl2d()
