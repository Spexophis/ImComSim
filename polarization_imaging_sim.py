import os
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import tifffile as tf

matplotlib.use('QtAgg')
plt.ion()


class POLAR:

    def __init__(self):

        self.nxh = 128
        self.nyh = 128
        self.dx = 0.08  # um
        self.dy = 0.08  # um
        self.wl = 0.505  # um
        self.na = 0.75
        self.n2 = 1.512
        self.num_of_fluo = 0
        self.xps = np.array([])
        self.yps = np.array([])
        self.d_x = np.array([])
        self.d_y = np.array([])
        self.illu = 100
        self.out = None
        self.cam_offset = 80.0

    def get_polynomial_objects(self, number_of_polynomials):
        number_of_fluo_per_polynomial = np.random.randint(256, 1024, number_of_polynomials)
        cords_x = np.array([])
        cords_y = np.array([])
        dev_x = np.array([])
        dev_y = np.array([])
        x_start = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_polynomials) + 0.1)
        y_start = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_polynomials) + 0.1)
        x_end = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_polynomials) + 0.1)
        y_end = (self.dx * self.nxh * 2) * (0.8 * np.random.rand(number_of_polynomials) + 0.1)
        for i in range(number_of_polynomials):
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
        return number_of_fluo_per_polynomial.sum(), cords_x, cords_y, dev_x, dev_y

    def get_objects(self, number_of_polynomials=8, verbose=False):
        _n, _x, _y, _dx, _dy = self.get_polynomial_objects(number_of_polynomials)
        self.num_of_fluo = _n
        self.xps = _x
        self.yps = _y
        magnitude = np.sqrt(_dx ** 2 + _dy ** 2)
        unit_dx = _dx / magnitude
        unit_dy = _dy / magnitude
        self.d_x = unit_dx
        self.d_y = unit_dy
        if verbose:
            plt.figure(figsize=(10, 10))
            plt.scatter(_x, _y, s=8, label="Points")
            plt.quiver(_x, _y, unit_dx, unit_dy, angles="xy", scale_units="xy", scale=1,
                       color="red", alpha=0.5, label="Derivative Vectors")
            plt.title("Polynomial Points and Derivative Vectors")
            plt.xlabel("X Coordinate")
            plt.ylabel("Y Coordinate")
            plt.legend()
            plt.axis("equal")
            plt.show()

    def get_pupil(self):
        dp = 1 / (self.nxh * 2 * self.dx)
        radius = (self.na / self.wl) / dp
        msk = self._shift(self._disc_array(shape=(self.nxh * 2, self.nyh * 2), radius=radius)) / np.sqrt(
            np.pi * radius ** 2) / (self.nxh * 2)
        phi = np.zeros((self.nxh * 2, self.nyh * 2))
        self.wf = msk * np.exp(1j * phi).astype(np.complex64)

    def add_psf_2d(self, x, y, n_photons):
        nx = self.nxh * 2
        ny = self.nyh * 2
        alpha = 2 * np.pi / nx / self.dx
        gxy = lambda m, n: np.exp(1j * alpha * (m * x + n * y)).astype(np.complex64)
        ph = self._shift(np.fromfunction(gxy, (nx, ny), dtype=np.float32))
        wfp = ph * self.wf
        psf_dist = np.abs(np.fft.fft2(wfp)) ** 2
        psf_dist /= psf_dist.sum()
        psf_img = self.generate_photon_distributions(int(n_photons), psf_dist)
        return psf_img

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

    def get_one_img_2d(self, idx):
        nx = self.nxh * 2
        ny = self.nyh * 2
        pix_map = self._vector_to_pixel_map(self.d_x[idx], self.d_y[idx])
        pol_msk = np.tile(pix_map, (self.nxh, self.nyh))
        pol_cam = pol_msk * self.add_psf_2d(self.xps[idx], self.yps[idx], self.illu)
        self.out += pol_cam
        return idx, 'done'

    def generate_data_2d(self, intensity=100):
        nx = self.nxh * 2
        ny = self.nxh * 2
        self.illu = intensity
        self.out = np.zeros((nx, ny)) + self.cam_offset
        for i in range(self.num_of_fluo):
            self.get_one_img_2d(i)
        self.out = rd.poisson(self.out)

    def save_result_2d(self, fd=None):
        t = time.strftime("%Y%m%d%H%M")
        if fd is None:
            tf.imwrite(t + '_pol_cam_simulation_image.tif', self.out)
        else:
            tf.imwrite(os.path.join(fd, t + '_pol_cam_simulation_image.tif'), self.out)

    @staticmethod
    def _normalize(coord, range_):
        r = np.max(coord) - np.min(coord)
        if r == 0:
            return np.full(coord.shape, (range_[1] + range_[0]) / 2)
        else:
            return (coord - np.min(coord)) / (np.max(coord) - np.min(coord)) * np.abs(range_[1] - range_[0]) + np.min(
                range_)

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

    @staticmethod
    def _shift(arr, shifts=None):
        if shifts is None:
            shifts = np.array(arr.shape) / 2
        if len(arr.shape) == len(shifts):
            for m, p in enumerate(shifts):
                arr = np.roll(arr, int(p), m)
        return arr

    @staticmethod
    def _disc_array(shape=(128, 128), radius=64, origin=None):
        nx, ny = shape
        ox = nx / 2
        oy = ny / 2
        x = np.linspace(-ox, ox - 1, nx)
        y = np.linspace(-oy, oy - 1, ny)
        xv, yv = np.meshgrid(x, y)
        rho = np.sqrt(xv ** 2 + yv ** 2)
        disc = (rho < radius)
        if not origin is None:
            s0 = origin[0] - int(nx / 2)
            s1 = origin[1] - int(ny / 2)
            disc = np.roll(np.roll(disc, int(s0), 0), int(s1), 1)
        return disc


if __name__ == '__main__':
    s = POLAR()
    s.get_objects(8)
    s.get_pupil()
    s.generate_data_2d()
    s.save_result_2d()
