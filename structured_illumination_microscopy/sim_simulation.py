"""
Optimized SIM (Structured Illumination Microscopy) simulator.
Supports 2D and 3D linear SIM.
"""

import time
import numpy as np
import numexpr as ne
import tifffile as tf
from numpy.random import rand, randint, uniform
import psf_generator as pg

TWO_PI = 2 * np.pi


class SIM:

    def __init__(self, nxh=64, nyh=64, nzh=16, dx=0.08, dy=0.08, dz=0.16,
                 wl=0.505, na=1.4, n2=1.512, sp=0.24):
        self.nxh = nxh
        self.nyh = nyh
        self.nzh = nzh
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.wl = wl
        self.na = na
        self.n2 = n2
        self.sp = sp

        self.nx = 2 * nxh
        self.ny = 2 * nyh
        self.nz = 2 * nzh

        self.number_of_angles = 0
        self.number_of_phases = 0
        self.number_of_fluorophores = 0
        self.xps = np.empty(0)
        self.yps = np.empty(0)
        self.zps = np.empty(0)
        self.focal_plane = nzh
        self.I = 10
        self.out = None
        self.cam_offset = 80.0
        self.psf = pg.PSF(wl, na, n2, dx, self.nx)


    def _rand_xy(self, n):
        return (self.dx * self.nx) * (0.8 * rand(n) + 0.1)

    def _rand_z_start(self, n):
        return (self.dz * self.nz) * (0.4 * rand(n) - 0.4)

    def _rand_z_end(self, n):
        return (self.dz * self.nz) * (0.4 * rand(n))

    def get_point_objects(self, n):
        x = self._rand_xy(n)
        y = (self.dy * self.ny) * (0.8 * rand(n) + 0.1)
        z = (self.dz * self.nz) * (0.8 * rand(n) - 0.4)
        return n, x, y, z

    def get_line_objects(self, n_lines):
        counts = randint(128, 512, n_lines)
        x_s, x_e = self._rand_xy(n_lines), self._rand_xy(n_lines)
        y_s, y_e = self._rand_xy(n_lines), self._rand_xy(n_lines)
        z_s, z_e = self._rand_z_start(n_lines), self._rand_z_end(n_lines)

        total = counts.sum()
        cx, cy, cz = np.empty(total), np.empty(total), np.empty(total)

        offset = 0
        for i, c in enumerate(counts):
            sl = slice(offset, offset + c)
            cx[sl] = np.linspace(x_s[i], x_e[i], c)
            cy[sl] = np.linspace(y_s[i], y_e[i], c)
            cz[sl] = np.linspace(z_s[i], z_e[i], c)
            offset += c
        return total, cx, cy, cz

    def get_polynomial_objects(self, n_polys):
        counts = randint(128, 512, n_polys)
        x_s, x_e = self._rand_xy(n_polys), self._rand_xy(n_polys)
        y_s, y_e = self._rand_xy(n_polys), self._rand_xy(n_polys)
        z_s, z_e = self._rand_z_start(n_polys), self._rand_z_end(n_polys)

        total = counts.sum()
        cx, cy, cz = np.empty(total), np.empty(total), np.empty(total)

        offset = 0
        for i, c in enumerate(counts):
            degrees = randint(2, 8, 3)
            t = np.linspace(-1, 1, c)
            x = np.poly1d(uniform(-1, 1, degrees[0] + 1))(t)
            y = np.poly1d(uniform(-1, 1, degrees[1] + 1))(t)
            z = np.poly1d(uniform(-1, 1, degrees[2] + 1))(t)

            sl = slice(offset, offset + c)
            cx[sl] = self._normalize(x, x_s[i], x_e[i])
            cy[sl] = self._normalize(y, y_s[i], y_e[i])
            cz[sl] = self._normalize(z, z_s[i], z_e[i])
            offset += c
        return total, cx, cy, cz

    def get_curve_objects(self, n_curves):
        counts = randint(128, 512, n_curves)
        x_s, x_e = self._rand_xy(n_curves), self._rand_xy(n_curves)
        y_s, y_e = self._rand_xy(n_curves), self._rand_xy(n_curves)
        z_s, z_e = self._rand_z_start(n_curves), self._rand_z_end(n_curves)

        total = counts.sum()
        cx, cy, cz = np.empty(total), np.empty(total), np.empty(total)

        offset = 0
        for i, c in enumerate(counts):
            num_coeffs = randint(2, 8)
            coeffs = uniform(-1, 1, (3, num_coeffs))
            phases = uniform(0, TWO_PI, (3, num_coeffs))
            t = np.linspace(0, TWO_PI, c)

            sl = slice(offset, offset + c)
            cx[sl] = self._normalize(self._fourier_curve(t, coeffs[0], phases[0]), x_s[i], x_e[i])
            cy[sl] = self._normalize(self._fourier_curve(t, coeffs[1], phases[1]), y_s[i], y_e[i])
            cz[sl] = self._normalize(self._fourier_curve(t, coeffs[2], phases[2]), z_s[i], z_e[i])
            offset += c
        return total, cx, cy, cz

    @staticmethod
    def _fourier_curve(t, coeffs, phases):
        i = np.arange(1, len(coeffs) + 1)
        return (coeffs[:, np.newaxis] * np.sin(
            i[:, np.newaxis] * t[np.newaxis, :] + phases[:, np.newaxis]
        )).sum(axis=0)

    @staticmethod
    def _normalize(coord, lo, hi):
        cmin = coord.min()
        cmax = coord.max()
        r = cmax - cmin
        if r == 0:
            return np.full(coord.shape, (lo + hi) * 0.5)
        return (coord - cmin) * (abs(hi - lo) / r) + min(lo, hi)

    def get_objects(self, number_of_dots=None, number_of_lines=None,
                    number_of_polynomials=None, number_of_curves=None):
        generators = [
            (number_of_dots, self.get_point_objects),
            (number_of_lines, self.get_line_objects),
            (number_of_polynomials, self.get_polynomial_objects),
            (number_of_curves, self.get_curve_objects),
        ]

        chunks_x, chunks_y, chunks_z = [], [], []
        total = 0
        for count, gen_func in generators:
            if count is not None:
                n, x, y, z = gen_func(count)
                total += n
                chunks_x.append(x)
                chunks_y.append(y)
                chunks_z.append(z)

        if chunks_x:
            self.number_of_fluorophores += total
            self.xps = np.concatenate([self.xps, *chunks_x]) if len(self.xps) else np.concatenate(chunks_x)
            self.yps = np.concatenate([self.yps, *chunks_y]) if len(self.yps) else np.concatenate(chunks_y)
            self.zps = np.concatenate([self.zps, *chunks_z]) if len(self.zps) else np.concatenate(chunks_z)

    def _setup_sim_params(self, nang, nph, I):
        self.number_of_angles = nang
        self.number_of_phases = nph
        self.I = I

        angles = np.arange(nang) * (TWO_PI / nang)
        phases = np.arange(nph) * (TWO_PI / nph)
        self.angle = angles
        self.phase = phases
        self.kx = TWO_PI * np.cos(angles) / self.sp
        self.ky = TWO_PI * np.sin(angles) / self.sp

    def _precompute_psfs_2d(self):
        """Pre-compute all 2D PSFs; store flat (n_fluor, nx*ny) for direct matmul use."""
        n_fluor = self.number_of_fluorophores
        psfs = np.empty((n_fluor, self.nx * self.ny), dtype=np.float64)
        for m in range(n_fluor):
            psfs[m] = self.psf.get_2d_psf((self.xps[m], self.yps[m], self.zps[m])).ravel()
        return psfs

    def _precompute_psfs_3d(self, zplane_offsets):
        """Pre-compute all 3D PSFs; store as (nz, n_fluor, nx*ny) for batched matmul use."""
        nz = len(zplane_offsets)
        n_fluor = self.number_of_fluorophores
        psfs = np.empty((nz, n_fluor, self.nx * self.ny), dtype=np.float64)
        for zp, zoff in enumerate(zplane_offsets):
            for m in range(n_fluor):
                dz_val = self.zps[m] - zoff
                psfs[zp, m] = self.psf.get_2d_psf((self.xps[m], self.yps[m], dz_val)).ravel()
        return psfs

    def _illumination_2d(self, ang_idx):
        """Compute illumination for all fluorophores and phases at once.

        Returns: (nph, n_fluor) array of intensities.
        """
        kx_a = self.kx[ang_idx]
        ky_a = self.ky[ang_idx]
        xps = self.xps
        yps = self.yps
        dot = ne.evaluate('kx_a * xps + ky_a * yps')  # (n_fluor,)

        ph = self.phase
        I = self.I

        dot_2d = dot[np.newaxis, :]  # (1, n_fluor)
        ph_2d = ph[:, np.newaxis]    # (nph, 1)
        return ne.evaluate('I * 0.5 * (1 + cos(dot_2d + ph_2d))')  # (nph, n_fluor)

    def _illumination_3d(self, ang_idx):
        """Compute 3D SIM illumination for all fluorophores and phases.

        Returns: (nph, nz, n_fluor) array of intensities.
        """
        kx_a = self.kx[ang_idx]
        ky_a = self.ky[ang_idx]
        xps = self.xps
        yps = self.yps
        zps = self.zps
        kz = self.kz

        dot = ne.evaluate('kx_a * xps + ky_a * yps')  # (n_fluor,)

        ph = self.phase          # (nph,)
        zplane = self._zplane_offsets  # (nz,)
        I = self.I

        dot_3 = dot[np.newaxis, np.newaxis, :]     # (1, 1, n_fluor)
        ph_3 = ph[:, np.newaxis, np.newaxis]        # (nph, 1, 1)
        zps_3 = zps[np.newaxis, np.newaxis, :]      # (1, 1, n_fluor)
        zpl_3 = zplane[np.newaxis, :, np.newaxis]   # (1, nz, 1)

        return ne.evaluate(
            'I * (3 + 2 * cos(2 * (0.5 * dot_3 + ph_3)) + 4 * cos(kz * (zps_3 - zpl_3)) * cos(0.5 * dot_3 + ph_3))'
        )  # (nph, nz, n_fluor)

    def _generate_2d_images(self):
        """Generate all 2D SIM images using a single matmul per angle."""
        nang = self.number_of_angles
        nph = self.number_of_phases

        # psfs: (n_fluor, nx*ny) — pre-flattened for direct matmul
        psfs = self._precompute_psfs_2d()

        for a in range(nang):
            illum = self._illumination_2d(a)  # (nph, n_fluor)
            # One matmul replaces nph separate dot products:
            # (nph, n_fluor) @ (n_fluor, nx*ny) → (nph, nx*ny)
            imgs = illum @ psfs
            block = self.cam_offset + imgs.reshape(nph, self.nx, self.ny)
            self.out[a * nph:(a + 1) * nph] = np.random.poisson(np.clip(block, 0, None))

    def _generate_3d_images(self):
        """Generate all 3D SIM images using batched matmul — no phase/z inner loops."""
        nang = self.number_of_angles
        nph = self.number_of_phases
        nz = self.nz

        # psfs: (nz, n_fluor, nx*ny) — stored in z-major order for batched matmul
        psfs = self._precompute_psfs_3d(self._zplane_offsets)

        for a in range(nang):
            illum = self._illumination_3d(a)  # (nph, nz, n_fluor)

            # Batched matmul over z-planes:
            #   illum_znp: (nz, nph, n_fluor)  [transpose is a free view]
            #   psfs:      (nz, n_fluor, nx*ny) [already contiguous in this layout]
            #   result:    (nz, nph, nx*ny)
            illum_znp = illum.transpose(1, 0, 2)          # (nz, nph, n_fluor) — view
            imgs_znp = np.matmul(illum_znp, psfs)          # (nz, nph, nx*ny)
            imgs = imgs_znp.transpose(1, 0, 2).reshape(nph, nz, self.nx, self.ny)

            block = self.cam_offset + imgs
            self.out[a] = np.random.poisson(np.clip(block, 0, None))

    def sim_2d(self, nang=3, nph=3, I=1000):
        self._setup_sim_params(nang, nph, I)
        self.out = np.zeros((nang * nph, self.nx, self.ny), dtype=np.float64)
        print(f"Generating 2D SIM: {nang} angles × {nph} phases, {self.number_of_fluorophores} fluorophores")
        t0 = time.perf_counter()
        self._generate_2d_images()
        print(f"Done in {time.perf_counter() - t0:.2f}s")

    def sim_3d(self, nang=3, nph=5, I=1000):
        self._setup_sim_params(nang, nph, I)
        phim = self.na / self.n2
        self.kz = (TWO_PI / self.sp) * (1 - np.sqrt(1 - phim ** 2))
        # Vectorized z-plane offsets: replaces list comprehension over range(nz)
        self._zplane_offsets = self.dz * (np.arange(self.nz) - self.focal_plane)
        self.out = np.zeros((nang, nph, self.nz, self.nx, self.ny), dtype=np.float64)
        print(f"Generating 3D SIM: {nang} angles × {nph} phases × {self.nz} z-planes, {self.number_of_fluorophores} fluorophores")
        t0 = time.perf_counter()
        self._generate_3d_images()
        print(f"Done in {time.perf_counter() - t0:.2f}s")

    def save_result_2d(self, prefix=None):
        if prefix is None:
            prefix = time.strftime("%Y%m%d%H%M") + '_'
        tf.imwrite(prefix + 'sim2d_simulation_image_stack.tif', self.out.astype(np.float32),
                   photometric='minisblack',
                   metadata={'number of phases': self.number_of_phases,
                             'number of angles': self.number_of_angles,
                             'pixel size (xy)': self.dx,
                             'wavelength': self.wl, 'numerical aperture': self.na,
                             'pattern spacing': self.sp})

    def save_result_3d(self, prefix=None):
        if prefix is None:
            prefix = time.strftime("%Y%m%d%H%M") + '_'

        data = self.out.transpose(2, 0, 1, 3, 4)  # (nz, nang, nph, nx, ny)
        data = data.reshape(-1, self.nx, self.ny).astype(np.float32)  # (nz*nang*nph, nx, ny)

        tf.imwrite(prefix + 'sim3d_simulation_image_stack.tif', data,
                   photometric='minisblack',
                   metadata={'number of phases': self.number_of_phases,
                             'number of angles': self.number_of_angles,
                             'number of z-planes': self.nz,
                             'pixel size (xy)': self.dx,
                             'pixel size (z)': self.dz,
                             'wavelength': self.wl,
                             'numerical aperture': self.na,
                             'pattern spacing': self.sp})


if __name__ == '__main__':
    s = SIM(nxh=128, nyh=128, nzh=16)
    s.get_objects(256, 2, 2, 2)
    s.sim_3d()
    s.save_result_3d()
