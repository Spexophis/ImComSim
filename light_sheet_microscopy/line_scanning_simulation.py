import numpy as np
from numpy.fft import fft2, fftshift


class LSS:

    def __init__(self):
        self.wl = 0.488  # wavelength in microns
        self.na = 1.4  # numerical aperture
        self.n2 = 1.512  # index at point
        self.dx = 0.01  # pixel size in microns
        self.nx = 1024  # size of region
        self.dp = 1 / (self.nx * self.dx)  # pixel size in frequency space (pupil)
        self.radius = (self.na / self.wl) / self.dp  # radius of pupil (NA/lambda) in pixels
        self.xv, self.yv = None, None
        self.rho = None
        self.bpp = None
        self.msk = None

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

    def flat_pupil(self):
        self.msk = self.disc_array(self.radius)
        self.bpp = self.msk * np.exp(2j * np.pi * np.zeros((self.nx, self.nx)))

    def add_cylindrical(self, r):
        phi = np.arctan2(self.yv, self.xv)
        mask = self.rho <= r
        cylindrical_phase = np.zeros_like(phi)
        cylindrical_phase[mask] = phi[mask]
        return cylindrical_phase

    def focus_mode(self, d):
        ro = self.rho / self.radius
        mk = (ro <= 1.0).astype(np.float64)
        wf = mk * (self.n2 * d / self.wl) * np.sqrt(1 - (self.na * mk * ro / self.n2) ** 2)
        return wf

    def get_3d_psf(self, start, stop, step):
        n_steps = int((stop - start) / step + 1)
        zarr = np.linspace(start / step, stop / step, n_steps).astype(np.int64)
        zarr = zarr[0:n_steps - 1]
        zarr = np.roll(zarr, int((n_steps - 1) / 2))
        stack = np.zeros((n_steps - 1, self.nx, self.nx))
        for m, z in enumerate(zarr):
            ph = self.focus_mode(z * step)
            wf = self.bpp * np.exp(2j * np.pi * ph)
            stack[m] = np.abs(fft2(wf)) ** 2
        return fftshift(stack)

    def disc_array(self, radius=64.0, origin=None, dtype=np.float64):
        disc = (self.rho < radius).astype(dtype)
        if origin is not None:
            xh = int(self.nx / 2)
            s0 = origin[0] - xh
            s1 = origin[1] - xh
            disc = np.roll(np.roll(disc, int(s0), 0), int(s1), 1)
        return disc
