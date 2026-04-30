from os import cpu_count

import numexpr as ne
import numpy as np
import pyfftw

import zernike_polynomials as zernike

TWO_PI = 2 * np.pi
NTHREADS = cpu_count() or 4


class PSF:
    """Point spread function with optimized FFT and element-wise operations"""

    def __init__(self, wl=0.505, na=1.4, n2=1.512, dx=0.064, nx=256):
        self.wl = wl
        self.na = na
        self.n2 = n2
        self.dx = dx
        self.nx = nx
        self._update_derived_params()
        self.zer = None
        self.bpp = np.zeros((nx, nx), dtype=np.complex64)
        self._grid = None
        self._rho = None
        self._mask = None

        # --- pyfftw setup ---
        # Aligned input/output arrays for SIMD acceleration
        self._fft_in = pyfftw.empty_aligned((nx, nx), dtype='complex128')
        self._fft_out = pyfftw.empty_aligned((nx, nx), dtype='complex128')

        # Pre-plan the FFT (FFTW_MEASURE finds the fastest algorithm)
        self._fft_plan = pyfftw.FFTW(
            self._fft_in, self._fft_out,
            axes=(0, 1),
            direction='FFTW_FORWARD',
            flags=['FFTW_MEASURE'],  # benchmarks to pick optimal algo
            threads=NTHREADS,
        )

    def _update_derived_params(self):
        self.dp = 1.0 / (self.nx * self.dx)
        self.radius = (self.na / self.wl) / self.dp
        self._grid = None
        self._rho = None
        self._mask = None

    def set_params(self, **kwargs):
        for key in ('wl', 'na', 'dx', 'nx'):
            if key in kwargs:
                setattr(self, key, kwargs[key])
        self._update_derived_params()
        # Re-plan FFT if nx changed
        if 'nx' in kwargs:
            self._rebuild_fft_plan()

    def _rebuild_fft_plan(self):
        nx = self.nx
        self._fft_in = pyfftw.empty_aligned((nx, nx), dtype='complex128')
        self._fft_out = pyfftw.empty_aligned((nx, nx), dtype='complex128')
        self._fft_plan = pyfftw.FFTW(
            self._fft_in, self._fft_out,
            axes=(0, 1),
            direction='FFTW_FORWARD',
            flags=['FFTW_MEASURE'],
            threads=NTHREADS,
        )

    @property
    def grid(self):
        if self._grid is None:
            x = np.arange(-self.nx // 2, self.nx // 2, dtype=np.float64)
            self._grid = np.meshgrid(x, x, copy=False, sparse=True)
        return self._grid

    @property
    def rho(self):
        if self._rho is None:
            x, y = self.grid
            self._rho = np.hypot(x, y) / self.radius
        return self._rho

    @property
    def mask(self):
        if self._mask is None:
            self._mask = (self.rho <= 1.0).astype(np.float64)
        return self._mask

    def flat_wavefront(self):
        self.bpp = self._disc(self.radius)

    def aberration_wavefront(self, zer):
        self.zer = zer
        msk = self._disc(self.radius)
        ph = zernike.zernike_sum(zer, rad=self.radius, nx=self.nx)
        self.bpp = ne.evaluate('msk * exp(1j * ph)')

    def shift_phase(self, dx_shift, dy_shift):
        alpha = TWO_PI / (self.nx * self.dx)
        i = np.arange(self.nx, dtype=np.float64) - self.nx // 2
        j = np.arange(self.nx, dtype=np.float64) - self.nx // 2
        phase = alpha * (i[:, np.newaxis] * dx_shift + j[np.newaxis, :] * dy_shift)
        return ne.evaluate('exp(1j * phase)')

    def focus_mode(self, d):
        rho = self.rho
        msk = self.mask
        na, n2, wl = self.na, self.n2, self.wl
        c = n2 * d / wl
        # numexpr: fused element-wise, avoids all temporaries
        return ne.evaluate(
            'msk * c * sqrt(1.0 - where(msk > 0, (na * rho / n2) ** 2, 0.0))'
        )

    def defocus(self, dz):
        focus = self.focus_mode(dz)
        return ne.evaluate('exp(1j * TWO_PI * focus)')

    def _fft2(self, data):
        """Execute pre-planned FFT"""
        # Copy into aligned input buffer
        np.copyto(self._fft_in, data)
        self._fft_plan()
        return self._fft_out

    def get_2d_psf(self, coords):
        sx, sy, sz = coords

        if self.bpp is None or not self.bpp.any():
            self.flat_wavefront()

        bpp = self.bpp  # local ref for numexpr

        if sx != 0.0 or sy != 0.0:
            xy_phase = self.shift_phase(sx, sy)
            bpp_xy = ne.evaluate('xy_phase * bpp')
        else:
            bpp_xy = bpp

        if sz != 0.0:
            z_phase = self.defocus(sz)
            wf = ne.evaluate('bpp_xy * z_phase')
        else:
            wf = bpp_xy

        ft = self._fft2(wf)
        fr = ft.real
        fi = ft.imag

        res = ne.evaluate('fr * fr + fi * fi')

        res /= res.sum()
        return res

    def get_3d_psf(self, coords, start, stop, step):
        sx, sy, sz = coords
        n_steps = int(round((stop - start) / step)) + 1
        n_slices = n_steps - 1

        psf_stack = np.empty((n_slices, self.nx, self.nx), dtype=np.float64)

        zarr = np.linspace(start / step, stop / step, n_steps, dtype=np.int64)
        zarr = np.roll(zarr[:-1], n_slices // 2)

        if self.bpp is None or not self.bpp.any():
            self.flat_wavefront()

        bpp = self.bpp  # local ref for numexpr

        if sx != 0.0 or sy != 0.0:
            xy_phase = self.shift_phase(sx, sy)
            bpp_xy = ne.evaluate('xy_phase * bpp')
        else:
            bpp_xy = bpp

        for m, z in enumerate(zarr):
            ph = self.focus_mode(sz - z * step)

            # numexpr: fused bpp_xy * exp(1j * TWO_PI * ph) — no temporaries
            wf = ne.evaluate('bpp_xy * exp(1j * TWO_PI * ph)')

            # pyfftw: pre-planned, multithreaded FFT
            ft = self._fft2(wf)

            # numexpr: fused |ft|^2 — avoids abs() + square() temporaries
            fr = ft.real
            fi = ft.imag
            ne.evaluate('fr * fr + fi * fi', out=psf_stack[m])

        psf_stack /= psf_stack.sum()

        return psf_stack

    def _disc(self, radius=64.0, origin=None):
        rho_abs = self.rho * self.radius
        disc = (rho_abs < radius).astype(np.float64)
        if origin is not None:
            ox = self.nx // 2
            disc = np.roll(disc, (int(origin[0] - ox), int(origin[1] - ox)), axis=(0, 1))
        return disc


if __name__ == "__main__":
    p = PSF(dx=0.064, nx=256)
    p.flat_wavefront()
    psf3d = p.get_3d_psf((0, 0, 0), -1.6, 1.6, 0.16)
