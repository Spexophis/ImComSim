"""
This script reconstructs the raw data obtained from 2-dimensional linear structured illumination microscopy.
Ruizhe Lin
2024-01-11
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pylab import imshow, subplot, subplots, figure, colorbar
from scipy.fft import fft2 as _fft2, ifft2 as _ifft2, fftshift
import numpy as np
import tifffile as tf
from skimage.filters import window
import psf_generator as pg

_WORKERS = -1


def fft2(a):
    return _fft2(a, workers=_WORKERS)


def ifft2(a):
    return _ifft2(a, workers=_WORKERS)


n_w = {0: 'zero', 1: 'first'}
w_n = {'zero': 0, 'first': 1}


class SIM_RECON:

    def __init__(self, **kwargs):
        self.na = kwargs['numerical_aperture']
        self.nph = kwargs['number_of_shifted_phases']
        self.angs = kwargs['pattern_orientations']
        self.nang = len(self.angs)
        self.sps = kwargs['pattern_spacings']
        self.wl = kwargs['emission_wavelength']
        self.norders = kwargs['number_of_frequency_orders']
        self.img = kwargs['image_stack']
        nz, nx, ny = self.img.shape
        self.img = self.img.reshape(self.nang, self.nph, nx, ny)
        dx = kwargs['image_pixel_size']
        self.ratio = 2
        self.dx = dx / self.ratio
        self.nx = nx * self.ratio
        self.ny = ny * self.ratio
        _psf_obj = pg.PSF(wl=self.wl, na=self.na, dx=self.dx, nx=self.nx)
        _psf_obj.flat_wavefront()
        self.psf = _psf_obj.get_2d_psf((0, 0, 0))
        self.radius = _psf_obj.radius
        self.xv, self.yv = self.meshgrid()
        self.sep_mat = self.sepmatrix()
        self.mu = 0.08
        self.cutoff = 0.01
        self.strength = 1.
        self.sigma = 4.
        self.eh = []
        self.eta = 2
        self.alpha = 0.04
        self.winf = self.window_function(self.alpha)
        self._apod_cache = {}
        # parameters
        self.angles = {i: {'first': self.angs[i]} for i in range(self.nang)}
        self.spacings = {i: {'first': self.sps[i]} for i in range(self.nang)}
        self.magnitudes = {i: {'first': 0.8} for i in range(self.nang)}
        self.phases = {i: {'first': 0} for i in range(self.nang)}

    def meshgrid(self):
        x = np.arange(-self.nx / 2, self.nx / 2)
        y = np.arange(-self.ny / 2, self.ny / 2)
        xv, yv = np.meshgrid(x, y, indexing='ij', sparse=True)
        return np.roll(xv, int(self.nx / 2)), np.roll(yv, int(self.ny / 2))

    def sepmatrix(self):
        nphases = self.nph
        norders = int((self.norders + 1) / 2)
        sepmat = np.zeros((self.norders, nphases), dtype=np.float32)
        j_arr = np.arange(nphases)
        sepmat[0, :] = 1.0
        for order in range(1, norders):
            sepmat[2 * order - 1, :] = 2 * np.cos(2 * np.pi * j_arr * order / nphases) / nphases
            sepmat[2 * order, :] = 2 * np.sin(2 * np.pi * j_arr * order / nphases) / nphases
        return np.linalg.inv(np.transpose(sepmat))

    def separate(self, nangle):
        self.angle_index = nangle
        angle, nph, Nw, Nw = self.img.shape
        outr = np.dot(self.sep_mat, self.img[nangle].reshape(nph, Nw ** 2))
        self.separr = np.zeros((self.norders, self.ratio * Nw, self.ratio * Nw), dtype=np.complex64)
        self.separr[0] = fftshift(self._interp(outr[0].reshape(Nw, Nw), self.ratio) * self.winf)
        for i in range(int(self.norders / 2)):
            self.separr[1 + 2 * i] = fftshift(
                self._interp(((outr[1 + 2 * i] + 1j * outr[2 + 2 * i]) / 2).reshape(Nw, Nw), self.ratio) * self.winf)
            self.separr[2 + 2 * i] = fftshift(
                self._interp(((outr[1 + 2 * i] - 1j * outr[2 + 2 * i]) / 2).reshape(Nw, Nw), self.ratio) * self.winf)
        self.otfs = {}
        self.imgfs = {}
        self.otf_zero_order()

    def otf_zero_order(self):
        self.otfs['zero'] = fft2(self.psf)      # zero_suppression always returns 1
        self.imgfs['zero'] = fft2(self.separr[0])

    def _compute_shift_indices(self, kx, ky):
        """Analytically compute FFT peak indices for exp(2πi(kx·xv + ky·yv)).

        Avoids an O(N²·log N) FFT call that was previously used just for argmax.
        For the rolled coordinate grids used here, the DFT peak is at index
        round(k * N) mod N, remapped to [-N/2, N/2).
        """
        Nw = self.nx
        sx = int(round(kx * Nw)) % Nw
        sy = int(round(ky * Nw)) % Nw
        if sx >= Nw // 2:
            sx -= Nw
        if sy >= Nw // 2:
            sy -= Nw
        return sx, sy

    def shift_otfs_n_imgfs(self, pattern_orientation=0, pattern_spacing=0.24, frequency_order='first'):
        """ shift data in freq space by multiplication in real space """
        order = w_n[frequency_order]
        dx = self.dx
        kx = dx * np.cos(pattern_orientation) / pattern_spacing
        ky = dx * np.sin(pattern_orientation) / pattern_spacing
        ysh_pos = np.exp(2j * np.pi * (kx * self.xv + ky * self.yv)).astype(np.complex64)
        ysh_neg = ysh_pos.conj()
        # shift to positive — zero_suppression always returns 1, so zsp is omitted
        sx, sy = self._compute_shift_indices(kx, ky)
        self.eh = np.append(self.eh, np.sqrt(sx ** 2 + sy ** 2) / (2 * self.radius))
        self.otfs[frequency_order + '_positive'] = fft2(self.psf * ysh_pos)
        self.imgfs[frequency_order + '_positive'] = fft2(self.separr[2 * order - 1] * ysh_pos)
        # shift to negative
        sx, sy = self._compute_shift_indices(-kx, -ky)
        self.eh = np.append(self.eh, np.sqrt(sx ** 2 + sy ** 2) / (2 * self.radius))
        self.otfs[frequency_order + '_negative'] = fft2(self.psf * ysh_neg)
        self.imgfs[frequency_order + '_negative'] = fft2(self.separr[2 * order] * ysh_neg)

    def get_overlap_w_zero(self, shift_orientation=0, shift_spacing=0.24, order_to_be_computed='first', verbose=False):
        """ shift data in freq space by multiplication in real space """
        order = w_n[order_to_be_computed]
        dx = self.dx
        cutoff = self.cutoff
        imgf0 = self.imgfs['zero']
        otf0 = self.otfs['zero']
        kx = dx * np.cos(shift_orientation) / shift_spacing
        ky = dx * np.sin(shift_orientation) / shift_spacing
        ysh = np.exp(2j * np.pi * (kx * self.xv + ky * self.yv))
        # zero_suppression always returns 1 — skip the argmax FFT entirely
        otf = fft2(self.psf * ysh)
        imgf = fft2(self.separr[2 * order - 1] * ysh)
        # calculate overlapping area
        wimgf0 = otf * imgf0
        wimgf1 = otf0 * imgf
        msk = abs(otf0 * otf) > cutoff
        a = np.sum(msk * wimgf1 * wimgf0.conj()) / np.sum(msk * wimgf0 * wimgf0.conj())
        mag = np.abs(a)
        phase = np.angle(a)
        if verbose:
            t = (msk * wimgf1 * wimgf0.conj()) / (msk * wimgf0 * wimgf0.conj())
            t[np.isnan(t)] = 0.0
            figure()
            imshow(np.abs(fftshift(t)), interpolation='nearest', vmin=0.0, vmax=2.0)
            colorbar()
            figure()
            imshow(np.angle(fftshift(t)), interpolation='nearest')
            colorbar()
        return mag, phase

    def map_overlap_w_zero(self, order='first', nps=10, r_ang=0.005, r_sp=0.005, verbose=True):
        angle = self.angles[self.angle_index][order]
        spacing = self.spacings[self.angle_index][order]
        d_ang = 2 * r_ang / nps
        d_sp = 2 * r_sp / nps
        ang_iter = np.arange(-r_ang, r_ang + d_ang / 2, d_ang) + angle
        sp_iter = np.arange(-r_sp, r_sp + d_sp / 2, d_sp) + spacing
        magarr = np.zeros((nps + 1, nps + 1))
        pharr = np.zeros((nps + 1, nps + 1))
        for m, ang in enumerate(ang_iter):
            for n, sp in enumerate(sp_iter):
                print(m, n)
                mag, phase = self.get_overlap_w_zero(ang, sp, order)
                if np.isnan(mag):
                    magarr[m, n] = 0.0
                else:
                    magarr[m, n] = mag
                    pharr[m, n] = phase
        if verbose:
            figure()
            subplot(211)
            imshow(magarr, vmin=magarr.min(), vmax=magarr.max(),
                   extent=[sp_iter.min(), sp_iter.max(), ang_iter.max(), ang_iter.min()], interpolation=None)
            subplot(212)
            imshow(pharr, interpolation=None)
        # get maximum
        k, l = np.where(magarr == magarr.max())
        angmax = k[0] * d_ang - r_ang + angle
        spmax = l[0] * d_sp - r_sp + spacing
        self.angles[self.angle_index][order] = angmax
        self.spacings[self.angle_index][order] = spmax
        self.magnitudes[self.angle_index][order] = magarr[k, l][0]
        self.phases[self.angle_index][order] = pharr[k, l][0]

    # --- Reconstruction helpers ---

    def _init_reconstruction(self, zero_order):
        nx, ny = self.nx, self.ny
        self.Snum = np.zeros((nx, ny), dtype=np.complex64)
        self.Sden = np.full((nx, ny), self.mu ** 2, dtype=np.complex64)
        if zero_order:
            otf = self.otfs['zero']
            self.Snum += otf.conj() * self.imgfs['zero']
            self.Sden += otf.real ** 2 + otf.imag ** 2

    def _accumulate_order(self, angle_idx, od):
        key = n_w[od + 1]
        self.shift_otfs_n_imgfs(pattern_orientation=self.angles[angle_idx][key],
                                pattern_spacing=self.spacings[angle_idx][key], frequency_order=key)
        ph = self.magnitudes[angle_idx][key] * np.exp(-1j * self.phases[angle_idx][key])
        otf_pos = self.otfs[key + '_positive']
        otf_neg = self.otfs[key + '_negative']
        self.Snum += ph * otf_pos.conj() * self.imgfs[key + '_positive']
        self.Snum += ph.conj() * otf_neg.conj() * self.imgfs[key + '_negative']
        self.Sden += otf_pos.real ** 2 + otf_pos.imag ** 2
        self.Sden += otf_neg.real ** 2 + otf_neg.imag ** 2

    def _finalize_reconstruction(self):
        A = self.apod(self.eta)
        self.S = A * self.Snum / self.Sden
        self.finalimage = fftshift(ifft2(self.S))

    def reconstruct_by_angle(self, angle_indices=None, zero_order=True):
        if angle_indices is None:
            angle_indices = [0, 2]
        self._init_reconstruction(zero_order)
        for i, ai in enumerate(angle_indices):
            self.separate(ai)
            for od in range(int(self.norders / 2)):
                self._accumulate_order(i, od)
        self._finalize_reconstruction()

    def reconstruct_by_order(self, order=1, zero_order=True):
        self._init_reconstruction(zero_order)
        for i in range(self.nang):
            self.separate(i)
            for od in range(order):
                self._accumulate_order(i, od)
        self._finalize_reconstruction()

    def reconstruct_all(self, zero_order=True):
        self._init_reconstruction(zero_order)
        for i in range(self.nang):
            self.separate(i)
            for od in range(int(self.norders / 2)):
                self._accumulate_order(i, od)
        self._finalize_reconstruction()

    def save_reconstruction(self, fn=''):
        tf.imwrite(fn + 'sim2d_final_image.tif', self.finalimage.real.astype(np.float32), photometric='minisblack')
        tf.imwrite(fn + 'sim2d_effective_OTF.tif', np.abs(fftshift(self.S)).astype(np.float32),
                   photometric='minisblack')

    def window_function(self, alpha):
        wxy = window(('tukey', alpha), self.nx)
        return np.outer(wxy, wxy)

    def zero_suppression(self, sx, sy, h=False):
        if h:
            return 1 - self.strength * np.exp(-((self.xv - sx) ** 2. + (self.yv - sy) ** 2.) / (2. * self.sigma ** 2.))
        else:
            return 1

    def apod(self, eta):
        if eta not in self._apod_cache:
            self._apod_cache[eta] = fftshift(window(('kaiser', eta), (self.nx, self.nx)))
        return self._apod_cache[eta]

    @staticmethod
    def _interp(arr, ratio):
        nx, ny = arr.shape
        px = int((nx * (ratio - 1)) / 2)
        py = int((ny * (ratio - 1)) / 2)
        arrf = fft2(arr)
        arro = np.pad(fftshift(arrf), ((px, px), (py, py)), 'constant', constant_values=0)
        return ifft2(fftshift(arro))


if __name__ == '__main__':
    img = tf.imread(r"sim2d_simulation_data.tif")
    p = SIM_RECON(image_stack=img,
                  image_pixel_size=0.08,
                  numerical_aperture=1.4,
                  emission_wavelength=0.505,
                  number_of_shifted_phases=3,
                  number_of_frequency_orders=3,
                  pattern_orientations=[0, 2 * np.pi / 3, 4 * np.pi / 3],
                  pattern_spacings=[0.24, 0.24, 0.24])
    for i in range(3):
        p.separate(i)
        p.map_overlap_w_zero(order='first', nps=10, r_ang=0.005, r_sp=0.005, verbose=False)
    print(p.angles)
    print(p.spacings)
    print(p.magnitudes)
    # p.magnitudes = {i: {'first': 0.8, 'second': 0.4} for i in range(p.nang)}
    p.reconstruct_all(zero_order=True)
    p.save_reconstruction()
