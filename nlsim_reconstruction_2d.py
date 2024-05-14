"""
This script reconstructs the raw data obtained from 2-dimensional nonlinear structured illumination microscopy.
Ruizhe Lin
2024-01-10
"""

from matplotlib.pyplot import imshow, subplot, subplots, figure, colorbar
from scipy.special import factorial
import numpy as np
import tifffile as tf
from skimage.filters import window

fft2 = np.fft.fft2
ifft2 = np.fft.ifft2
fftshift = np.fft.fftshift

n_w = {0: 'zero', 1: 'first', 2: 'second', 3: 'third', 4: 'fourth'}
w_n = {'zero': 0, 'first': 1, 'second': 2, 'third': 3, 'fourth': 4}


class NLSIM_RECON:

    def __init__(self, **kwargs):
        self.na = kwargs['numerical_aperture']
        self.nph = kwargs['number_of_shifted_phases']
        self.angs = kwargs['pattern_orientations']
        self.nang = len(self.angs)
        self.sps = kwargs['pattern_spacings']
        self.wl = kwargs['emission_wavelength']
        self.norders = kwargs['number_of_frequency_orders']
        resolution_target = (self.wl / (2 * self.na)) / int(self.norders / 2)
        self.img = kwargs['image_stack']
        nz, nx, ny = self.img.shape
        self.img = self.img.reshape(self.nang, self.nph, nx, ny)
        dx = kwargs['image_pixel_size']
        self.ratio = int(dx / (resolution_target / 2)) + 1
        self.dx = dx / self.ratio
        self.nx = nx * self.ratio
        self.ny = ny * self.ratio
        self.psf, self.radius = self.get_psf()
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
        # parameters
        self.angles = {i: {'first': self.angs[i], 'second': self.angs[i], 'third': self.angs[i], 'fourth': self.angs[i]}
                       for i in range(self.nang)}
        self.spacings = {
            i: {'first': self.sps[i], 'second': self.sps[i] / 2, 'third': self.sps[i] / 3, 'fourth': self.sps[i] / 4}
            for i in range(self.nang)}
        self.magnitudes = {i: {'first': 0.8, 'second': 0.4, 'third': 0.2, 'fourth': 0.1} for i in range(self.nang)}
        self.phases = {i: {'first': 0, 'second': 0, 'third': 0, 'fourth': 0} for i in range(self.nang)}

    def _get_pupil(self, zarr=None):
        dp = 1 / (self.nx * self.dx)
        radius = (self.na / self.wl) / dp
        msk = self._shift(self._disc_array(shape=(self.nx, self.ny), radius=radius)) / np.sqrt(np.pi * radius ** 2) / (
            self.nx)
        phi = np.zeros((self.nx, self.ny))
        wf = msk * np.exp(1j * phi).astype(np.complex64)
        if zarr is not None:
            for z in range(len(zarr)):
                n, m = self._zernike_j_nm(z + 1)
                phi += zarr[z] * self._zernike(n, m, radius=radius, shape=(self.nxh * 2, self.nyh * 2))
            wf *= np.exp(1j * phi).astype(np.complex64)
        return wf, radius

    def get_psf(self, zernike_arr=None):
        bpp, radius = self._get_pupil(zernike_arr)
        psf1 = np.abs((fft2(fftshift(bpp)))) ** 2
        return psf1 / psf1.sum(), radius

    def meshgrid(self):
        x = np.arange(-self.nx / 2, self.nx / 2)
        y = np.arange(-self.ny / 2, self.ny / 2)
        xv, yv = np.meshgrid(x, y, indexing='ij', sparse=True)
        return np.roll(xv, int(self.nx / 2)), np.roll(yv, int(self.ny / 2))

    def sepmatrix(self):
        nphases = self.nph
        norders = int((self.norders + 1) / 2)
        sepmat = np.zeros((self.norders, nphases), dtype=np.float32)
        for j in range(nphases):
            sepmat[0, j] = 1.0
            for order in range(1, norders):
                sepmat[2 * order - 1, j] = 2 * np.cos(2 * np.pi * (j * order) / nphases) / nphases
                sepmat[2 * order, j] = 2 * np.sin(2 * np.pi * (j * order) / nphases) / nphases
        return np.linalg.inv(np.transpose(sepmat))

    def separate(self, nangle):
        self.angle_index = nangle
        angle, nph, Nw, Nw = self.img.shape
        outr = np.dot(self.sep_mat, self.img[nangle].reshape(nph, Nw ** 2))
        self.separr = np.zeros((self.norders, self.ratio * Nw, self.ratio * Nw), dtype=np.complex64)
        self.separr[0] = np.fft.fftshift(self._interp(outr[0].reshape(Nw, Nw), self.ratio) * self.winf)
        for i in range(int(self.norders / 2)):
            self.separr[1 + 2 * i] = np.fft.fftshift(
                self._interp(((outr[1 + 2 * i] + 1j * outr[2 + 2 * i]) / 2).reshape(Nw, Nw), self.ratio) * self.winf)
            self.separr[2 + 2 * i] = np.fft.fftshift(
                self._interp(((outr[1 + 2 * i] - 1j * outr[2 + 2 * i]) / 2).reshape(Nw, Nw), self.ratio) * self.winf)
        self.otfs = {}
        self.imgfs = {}
        self.otf_zero_order()

    def otf_zero_order(self):
        zsp = self.zero_suppression(0, 0)
        self.otfs['zero'] = fft2(self.psf) * zsp
        self.imgfs['zero'] = fft2(self.separr[0])

    def shift_otfs_imgfs(self, pattern_orientation=0, pattern_spacing=0.24, frequency_order='first'):
        """ shift data in freq space by multiplication in real space """
        order = w_n[frequency_order]
        Nw = self.nx
        dx = self.dx
        kx = dx * np.cos(pattern_orientation) / pattern_spacing
        ky = dx * np.sin(pattern_orientation) / pattern_spacing
        # shift matrix
        ysh = np.zeros((2, Nw, Nw), dtype=np.complex64)
        ysh[0, :, :] = np.exp(2j * np.pi * (kx * self.xv + ky * self.yv))
        ysh[1, :, :] = np.exp(2j * np.pi * (-kx * self.xv - ky * self.yv))
        # shift otf and imgf to positive
        yshf = np.abs(fft2(ysh[0]))
        sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if sx < Nw / 2:
            sx = sx
        else:
            sx = sx - Nw
        if sy < Nw / 2:
            sy = sy
        else:
            sy = sy - Nw
        temp = np.sqrt(sx ** 2 + sy ** 2) / (2 * self.radius)
        self.eh = np.append(self.eh, temp)
        zsp = self.zero_suppression(sx, sy)
        self.otfs[frequency_order + '_positive'] = fft2(self.psf * ysh[0]) * zsp
        self.imgfs[frequency_order + '_positive'] = fft2(self.separr[2 * order - 1] * ysh[0])
        # shift otf and imgf to negative
        yshf = np.abs(fft2(ysh[1]))
        sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if sx < Nw / 2:
            sx = sx
        else:
            sx = sx - Nw
        if sy < Nw / 2:
            sy = sy
        else:
            sy = sy - Nw
        temp = np.sqrt(sx ** 2 + sy ** 2) / (2 * self.radius)
        self.eh = np.append(self.eh, temp)
        zsp = self.zero_suppression(sx, sy)
        self.otfs[frequency_order + '_negative'] = fft2(self.psf * ysh[1]) * zsp
        self.imgfs[frequency_order + '_negative'] = fft2(self.separr[2 * order] * ysh[1])

    def getoverlap_w_zero(self, shift_orientation=0, shift_spacing=0.24, order_to_be_computed='first', verbose=False):
        """ shift data in freq space by multiplication in real space """
        order = w_n[order_to_be_computed]
        dx = self.dx
        Nw = self.nx
        cutoff = self.cutoff
        imgf0 = self.imgfs['zero']
        otf0 = self.otfs['zero']
        kx = dx * np.cos(shift_orientation) / shift_spacing
        ky = dx * np.sin(shift_orientation) / shift_spacing
        ysh = np.exp(2j * np.pi * (kx * self.xv + ky * self.yv))
        yshf = np.abs(fft2(ysh))
        sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if sx < Nw / 2:
            sx = sx
        else:
            sx = sx - Nw
        if sy < Nw / 2:
            sy = sy
        else:
            sy = sy - Nw
        zsp = self.zero_suppression(sx, sy)
        otf = fft2(self.psf * ysh) * zsp
        imgf = fft2(self.separr[2 * order - 1] * ysh)
        # calculate ovelapping area
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
            imshow(np.abs(np.fft.fftshift(t)), interpolation='nearest', vmin=0.0, vmax=2.0)
            colorbar()
            figure()
            imshow(np.angle(np.fft.fftshift(t)), interpolation='nearest')
            colorbar()
        return mag, phase

    def mapoverlap_w_zero(self, order='first', nps=10, r_ang=0.005, r_sp=0.005, verbose=True):
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
                mag, phase = self.getoverlap_w_zero(ang, sp, order)
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

    def getoverlap(self, shift_orientations=[0, 0], shift_spacings=[0.24, 0.12],
                   orders_to_be_computed=['first', 'second'], verbose=False):
        """ shift data in freq space by multiplication in real space """
        dx = self.dx
        Nw = self.nx
        cutoff = self.cutoff
        # shift first term
        kx = dx * np.cos(shift_orientations[0]) / shift_spacings[0]
        ky = dx * np.sin(shift_orientations[0]) / shift_spacings[0]
        ysh = np.exp(2j * np.pi * (kx * self.xv + ky * self.yv))
        yshf = np.abs(fft2(ysh))
        sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if sx < Nw / 2:
            sx = sx
        else:
            sx = sx - Nw
        if sy < Nw / 2:
            sy = sy
        else:
            sy = sy - Nw
        zsp = self.zero_suppression(sx, sy)
        otf0 = fft2(self.psf * ysh) * zsp
        imgf0 = fft2(self.separr[2 * w_n[orders_to_be_computed[0]] - 1] * ysh)
        # shift second term
        kx = dx * np.cos(shift_orientations[1]) / shift_spacings[1]
        ky = dx * np.sin(shift_orientations[1]) / shift_spacings[1]
        ysh = np.exp(2j * np.pi * (kx * self.xv + ky * self.yv))
        yshf = np.abs(fft2(ysh))
        sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if sx < Nw / 2:
            sx = sx
        else:
            sx = sx - Nw
        if sy < Nw / 2:
            sy = sy
        else:
            sy = sy - Nw
        zsp = self.zero_suppression(sx, sy)
        otf = fft2(self.psf * ysh) * zsp
        imgf = fft2(self.separr[2 * w_n[orders_to_be_computed[1]] - 1] * ysh)
        # calculate ovelapping area
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
            imshow(np.abs(np.fft.fftshift(t)), interpolation='nearest', vmin=0.0, vmax=2.0)
            colorbar()
            figure()
            imshow(np.angle(np.fft.fftshift(t)), interpolation='nearest')
            colorbar()
        return mag, phase

    def mapoverlap(self, orders=['first', 'second'], nps=10, r_ang=0.005, r_sp=0.005, verbose=True):
        angles = [self.angles[self.angle_index][orders[0]], self.angles[self.angle_index][orders[1]]]
        spacings = [self.spacings[self.angle_index][orders[0]], self.spacings[self.angle_index][orders[1]]]
        d_ang = 2 * r_ang / nps
        d_sp = 2 * r_sp / nps
        ang_iter = np.arange(-r_ang, r_ang + d_ang / 2, d_ang) + angles[1]
        sp_iter = np.arange(-r_sp, r_sp + d_sp / 2, d_sp) + spacings[1]
        magarr = np.zeros((nps + 1, nps + 1))
        pharr = np.zeros((nps + 1, nps + 1))
        for m, ang in enumerate(ang_iter):
            for n, sp in enumerate(sp_iter):
                print(m, n)
                mag, phase = self.getoverlap([angles[0], ang], [spacings[0], sp], orders)
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
        angmax = k[0] * d_ang - r_ang + angles[1]
        spmax = l[0] * d_sp - r_sp + spacings[1]
        self.angles[self.angle_index][orders[1]] = angmax
        self.spacings[self.angle_index][orders[1]] = spmax
        self.magnitudes[self.angle_index][orders[1]] = self.magnitudes[self.angle_index][orders[0]] * magarr[k, l][0]
        self.phases[self.angle_index][orders[1]] = self.phases[self.angle_index][orders[0]] + pharr[k, l][0]

    def check_components(self):
        """ plot components in Fourier space """
        fig, axs = subplots(1, 5)
        for m in range(5):
            temp = np.abs(fftshift(fft2(self.separr[m]))) ** 0.5
            axs[m].imshow(temp)

    def reconstruct_by_angle(self, angle_indices=[0, 2], zero_order=True):
        # reconstruct n angles
        nx = self.nx
        ny = self.ny
        mu = self.mu
        self.Snum = np.zeros((nx, ny), dtype=np.complex64)
        self.Sden = np.zeros((nx, ny), dtype=np.complex64)
        self.Sden += mu ** 2
        if zero_order:
            imgf = self.imgfs['zero']
            otf = self.otfs['zero']
            self.Snum += otf.conj() * imgf
            self.Sden += abs(otf) ** 2
        for i in range(len(angle_indices)):
            self.separate(angle_indices[i])
            for od in range(int(self.norders / 2)):
                self.shift_otfs_imgfs(pattern_orientation=self.angles[i][n_w[od + 1]],
                                      pattern_spacing=self.spacings[i][n_w[od + 1]], frequency_order=n_w[od + 1])
                ph = self.magnitudes[i][n_w[od + 1]] * np.exp(-1j * self.phases[i][n_w[od + 1]])
                self.Snum += ph * self.otfs[n_w[od + 1] + '_positive'].conj() * self.imgfs[n_w[od + 1] + '_positive']
                self.Sden += abs(self.otfs[n_w[od + 1] + '_positive']) ** 2
                self.Snum += ph.conj() * self.otfs[n_w[od + 1] + '_negative'].conj() * self.imgfs[
                    n_w[od + 1] + '_negative']
                self.Sden += abs(self.otfs[n_w[od + 1] + '_negative']) ** 2
        A = self.apod(self.eta)
        self.S = A * self.Snum / self.Sden
        self.finalimage = fftshift(ifft2(self.S))

    def reconstruct_by_order(self, order=1, zero_order=True):
        n = self.nang
        nx = self.nx
        ny = self.ny
        mu = self.mu
        self.Snum = np.zeros((nx, ny), dtype=np.complex64)
        self.Sden = np.zeros((nx, ny), dtype=np.complex64)
        self.Sden += mu ** 2
        if zero_order:
            imgf = self.imgfs['zero']
            otf = self.otfs['zero']
            self.Snum += otf.conj() * imgf
            self.Sden += abs(otf) ** 2
        for i in range(n):
            self.separate(i)
            for od in range(order):
                self.shift_otfs_imgfs(pattern_orientation=self.angles[i][n_w[od + 1]],
                                      pattern_spacing=self.spacings[i][n_w[od + 1]], frequency_order=n_w[od + 1])
                ph = self.magnitudes[i][n_w[od + 1]] * np.exp(-1j * self.phases[i][n_w[od + 1]])
                self.Snum += ph * self.otfs[n_w[od + 1] + '_positive'].conj() * self.imgfs[n_w[od + 1] + '_positive']
                self.Sden += abs(self.otfs[n_w[od + 1] + '_positive']) ** 2
                self.Snum += ph.conj() * self.otfs[n_w[od + 1] + '_negative'].conj() * self.imgfs[
                    n_w[od + 1] + '_negative']
                self.Sden += abs(self.otfs[n_w[od + 1] + '_negative']) ** 2
        A = self.apod(self.eta)
        self.S = A * self.Snum / self.Sden
        self.finalimage = fftshift(ifft2(self.S))

    def reconstruct_all(self, zero_order=True):
        n = self.nang
        nx = self.nx
        ny = self.ny
        mu = self.mu
        self.Snum = np.zeros((nx, ny), dtype=np.complex64)
        self.Sden = np.zeros((nx, ny), dtype=np.complex64)
        self.Sden += mu ** 2
        if zero_order:
            imgf = self.imgfs['zero']
            otf = self.otfs['zero']
            self.Snum += otf.conj() * imgf
            self.Sden += abs(otf) ** 2
        for i in range(n):
            self.separate(i)
            for od in range(int(self.norders / 2)):
                self.shift_otfs_imgfs(pattern_orientation=self.angles[i][n_w[od + 1]],
                                      pattern_spacing=self.spacings[i][n_w[od + 1]], frequency_order=n_w[od + 1])
                ph = self.magnitudes[i][n_w[od + 1]] * np.exp(-1j * self.phases[i][n_w[od + 1]])
                self.Snum += ph * self.otfs[n_w[od + 1] + '_positive'].conj() * self.imgfs[n_w[od + 1] + '_positive']
                self.Sden += abs(self.otfs[n_w[od + 1] + '_positive']) ** 2
                self.Snum += ph.conj() * self.otfs[n_w[od + 1] + '_negative'].conj() * self.imgfs[
                    n_w[od + 1] + '_negative']
                self.Sden += abs(self.otfs[n_w[od + 1] + '_negative']) ** 2
        A = self.apod(self.eta)
        self.S = A * self.Snum / self.Sden
        self.finalimage = fftshift(ifft2(self.S))

    def save_reconstruction(self, fn=''):
        tf.imwrite(fn + 'nlsim2d_final_image.tif', self.finalimage.real.astype(np.float32), photometric='minisblack')
        tf.imwrite(fn + 'nlsim2d_effective_OTF.tif', np.abs(fftshift(self.S)).astype(np.float32),
                   photometric='minisblack')

    def window_function(self, alpha):
        wxy = window(('tukey', alpha), self.nx)
        wx = np.tile(wxy, (self.nx, 1))
        wy = wx.swapaxes(0, 1)
        w = wx * wy
        return w

    def zero_suppression(self, sx, sy, h=False):
        if h:
            return 1 - self.strength * np.exp(-((self.xv - sx) ** 2. + (self.yv - sy) ** 2.) / (2. * self.sigma ** 2.))
        else:
            return 1

    def apod(self, eta):
        return fftshift(
            window(('kaiser', eta), (self.nx, self.nx)))

    @staticmethod
    def _interp(arr, ratio):
        nx, ny = arr.shape
        px = int((nx * (ratio - 1)) / 2)
        py = int((ny * (ratio - 1)) / 2)
        arrf = fft2(arr)
        arro = np.pad(np.fft.fftshift(arrf), ((px, px), (py, py)), 'constant', constant_values=(0))
        out = ifft2(np.fft.fftshift(arro))
        return out

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
    img = tf.imread(r"202403131228_nlsim2d_simulation_image_stack.tif")
    p = NLSIM_RECON(image_stack=img,
                    image_pixel_size=0.08,
                    numerical_aperture=1.4,
                    emission_wavelength=0.505,
                    number_of_shifted_phases=7,
                    number_of_frequency_orders=7,
                    pattern_orientations=[0, 2 * np.pi / 7, 4 * np.pi / 7, 6 * np.pi / 7, 8 * np.pi / 7,
                                          10 * np.pi / 7,
                                          12 * np.pi / 7],
                    pattern_spacings=[0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24])
    for i in range(7):
        p.separate(i)
        p.mapoverlap_w_zero(order='first', nps=10, r_ang=0.005, r_sp=0.005, verbose=True)
        p.mapoverlap(orders=['first', 'second'], nps=10, r_ang=0.005, r_sp=0.005, verbose=True)
        p.mapoverlap(orders=['second', 'third'], nps=10, r_ang=0.005, r_sp=0.005, verbose=True)
    print(p.angles)
    print(p.spacings)
    print(p.magnitudes)
    # p.magnitudes = {i: {'first': 0.8, 'second': 0.4, 'third': 0.2, 'fourth': 0.1} for i in range(p.nang)}
    p.reconstruct_all(zero_order=True)
    p.save_reconstruction(fn=r"202403131228_")
