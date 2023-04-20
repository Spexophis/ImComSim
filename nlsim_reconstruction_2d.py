import re

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf
from skimage.filters import window

fft2 = np.fft.fft2
ifft2 = np.fft.ifft2
fftshift = np.fft.fftshift

n_w = {0: 'zero', 1: 'first', 2: 'second', 3: 'third', 4: 'fourth'}
w_n = {'zero': 0, 'first': 1, 'second': 2, 'third': 3, 'fourth': 4}


class NLSIM2D():

    def __init__(self, **kwargs):
        self.na = kwargs['numerical_aperture']
        self.nph = kwargs['number_of_shifted_phases']
        angs = re.findall(r"[-+]?\d*\.\d+|\d+", kwargs['pattern_orientations'])
        self.angs = [float(item) for item in angs]
        self.nang = len(self.angs)
        sps = re.findall(r"[-+]?\d*\.\d+|\d+", kwargs['pattern_spacings'])
        self.sps = [float(item) for item in sps]
        self.wl = kwargs['emission_wavelength']
        self.norders = kwargs['number_of_frequency_orders']
        resolution_target = (self.wl / (2 * self.na)) / int(self.norders / 2)
        self.img = kwargs['image_stack']
        nz, ny, nx = self.img.shape
        self.img = self.img.reshape(self.nang, self.nph, ny, nx)
        dx = kwargs['image_pixel_size']
        self.ratio = int(dx / (resolution_target / 2)) + 1
        self.dx = dx / self.ratio
        self.nx = nx * self.ratio
        self.ny = ny * self.ratio
        self.radius = None
        self.psf = self.get_psf()
        self.meshgrid()
        self.sepmat = self.sepmatrix()
        self.mu = 0.08
        self.cutoff = 0.01
        self.strength = 1.
        self.sigma = 4.
        self.eh = []
        self.eta = 2
        self.alpha = 0.04
        self.winf = self._window(self.alpha)
        # parameters
        self.angles = {i: {'first': self.angs[i], 'second': self.angs[i], 'third': self.angs[i], 'fourth': self.angs[i]}
                       for i in range(self.nang)}
        self.spacings = {
            i: {'first': self.sps[i], 'second': self.sps[i] / 2, 'third': self.sps[i] / 3, 'fourth': self.sps[i] / 4}
            for i in range(self.nang)}
        self.magnitudes = {i: {'first': 0.8, 'second': 0.4, 'third': 0.2, 'fourth': 0.1} for i in range(self.nang)}
        self.phases = {i: {'first': 0, 'second': 0, 'third': 0, 'fourth': 0} for i in range(self.nang)}

    def get_psf(self):
        dp = 1 / (self.nx * self.dx)
        self.radius = (self.na / self.wl) / dp
        bpp = self._disc_array(shape=(self.ny, self.nx), radius=self.radius)
        psf = np.abs((fft2(fftshift(bpp)))) ** 2
        return psf / psf.sum()

    def _disc_array(self, shape=(128, 128), radius=64):
        ny, nx = shape
        ox = nx / 2
        oy = ny / 2
        x = np.linspace(-ox, ox - 1, nx)
        y = np.linspace(-oy, oy - 1, ny)
        yv, xv = np.meshgrid(y, x, indexing='ij')
        rho = np.sqrt(xv ** 2 + yv ** 2)
        return (rho < radius).astype(np.float64)

    def meshgrid(self):
        x = np.arange(-self.nx / 2, self.nx / 2)
        y = np.arange(-self.ny / 2, self.ny / 2)
        xv, yv = np.meshgrid(x, y, indexing='ij', sparse=True)
        self.xv = np.roll(xv, int(self.nx / 2))
        self.yv = np.roll(yv, int(self.ny / 2))

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
        outr = np.dot(self.sepmat, self.img[nangle].reshape(nph, Nw ** 2))
        self.separr = np.zeros((self.norders, self.ratio * Nw, self.ratio * Nw), dtype=np.complex64)
        self.separr[0] = np.fft.fftshift(self.interp(outr[0].reshape(Nw, Nw), self.ratio) * self.winf)
        for i in range(int(self.norders / 2)):
            self.separr[1 + 2 * i] = np.fft.fftshift(
                self.interp(((outr[1 + 2 * i] + 1j * outr[2 + 2 * i]) / 2).reshape(Nw, Nw), self.ratio) * self.winf)
            self.separr[2 + 2 * i] = np.fft.fftshift(
                self.interp(((outr[1 + 2 * i] - 1j * outr[2 + 2 * i]) / 2).reshape(Nw, Nw), self.ratio) * self.winf)
        self.otfs = {}
        self.imgfs = {}
        self.otf_zero_order()

    def otf_zero_order(self):
        zsp = self.zerosuppression(0, 0)
        self.otfs['zero'] = fft2(self.psf) * zsp
        self.imgfs['zero'] = fft2(self.separr[0])

    def shift_otfs_imgfs(self, pattern_orientation=0, pattern_spacing=0.24, frequency_order='first'):
        ''' shift data in freq space by multiplication in real space '''
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
        zsp = self.zerosuppression(sx, sy)
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
        zsp = self.zerosuppression(sx, sy)
        self.otfs[frequency_order + '_negative'] = fft2(self.psf * ysh[1]) * zsp
        self.imgfs[frequency_order + '_negative'] = fft2(self.separr[2 * order] * ysh[1])

    def getoverlap_w_zero(self, shift_orientation=0, shift_spacing=0.24, order_to_be_computed='first', verbose=False):
        ''' shift data in freq space by multiplication in real space '''
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
        zsp = self.zerosuppression(sx, sy)
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
            plt.figure()
            plt.imshow(np.abs(np.fft.fftshift(t)), interpolation='nearest', vmin=0.0, vmax=2.0)
            plt.colorbar()
            plt.figure()
            plt.imshow(np.angle(np.fft.fftshift(t)), interpolation='nearest')
            plt.colorbar()
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
            plt.figure()
            plt.subplot(211)
            plt.imshow(magarr, vmin=magarr.min(), vmax=magarr.max(),
                       extent=[sp_iter.min(), sp_iter.max(), ang_iter.max(), ang_iter.min()], interpolation=None)
            plt.subplot(212)
            plt.imshow(pharr, interpolation=None)
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
        ''' shift data in freq space by multiplication in real space '''
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
        zsp = self.zerosuppression(sx, sy)
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
        zsp = self.zerosuppression(sx, sy)
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
            plt.figure()
            plt.imshow(np.abs(np.fft.fftshift(t)), interpolation='nearest', vmin=0.0, vmax=2.0)
            plt.colorbar()
            plt.figure()
            plt.imshow(np.angle(np.fft.fftshift(t)), interpolation='nearest')
            plt.colorbar()
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
            plt.figure()
            plt.subplot(211)
            plt.imshow(magarr, vmin=magarr.min(), vmax=magarr.max(),
                       extent=[sp_iter.min(), sp_iter.max(), ang_iter.max(), ang_iter.min()], interpolation=None)
            plt.subplot(212)
            plt.imshow(pharr, interpolation=None)
        # get maximum
        k, l = np.where(magarr == magarr.max())
        angmax = k[0] * d_ang - r_ang + angles[1]
        spmax = l[0] * d_sp - r_sp + spacings[1]
        self.angles[self.angle_index][orders[1]] = angmax
        self.spacings[self.angle_index][orders[1]] = spmax
        self.magnitudes[self.angle_index][orders[1]] = self.magnitudes[self.angle_index][orders[0]] * magarr[k, l][0]
        self.phases[self.angle_index][orders[1]] = self.phases[self.angle_index][orders[0]] + pharr[k, l][0]

    def reconstruct_by_angle(self, angle_indices=[0, 2], zero_order=True):
        # reconstruct n angles
        nx = self.nx
        ny = self.ny
        mu = self.mu
        self.Snum = np.zeros((nx, ny), dtype=np.complex64)
        self.Sden = np.zeros((nx, ny), dtype=np.complex64)
        self.Sden += mu ** 2
        if zero_order == True:
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
        if zero_order == True:
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
        if zero_order == True:
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

    def interp(self, arr, ratio):
        nx, ny = arr.shape
        px = int((nx * (ratio - 1)) / 2)
        py = int((ny * (ratio - 1)) / 2)
        arrf = fft2(arr)
        arro = np.pad(np.fft.fftshift(arrf), ((px, px), (py, py)), 'constant', constant_values=(0))
        out = ifft2(np.fft.fftshift(arro))
        return out

    def _window(self, alpha):
        wxy = window(('tukey', alpha), (self.nx))
        wx = np.tile(wxy, (self.nx, 1))
        wy = wx.swapaxes(0, 1)
        w = wx * wy
        return w

    def zerosuppression(self, sx, sy):
        x = self.xv
        y = self.yv
        g = 1 - self.strength * np.exp(-((x - sx) ** 2. + (y - sy) ** 2.) / (2. * self.sigma ** 2.))
        return g

    def apod(self, eta):
        return fftshift(
            window(('kaiser', eta), (self.ny, self.nx)))  # window(('general_hamming', alpha=0.72), (self.nx, self.nx))

# if __name__ == '__main__':

# p = si2D(file=r'C:/Users/Ruizhe.Lin/Documents/PythonProject/nlsim_recon/sim_nsi2d.tif',
#           image_pixel_size=0.075,
#           numerical_aperture=1.4,
#           emission_wavelength=0.505,
#           number_of_shifted_phases=7,
#           number_of_frequency_orders=7,
#           pattern_orientations=[0, 2*np.pi/7, 4*np.pi/7, 6*np.pi/7, 8*np.pi/7, 10*np.pi/7, 12*np.pi/7],
#           pattern_spacings=[0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24])
#     for i in range(7):
#         p.separate(i)
#         p.mapoverlap_w_zero(order='first', nps=10, r_ang=0.005, r_sp=0.005)
#         p.mapoverlap(orders=['first', 'second'], nps=10, r_ang=0.005, r_sp=0.005)
#         p.mapoverlap(orders=['second', 'third'], nps=10, r_ang=0.005, r_sp=0.005)
#     # p.magnitudes = {i: {'first': 0.8, 'second': 0.4, 'third': 0.2, 'fourth': 0.1} for i in range(p.nang)}
#     p.reconstruct_all()
#     p.save_reconstruction()
