"""
This script reconstructs the raw data obtained from 3-dimensional linear structured illumination microscopy.
Ruizhe Lin
2024-01-12
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tifffile as tf
from pylab import imshow, subplot, figure, plot
import numpy as np
from scipy import signal
from scipy.fft import fftn as _fftn, ifftn as _ifftn, fft2 as _fft2, ifft2 as _ifft2, fftshift
import psf_generator as pg

_WORKERS = -1


def fftn(a):
    return _fftn(a, workers=_WORKERS)


def ifftn(a):
    return _ifftn(a, workers=_WORKERS)


def fft2(a):
    return _fft2(a, workers=_WORKERS)


def ifft2(a):
    return _ifft2(a, workers=_WORKERS)


class SIM_RECON:

    def __init__(self, **kwargs):
        self.na = kwargs['numerical_aperture']
        self.n2 = 1.512
        self.nph = kwargs['number_of_shifted_phases']
        self.angs = kwargs['pattern_orientations']
        self.nang = len(self.angs)
        self.sps = kwargs['pattern_lateral_spacings']
        self.spz = kwargs['pattern_axial_spacings']
        self.wl = kwargs['emission_wavelength']
        self.norders = kwargs['number_of_frequency_orders']
        self.dxd, self.dzd = kwargs['image_pixel_size']
        self.dx = self.dxd / 2
        self.dz = self.dzd / 2
        self.img = kwargs['image_stack']
        self.nzh, self.nxh, self.nyh = self.img.shape
        self.nzh = int(self.nzh / self.nph / self.nang)
        self.img = self.img.reshape(self.nzh, self.nang, self.nph, self.nxh, self.nyh).swapaxes(0, 1).swapaxes(1, 2)
        self.nx = self.nxh * 2
        self.ny = self.nyh * 2
        self.nz = self.nzh * 2
        self.dpx = 1 / (self.nx * self.dx)
        self.dpz = 1 / (self.nz * self.dz)
        self.radius_xy = (self.na / self.wl) / self.dpx
        self.radius_z = ((self.na ** 2) / self.wl) / self.dpz
        self.mu = 1e-2
        self.cutoff = 1e-3
        self.strength = 1.
        self.sigma = 8.
        self.eta = 0.08
        self.expn = 1.
        self.axy = 0.8
        self.az = 0.8
        self.zoa = 10e-2
        _psf_obj = pg.PSF(wl=self.wl, na=self.na, n2=self.n2, dx=self.dx, nx=self.nx)
        _psf_obj.flat_wavefront()
        self.psf = _psf_obj.get_3d_psf((0, 0, 0), -2.56, 2.56, self.dz)
        self.zv, self.xv, self.yv = self.meshgrid()
        self.sep_mat = self.sepmatrix()
        self.winf = self.window(self.eta)
        self.apd = self.apod()
        # parameters
        self.angles = {i: {'first': self.angs[i], 'second': self.angs[i]} for i in range(self.nang)}
        self.lateral_spacings = {i: {'first': self.sps[i], 'second': self.sps[i] / 2} for i in range(self.nang)}
        self.axial_spacings = {i: {'first': self.spz[i]} for i in range(self.nang)}
        self.magnitudes = {i: {'first': 0.8, 'second': 0.4} for i in range(self.nang)}
        self.phases = {i: {'first': 0, 'second': 0} for i in range(self.nang)}

    def sub_bg(self, img):
        hist, bin_edges = np.histogram(img, bins=np.arange(img.min(), img.max(), 256))
        ind = np.where(hist == hist.max())
        bg = bin_edges[np.max(ind[0] * 2)]
        img[img <= bg] = 0.
        img[img > bg] = img[img > bg] - bg
        return img

    def meshgrid(self):
        x = np.arange(-self.nxh, self.nxh)
        y = np.arange(-self.nyh, self.nyh)
        z = np.arange(-self.nzh, self.nzh)
        zv, xv, yv = np.meshgrid(z, x, y, indexing='ij', sparse=True)
        return np.roll(zv, int(self.nzh)), np.roll(xv, int(self.nxh)), np.roll(yv, int(self.nyh))

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

    def separate(self, ang_ind=0):
        self.ang_ind = ang_ind
        out = np.dot(self.sep_mat, self.img[ang_ind].reshape(self.nph, self.nzh * self.nxh * self.nyh))
        self.img_0 = fftshift(self._interp(out[0].reshape(self.nzh, self.nxh, self.nyh)) * self.winf)
        self.img_1_p = fftshift(self._interp((out[1] + 1j * out[2]).reshape(self.nzh, self.nxh, self.nyh)) * self.winf)
        self.img_1_n = fftshift(self._interp((out[1] - 1j * out[2]).reshape(self.nzh, self.nxh, self.nyh)) * self.winf)
        self.img_2_p = fftshift(self._interp((out[3] + 1j * out[4]).reshape(self.nzh, self.nxh, self.nyh)) * self.winf)
        self.img_2_n = fftshift(self._interp((out[3] - 1j * out[4]).reshape(self.nzh, self.nxh, self.nyh)) * self.winf)

    def shift_mat(self, kz, kx, ky):
        return np.exp(2j * np.pi * (kx * self.xv + ky * self.yv)) * np.cos(2 * np.pi * kz * self.zv)

    def get_overlap_1st(self, angle, spacingx, spacingz, verbose=False):
        kx = self.dx * np.cos(angle) / (spacingx * 2)
        ky = self.dx * np.sin(angle) / (spacingx * 2)
        kz = self.dz / spacingz

        # zero_suppression always returns 1 — skip the argmax fftn entirely
        ysh = np.exp(2j * np.pi * (kx * self.xv + ky * self.yv + kz * self.zv))
        otf = fftn(self.psf * ysh)
        ysh = np.exp(2j * np.pi * (kx * self.xv + ky * self.yv + 0. * self.zv))
        imgf = fftn(self.img_1_p * ysh)

        cutoff = self.cutoff
        imgf0 = self.imgf_0
        otf0 = self.otf_0
        wimgf0 = otf * imgf0
        wimgf1 = otf0 * imgf
        msk = (np.abs(otf0 * otf) > cutoff).astype(np.complex64)
        denom = np.sum(msk * wimgf0 * wimgf0.conj())
        if denom.real == 0:
            return np.nan, np.nan
        if verbose:
            tf.imshow(np.abs((msk * wimgf1 * wimgf0.conj()) / (msk * wimgf0 * wimgf0.conj())))
            tf.imshow(np.angle((msk * wimgf1 * wimgf0.conj()) / (msk * wimgf0 * wimgf0.conj())))
        a = np.sum(msk * wimgf1 * wimgf0.conj()) / denom
        return np.abs(a), np.angle(a)

    def map_overlap_1st(self, nps=10, r_ang=0.02, r_sp=0.008, verbose=False):
        angle = self.angles[self.ang_ind]["first"]
        spacing = self.lateral_spacings[self.ang_ind]["first"]
        spz = self.axial_spacings[self.ang_ind]["first"]
        d_ang = 2 * r_ang / nps
        d_sp = 2 * r_sp / nps
        ang_iter = np.arange(-r_ang, r_ang + d_ang / 2, d_ang) + angle
        sp_iter = np.arange(-r_sp, r_sp + d_sp / 2, d_sp) + spacing
        magarr = np.zeros((nps + 1, nps + 1))
        pharr = np.zeros((nps + 1, nps + 1))
        for m, ang in enumerate(ang_iter):
            for n, sp in enumerate(sp_iter):
                print(m, n)
                mag, phase = self.get_overlap_1st(ang, sp, spz)
                if np.isnan(mag):
                    magarr[m, n] = 0.0
                else:
                    magarr[m, n] = mag
                    pharr[m, n] = phase
        if verbose:
            figure()
            subplot(211)
            imshow(magarr, interpolation='nearest')
            subplot(212)
            imshow(pharr, interpolation='nearest')
        # get maximum
        k, l = np.where(magarr == magarr.max())
        angmax = k[0] * d_ang - r_ang + angle
        spmax = l[0] * d_sp - r_sp + spacing
        self.angles[self.ang_ind]["first"] = angmax
        self.lateral_spacings[self.ang_ind]["first"] = spmax
        self.magnitudes[self.ang_ind]["first"] = magarr[k, l][0]
        self.phases[self.ang_ind]["first"] = pharr[k, l][0]

    def get_overlap_z(self, angle, spacingx, spacingz):
        kx = self.dx * np.cos(angle) / (spacingx * 2)
        ky = self.dx * np.sin(angle) / (spacingx * 2)
        kz = self.dz / spacingz

        # zero_suppression always returns 1 — skip the argmax fftn entirely
        ysh = np.exp(2j * np.pi * (kx * self.xv + ky * self.yv + kz * self.zv))
        otf = fftn(self.psf * ysh)

        ysh = self.shift_mat(0., kx, ky).astype(np.complex64)
        imgf = fftn(self.img_1_p * ysh)
        return (np.abs(imgf * otf) ** 2).sum()

    def map_overlap_z(self, nps=10, r_spz=0.1, verbose=False):
        angle = self.angles[self.ang_ind]["first"]
        spacing = self.lateral_spacings[self.ang_ind]["first"]
        spz = self.axial_spacings[self.ang_ind]["first"]
        d_spz = 2 * r_spz / nps
        spz_iter = np.arange(-r_spz, r_spz + d_spz / 2, d_spz) + spz
        magarr = np.zeros((nps + 1))
        for m, z in enumerate(spz_iter):
            print(m)
            temp = self.get_overlap_z(angle, spacing, z)
            if np.isnan(temp):
                magarr[m] = 0.0
            else:
                magarr[m] = temp
        print(spz_iter)
        print(magarr)
        if verbose:
            figure()
            plot(spz_iter, magarr)
        k = np.where(magarr == magarr.max())
        spzmax = k[0] * d_spz - r_spz + spz
        self.axial_spacings[self.ang_ind]["first"] = spzmax

    def get_overlap_2nd(self, angle, spacingx, verbose=False):
        kx = self.dx * np.cos(angle) / spacingx
        ky = self.dx * np.sin(angle) / spacingx

        # zero_suppression always returns 1 — skip the argmax fftn entirely
        ysh = np.exp(2j * np.pi * (kx * self.xv + ky * self.yv))
        otf = fftn(self.psf * ysh)
        imgf = fftn(self.img_2_p * ysh)

        cutoff = self.cutoff
        imgf0 = self.imgf_0
        otf0 = self.otf_0
        wimgf0 = otf * imgf0
        wimgf1 = otf0 * imgf
        msk = (np.abs(otf0 * otf) > cutoff).astype(np.complex64)
        denom = np.sum(msk * wimgf0 * wimgf0.conj())
        if denom.real == 0:
            return np.nan, np.nan
        if verbose:
            tf.imshow(np.abs((msk * wimgf1 * wimgf0.conj()) / (msk * wimgf0 * wimgf0.conj())))
            tf.imshow(np.angle((msk * wimgf1 * wimgf0.conj()) / (msk * wimgf0 * wimgf0.conj())))
        a = np.sum(msk * wimgf1 * wimgf0.conj()) / denom
        return np.abs(a), np.angle(a)

    def map_overlap_2nd(self, nps=10, r_ang=0.02, r_sp=0.008, verbose=False):
        angle = self.angles[self.ang_ind]["second"]
        spacing = self.lateral_spacings[self.ang_ind]["second"]
        d_ang = 2 * r_ang / nps
        d_sp = 2 * r_sp / nps
        ang_iter = np.arange(-r_ang, r_ang + d_ang / 2, d_ang) + angle
        sp_iter = np.arange(-r_sp, r_sp + d_sp / 2, d_sp) + spacing
        magarr = np.zeros((nps + 1, nps + 1))
        pharr = np.zeros((nps + 1, nps + 1))
        for m, ang in enumerate(ang_iter):
            for n, sp in enumerate(sp_iter):
                print(m, n)
                mag, phase = self.get_overlap_2nd(ang, sp)
                if np.isnan(mag):
                    magarr[m, n] = 0.0
                else:
                    magarr[m, n] = mag
                    pharr[m, n] = phase
        if verbose:
            figure()
            subplot(211)
            imshow(magarr, interpolation='nearest')
            subplot(212)
            imshow(pharr, interpolation='nearest')
        # get maximum
        k, l = np.where(magarr == magarr.max())
        angmax = k[0] * d_ang - r_ang + angle
        spmax = l[0] * d_sp - r_sp + spacing
        self.angles[self.ang_ind]["first"] = angmax
        self.lateral_spacings[self.ang_ind]["first"] = spmax
        self.magnitudes[self.ang_ind]["first"] = magarr[k, l][0]
        self.phases[self.ang_ind]["first"] = pharr[k, l][0]

    def shift_0(self, verbose=False):
        # zero_suppression always returns 1; store results in memory instead of disk
        self.otf_0 = fftn(self.psf)
        self.imgf_0 = fftn(self.img_0)
        if verbose:
            tf.imshow(np.abs(fftshift(self.otf_0)), photometric='minisblack',
                      title='Angle %d _ 0 order OTF' % self.ang_ind)
            tf.imshow(np.abs(fftshift(self.imgf_0)), photometric='minisblack',
                      title='Angle %d _ 0 order frequency spectrum' % self.ang_ind)

    def shift_1st(self, verbose=False):
        angle = self.angles[self.ang_ind]["first"]
        spacingx = self.lateral_spacings[self.ang_ind]["first"]
        spz = self.axial_spacings[self.ang_ind]["first"]
        kx = self.dx * np.cos(angle) / (spacingx * 2)
        ky = self.dx * np.sin(angle) / (spacingx * 2)
        kz = self.dz / spz

        # zero_suppression always returns 1 — skip all argmax fftn calls
        ysh_pos = self.shift_mat(kz, kx, ky).astype(np.complex64)
        ysh_neg = self.shift_mat(kz, -kx, -ky).astype(np.complex64)

        self.otf_1_p = fftn(self.psf * ysh_pos)
        if verbose:
            tf.imshow(np.abs(fftshift(self.otf_1_p)), photometric='minisblack',
                      title='Angle %d _ 1st order +1 OTF' % self.ang_ind)

        self.otf_1_n = fftn(self.psf * ysh_neg)
        if verbose:
            tf.imshow(np.abs(fftshift(self.otf_1_n)), photometric='minisblack',
                      title='Angle %d _ 1st order -1 OTF' % self.ang_ind)

        ysh_pos_xy = self.shift_mat(0, kx, ky).astype(np.complex64)
        ysh_neg_xy = self.shift_mat(0, -kx, -ky).astype(np.complex64)

        self.imgf_1_p = fftn(self.img_1_p * ysh_pos_xy)
        if verbose:
            tf.imshow(np.abs(fftshift(self.imgf_1_p)), photometric='minisblack',
                      title='Angle %d _ 1st order +1 frequency spectrum' % self.ang_ind)
        self.imgf_1_n = fftn(self.img_1_n * ysh_neg_xy)
        if verbose:
            tf.imshow(np.abs(fftshift(self.imgf_1_n)), photometric='minisblack',
                      title='Angle %d _ 1st order -1 frequency spectrum' % self.ang_ind)

    def shift_2nd(self, verbose=False):
        angle = self.angles[self.ang_ind]["second"]
        spacing = self.lateral_spacings[self.ang_ind]["second"]
        kx = self.dx * np.cos(angle) / spacing
        ky = self.dx * np.sin(angle) / spacing

        # zero_suppression always returns 1 — skip all argmax fftn calls
        ysh_pos = self.shift_mat(0., kx, ky).astype(np.complex64)
        ysh_neg = self.shift_mat(0., -kx, -ky).astype(np.complex64)

        self.otf_2_p = fftn(self.psf * ysh_pos)
        if verbose:
            tf.imshow(np.abs(fftshift(self.otf_2_p)), photometric='minisblack',
                      title='Angle %d _ 2nd order +1 OTF' % self.ang_ind)

        self.otf_2_n = fftn(self.psf * ysh_neg)
        if verbose:
            tf.imshow(np.abs(fftshift(self.otf_2_n)), photometric='minisblack',
                      title='Angle %d _ 2nd order -1 OTF' % self.ang_ind)

        self.imgf_2_p = fftn(self.img_2_p * ysh_pos)
        if verbose:
            tf.imshow(np.abs(fftshift(self.imgf_2_p)), photometric='minisblack',
                      title='Angle %d _ 2nd order +1 frequency spectrum' % self.ang_ind)
        self.imgf_2_n = fftn(self.img_2_n * ysh_neg)
        if verbose:
            tf.imshow(np.abs(fftshift(self.imgf_2_n)), photometric='minisblack',
                      title='Angle %d _ 2nd order -1 frequency spectrum' % self.ang_ind)

    def recon_one(self, ang=0):
        ph1 = self.magnitudes[ang]['first'] * np.exp(1j * self.phases[ang]['first'])
        ph2 = self.magnitudes[ang]['second'] * np.exp(1j * self.phases[ang]['second'])
        # Use in-memory attributes set by shift_0 / shift_1st / shift_2nd
        self.Snum += self.zoa * self.otf_0.conj() * self.imgf_0
        self.Sden += self.otf_0.real ** 2 + self.otf_0.imag ** 2
        self.Snum += ph1 * self.otf_1_p.conj() * self.imgf_1_p
        self.Sden += self.otf_1_p.real ** 2 + self.otf_1_p.imag ** 2
        self.Snum += ph1.conj() * self.otf_1_n.conj() * self.imgf_1_n
        self.Sden += self.otf_1_n.real ** 2 + self.otf_1_n.imag ** 2
        self.Snum += ph2 * self.otf_2_p.conj() * self.imgf_2_p
        self.Sden += self.otf_2_p.real ** 2 + self.otf_2_p.imag ** 2
        self.Snum += ph2.conj() * self.otf_2_n.conj() * self.imgf_2_n
        self.Sden += self.otf_2_n.real ** 2 + self.otf_2_n.imag ** 2

    def recon_all(self):
        self.Snum = np.zeros((self.nz, self.nx, self.ny), dtype=np.complex64)
        self.Sden = np.full((self.nz, self.nx, self.ny), self.mu ** 2, dtype=np.complex64)
        for i in range(3):
            self.separate(i)
            self.shift_0()
            self.shift_1st()
            self.shift_2nd()
            self.recon_one(i)
        self.S = self.Snum / self.Sden  # * self.apd
        self.final_image = fftshift(ifftn(self.S))

    def save_reconstruction(self, fn=''):
        tf.imwrite(fn + 'sim3d_final_image.tif', self.final_image.real.astype(np.float32), photometric='minisblack')
        tf.imwrite(fn + 'sim3d_effective_OTF.tif', np.abs(fftshift(self.S)).astype(np.float32), photometric='minisblack')

    def zero_suppression(self, sz, sx, sy):
        return 1

    def window(self, eta):
        wind = signal.windows.tukey(self.nx, alpha=eta, sym=True)
        wz = signal.windows.tukey(self.nz, alpha=eta, sym=True)
        # Broadcasting replaces the Python loop over z-planes
        return wz[:, np.newaxis, np.newaxis] * np.outer(wind, wind)[np.newaxis, :, :]

    def apod(self):
        rxy = 2. * self.radius_xy
        rz = 2. * self.radius_z
        apo = (1 - self.axy * np.sqrt(self.xv ** 2 + self.yv ** 2) / rxy) ** self.expn * (
                1 - self.az * np.sqrt(self.zv ** 2) / rz) ** self.expn
        rhxy = np.sqrt(self.xv ** 2 + self.yv ** 2 + 0. * self.zv ** 2) / rxy
        rhz = np.sqrt(0. * self.xv ** 2 + 0. * self.yv ** 2 + self.zv ** 2) / rz
        msk_xy = (rhxy <= 1.0).astype(np.float64)
        msk_z = (rhz <= 1.0).astype(np.float64)
        return apo * msk_xy * msk_z

    @staticmethod
    def _interp(arr, ratio=2):
        nz, nx, ny = arr.shape
        px = int((nx * (ratio - 1)) / 2)
        py = int((ny * (ratio - 1)) / 2)
        pz = int((nz * (ratio - 1)) / 2)
        arrf = fftn(arr)
        arro = np.pad(fftshift(arrf), ((pz, pz), (px, px), (py, py)), 'constant', constant_values=0)
        return ifftn(fftshift(arro))


if __name__ == '__main__':
    img = tf.imread(r"C:\Users\Ruiz\Documents\GitHub\ImComSim\structured_illumination_microscopy\202602062331_sim3d_simulation_image_stack.tif")
    p = SIM_RECON(image_stack=img,
                  image_pixel_size=(0.08, 0.16),
                  numerical_aperture=1.4,
                  emission_wavelength=0.505,
                  number_of_shifted_phases=5,
                  number_of_frequency_orders=5,
                  pattern_orientations=[0, 2 * np.pi / 3, 4 * np.pi / 3],
                  pattern_lateral_spacings=[0.24, 0.24, 0.24],
                  pattern_axial_spacings=[1.5, 1.5, 1.5])
    for i in range(3):
        p.separate(i)
        p.shift_0()
        p.map_overlap_2nd(nps=10, r_ang=0.005, r_sp=0.005)
        p.map_overlap_z(nps=50, r_spz=0.25)
        p.map_overlap_1st(nps=10, r_ang=0.005, r_sp=0.005)
    p.recon_all()
    p.save_reconstruction()
