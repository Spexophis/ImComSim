"""
This script reconstructs the raw data obtained from 3-dimensional linear structured illumination microscopy.
Ruizhe Lin
2024-01-12
"""

import os

temppath = r'C:/Users/ruizhe.lin/Documents/python_codes/sim3d/temp'
join = lambda fn: os.path.join(temppath, fn)

import tifffile as tf
from pylab import imshow, subplot, figure, plot
import numpy as np
from scipy.special import factorial
from scipy import signal
# from cupyfft import fftn, ifftn
from numpy.fft import fftshift, fft2, ifft2, fftn, ifftn


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
        self.psf = self.get_psf()
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
        for j in range(nphases):
            sepmat[0, j] = 1.0
            for order in range(1, norders):
                sepmat[2 * order - 1, j] = 2 * np.cos(2 * np.pi * (j * order) / nphases) / nphases
                sepmat[2 * order, j] = 2 * np.sin(2 * np.pi * (j * order) / nphases) / nphases
        return np.linalg.inv(np.transpose(sepmat))

    def _get_pupil(self, zarr=None):
        msk = self._shift(self._disc_array(shape=(self.nx, self.ny), radius=self.radius_xy)) / np.sqrt(
            np.pi * self.radius_xy ** 2) / self.nx
        phi = np.zeros((self.nx, self.ny))
        wf = msk * np.exp(1j * phi).astype(np.complex64)
        if zarr is not None:
            for z in range(len(zarr)):
                n, m = self._zernike_j_nm(z + 1)
                phi += zarr[z] * self._zernike(n, m, radius=self.radius_xy, shape=(self.nx, self.ny))
            wf *= np.exp(1j * phi).astype(np.complex64)
        return wf

    def _focus_mode(self, depth=0.):
        x = np.arange(-self.nxh, self.nxh)
        xv, yv = np.meshgrid(x, x)
        rho = np.sqrt(xv ** 2 + yv ** 2) / self.radius_xy
        msk = (rho <= 1.0).astype(np.float64)
        return msk * (self.n2 * depth / self.wl) * np.sqrt(1 - (self.na * msk * rho / self.n2) ** 2)

    def get_psf(self, axial=(-1.6, 1.6, 0.16), zernike_arr=None):
        bpp = self._get_pupil(zernike_arr)
        start, stop, step = axial
        nsteps = int((stop - start) / step + 1)
        zarr = np.linspace(start / step, stop / step, nsteps).astype(np.int64)
        zarr = zarr[0:nsteps - 1]
        zarr = np.roll(zarr, int((nsteps - 1) / 2))
        stack = np.zeros((nsteps - 1, self.nx, self.ny))
        for m, z in enumerate(zarr):
            d = z * step
            ph = self._focus_mode(d)
            wf = bpp * np.exp(2j * np.pi * ph)
            stack[m] = np.abs(fft2(fftshift(wf))) ** 2
        return stack / stack.sum()

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

        ysh = np.exp(2j * np.pi * (kx * self.xv + ky * self.yv + kz * self.zv))
        otf = fftn((self.psf * ysh))
        yshf = np.abs(fftn(ysh))
        sz, sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if sx < self.nxh:
            sx = sx
        else:
            sx = sx - self.nx
        if sy < self.nyh:
            sy = sy
        else:
            sy = sy - self.ny
        zsp = self.zero_suppression(sz, sx, sy)
        otf = otf * zsp
        ysh = np.exp(2j * np.pi * (kx * self.xv + ky * self.yv + 0. * self.zv))
        imgf = fftn((self.img_1_p * ysh))

        cutoff = self.cutoff
        imgf0 = self.imgf_0
        otf0 = self.otf_0
        wimgf0 = otf * imgf0
        wimgf1 = otf0 * imgf
        msk = (np.abs(otf0 * otf) > cutoff).astype(np.complex64)
        if verbose:
            tf.imshow(np.abs((msk * wimgf1 * wimgf0.conj()) / (msk * wimgf0 * wimgf0.conj())))
            tf.imshow(np.angle((msk * wimgf1 * wimgf0.conj()) / (msk * wimgf0 * wimgf0.conj())))
        a = np.sum(msk * wimgf1 * wimgf0.conj()) / np.sum(msk * wimgf0 * wimgf0.conj())
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

        ysh = np.exp(2j * np.pi * (kx * self.xv + ky * self.yv + kz * self.zv))
        otf = fftn((self.psf * ysh))
        yshf = np.abs(fftn(ysh))
        sz, sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if sx < self.nxh:
            sx = sx
        else:
            sx = sx - self.nx
        if sy < self.nyh:
            sy = sy
        else:
            sy = sy - self.ny
        zsp = self.zero_suppression(sz, sx, sy)
        otf = otf * zsp

        ysh = self.shift_mat(0., kx, ky).astype(np.complex64)
        imgf = fftn((self.img_1_p * ysh))
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
        kz = 0

        ysh = np.exp(2j * np.pi * (kx * self.xv + ky * self.yv + kz * self.zv))
        otf = fftn(self.psf * ysh)
        yshf = np.abs(fftn(ysh))
        sz, sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if sx < self.nxh:
            sx = sx
        else:
            sx = sx - self.nx
        if sy < self.nyh:
            sy = sy
        else:
            sy = sy - self.ny
        zsp = self.zero_suppression(sz, sx, sy)
        otf = otf * zsp

        imgf = fftn(self.img_2_p * ysh)

        cutoff = self.cutoff
        imgf0 = self.imgf_0
        otf0 = self.otf_0
        wimgf0 = otf * imgf0
        wimgf1 = otf0 * imgf
        msk = (np.abs(otf0 * otf) > cutoff).astype(np.complex64)
        if verbose:
            tf.imshow(np.abs((msk * wimgf1 * wimgf0.conj()) / (msk * wimgf0 * wimgf0.conj())))
            tf.imshow(np.angle((msk * wimgf1 * wimgf0.conj()) / (msk * wimgf0 * wimgf0.conj())))
        a = np.sum(msk * wimgf1 * wimgf0.conj()) / np.sum(msk * wimgf0 * wimgf0.conj())
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
        self.otf_0 = np.zeros((self.nz, self.nx, self.ny), dtype=np.complex64)
        zsp = self.zero_suppression(0., 0., 0.)
        self.otf_0 = fftn(self.psf) * zsp
        self.imgf_0 = np.zeros((self.nz, self.nx, self.ny), dtype=np.complex64)
        self.imgf_0 = fftn(self.img_0)
        tf.imwrite(join('otf_0.tif'), self.otf_0)
        tf.imwrite(join('imgf_0.tif'), self.imgf_0)
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

        ysh = np.zeros((2, self.nz, self.nx, self.ny), dtype=np.complex64)
        ysh[0] = self.shift_mat(kz, kx, ky).astype(np.complex64)
        ysh[1] = self.shift_mat(kz, -kx, -ky).astype(np.complex64)

        otf = fftn((self.psf * ysh[0]))
        yshf = np.abs(fftn(ysh[0]))
        sz, sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if sx < self.nxh:
            sx = sx
        else:
            sx = sx - self.nx
        if sy < self.nyh:
            sy = sy
        else:
            sy = sy - self.ny
        zsp = self.zero_suppression(sz, sx, sy)
        otf = otf * zsp
        tf.imwrite(join('otf_1_p.tif'), otf)
        if verbose:
            tf.imshow(np.abs(fftshift(otf)), photometric='minisblack',
                      title='Angle %d _ 1st order +1 OTF' % self.ang_ind)

        otf = fftn((self.psf * ysh[1]))
        yshf = np.abs(fftn(ysh[1]))
        sz, sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if sx < self.nxh:
            sx = sx
        else:
            sx = sx - self.nx
        if sy < self.nyh:
            sy = sy
        else:
            sy = sy - self.ny
        zsp = self.zero_suppression(sz, sx, sy)
        otf = otf * zsp
        tf.imwrite(join('otf_1_n.tif'), otf)
        if verbose:
            tf.imshow(np.abs(fftshift(otf)), photometric='minisblack',
                      title='Angle %d _ 1st order -1 OTF' % self.ang_ind)

        ysh = np.zeros((2, self.nz, self.nx, self.ny), dtype=np.complex64)
        ysh[0] = self.shift_mat(0, kx, ky).astype(np.complex64)
        ysh[1] = self.shift_mat(0, -kx, -ky).astype(np.complex64)

        imgf = fftn(self.img_1_p * ysh[0])
        tf.imwrite(join('imgf_1_p.tif'), imgf)
        if verbose:
            tf.imshow(np.abs(fftshift(imgf)), photometric='minisblack',
                      title='Angle %d _ 1st order +1 frequency spectrum' % self.ang_ind)
        imgf = fftn(self.img_1_n * ysh[1])
        tf.imwrite(join('imgf_1_n.tif'), imgf)
        if verbose:
            tf.imshow(np.abs(fftshift(imgf)), photometric='minisblack',
                      title='Angle %d _ 1st order -1 frequency spectrum' % self.ang_ind)

    def shift_2nd(self, verbose=False):
        angle = self.angles[self.ang_ind]["second"]
        spacing = self.lateral_spacings[self.ang_ind]["second"]
        kx = self.dx * np.cos(angle) / spacing
        ky = self.dx * np.sin(angle) / spacing

        ysh = np.zeros((2, self.nz, self.nx, self.ny), dtype=np.complex64)
        ysh[0] = self.shift_mat(0., kx, ky).astype(np.complex64)
        ysh[1] = self.shift_mat(0., -kx, -ky).astype(np.complex64)

        otf = fftn(self.psf * ysh[0])
        yshf = np.abs(fftn(ysh[0]))
        sz, sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if sx < self.nxh:
            sx = sx
        else:
            sx = sx - self.nx
        if sy < self.nyh:
            sy = sy
        else:
            sy = sy - self.ny
        zsp = self.zero_suppression(sz, sx, sy)
        otf = otf * zsp
        tf.imwrite(join('otf_2_p.tif'), otf)
        if verbose:
            tf.imshow(np.abs(fftshift(otf)), photometric='minisblack',
                      title='Angle %d _ 2nd order +1 OTF' % self.ang_ind)

        otf = fftn(self.psf * ysh[1])
        yshf = np.abs(fftn(ysh[1]))
        sz, sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if sx < self.nxh:
            sx = sx
        else:
            sx = sx - self.nx
        if sy < self.nyh:
            sy = sy
        else:
            sy = sy - self.ny
        zsp = self.zero_suppression(sz, sx, sy)
        otf = otf * zsp
        tf.imwrite(join('otf_2_n.tif'), otf)
        if verbose:
            tf.imshow(np.abs(fftshift(otf)), photometric='minisblack',
                      title='Angle %d _ 2nd order -1 OTF' % self.ang_ind)

        imgf = fftn(self.img_2_p * ysh[0])
        tf.imwrite(join('imgf_2_p.tif'), imgf)
        if verbose:
            tf.imshow(np.abs(fftshift(imgf)), photometric='minisblack',
                      title='Angle %d _ 2nd order +1 frequency spectrum' % self.ang_ind)
        imgf = fftn(self.img_2_n * ysh[1])
        tf.imwrite(join('imgf_2_n.tif'), imgf)
        if verbose:
            tf.imshow(np.abs(fftshift(imgf)), photometric='minisblack',
                      title='Angle %d _ 2nd order -1 frequency spectrum' % self.ang_ind)

    def recon_one(self, ang=0):
        ph1 = self.magnitudes[ang]['first'] * np.exp(1j * self.phases[ang]['first'])
        ph2 = self.magnitudes[ang]['second'] * np.exp(1j * self.phases[ang]['second'])
        # 0th order
        imgf = tf.imread(join('imgf_0.tif'))
        tf.imwrite('angle%d_imgf_0.tif' % self.ang_ind, np.abs(np.fft.fftshift(imgf)).astype(np.float32),
                   photometric='minisblack')
        otf = tf.imread(join('otf_0.tif'))
        tf.imwrite('angle%d_otf_0.tif' % self.ang_ind, np.abs(np.fft.fftshift(otf)).astype(np.float32),
                   photometric='minisblack')
        self.Snum += self.zoa * otf.conj() * imgf
        self.Sden += np.abs(otf) ** 2
        # +1st order
        imgf = tf.imread(join('imgf_1_p.tif'))
        tf.imwrite('angle%d_imgf_1_p.tif' % self.ang_ind, np.abs(np.fft.fftshift(imgf)).astype(np.float32),
                   photometric='minisblack')
        otf = tf.imread(join('otf_1_p.tif'))
        tf.imwrite('angle%d_otf_1_p.tif' % self.ang_ind, np.abs(np.fft.fftshift(otf)).astype(np.float32),
                   photometric='minisblack')
        self.Snum += ph1 * otf.conj() * imgf
        self.Sden += np.abs(otf) ** 2
        # -1 order
        imgf = tf.imread(join('imgf_1_n.tif'))
        tf.imwrite('angle%d_imgf_1_n.tif' % self.ang_ind, np.abs(np.fft.fftshift(imgf)).astype(np.float32),
                   photometric='minisblack')
        otf = tf.imread(join('otf_1_n.tif'))
        tf.imwrite('angle%d_otf_1_n.tif' % self.ang_ind, np.abs(np.fft.fftshift(otf)).astype(np.float32),
                   photometric='minisblack')
        self.Snum += ph1.conj() * otf.conj() * imgf
        self.Sden += np.abs(otf) ** 2
        # +2nd order
        imgf = tf.imread(join('imgf_2_p.tif'))
        tf.imwrite('angle%d_imgf_2_p.tif' % self.ang_ind, np.abs(np.fft.fftshift(imgf)).astype(np.float32),
                   photometric='minisblack')
        otf = tf.imread(join('otf_2_p.tif'))
        tf.imwrite('angle%d_otf_2_p.tif' % self.ang_ind, np.abs(np.fft.fftshift(otf)).astype(np.float32),
                   photometric='minisblack')
        self.Snum += ph2 * otf.conj() * imgf
        self.Sden += np.abs(otf) ** 2
        # -2nd order
        imgf = tf.imread(join('imgf_2_n.tif'))
        tf.imwrite('angle%d_imgf_2_n.tif' % self.ang_ind, np.abs(np.fft.fftshift(imgf)).astype(np.float32),
                   photometric='minisblack')
        otf = tf.imread(join('otf_2_n.tif'))
        tf.imwrite('angle%d_otf_2_n.tif' % self.ang_ind, np.abs(np.fft.fftshift(otf)).astype(np.float32),
                   photometric='minisblack')
        self.Snum += ph2.conj() * otf.conj() * imgf
        self.Sden += np.abs(otf) ** 2

    def recon_all(self):
        self.Snum = np.zeros((self.nz, self.nx, self.ny), dtype=np.complex64)
        self.Sden = np.zeros((self.nz, self.nx, self.ny), dtype=np.complex64)
        self.Sden += self.mu ** 2
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
        x = self.xv
        y = self.yv
        z = self.zv
        g = 1 - self.strength * np.exp(
            -((x - sx) ** 2. + (y - sy) ** 2. + 0. * (z - sz) ** 2.) / (2. * self.sigma ** 2.))
        g[g < 0.5] = 0.0
        g[g >= 0.5] = 1.0
        g = 1
        return g

    def window(self, eta):
        nz = self.nz * 2
        nx = self.nx * 2
        ny = self.ny * 2
        wd = np.zeros((nz, nx, ny))
        wind = signal.tukey(nx, alpha=eta, sym=True)
        wz = signal.tukey(nz, alpha=eta, sym=True)
        wx = np.tile(wind, (nx, 1))
        wy = wx.swapaxes(0, 1)
        w = wx * wy
        for i in range(nz):
            wd[i, :, :, ] = w * wz[i]
        return wd

    def apod(self):
        rxy = 2. * self.radius_xy
        rz = 2. * self.radius_z
        apo = (1 - self.axy * np.sqrt(self.xv ** 2 + self.yv ** 2) / rxy) ** self.expn * (
                1 - self.az * np.sqrt(self.zv ** 2) / rz) ** self.expn
        rhxy = np.sqrt(self.xv ** 2 + self.yv ** 2 + 0. * self.zv ** 2) / rxy
        rhz = np.sqrt(0. * self.xv ** 2 + 0. * self.yv ** 2 + self.zv ** 2) / rz
        msk_xy = (rhxy <= 1.0).astype(np.float64)
        msk_z = (rhz <= 1.0).astype(np.float64)
        msk = msk_xy * msk_z
        apodiz = apo * msk
        return apodiz

    @staticmethod
    def _interp(arr, ratio=2):
        nz, nx, ny = arr.shape
        px = int((nx * (ratio - 1)) / 2)
        py = int((ny * (ratio - 1)) / 2)
        pz = int((nz * (ratio - 1)) / 2)
        arrf = fftn(arr)
        arro = np.pad(np.fft.fftshift(arrf), ((pz, pz), (px, px), (py, py)), 'constant', constant_values=0)
        out = ifftn(np.fft.fftshift(arro))
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
    img = tf.imread(r"sim3d_simulation_data.tif")
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
        # p.map_overlap_2nd(nps=4, r_ang=0.02, r_sp=0.02)
        p.map_overlap_2nd(nps=10, r_ang=0.005, r_sp=0.005)
        # p.map_overlap_2nd(nps=10, r_ang=0.005, r_sp=0.005)
        p.map_overlap_z(nps=50, r_spz=0.25)
        p.map_overlap_1st(nps=10, r_ang=0.005, r_sp=0.005)
        # p.map_overlap_1st(nps=10, r_ang=0.005, r_sp=0.005)
    p.recon_all()
    p.save_reconstruction()
