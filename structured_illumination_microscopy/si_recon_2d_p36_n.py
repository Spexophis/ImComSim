"""
Optimized 2D structured illumination reconstruction
python3.6
Ruizhe Lin 04/15/2019
"""

import numpy as np
import tifffile as tf
from matplotlib.pyplot import imshow, subplot, plot, figure, colorbar, show
from numpy.fft import fftshift
from scipy.signal.windows import tukey

import psf_generator as pg

try:
    from scipy.fft import rfft2 as _rfft2, fft2 as _fft2, ifft2 as _ifft2

    _FFT_KW = {"workers": -1}
except ImportError:
    from numpy.fft import rfft2 as _rfft2, fft2 as _fft2, ifft2 as _ifft2

    _FFT_KW = {}

_WORKERS = -1


def fft2(a):
    return _fft2(a, workers=_WORKERS)


def ifft2(a):
    return _ifft2(a, workers=_WORKERS)


class SIM_RECON:

    def __init__(self, img_stack, nangles, nphs, norders, dx, wavelength, na):
        self.dx = dx
        self.na = na
        self.wl = wavelength
        self.nph = nphs
        self.nang = nangles
        self.norders = norders
        self.n, self.nx, self.ny = img_stack.shape
        self.eta = 0.1
        self.raps_stack(img_stack, "tukey", np.float64, True)
        # self.img = self.sub_bg(img_stack)
        self.img = img_stack.reshape(self.nang, self.nph, self.nx, self.ny).astype(np.float64)
        self.getpsf()
        self.meshgrid()

        self.eps = 0.8
        self.mu = 0.02
        self.cutoff = 0.04
        self.strength = 0.99
        self.fwhm = 0.99
        self.minv = 0.0
        self.eh = []

        self.winf = np.outer(tukey(self.nx * 2, alpha=self.eta), tukey(self.nx * 2, alpha=self.eta)).astype(float)

    def raps_stack(self, stack, window="tukey", dtype=np.float64, verbose=True):
        stack = np.asarray(stack, float, copy=True)

        if window == "hanning":
            w = np.outer(np.hanning(self.ny), np.hanning(self.nx)).astype(dtype)
            wnorm = dtype(np.sum(w.astype(np.float64) ** 2))
        elif window == "tukey":
            w = np.outer(tukey(self.ny, alpha=self.eta), tukey(self.nx, alpha=self.eta)).astype(dtype)
            wnorm = dtype(np.sum(w.astype(np.float64) ** 2))
        else:
            w, wnorm = None, dtype(self.ny * self.nx)

        self.fy = np.fft.fftfreq(self.ny, self.dx)[:, None]
        self.fx = np.fft.fftfreq(self.nx, self.dx)[None, :]
        self.fr = np.hypot(self.fx, self.fy)

        self.df = 1.0 / min(self.nx * self.dx, self.ny * self.dx)
        f_nyq = min(1.0 / (2 * self.dx), 1.0 / (2 * self.dx))

        nb = int(round(f_nyq / self.df)) + 1
        k = np.round(self.fr / self.df).astype(np.int32)
        k[self.fr > f_nyq] = nb
        k = k.ravel()

        n = np.bincount(k, minlength=nb + 1)[:nb]

        profs = np.empty((self.n, nb), dtype)
        buf = np.empty((self.ny, self.nx), dtype)

        for i in range(self.n):
            np.subtract(stack[i], stack[i].mean(), out=buf, casting="unsafe")
            if w is not None:
                buf *= w
            F = _fft2(buf, **_FFT_KW)
            P = F.real ** 2 + F.imag ** 2
            profs[i] = np.bincount(k, P.ravel(), minlength=nb + 1)[:nb] / (n * wnorm)

        if verbose:
            f = np.arange(nb) * self.df
            figure()
            for i in range(self.n):
                plot(f, np.log(profs[i]))
                show()

    def sub_bg(self, img):
        nz, nx, ny = img.shape
        for i in range(nz):
            data = img[i, :, :]
            hist, bin_edges = np.histogram(img[i, :, :], bins=np.arange(img[i, :, :].min(), img[i, :, :].max()))
            ind = np.where(hist == hist.max())
            bg = bin_edges[np.max(ind) + 1]
            data[data <= bg] = 0.
            data[data > bg] = data[data > bg] - bg
            img[i, :, :] = data
        return img

    def getpsf(self):
        dx = self.dx / 2
        nx = self.nx * 2
        _psf_obj = pg.PSF(wl=self.wl, na=self.na, dx=dx, nx=nx)
        _psf_obj.flat_wavefront()
        self.psf = _psf_obj.get_2d_psf((0, 0, 0))
        self.radius = _psf_obj.radius

    def meshgrid(self):
        nx = self.nx
        ny = self.ny
        x = np.arange(2 * nx)
        y = np.arange(2 * ny)
        xv, yv = np.meshgrid(x, y, indexing='ij', sparse=True)
        i_xv = xv[1:nx + 1, 0]
        xv[nx:2 * nx, 0] = -np.flip(i_xv, 0)
        i_yv = yv[0, 1:ny + 1]
        yv[0, ny:2 * ny] = -np.flip(i_yv, 0)
        self.xv = xv
        self.yv = yv

    def sepmatrix(self):
        nphases = self.nph
        sepmat = np.zeros((self.norders, nphases), dtype=np.float32)
        norders = int((self.norders + 1) / 2)
        phi = 2 * np.pi / nphases
        for j in range(nphases):
            sepmat[0, j] = 1.0 / nphases
            for order in range(1, norders):
                sepmat[2 * order - 1, j] = 2.0 * np.cos(j * order * phi) / nphases
                sepmat[2 * order, j] = 2.0 * np.sin(j * order * phi) / nphases
        return sepmat

    def lockin_demod(self, ang, alpha):
        a = self.img[ang]
        d = a - np.roll(a, 1, axis=0)
        idc2 = np.sqrt((2.0 / self.nph ** 2) * np.einsum('ijk,ijk->jk', d, d))
        return a - alpha * a.mean(0) + idc2

    def separate(self, nangle, lockin=False):
        Nw = self.nx
        mat = self.sepmatrix()
        if lockin:
            data = self.lockin_demod(nangle, self.eps)
            outr = np.dot(mat, data.reshape(self.nph, Nw ** 2))
        else:
            outr = np.dot(mat, self.img[nangle].reshape(self.nph, Nw ** 2))
        if self.norders == 3:
            self.separr = np.zeros((3, Nw * 2, Nw * 2), dtype=np.complex64)
            self.separr[0] = np.fft.fftshift(self.interp(outr[0].reshape(Nw, Nw)) * self.winf)
            self.separr[1] = np.fft.fftshift(self.interp((outr[1] + 1j * outr[2]).reshape(Nw, Nw)) * self.winf)
            self.separr[2] = np.fft.fftshift(self.interp((outr[1] - 1j * outr[2]).reshape(Nw, Nw)) * self.winf)
        if self.norders == 5:
            self.separr = np.zeros((5, Nw * 2, Nw * 2), dtype=np.complex64)
            self.separr[0] = np.fft.fftshift(self.interp(outr[0].reshape(Nw, Nw)) * self.winf)
            self.separr[1] = np.fft.fftshift(self.interp((outr[1] + 1j * outr[2]).reshape(Nw, Nw)) * self.winf)
            self.separr[2] = np.fft.fftshift(self.interp((outr[1] - 1j * outr[2]).reshape(Nw, Nw)) * self.winf)
            self.separr[3] = np.fft.fftshift(self.interp((outr[3] + 1j * outr[4]).reshape(Nw, Nw)) * self.winf)
            self.separr[4] = np.fft.fftshift(self.interp((outr[3] - 1j * outr[4]).reshape(Nw, Nw)) * self.winf)

    def shift0(self):
        self.otf0 = fft2(self.psf)
        zsp = self.zero_suppression(0, 0)
        self.otf0 = self.otf0 * zsp
        self.imgf0 = fft2(self.separr[0])

    def shift(self, order, angle, spacing):
        """ shift data in freq space by multiplication in real space """
        Nw = self.nx
        dx = self.dx / 2
        kx = dx * np.cos(angle) / spacing
        ky = dx * np.sin(angle) / spacing
        # shift matrix
        ysh = np.zeros((2, 2 * Nw, 2 * Nw), dtype=np.complex64)
        ysh[0, :, :] = np.exp(2j * np.pi * (kx * self.xv + ky * self.yv))
        ysh[1, :, :] = np.exp(2j * np.pi * (-kx * self.xv - ky * self.yv))
        # 1st order positive
        self.otf_1_0 = fft2(self.psf * ysh[0])
        yshf = np.abs(fft2(ysh[0]))
        sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if sx < Nw:
            sx = sx
        else:
            sx = sx - 2 * Nw
        if sy < Nw:
            sy = sy
        else:
            sy = sy - 2 * Nw
        temp = np.sqrt(sx ** 2 + sy ** 2) / (2 * self.radius)
        self.eh = np.append(self.eh, temp)
        zsp = self.zero_suppression(sx, sy)
        self.otf_1_0 = self.otf_1_0 * zsp
        self.imgf_1_0 = fft2(self.separr[2 * (order - 1) + 1] * ysh[0])
        # 1st order negative
        self.otf_1_1 = fft2(self.psf * ysh[1])
        yshf = np.abs(fft2(ysh[1]))
        sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if sx < Nw:
            sx = sx
        else:
            sx = sx - 2 * Nw
        if sy < Nw:
            sy = sy
        else:
            sy = sy - 2 * Nw
        zsp = self.zero_suppression(sx, sy)
        temp = np.sqrt(sx ** 2 + sy ** 2) / (2 * self.radius)
        self.eh = np.append(self.eh, temp)
        self.otf_1_1 = self.otf_1_1 * zsp
        self.imgf_1_1 = fft2(self.separr[2 * (order - 1) + 2] * ysh[1])

    def getoverlap(self, order, angle, spacing, verbose=False):
        dx = self.dx / 2
        Nw = self.nx
        kx = dx * np.cos(angle) / spacing
        ky = dx * np.sin(angle) / spacing

        ysh = np.exp(2j * np.pi * (kx * self.xv + ky * self.yv))
        otf = fft2(self.psf * ysh)
        yshf = np.abs(fft2(ysh))
        sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if sx < Nw:
            sx = sx
        else:
            sx = sx - 2 * Nw
        if sy < Nw:
            sy = sy
        else:
            sy = sy - 2 * Nw
        zsp = self.zero_suppression(sx, sy)
        otf = otf * zsp
        imgf = fft2(self.separr[order] * ysh)
        cutoff = self.cutoff
        imgf0 = self.imgf0
        otf0 = self.otf0
        wimgf0 = otf * imgf0
        wimgf1 = otf0 * imgf
        # msk = abs(otf0 * otf) > cutoff
        msk = zsp * self.zero_suppression(0, 0)
        a = np.sum(msk * wimgf1 * wimgf0.conj()) / np.sum(msk * wimgf0 * wimgf0.conj())
        mag = np.abs(a)
        phase = np.angle(a)
        if verbose:
            t = (msk * wimgf1 * wimgf0.conj()) / (msk * wimgf0 * wimgf0.conj())
            t[np.isnan(t)] = 0.0
            figure()
            imshow(np.abs(np.fft.fftshift(t)), interpolation='nearest', vmin=0.0, vmax=2.0)
            colorbar()
            tf.imwrite('overlap_abs.tif', np.abs(np.fft.fftshift(t)).astype(np.float32), photometric='minisblack')
            figure()
            imshow(np.angle(np.fft.fftshift(t)), interpolation='nearest')
            colorbar()
            tf.imwrite('overlap_phs.tif', np.angle(np.fft.fftshift(t)).astype(np.float32), photometric='minisblack')
        return mag, phase

    def mapoverlap(self, order, angle, spacing, nps=10, r_ang=0.02, r_sp=0.008):
        d_ang = 2 * r_ang / nps
        d_sp = 2 * r_sp / nps
        ang_iter = np.arange(-r_ang, r_ang + d_ang / 2, d_ang) + angle
        sp_iter = np.arange(-r_sp, r_sp + d_sp / 2, d_sp) + spacing
        magarr = np.zeros((nps + 1, nps + 1))
        pharr = np.zeros((nps + 1, nps + 1))
        for m, ang in enumerate(ang_iter):
            for n, sp in enumerate(sp_iter):
                print(m, n)
                mag, phase = self.getoverlap(2 * (order - 1) + 1, ang, sp)
                if np.isnan(mag):
                    magarr[m, n] = 0.0
                else:
                    magarr[m, n] = mag
                    pharr[m, n] = phase
        figure()
        subplot(211)
        imshow(magarr, interpolation='nearest')
        subplot(212)
        imshow(pharr, interpolation='nearest')
        # get maximum
        k, l = np.where(magarr == magarr.max())
        angmax = k[0] * d_ang - r_ang + angle
        spmax = l[0] * d_sp - r_sp + spacing
        return angmax, spmax, magarr.max(), magarr

    def init_recon(self):
        nx = 2 * self.nx
        ny = 2 * self.ny
        self.Snum = np.zeros((nx, ny), dtype=np.complex64)
        self.Sden = np.zeros((nx, ny), dtype=np.complex64)
        self.Sden += self.mu ** 2

    def add_recon(self, o, ang, spacing, phase, mag, crop=False, zero=True):
        if crop:
            zsp = 1 - self.zero_suppression(0, 0)
        else:
            zsp = 1

        if zero:
            # 0th order
            imgf = self.imgf0
            otf = self.otf0
            self.Snum += otf.conj() * imgf
            self.Sden += abs(mag * otf) ** 2
        self.shift(o, ang, spacing)
        ph = mag * np.exp(1j * phase)
        # +1st order
        imgf = zsp * self.imgf_1_0
        otf = zsp * self.otf_1_0
        self.Snum += ph * otf.conj() * imgf
        self.Sden += abs(mag * otf) ** 2
        # -1 order
        imgf = zsp * self.imgf_1_1
        otf = zsp * self.otf_1_1
        self.Snum += ph.conj() * otf.conj() * imgf
        self.Sden += abs(mag * otf) ** 2

    def finish_recon(self):
        # ss = np.max(self.eh)
        # A = self.apod(ss)
        self.S = self.Snum / self.Sden  # * A
        self.finalimage = fftshift(ifft2(self.S))

    def save_results(self, fd):
        tf.imwrite(fd + r'\final_image.tif', np.abs(self.finalimage).astype(np.float32), photometric='minisblack')
        tf.imwrite(fd + r'\effective_OTF.tif', np.abs(fftshift(self.S)).astype(np.float32), photometric='minisblack')

    def hessian_denoise(self, mu=1.0, lam=1.0, w_xy=2.0, n_iter=50):
        """Split-Bregman Hessian denoiser for a Wiener-SIM reconstruction g.
        min ||f_xx||_1 + ||f_yy||_1 + w_xy||f_xy||_1 + (mu/2)||f - g||^2 ."""
        g = np.real(self.finalimage)
        ny, nx = g.shape
        wx = 2 * np.pi * np.fft.fftfreq(nx)[None, :]
        wy = 2 * np.pi * np.fft.fftfreq(ny)[:, None]

        # Transfer functions of the discrete 2nd-difference operators
        Dxx = -4 * np.sin(wx / 2) ** 2 + 0j
        Dyy = -4 * np.sin(wy / 2) ** 2 + 0j
        Dxy = (1 - np.exp(-1j * wx)) * (1 - np.exp(-1j * wy))
        ops, weights = [Dxx, Dyy, Dxy], [1.0, 1.0, w_xy]

        denom = mu + lam * sum(np.abs(D) ** 2 for D in ops)  # precompute once
        G = np.fft.fftshift(self.S)

        f = g.copy()
        d = [np.zeros_like(g) for _ in ops]  # auxiliary  d_i ~ D_i f
        b = [np.zeros_like(g) for _ in ops]  # Bregman variables

        for _ in range(n_iter):
            # f-update: diagonal in Fourier -> one division
            rhs = mu * G + lam * sum(np.conj(D) * np.fft.fft2(di - bi) for D, di, bi in zip(ops, d, b))
            f = np.real(np.fft.ifft2(rhs / denom))
            F = np.fft.fft2(f)
            # d-update (weighted shrinkage) + Bregman update
            for i, (D, wgt) in enumerate(zip(ops, weights)):
                t = np.real(np.fft.ifft2(D * F)) + b[i]
                d[i] = np.sign(t) * np.maximum(np.abs(t) - wgt / lam, 0.0)
                b[i] = t - d[i]
        return f

    def interp(self, arr):
        nx, ny = arr.shape
        px = int(nx / 2)
        py = int(ny / 2)
        arrf = fft2(arr)
        arro = np.pad(np.fft.fftshift(arrf), ((px, px), (py, py)), 'constant', constant_values=(0, 0))
        out = ifft2(np.fft.fftshift(arro))
        return out

    def zero_suppression(self, sx, sy, g=False):
        x = self.xv
        y = self.yv
        msk = (np.sqrt((x - sx) ** 2 + (y - sy) ** 2) < self.radius) * 1.0
        if g:
            g = 1 - self.strength * np.exp(-(np.abs((x - sx) ** 2) + np.abs((y - sy) ** 2))) / (2 * (self.fwhm ** 2))
            g[g < 1.0] = self.minv
            msk = g * msk
        return msk

    def apod(self, mp):
        r = (2 * mp) * self.radius
        ap = 1 - np.sqrt(self.xv ** 2 + self.yv ** 2) / r
        ap[ap < 0.] = 0.
        return ap


if __name__ == '__main__':
    fns = r"C:\Users\Ruiz\Desktop\New folder (3)\2D_SIM_test_000000_ids_1_12.tiff"
    img = tf.imread(fns)
    p = SIM_RECON(img, 2, 6, 5, 0.054, 0.505, 1.3)
    p.radius = 4.0 * 1024 * 0.054
    p.init_recon()
    p.separate(0, True)
    p.shift0()
    x1 = p.mapoverlap(1, 1.57 + 1.538, 0.246 * 2, nps=8, r_ang=0.04, r_sp=0.04)
    x1 = p.mapoverlap(1, x1[0], x1[1], nps=10, r_ang=0.005, r_sp=0.005)
    x2 = p.mapoverlap(2, x1[0], x1[1] / 2, nps=8, r_ang=0.04, r_sp=0.04)
    x2 = p.mapoverlap(2, x2[0], x2[1], nps=10, r_ang=0.005, r_sp=0.005)
    a1 = p.getoverlap(1, x1[0], x1[1])
    a2 = p.getoverlap(3, x2[0], x2[1])
    p.add_recon(1, x1[0], x1[1], -a1[1], 0.8, False, True)
    p.add_recon(2, x2[0], x2[1], -a2[1], 0.3, False, False)
    p.separate(1, True)
    p.shift0()
    x1 = p.mapoverlap(1, 1.57 - 0.034, 0.245 * 2, nps=8, r_ang=0.04, r_sp=0.04)
    x1 = p.mapoverlap(1, x1[0], x1[1], nps=10, r_ang=0.005, r_sp=0.005)
    x2 = p.mapoverlap(2, x1[0], x1[1] / 2, nps=8, r_ang=0.04, r_sp=0.04)
    x2 = p.mapoverlap(2, x2[0], x2[1], nps=10, r_ang=0.005, r_sp=0.005)
    a1 = p.getoverlap(1, x1[0], x1[1])
    a2 = p.getoverlap(3, x2[0], x2[1])
    p.add_recon(1, x1[0], x1[1], -a1[1], 0.8, False, True)
    p.add_recon(2, x2[0], x2[1], -a2[1], 0.3, False, False)
    p.finish_recon()
    p.save_results(r'C:\Users\Ruiz\Desktop\New folder (3)')

    import heissian_deconv
    heissian_deconv.attach(SIM_RECON)
    f = p.hessian_deconv(lam=3e-3, n_iter=60)
    tf.imwrite(r'C:\Users\Ruiz\Desktop\New folder (3)\final_image_heissan.tif',
               np.abs(p.finalimage).astype(np.float32), photometric='minisblack')
    tf.imwrite(r'C:\Users\Ruiz\Desktop\New folder (3)\effective_OTF_heissan.tif',
               np.abs(np.fft.fftshift(p.S)).astype(np.float32), photometric='minisblack')
    h = p.effective_psf()  # inspect the star
    f = p.rl_deconv(n_iter=50, apodize=True)  # RL with it
    p.save_hessian(r'C:\Users\Ruiz\Desktop\New folder (3)\hessian_image.tif')
