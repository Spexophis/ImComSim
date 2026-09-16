"""
Optimized 2D structured illumination reconstruction
python3.6
Ruizhe Lin 04/15/2019

Performance revision (2026)
"""

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tifffile as tf
from matplotlib.pyplot import imshow, subplot, plot, figure, colorbar, show
from numpy.fft import fftshift
from scipy.signal.windows import tukey

import psf_generator as pg

try:
    from scipy.fft import fft as _fft, fft2 as _fft2, ifft2 as _ifft2

    _FFT_KW = {"workers": -1}
    _SCIPY_FFT = True
except ImportError:
    from numpy.fft import fft as _fft, fft2 as _fft2, ifft2 as _ifft2

    _FFT_KW = {}
    _SCIPY_FFT = False

_WORKERS = -1


def fft2(a, overwrite=False):
    """Forward 2-D FFT.  ``overwrite`` lets the transform reuse ``a`` as its
    own workspace; the returned values are bit-identical either way."""
    if _SCIPY_FFT:
        return _fft2(a, workers=_WORKERS, overwrite_x=overwrite)
    return _fft2(a)


def ifft2(a, overwrite=False):
    if _SCIPY_FFT:
        return _ifft2(a, workers=_WORKERS, overwrite_x=overwrite)
    return _ifft2(a)


class SIM_RECON:

    def __init__(self, img_stack, nangles, nphs, norders, dx, wavelength, na, exact_modulation=False):
        self.dx = dx
        self.na = na
        self.wl = wavelength
        self.nph = nphs
        self.nang = nangles
        self.norders = norders
        self.n, self.nx, self.ny = img_stack.shape
        self.eta = 0.1
        # caches and scratch space (set up before anything that may use them)
        self.exact_modulation = exact_modulation
        self._tls = threading.local()
        self._zsp_cache = {}
        self._support_cache = {}
        self._sepmat = None
        self.raps = self.raps_stack(img_stack, "tukey", np.float64, True)
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

    # ------------------------------------------------------------------
    # scratch buffers
    # ------------------------------------------------------------------

    def _scratch(self, name, dtype):
        """Thread-local 2N x 2N buffer.  Buffers are only used for values that
        do not escape the call that filled them."""
        buf = getattr(self._tls, 'buf', None)
        if buf is None:
            buf = self._tls.buf = {}
        key = (name, np.dtype(dtype))
        arr = buf.get(key)
        if arr is None:
            arr = buf[key] = np.empty((2 * self.nx, 2 * self.ny), dtype)
        return arr

    def release_buffers(self):
        """Drop the scratch buffers of the calling thread (3 x 2N x 2N
        complex128 each) once the overlap search is done."""
        self._tls.buf = {}

    # ------------------------------------------------------------------

    def raps_stack(self, stack, window="tukey", dtype=np.float64, verbose=True):
        # no copy: the loop reads ``stack`` and writes into ``buf``
        stack = np.asarray(stack, float)

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
        return profs

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
        # 1-D views of the same coordinates, for the separable operations
        self._x1 = xv[:, 0]
        self._y1 = yv[0, :]
        self._zsp_cache.clear()
        self._support_cache.clear()

    def sepmatrix(self):
        if self._sepmat is not None:
            return self._sepmat
        nphases = self.nph
        sepmat = np.zeros((self.norders, nphases), dtype=np.float32)
        norders = int((self.norders + 1) / 2)
        phi = 2 * np.pi / nphases
        for j in range(nphases):
            sepmat[0, j] = 1.0 / nphases
            for order in range(1, norders):
                sepmat[2 * order - 1, j] = 2.0 * np.cos(j * order * phi) / nphases
                sepmat[2 * order, j] = 2.0 * np.sin(j * order * phi) / nphases
        self._sepmat = sepmat
        return sepmat

    def lockin_demod(self, ang, alpha):
        a = self.img[ang]
        d = np.empty_like(a)
        d[0] = a[0] - a[-1]                     # == a - np.roll(a, 1, axis=0)
        np.subtract(a[1:], a[:-1], out=d[1:])
        idc2 = np.sqrt((2.0 / self.nph ** 2) * np.einsum('ijk,ijk->jk', d, d))
        return a - alpha * a.mean(0) + idc2

    def _modulation(self, kx, ky, dtype, out=None, exact=False):
        if exact or self.exact_modulation:
            ysh = np.exp(2j * np.pi * (kx * self.xv + ky * self.yv))
            if out is None:
                return ysh.astype(dtype, copy=False)
            out[...] = ysh
            return out
        ex = np.exp(2j * np.pi * (kx * self._x1))
        ey = np.exp(2j * np.pi * (ky * self._y1))
        if out is None:
            out = np.empty((2 * self.nx, 2 * self.ny), dtype)
        # products are formed in complex128 and stored (rounded) into ``out``,
        # matching the original "exponentiate in double, then cast" order
        np.multiply(ex[:, None], ey[None, :], out=out)
        return out

    def _peak_index(self, kx, ky):
        Nw = self.nx
        ax = np.abs(_fft(np.exp(2j * np.pi * (kx * self._x1)), **_FFT_KW))
        ay = np.abs(_fft(np.exp(2j * np.pi * (ky * self._y1)), **_FFT_KW))
        sx = int(ax.argmax())
        sy = int(ay.argmax())
        if sx >= Nw:
            sx -= 2 * Nw
        if sy >= Nw:                            # the original tests both
            sy -= 2 * Nw                        # against Nw = self.nx
        return sx, sy

    def separate(self, nangle, lockin=False):
        Nw = self.nx
        mat = self.sepmatrix()
        if lockin:
            data = self.lockin_demod(nangle, self.eps)
            outr = np.dot(mat, data.reshape(self.nph, Nw ** 2))
        else:
            outr = np.dot(mat, self.img[nangle].reshape(self.nph, Nw ** 2))
        self.separr = np.zeros((self.norders, Nw * 2, Nw * 2), dtype=np.complex64)
        self.separr[0] = fftshift(self.interp(outr[0].reshape(Nw, Nw)) * self.winf)
        for order in range(1, (self.norders + 1) // 2):
            re = outr[2 * order - 1]
            im = outr[2 * order]
            self.separr[2 * order - 1] = fftshift(self.interp((re + 1j * im).reshape(Nw, Nw)) * self.winf)
            self.separr[2 * order] = fftshift(self.interp((re - 1j * im).reshape(Nw, Nw)) * self.winf)

    def shift0(self):
        self.otf0 = fft2(self.psf)
        zsp = self.zero_suppression(0, 0)
        self.otf0 *= zsp
        self.imgf0 = fft2(self.separr[0])

    def shift(self, order, angle, spacing):
        """ shift data in freq space by multiplication in real space """
        dx = self.dx / 2
        kx = dx * np.cos(angle) / spacing
        ky = dx * np.sin(angle) / spacing
        # shift matrix (complex64, as the original (2, 2N, 2N) buffer was);
        # exp(-z) is bit-identical to conj(exp(z)) for these arguments
        ysh_0 = self._modulation(kx, ky, np.complex64, exact=True)
        ysh_1 = ysh_0.conj()
        # 1st order positive
        self.otf_1_0 = fft2(self.psf * ysh_0, overwrite=True)
        sx, sy = self._peak_index(kx, ky)
        temp = np.sqrt(sx ** 2 + sy ** 2) / (2 * self.radius)
        self.eh = np.append(self.eh, temp)
        self.otf_1_0 *= self.zero_suppression(sx, sy)
        self.imgf_1_0 = fft2(self.separr[2 * (order - 1) + 1] * ysh_0, overwrite=True)
        # 1st order negative
        self.otf_1_1 = fft2(self.psf * ysh_1, overwrite=True)
        sx, sy = self._peak_index(-kx, -ky)
        temp = np.sqrt(sx ** 2 + sy ** 2) / (2 * self.radius)
        self.eh = np.append(self.eh, temp)
        self.otf_1_1 *= self.zero_suppression(sx, sy)
        self.imgf_1_1 = fft2(self.separr[2 * (order - 1) + 2] * ysh_1, overwrite=True)

    def getoverlap(self, order, angle, spacing, verbose=False):
        dx = self.dx / 2
        kx = dx * np.cos(angle) / spacing
        ky = dx * np.sin(angle) / spacing
        imgf0 = self.imgf0
        otf0 = self.otf0
        sx, sy = self._peak_index(kx, ky)

        if verbose:
            ysh = self._modulation(kx, ky, np.complex128)
            otf = fft2(self.psf * ysh, overwrite=True)
            zsp = self.zero_suppression(sx, sy)
            otf = otf * zsp
            imgf = fft2(self.separr[order] * ysh, overwrite=True)
            msk = zsp * self.zero_suppression(0, 0)
            wimgf0 = otf * imgf0
            wimgf1 = otf0 * imgf
            a = np.sum(msk * wimgf1 * wimgf0.conj()) / np.sum(msk * wimgf0 * wimgf0.conj())
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
            return np.abs(a), np.angle(a)

        ysh = self._modulation(kx, ky, np.complex128, out=self._scratch('ysh', np.complex128))
        otf = fft2(np.multiply(self.psf, ysh, out=self._scratch('p0', np.complex128)), overwrite=True)
        imgf = fft2(np.multiply(self.separr[order], ysh, out=self._scratch('p1', np.complex128)), overwrite=True)
        idx = self._support(sx, sy)
        wimgf0 = np.take(otf.reshape(-1), idx) * np.take(imgf0.reshape(-1), idx)
        wimgf1 = np.take(otf0.reshape(-1), idx) * np.take(imgf.reshape(-1), idx)
        with np.errstate(invalid='ignore', divide='ignore'):
            a = np.sum(wimgf1 * wimgf0.conj()) / np.sum(wimgf0 * wimgf0.conj())
        return np.abs(a), np.angle(a)

    def mapoverlap(self, order, angle, spacing, nps=10, r_ang=0.02, r_sp=0.008,
                   nthreads=4, verbose=True):
        d_ang = 2 * r_ang / nps
        d_sp = 2 * r_sp / nps
        ang_iter = np.arange(-r_ang, r_ang + d_ang / 2, d_ang) + angle
        sp_iter = np.arange(-r_sp, r_sp + d_sp / 2, d_sp) + spacing
        magarr = np.zeros((nps + 1, nps + 1))
        pharr = np.zeros((nps + 1, nps + 1))
        od = 2 * (order - 1) + 1

        def _cell(m, n):
            return self.getoverlap(od, ang_iter[m], sp_iter[n])

        cells = [(m, n) for m in range(len(ang_iter)) for n in range(len(sp_iter))]
        if nthreads and nthreads > 1:
            with ThreadPoolExecutor(max_workers=int(nthreads)) as pool:
                results = list(pool.map(lambda mn: _cell(*mn), cells))
        else:
            results = []
            for m, n in cells:
                if verbose:
                    print(m, n)
                results.append(_cell(m, n))
        for (m, n), (mag, phase) in zip(cells, results):
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
        return angmax, spmax, magarr.max(), magarr

    def init_recon(self):
        nx = 2 * self.nx
        ny = 2 * self.ny
        self.Snum = np.zeros((nx, ny), dtype=np.complex64)
        self.Sden = np.full((nx, ny), self.mu ** 2, dtype=np.complex64)

    def add_recon(self, o, ang, spacing, phase, mag, crop=False, zero=True):
        if zero:
            # 0th order
            imgf = self.imgf0
            otf = self.otf0
            self.Snum += otf.conj() * imgf
            self.Sden += abs(mag * otf) ** 2
        self.shift(o, ang, spacing)
        ph = mag * np.exp(1j * phase)
        if crop:
            zsp = 1 - self.zero_suppression(0, 0)
            imgf_p, otf_p = zsp * self.imgf_1_0, zsp * self.otf_1_0
            imgf_n, otf_n = zsp * self.imgf_1_1, zsp * self.otf_1_1
        else:
            # zsp == 1: use the arrays as they are instead of copying them
            imgf_p, otf_p = self.imgf_1_0, self.otf_1_0
            imgf_n, otf_n = self.imgf_1_1, self.otf_1_1
        # +1st order
        self.Snum += ph * otf_p.conj() * imgf_p
        self.Sden += abs(mag * otf_p) ** 2
        # -1 order
        self.Snum += ph.conj() * otf_n.conj() * imgf_n
        self.Sden += abs(mag * otf_n) ** 2

    def finish_recon(self):
        # ss = np.max(self.eh)
        # A = self.apod(ss)
        self.S = self.Snum / self.Sden  # * A
        self.finalimage = fftshift(ifft2(self.S))

    def save_results(self, fd):
        tf.imwrite(fd + r'\final_image.tif', np.abs(self.finalimage).astype(np.float32), photometric='minisblack')
        tf.imwrite(fd + r'\effective_OTF.tif', np.abs(fftshift(self.S)).astype(np.float32), photometric='minisblack')

    @staticmethod
    def interp(arr):
        nx, ny = arr.shape
        px = int(nx / 2)
        py = int(ny / 2)
        arrf = fft2(arr)
        arro = np.zeros((2 * nx, 2 * ny), dtype=arrf.dtype)
        arro[:px, :py] = arrf[:px, :py]
        arro[:px, -py:] = arrf[:px, py:]
        arro[-px:, :py] = arrf[px:, :py]
        arro[-px:, -py:] = arrf[px:, py:]
        out = ifft2(arro, overwrite=True)
        return out

    def zero_suppression(self, sx, sy, g=False):
        x = self.xv
        y = self.yv
        r2 = float(self.radius) ** 2
        if g:
            msk = ((x - sx) ** 2 + (y - sy) ** 2 < r2) * 1.0
            g = 1 - self.strength * np.exp(-(np.abs((x - sx) ** 2) + np.abs((y - sy) ** 2))) / (2 * (self.fwhm ** 2))
            g[g < 1.0] = self.minv
            return g * msk
        key = (sx, sy, r2)
        msk = self._zsp_cache.get(key)
        if msk is None:
            if len(self._zsp_cache) >= 8:
                self._zsp_cache.clear()
            msk = self._zsp_cache[key] = ((x - sx) ** 2 + (y - sy) ** 2 < r2) * 1.0
        return msk

    def _support(self, sx, sy):
        r2 = float(self.radius) ** 2
        key = (sx, sy, r2)
        idx = self._support_cache.get(key)
        if idx is None:
            x = self._x1
            y = self._y1
            rows = np.flatnonzero(((x - sx) ** 2 < r2) & (x ** 2 < r2))
            cols = np.flatnonzero(((y - sy) ** 2 < r2) & (y ** 2 < r2))
            xr = x[rows][:, None]
            yc = y[cols][None, :]
            msk = (((xr - sx) ** 2 + (yc - sy) ** 2) < r2) & ((xr ** 2 + yc ** 2) < r2)
            idx = (rows[:, None] * (2 * self.ny) + cols[None, :])[msk]
            if len(self._support_cache) >= 64:
                self._support_cache.clear()
            self._support_cache[key] = idx
        return idx

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
    p.add_recon(1, x1[0], x1[1], -a1[1], 0.8, False, False)
    p.add_recon(2, x2[0], x2[1], -a2[1], 0.3, False, False)
    p.separate(1, True)
    p.shift0()
    x1 = p.mapoverlap(1, 1.57 - 0.034, 0.245 * 2, nps=8, r_ang=0.04, r_sp=0.04)
    x1 = p.mapoverlap(1, x1[0], x1[1], nps=10, r_ang=0.005, r_sp=0.005)
    x2 = p.mapoverlap(2, x1[0], x1[1] / 2, nps=8, r_ang=0.04, r_sp=0.04)
    x2 = p.mapoverlap(2, x2[0], x2[1], nps=10, r_ang=0.005, r_sp=0.005)
    a1 = p.getoverlap(1, x1[0], x1[1])
    a2 = p.getoverlap(3, x2[0], x2[1])
    p.add_recon(1, x1[0], x1[1], -a1[1], 0.8, False, False)
    p.add_recon(2, x2[0], x2[1], -a2[1], 0.3, False, False)
    p.release_buffers()
    p.finish_recon()
    p.save_results(r'C:\Users\Ruiz\Desktop\New folder (4)')

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