import Utility36 as U
import Zernike36 as Z

import tifffile as tf
import numpy as N
import numpy.random as rd

import time
import concurrent.futures

pi = N.pi
fft2 = N.fft.fft2
ifft2 = N.fft.ifft2
fftn = N.fft.fftn
ifftn = N.fft.ifftn
fftshift = N.fft.fftshift


class sim():

    def __init__(self, nx=128, nz=26):
        self.yps = None
        self.xps = None
        self.zps = None
        self.wf = None
        self.Np = 512  # number of particles
        self.dx = 0.075
        self.dz = 0.200  # 0.100
        self.nx = nx  # 256
        self.nz = nz
        self.fp = 13
        self.na = 1.4
        self.n2 = 1.512
        self.wl = 0.505
        self.sp = 0.2 * 2
        self.nzarr = 15
        self.zarr = 0.0 * rd.randn(15)
        self.img = N.zeros((self.nz, self.nx, self.nx))
        self.imgf = N.zeros((self.nx, self.nx), dtype=N.float32)

    def getobj(self, singleplane=True):
        """ put fluorophores in circle """
        Np = self.Np  # 500
        self.xps = (self.dx * self.nx) * (0.8 * rd.rand(Np) + 0.1)
        self.yps = (self.dx * self.nx) * (0.8 * rd.rand(Np) + 0.1)
        if singleplane:
            self.zps = N.zeros(Np)
        else:
            self.zps = (self.dz * self.nz) * (0.8 * rd.rand(Np) + 0.1 - 0.5)

    def getaberr(self):
        wl = self.wl
        na = self.na
        dp = 1 / (self.nx * self.dx)
        radius = (na / wl) / dp
        msk = U.shift(U.discArray((self.nx, self.nx), radius)) / N.sqrt(pi * radius ** 2) / self.nx
        phi = N.zeros((self.nx, self.nx))
        for m in range(1, self.nzarr):
            phi = phi + self.zarr[m] * Z.Zm(m, radius, [0, 0], self.nx)
        self.wf = msk * N.exp(1j * phi).astype(N.complex64)

    def addpsf(self, x, y, z, I):
        # create phase
        nx = self.nx
        alpha = 2 * pi / nx / self.dx
        g = lambda m, n: N.exp(1j * alpha * (m * x + n * y)).astype(N.complex64)
        ph = N.fromfunction(g, (nx, nx), dtype=N.float32)
        ph = U.shift(ph)
        defocus = N.exp(1j * self.focusmode(z)).astype(N.complex64)
        wfp = N.sqrt(I) * ph * defocus * self.wf
        self.imgf += abs(fft2(wfp)) ** 2
        return True

    def focusmode(self, w=5):  # wavefront phase for focusing
        """ focus mode, d is depth in microns, nap is num. ap.
        focuses, with depth correction """
        nap = self.na
        n2 = self.n2
        dx = self.dx
        Nx = self.nx
        wl = self.wl
        if (nap > n2):
            raise "Numerical aperture cannot be greater than n2!"
        dp = 1 / (Nx * dx)
        radius = (2 * nap / wl) / 2 / dp
        factor = Nx / 2. / radius
        msk = U.discArray(shape=(Nx, Nx), radius=radius, origin=(0, 0))
        dp = (2 * nap / wl) / (2 * radius)
        sinphim = (nap / n2)
        wf = N.zeros((Nx, Nx), dtype=N.float64)
        rho = msk * U.radialArray((Nx, Nx), lambda x: x, origin=(0, 0)) / radius
        wf = 2 * N.pi * msk * n2 * w * N.sqrt(1 - (sinphim * rho) ** 2) / wl
        return wf

    def getoneimg(self, angle, phase, Iph):
        # self.imgf[:,:] = 0.0
        # get points
        # create psfs
        phim = self.na / self.n2
        kx = 2 * pi * N.cos(angle) / self.sp
        ky = 2 * pi * N.sin(angle) / self.sp
        dkz = (2 * pi / self.sp) * (1 - N.sqrt(1 - phim ** 2))
        print(dkz)
        for zp in range(self.nz):
            zplane = self.dz * (zp - self.fp)
            self.imgf[:, :] = 20.0
            for m in range(self.Np):
                # Ip = Iph*0.5*(1+N.cos(kx*self.xps[m]+ky*self.yps[m]+phase))
                cs2xy = N.cos(2 * kx * self.xps[m] + 2 * ky * self.yps[m] + 2 * phase)
                csxy = N.cos(kx * self.xps[m] + ky * self.yps[m] + phase)
                csz = N.cos(dkz * (self.zps[m] - zplane))
                Ip = Iph * (3 + 2 * cs2xy + 4 * csz * csxy)
                self.addpsf(self.xps[m], self.yps[m], self.zps[m] - zplane, Ip)
            self.img[zp] = self.imgf  # fft2(self.imgf).real
        # noise
        self.img = rd.poisson(self.img)
        # done!

    def run(self, Iph=1000, nangles=3, nphases=5):
        out = N.zeros((nangles, nphases, self.nz, self.nx, self.nx), dtype=N.float32)
        for nph in range(nangles):
            for m in range(nphases):
                print('angle', nph, 'phase', m)
                self.getoneimg(nph * (2 * pi / nangles), m * (2 * pi / nphases), Iph)
                out[nph, m, :, :, :] = self.img
        tf.imwrite('sim_si3d_2.tif', out.swapaxes(1, 2).swapaxes(0, 1).reshape(3 * nphases * self.nz, self.nx, self.nx),
                  photometric='minisblack')


if __name__ == '__main__':
    d = N.load(r'C:/Users/ruizhe.lin/Documents/python_codes/sim3d/3DLines_500lines_5x5x5um_300f_per_um.npy')
    d[1] = 0.8 + (d[1] - d[1].min()) / 1000
    d[2] = 0.8 + (d[2] - d[2].min()) / 1000
    d[0] = - 2.5 + (d[0] - d[0].min()) / 1000
    s = sim(88, 26)
    s.getaberr()
    s.getobj()
    s.Np = 674990
    s.xps = d[2]
    s.yps = d[1]
    s.zps = d[0]
    s.runthreeangles(Iph=256, nphases=5)
