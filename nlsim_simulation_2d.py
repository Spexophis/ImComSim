# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 13:07:14 2023

@author: Ruizhe.Lin
"""


import Zernike36 as Z
import tifffile as tf
import numpy as np
import numpy.random as rd

pi = np.pi
fft2 = np.fft.fft2
ifft2 = np.fft.ifft2
fftn = np.fft.fftn
ifftn = np.fft.ifftn
fftshift = np.fft.fftshift

class nlsim_2d():

    def __init__(self, **kwargs):
        self.Np = kwargs['number_of_fluorophores']
        self.dx = kwargs['image_pixel_size']
        self.nx = kwargs['image_pixel_number']
        self.na = kwargs['numerical_aperture']
        self.wl = kwargs['emission_wavelength']
        self.sp = kwargs['linear_pattern_spacing']
        self.nphs = kwargs['number_of_shifted_phases']
        self.nangs = kwargs['number_of_rotated_angles']
        self.nzarr = kwargs['number_of_Zernike_modes']
        self.zarr = np.zeros(self.nzarr)
        self.img = np.zeros((self.nx, self.nx))
        self.texpo = 0.1 #unit in second
        self.taoff = 0.02 #unit in second


    def get_pupil_wf(self):
        wl = self.wl
        na = self.na
        dp = 1/(self.nx*self.dx)
        radius = (na/wl)/dp
        msk = self.shift(self.discArray((self.nx,self.nx),radius))/np.sqrt(pi*radius**2)/self.nx
        phi = np.zeros((self.nx,self.nx))
        for m in range(1,self.nzarr):
            phi = phi + self.zarr[m]*Z.Zm(m,radius,[0,0],self.nx)
        self.wf = msk*np.exp(1j*phi).astype(np.complex64)

    def getobj(self):
        Np = self.Np
        self.xps = (self.dx*self.nx)*(0.8*rd.rand(Np)+0.1)
        self.yps = (self.dx*self.nx)*(0.8*rd.rand(Np)+0.1)

    def getLines(self, number_of_lines):
        number_of_fluorophores_per_line = np.random.randint(128, 512, number_of_lines)
        x_start = (self.dx*self.nx)*(0.8*np.random.rand(number_of_lines)+0.1)
        y_start = (self.dx*self.nx)*(0.8*np.random.rand(number_of_lines)+0.1)
        x_end = (self.dx*self.nx)*(0.8*np.random.rand(number_of_lines)+0.1)
        y_end = (self.dx*self.nx)*(0.8*np.random.rand(number_of_lines)+0.1)
        xps = np.zeros(1)
        yps = np.zeros(1)
        for i in range(number_of_lines):
            xps = np.concatenate((xps, np.linspace(x_start[i], x_end[i], number_of_fluorophores_per_line[i])), axis=0)
            yps = np.concatenate((yps, np.linspace(y_start[i], y_end[i], number_of_fluorophores_per_line[i])), axis=0)
        self.xps = np.delete(xps, 0)
        self.yps = np.delete(yps, 0)
        self.Np = number_of_fluorophores_per_line.sum()

    def addpsf(self,x,y,I):
        # create phase
        nx = self.nx
        alpha = 2*pi/nx/self.dx
        g = lambda m, n: np.exp(1j*alpha*(m*x+n*y)).astype(np.complex64)
        ph = np.fromfunction(g, (nx,nx), dtype=np.float32)
        ph = self.shift(ph)
        wfp = np.sqrt(I)*ph*self.wf
        self.img = self.img + abs(fft2(wfp))**2

    def nlsimpattern(self, kx, ky, angle, phase, x, y, I):
        return 0.5*(1-np.cos(2*pi*kx*x+2*pi*ky*y+phase+pi)) * np.exp(-I*0.5*(1-np.cos(2*pi*kx*x+2*pi*ky*y+phase))*self.texpo/(I*self.taoff))

    def getoneimg(self, angle, phase, Iph):
        self.img[:,:] = 20.0
        # create psfs
        kx = np.cos(angle)/self.sp
        ky = np.sin(angle)/self.sp
        for m in range(self.Np):
            Ip = Iph *self.nlsimpattern(kx, ky, angle, phase, self.xps[m], self.yps[m], Iph)
            self.addpsf(self.xps[m],self.yps[m],Ip)
        # noise
        self.img = rd.poisson(self.img)
        # done!

    def runoneangle(self,Iph=256):
        out = np.zeros((self.nphs,self.nx,self.nx),dtype=np.float32)
        for i in range(self.nphs):
            self.getoneimg(0.0,i*2*pi/self.nphs,Iph)
            out[i,:,:] = self.img
        tf.imsave('sim_nsi2d.tif',out,photometric='minisblack')

    def runallangles(self,Iph=256,fn='sim_nsi2d.tif'):
        out = np.zeros((self.nangs*self.nphs,self.nx,self.nx),dtype=np.float32)
        for m in range(self.nangs):
            for n in range(self.nphs):
                print('angle', m, 'phase', n)
                self.getoneimg(m*2*pi/self.nangs,n*2*pi/self.nphs,Iph)
                out[self.nphs*m+n,:,:] = self.img
        tf.imsave(fn,out,photometric='minisblack')

    def discArray(self, shape=(128, 128), radius=64, origin=None, dtype=np.float64):
        nx = shape[0]
        ny = shape[1]
        ox = nx/2
        oy = ny/2
        x = np.linspace(-ox,ox-1,nx)
        y = np.linspace(-oy,oy-1,ny)
        X,Y = np.meshgrid(x,y)
        rho = np.sqrt(X**2 + Y**2)
        disc = (rho<radius).astype(dtype)
        if not origin==None:
            s0 = origin[0]-int(nx/2)
            s1 = origin[1]-int(ny/2)
            disc = np.roll(np.roll(disc,s0,0),s1,1)
        return disc

    def shift(self, arr, shifts=None):
        if shifts == None:
            shifts = ( np.array(arr.shape)/2 ).astype(np.uint16)
        if len(arr.shape)==len(shifts):
            for m,p in enumerate(shifts):
                arr = np.roll(arr,p,m)
        return arr


if __name__ == '__main__':
    sim = nlsim_2d(number_of_fluorophores=512,
                   image_pixel_size=0.075,
                   image_pixel_number=128,
                   numerical_aperture=1.4,
                   emission_wavelength=0.505,
                   linear_pattern_spacing=0.24,
                   number_of_shifted_phases=7,
                   number_of_rotated_angles=7,
                   number_of_Zernike_modes=26)
    sim.getobj()
    sim.get_pupil_wf()
    sim.runallangles()
