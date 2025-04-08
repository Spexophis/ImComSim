import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf
from tqdm import tqdm

matplotlib.use('Qt5Agg')
plt.ion()


def dft2(arr, k, a, arn):
    f1 = np.linspace(-1 / (2 * a[0]), 1 / (2 * a[0]), arn[0]) - k[0] / arr.shape[0]
    f1 = f1.reshape((-1, 1))
    f2 = np.linspace(-1 / (2 * a[1]), 1 / (2 * a[1]), arn[1]) - k[1] / arr.shape[1]
    x1 = np.arange(0, arr.shape[0])
    x2 = np.arange(0, arr.shape[1])
    x2 = x2.reshape((-1, 1))
    g1 = np.exp(-1j * 2 * np.pi * f1 * x1)
    g2 = np.exp(-1j * 2 * np.pi * x2 * f2)
    xht = np.matmul(g1, np.matmul(arr, g2))
    return xht


class Pupil:
    def __init__(self):

        self.na = 1.42
        self.ri = 1.512
        self.wl = 488E-9
        self.hpp = np.array([256, 256])
        px = np.linspace(-1, 1, self.hpp[0])
        py = np.linspace(-1, 1, self.hpp[1])
        self.px, self.py = np.meshgrid(px, py)
        self.r = np.sqrt(self.px ** 2 + self.py ** 2)
        self.r[self.r > 1] = 0
        self.theta = np.arcsin(self.na * self.r / self.ri)
        self.phi = np.arctan2(self.py, self.px) * 180 / np.pi
        self.stop = (np.sqrt(self.px ** 2 + self.py ** 2) <= 1)
        self.mask = np.ones(self.stop.shape)
        self.apo = 1 / np.cos(self.theta)
        self.amp = self.stop.astype(float)
        self.phase = np.zeros(self.stop.shape)
        self.k_0 = 2 * np.pi / self.wl
        self.k_xy = self.r * self.k_0 * self.na
        self.k_z = np.sqrt((self.k_0 * self.ri) ** 2 - self.k_xy ** 2)
        self.psf_size = np.array([256, 256, 256])
        self.psf_pitch = np.array([20E-9, 20E-9, 20E-9])
        self.ex = None
        self.ey = None
        self.ez = None

    def bessel_pupil(self, outer_na, inner_na):
        outer_r = outer_na / self.na
        inner_r = inner_na / self.na
        self.amp = (self.r <= outer_r) & (self.r >= inner_r)

    def pupil_mask(self, kind="bisected"):
        if kind == "bisected":
            nr = np.sqrt((self.px - 0.5) ** 2 + (self.py - 0.0) ** 2)
            nr = nr < 0.4
            nr = nr * 1
            nr_ = np.sqrt((self.px + 0.5) ** 2 + (self.py - 0.0) ** 2)
            nr_ = nr_ < 0.4
            nr_ = nr_ * 1
            self.mask = nr + nr_
            self.phase = np.pi * (np.abs(self.phi) > 90)

    def propagate(self, z):
        scaling = self.wl / (2 * self.na * self.psf_pitch[0:2])
        defocus = np.exp(1j * self.k_z * z)

        field_x = dft2(defocus * self.ex, np.array([0, 0]), scaling, self.psf_size[0:2])
        field_y = dft2(defocus * self.ey, np.array([0, 0]), scaling, self.psf_size[0:2])
        field_z = dft2(defocus * self.ez, np.array([0, 0]), scaling, self.psf_size[0:2])

        electric_field = np.zeros([self.psf_size[0], self.psf_size[1], 3], dtype=complex)

        electric_field[:, :, 0] = field_x
        electric_field[:, :, 1] = field_y
        electric_field[:, :, 2] = field_z

        intensity = np.abs(field_x) ** 2 + np.abs(field_y) ** 2 + np.abs(field_z) ** 2

        return electric_field, intensity

    def propagate3d(self):
        zpx = np.linspace(-(self.psf_size[2] - 1) / 2, (self.psf_size[2] - 1) / 2, self.psf_size[2])
        electric_field = np.zeros([self.psf_size[0], self.psf_size[1], self.psf_size[2], 3], dtype=complex)
        intensity = np.zeros(self.psf_size)

        for z_index in tqdm(range(1, self.psf_size[2])):
            z = zpx[z_index] * self.psf_pitch[2]
            electric_field[:, :, z_index, :], intensity[:, :, z_index] = self.propagate(z)

        return electric_field, intensity

    def apply_polarisation(self, polarisation):
        x, y = 0, 0
        if polarisation == 'vertical':
            x, y = 0, 1
        elif polarisation == 'horizontal':
            x, y = 1, 0
        elif polarisation == 'vertical_binary':
            x = 0
            y = 1
        elif polarisation == 'circular':
            x, y = 1, 1j
        elif polarisation == 'radial':
            x, y = np.cos(self.phi), np.sin(self.phi)
        elif polarisation == 'azimuthal':
            x, y = -np.sin(self.phi), np.cos(self.phi)
        elif polarisation == 'dipole_x':
            x = np.cos(self.theta) * np.cos(self.phi) ** 2 + np.sin(self.phi) ** 2
            y = (np.cos(self.theta) - 1) * np.sin(self.phi) * np.cos(pupil.phi)
        elif polarisation == 'dipole_y':
            x = (np.cos(self.theta) - 1) * np.sin(self.phi) * np.cos(pupil.phi)
            y = np.cos(self.theta) * np.cos(self.phi) ** 2 + np.sin(self.phi) ** 2
        elif polarisation == 'dipole_z':
            x = np.sin(self.theta) * np.cos(self.phi)
            y = np.sin(self.theta) * np.sin(self.phi)

        pol_x = x * (1 - np.cos(2 * self.phi) * (1 - np.cos(self.theta)) + np.cos(self.theta)) + y * (
                -1 + np.cos(self.theta)) * np.sin(2 * self.phi)
        pol_y = y * (1 - np.cos(2 * self.phi) * (1 - np.cos(self.theta)) + np.cos(self.theta)) + x * (
                -1 + np.cos(self.theta)) * np.sin(2 * self.phi)
        pol_z = -2 * x * np.cos(self.phi) * np.sin(self.theta) - 2 * y * np.sin(self.phi) * np.sin(self.theta)

        self.ex = pol_x * self.stop * self.apo * self.amp * self.mask * np.exp(1j * self.phase)
        self.ey = pol_y * self.stop * self.apo * self.amp * self.mask * np.exp(1j * self.phase)
        self.ez = pol_z * self.stop * self.apo * self.amp * self.mask * np.exp(1j * self.phase)


if __name__ == "__main__":
    pupil = Pupil()
    # pupil.set_bessel_pupil(0.9, 0.8)
    pupil.pupil_mask(kind="bisected")
    pupil.apply_polarisation('vertical')
    efd, itn = pupil.propagate(0)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(pupil.amp)
    ax[0, 0].title.set_text('Pupil amplitude')
    ax[0, 1].imshow(np.angle(pupil.phase))
    ax[0, 1].title.set_text('Pupil phase')
    ax[1, 0].imshow(itn)
    ax[1, 0].title.set_text('PSF')
    ax[1, 1].imshow(np.abs(np.fft.fftshift(np.fft.fft2(itn))))
    ax[1, 1].title.set_text('MTF')
    plt.tight_layout()
    plt.show()

    e3d, i3d = pupil.propagate3d()
    tf.imwrite(r"C:\Users\Ruiz\Desktop\bisect_h_prop3d.tif", i3d)
