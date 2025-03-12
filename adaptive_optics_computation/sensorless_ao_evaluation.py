import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf
from scipy.optimize import curve_fit
from skimage import filters

try:
    import cupy as cp

    cupy_available = True


    def fourier_transform(data):
        data_gpu = cp.asarray(data)
        fft2_gpu = cp.fft.fft2(data_gpu)
        return np.fft.fftshift(fft2_gpu.get())

except ImportError:
    cp = None
    cupy_available = False


    def fourier_transform(data):
        return np.fft.fftshift(np.fft.fft2(data))

matplotlib.use('Qt5Agg')
plt.ion()

wl = 0.5  # wavelength in microns
na = 1.4  # numerical aperture
dx = 0.081  # pixel size in microns
fs = 1 / dx  # Spatial sampling frequency, inverse microns


def rms(data):
    _nx, _ny = data.shape
    _n = _nx * _ny
    _m = np.mean(data, dtype=np.float64)
    _a = (data - _m) ** 2
    _r = np.sqrt(np.sum(_a) / _n)
    return _r


def sobel(image):
    edges = filters.sobel(image)
    focus_measure = np.var(edges)
    return focus_measure


def laplacian(image):
    laplacian_image = filters.laplace(image)
    focus_measure = np.var(laplacian_image)
    return focus_measure


def disc_array(shape=(128, 128), radi=64.0, origin=None, dtp=np.float64):
    _nx, _ny = shape
    _ox, _oy = _nx / 2, _ny / 2
    _x = np.linspace(-_ox, _ox - 1, _nx)
    _y = np.linspace(-_oy, _oy - 1, _ny)
    _xv, _yv = np.meshgrid(_x, _y)
    _rho = np.sqrt(_xv ** 2 + _yv ** 2)
    disc = (_rho < radi).astype(dtp)
    if origin is not None:
        s0 = origin[0] - int(_nx / 2)
        s1 = origin[1] - int(_ny / 2)
        disc = np.roll(np.roll(disc, int(s0), 0), int(s1), 1)
    return disc


def gaussian_filter(shape, sigma, pv, orig=None):
    _nx, _ny = shape
    if orig is None:
        ux = _nx / 2.
        uy = _ny / 2.
    else:
        ux, uy = orig
    g = np.fromfunction(lambda i, j: np.exp(-((i - ux) ** 2. + (j - uy) ** 2.) / (2. * sigma ** 2.)), (_nx, _ny))
    return pv * g


def fft_snr(data, lpr, hpr, relative=True, gau=True):
    _ny, _nx = data.shape
    _df = fs / _nx
    _radius = (na / wl) / _df
    msk = disc_array(shape=(_nx, _ny), radi=0.9 * _radius)
    if gau:
        lp = msk * gaussian_filter(shape=(_nx, _ny), sigma=lpr * _radius, pv=1, orig=None)
        hp = (1 - gaussian_filter(shape=(_nx, _ny), sigma=hpr * _radius, pv=1, orig=None)) * msk
    else:
        lp = disc_array(shape=(_nx, _ny), radi=lpr * _radius)
        hp = msk - disc_array(shape=(_nx, _ny), radi=hpr * _radius)
    wft = fourier_transform(data)
    if relative:
        num = (np.abs(hp * wft)).sum() / (np.abs(wft * msk)).sum()
        den = (np.abs(lp * wft)).sum() / (np.abs(wft * msk)).sum()
        return num / den
    else:
        return (np.abs(hp * wft)).sum() / (np.abs(lp * wft)).sum()


def fft_hpf(data, hpr, relative=True, gau=True):
    _nx, _ny = data.shape
    df = fs / _nx
    _radius = (na / wl) / df
    msk = disc_array(shape=(_nx, _ny), radi=0.9 * _radius)
    if gau:
        hp = (1 - gaussian_filter(shape=(_nx, _ny), sigma=hpr * _radius, pv=1, orig=None)) * msk
    else:
        hp = msk - disc_array(shape=(_nx, _ny), radi=hpr * _radius)
    wft = fourier_transform(data)
    if relative:
        return (np.abs(wft * hp)).sum() / (np.abs(wft * msk)).sum()
    else:
        return (np.abs(wft * hp)).sum()


def selected_frequency(data, freqs, relative=True):
    _ny, _nx = data.shape
    df = fs / _nx
    _radius = (na / wl) / df
    freq_x = np.fft.fftshift(np.fft.fftfreq(_nx, dx))
    freq_y = np.fft.fftshift(np.fft.fftfreq(_ny, dx))
    freq_x = np.divide(1.0, freq_x, where=freq_x != 0, out=np.zeros_like(freq_x))
    freq_y = np.divide(1.0, freq_y, where=freq_y != 0, out=np.zeros_like(freq_y))
    freq_coords = []
    for freq in freqs:
        horizontal_indices = np.argsort(np.abs(freq_x - freq))[:2]
        horizontal_coords = [(x, _ny // 2) for x in horizontal_indices]
        vertical_indices = np.argsort(np.abs(freq_y - freq))[:2]
        vertical_coords = [(_nx // 2, y) for y in vertical_indices]
        freq_coords += horizontal_coords + vertical_coords
    msk = disc_array(shape=(_ny, _nx), radi=0.9 * _radius)
    g = np.zeros((_ny, _nx))
    for freq_coord in freq_coords:
        g += disc_array(shape=(_ny, _nx), radi=9, origin=freq_coord)
    wft = fourier_transform(data)
    if relative:
        return (np.abs(wft * g)).sum() / (np.abs(wft * msk)).sum()
    else:
        return (np.abs(wft * g)).sum()


def binomial_model(_x, a, b, c):
    return a * _x ** 2 + b * _x + c


def find_peak(x_data, y_data, y_std=None):
    _x = np.asarray(x_data)
    _y = np.asarray(y_data)
    if y_std is not None:
        y_err = np.asarray(y_std)
        p_opt, p_cov = curve_fit(binomial_model, _x, _y, sigma=y_err, absolute_sigma=True)
        a, b, c = p_opt
        x_peak = -b / (2 * a)
        if a > 0:
            return "No peak"
        elif x_peak >= _x.max():
            return "Peak above maximum"
        elif x_peak <= _x.min():
            return "Peak below minimum"
        else:
            sigma_a, sigma_b, sigma_c = np.sqrt(np.diag(p_cov))
            x_peak_err = np.sqrt((b / (2 * a ** 2) * sigma_a) ** 2 + (-1 / (2 * a) * sigma_b) ** 2)
            print(x_peak_err)
            return x_peak
    else:
        a, b, c = np.polyfit(_x, _y, 2)
        x_peak = -b / (2 * a)
        if a > 0:
            return "No peak"
        elif x_peak >= _x.max():
            return "Peak above maximum"
        elif x_peak <= _x.min():
            return "Peak below minimum"
        else:
            return x_peak


def find_valley(x_data, y_data, y_std=None):
    _x = np.asarray(x_data)
    _y = np.asarray(y_data)
    if y_std is not None:
        y_err = np.asarray(y_std)
        p_opt, p_cov = curve_fit(binomial_model, _x, _y, sigma=y_err, absolute_sigma=True)
        a, b, c = p_opt
        x_valley = -b / (2 * a)
        if a > 0:
            return "No valley"
        elif x_valley >= _x.max():
            return "Valley above maximum"
        elif x_valley <= _x.min():
            return "Valley below minimum"
        else:
            sigma_a, sigma_b, sigma_c = np.sqrt(np.diag(p_cov))
            x_valley_err = np.sqrt((b / (2 * a ** 2) * sigma_a) ** 2 + (-1 / (2 * a) * sigma_b) ** 2)
            print(x_valley_err)
            return x_valley
    else:
        a, b, c = np.polyfit(_x, _y, 2)
        x_valley = -b / (2 * a)
        if a > 0:
            return "No peak"
        elif x_valley >= _x.max():
            return "Peak above maximum"
        elif x_valley <= _x.min():
            return "Peak below minimum"
        else:
            return x_valley


def find_peak_2d(image, coordinates=None):
    data = image - image.min()
    data = data / data.max()
    mx = np.average(data, axis=0)
    my = np.average(data, axis=1)
    if coordinates is not None:
        cdx, cdy = coordinates
        l_ = find_peak(cdy, my)
        k_ = find_peak(cdx, mx)
        return k_, l_
    else:
        l_ = np.where(my == my.max())
        k_ = np.where(mx == mx.max())
        return k_[0][0], l_[0][0]


def find_valley_2d(image, coordinates=None):
    data = image - image.min()
    data = data / data.max()
    mx = np.average(data, axis=0)
    my = np.average(data, axis=1)
    if coordinates is not None:
        cdx, cdy = coordinates
        l_ = find_valley(cdy, my)
        k_ = find_valley(cdx, mx)
        return k_, l_
    else:
        l_ = np.where(my == my.min())
        k_ = np.where(mx == mx.min())
        return k_[0][0], l_[0][0]


def svd(a, vn=None):
    u, s, vt = np.linalg.svd(a)
    s_inv = np.zeros_like(a.T)
    if vn is None:
        s_inv[:min(a.shape), :min(a.shape)] = np.diag(1 / s[:min(a.shape)])
    else:
        s_inv[:vn, :vn] = np.diag(1 / s[:vn])
    return vt.T @ s_inv @ u.T


if __name__ == '__main__':

    data_folder = r"C:\Users\ruizhe.lin\Documents\data\20241115\20241115_143610_sensorless_acquisitions"

    metric_values = {i: {} for i in range(16)}

    for filename in os.listdir(data_folder):
        if filename.endswith(".tiff"):
            fd = os.path.join(data_folder, filename)
            image_stack = tf.imread(fd)
            fn = os.path.splitext(filename)[0].split("_")
            zn, amp = int(fn[2]), float(fn[4])
            m = []
            for z in range(image_stack.shape[0]):
                img = image_stack[z, :, :]
                # res = hpf(img, 0.64, relative=True, gau=False)
                res = selected_frequency(img, freqs=[1.41, 2.82], relative=True)
                m.append(res)
            metric_values[zn][amp] = m

    for n in [3, 4, 5, 6, 7, 8, 9, 10]:
        x = list(metric_values[0].keys())
        x.extend(list(metric_values[n].keys()))
        y = [np.mean(values) for values in metric_values[0].values()]
        y.extend([np.mean(values) for values in metric_values[n].values()])
        y_er = [np.std(values) for values in metric_values[0].values()]
        y_er.extend([np.std(values) for values in metric_values[n].values()])
        x_pk, x_pk_er, y_pk_er = find_peak(x, y, y_er)
        plt.figure(figsize=(10, 8))
        plt.errorbar(x, y, yerr=y_er, fmt='o', capsize=5, label='mean value with error bar')
        plt.axvline(x_pk, color='red', linestyle='--', label=f'peak = {x_pk:.3f}')
        plt.fill_betweenx([min(y), max(y)], x_pk - x_pk_er, x_pk + x_pk_er,
                          color='red', alpha=0.3, label=f'peak_err = ±{x_pk_er:.3f}')
        plt.xlabel('Zernike Amplitude')
        plt.ylabel('Metric Value')
        plt.title(f'Metric Value of Mode #{n}')
        plt.legend()
        plt.grid(True)
        plt.show()
        fd_n = os.path.join(data_folder, f"metric_value_mode_#{n}_plot.png")
        plt.savefig(fd_n, dpi=600, bb_ox_inches='tight')
