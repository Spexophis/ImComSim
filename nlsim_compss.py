from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import *
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf
from skimage.filters import window
import pandas as pd


# @task(returns=1)
# def _fftshift(_x):
#     return np.fft.fftshift(_x)


# # @task(returns=1)
# def _fft2(_x):
#     return np.fft.fft2(_x)


def load_data(_fn, _nang, _nph):
    _img = tf.imread(_fn)
    _nz, _ny, _nx = _img.shape
    _img = _img.reshape(_nang, _nph, _ny, _nx)
    return _img, _nx, _ny


@task(returns=2)
def generate_coords(_nx, _ny):
    _xv, _yv = np.meshgrid(np.arange(-_nx / 2, _nx / 2), np.arange(-_ny / 2, _ny / 2), indexing='ij', sparse=True)
    _xv = np.roll(_xv, int(_nx / 2))
    _yv = np.roll(_yv, int(_ny / 2))
    return _xv, _yv


def disc_array(_shape=(128, 128), _radius=64):
    _ny, _nx = _shape
    _ox = _nx / 2
    _oy = _ny / 2
    _x = np.linspace(-_ox, _ox - 1, _nx)
    _y = np.linspace(-_oy, _oy - 1, _ny)
    _yv, _xv = np.meshgrid(_y, _x, indexing='ij')
    _rho = np.sqrt(_xv ** 2 + _yv ** 2)
    return (_rho < _radius) * 1.0


@task(returns=2)
def get_psf(_nx, _dx, _na, _wl):
    _dp = 1 / (_nx * _dx)
    _rad = (_na / _wl) / _dp
    _bpp = disc_array(_shape=(_nx, _nx), _radius=_rad)
    _psf = np.abs((np.fft.fft2(_bpp))) ** 2
    _psf = np.fft.fftshift(_psf)
    return _psf / _psf.sum(), _rad


@task(returns=1)
def separate_matrix(_norders, _nph):
    _orders = int((_norders + 1) / 2)
    _sep_mat = np.zeros((_norders, _nph), dtype=np.float32)
    for j in range(_nph):
        _sep_mat[0, j] = 1.0
        for _order in range(1, _orders):
            _sep_mat[2 * _order - 1, j] = 2 * np.cos(2 * np.pi * (j * _order) / _nph) / _nph
            _sep_mat[2 * _order, j] = 2 * np.sin(2 * np.pi * (j * _order) / _nph) / _nph
    return np.linalg.inv(np.transpose(_sep_mat))


@task(returns=1)
def window_function(_alpha, _nx):
    _w = window(('tukey', _alpha), _nx)
    _wx = np.tile(_w, (_nx, 1))
    _wy = _wx.swapaxes(0, 1)
    return _wx * _wy


def interp(arr, _ratio, _w):
    _nx, _ny = arr.shape
    _px = int((_nx * (_ratio - 1)) / 2)
    _py = int((_ny * (_ratio - 1)) / 2)
    _arr = np.fft.fftshift(np.fft.fft2(arr))
    _arr = np.pad(_arr, ((_px, _px), (_py, _py)), 'constant', constant_values=(0))
    return np.fft.ifft2(np.fft.fftshift(_arr)) * _w


@task(returns=1)
def separate(n, _img, _sep_mat, _norders, _ratio, _wd):
    _nang, _nph, _npx, _npy = _img.shape
    _temp = np.dot(_sep_mat, _img[n].reshape(_nph, _npx * _npy))
    _out = []
    _out.append(np.fft.fftshift(interp(_temp[0].reshape(_npx, _npy), _ratio, _wd)))
    for i in range(int(_norders / 2)):
        _out.append(np.fft.fftshift(interp(((_temp[1 + 2 * i] + 1j * _temp[2 + 2 * i]) / 2).reshape(_npx, _npy), _ratio, _wd)))
        _out.append(np.fft.fftshift(interp(((_temp[1 + 2 * i] - 1j * _temp[2 + 2 * i]) / 2).reshape(_npx, _npy), _ratio, _wd)))
    return _out


def separate_orders(_img, _psf, _nang, _norders, _sep_mat, _ratio, _wd):
    _cps = []
    _otfs_0 = []
    _imgfs_0 = []
    for n in range(_nang):
        s = separate(n, _img, _sep_mat, _norders, _ratio, _wd)
        _otf, _imgf = otf_zero_order(s, _psf)
        _otfs_0.append(_otf)
        _imgfs_0.append(_imgf)
        _cps.append(s)
    return _cps, _otfs_0, _imgfs_0


def zero_suppression(_strength, _sigma, _xv, _yv, _sx, _sy):
    return 1 - _strength * np.exp(-((_xv - _sx) ** 2. + (_yv - _sy) ** 2.) / (2. * _sigma ** 2.))


@task(returns=2)
def otf_zero_order(_s, _psf):
    return np.fft.fft2(_psf), np.fft.fft2(_s[0])


def get_shift_v(_dx, _xv, _yv, shift_orientation, shift_spacing):
    _kx = _dx * np.cos(shift_orientation) / shift_spacing
    _ky = _dx * np.sin(shift_orientation) / shift_spacing
    return np.exp(2j * np.pi * (_kx * _xv + _ky * _yv))


def get_origin(_ysh, _nx):
    _ysh = np.abs(np.fft.fft2(_ysh))
    _sx, _sy = np.unravel_index(_ysh.argmax(), _ysh.shape)
    _sx = _sx if _sx < _nx / 2 else _sx - _nx
    _sy = _sy if _sy < _nx / 2 else _sy - _nx
    return _sx, _sy


@task(returns=2)
def shift_otf_n_imgf(_s, _psf, pattern_orientation, pattern_spacing, frequency_order, _nx, _dx, _xv, _yv, _strength, _sigma):
    """ shift data in freq space by multiplication in real space """
    _otf_sh = []
    _imgf_sh = []
    # shift matrix
    yshp = get_shift_v(_dx, _xv, _yv, pattern_orientation, pattern_spacing)
    sxp, syp = get_origin(yshp, _nx)
    zsp = zero_suppression(_strength, _sigma, _xv, _yv, sxp, syp)
    # shift otf and imf
    _otf_sh.append(np.fft.fft2(_psf * yshp) * zsp)
    _imgf_sh.append(np.fft.fft2(_s[2 * frequency_order - 1] * yshp))
    # shift matrix
    yshn = get_shift_v(_dx, _xv, _yv, pattern_orientation, -pattern_spacing)
    sxn, syn = get_origin(yshn, _nx)
    zsn = zero_suppression(_strength, _sigma, _xv, _yv, sxn, syn)
    # shift otf and imf
    _otf_sh.append(np.fft.fft2(_psf * yshn) * zsn)
    _imgf_sh.append(np.fft.fft2(_s[2 * frequency_order] * yshn))
    return _otf_sh, _imgf_sh


def shift_otfs_n_imgfs(_s, _psf, pattern_orientations, pattern_spacings, frequency_order, _nx, _dx, _xv, _yv, _strength, _sigma):
    """ shift data in freq space by multiplication in real space """
    _otfs_sh = []
    _imgfs_sh = []
    for i in range(len(_s)):
        _otf_sh, _imgf_sh = shift_otf_n_imgf(_s[i], _psf, pattern_orientations[i], pattern_spacings[i], frequency_order, _nx, _dx, _xv, _yv, _strength, _sigma)
        _otfs_sh.append(_otf_sh)
        _imgfs_sh.append(_imgf_sh)
    return _otfs_sh, _imgfs_sh


def get_search(_angs, _sps, _steps, _ang_range, _sp_range):
    _r_ang = np.linspace(-_ang_range, _ang_range, _steps + 1)
    _r_sp = np.linspace(-_sp_range, _sp_range, _steps + 1)
    _ang_iters = np.add.outer(_angs, _r_ang)
    _sp_iters = np.add.outer(_sps, _r_sp)
    return _ang_iters, _sp_iters


def calculate_overlap(_otf_s, _otf_0, _imf_s, _imf_0, _cutoff):
    """ calculate overlapping area """
    _w_0 = _otf_s * _imf_0
    _w_1 = _otf_0 * _imf_s
    _msk = abs(_otf_0 * _otf_s) > _cutoff
    return np.sum(_msk * _w_1 * _w_0.conj()) / np.sum(_msk * _w_0 * _w_0.conj())


def get_overlap_w_zero(_s, _psf, otf_0, imf_0, shift_orientation, shift_spacing, order_to_be_computed, _dx, _xv, _yv, _cutoff):
    """ shift data in freq space by multiplication in real space """
    ysh = get_shift_v(_dx, _xv, _yv, shift_orientation, shift_spacing)
    otf_s = np.fft.fft2(_psf * ysh)
    imf_s = np.fft.fft2(_s[2 * order_to_be_computed - 1] * ysh)
    a = calculate_overlap(otf_s, otf_0, imf_s, imf_0, _cutoff)
    # _w_0 = otf_s * imf_0
    # _w_1 = otf_0 * imf_s
    # _msk = abs(otf_0 * otf_s) > _cutoff
    # a = np.sum(_msk * _w_1 * _w_0.conj()) / np.sum(_msk * _w_0 * _w_0.conj())
    return np.abs(a), np.angle(a)


@task(returns=2)
def map_overlap_w_zero_(_cp, od, _otf_0, _imgf_0, _ang_iter, _sp_iter, _psf, _dx, _xv, _yv, _cutoff):
    _mag_arr = []
    _ph_arr = []
    for m, ang in enumerate(_ang_iter):
        _temp_m = []
        _temp_p = []
        for n, sp in enumerate(_sp_iter):
            _mag, _ph = get_overlap_w_zero(_cp, _psf, _otf_0, _imgf_0, ang, sp, od, _dx, _xv, _yv, _cutoff)
            _temp_m.append(_mag)
            _temp_p.append(_ph)
        _mag_arr.append(_temp_m)
        _ph_arr.append(_temp_p)
    return _mag_arr, _ph_arr


def map_overlap_w_zero(_cps, od, _otfs_0, _imgfs_0, _ang_iters, _sp_iters, _psf, _dx, _xv, _yv, _cutoff):
    _mag_arrs = []
    _ph_arrs = []
    for i in range(len(_cps)):
        _mag_arr, _ph_arr = map_overlap_w_zero_(_cps[i], od, _otfs_0[i], _imgfs_0[i], _ang_iters[i], _sp_iters[i], _psf, _dx, _xv, _yv, _cutoff)
        _mag_arrs.append(_mag_arr)
        _ph_arrs.append(_ph_arr)
    return _mag_arrs, _ph_arrs


def get_overlap(_s, _psf, otf_0, imf_0, shift_orientation, shift_spacing, order_to_be_computed, _dx, _xv, _yv, _cutoff):
    """ shift data in freq space by multiplication in real space """
    ysh = get_shift_v(_dx, _xv, _yv, shift_orientation, shift_spacing)
    otf_s = np.fft.fft2(_psf * ysh)
    imf_s = np.fft.fft2(_s[2 * order_to_be_computed - 1] * ysh)
    a = calculate_overlap(otf_s, otf_0, imf_s, imf_0, _cutoff)
    # _w_0 = otf_s * imf_0
    # _w_1 = otf_0 * imf_s
    # _msk = abs(otf_0 * otf_s) > _cutoff
    # a = np.sum(_msk * _w_1 * _w_0.conj()) / np.sum(_msk * _w_0 * _w_0.conj())
    return np.abs(a), np.angle(a)


@task(returns=2)
def map_overlap_(_cp, od, _otf_0, _imgf_0, _ang_iter, _sp_iter, _psf, _dx, _xv, _yv, _cutoff):
    _mag_arr = []
    _ph_arr = []
    for m, ang in enumerate(_ang_iter):
        _temp_m = []
        _temp_p = []
        for n, sp in enumerate(_sp_iter):
            _mag, _ph = get_overlap(_cp, _psf, _otf_0, _imgf_0, ang, sp, od, _dx, _xv, _yv, _cutoff)
            _temp_m.append(_mag)
            _temp_p.append(_ph)
        _mag_arr.append(_temp_m)
        _ph_arr.append(_temp_p)
    return _mag_arr, _ph_arr


def map_overlap(_cps, od, _otfs_0, _imgfs_0, _ang_iters, _sp_iters, _psf, _dx, _xv, _yv, _cutoff):
    _mag_arrs = []
    _ph_arrs = []
    for i in range(len(_cps)):
        _mag_arr, _ph_arr = map_overlap_(_cps[i], od, _otfs_0[i][0], _imgfs_0[i][0], _ang_iters[i], _sp_iters[i], _psf, _dx, _xv, _yv, _cutoff)
        _mag_arrs.append(_mag_arr)
        _ph_arrs.append(_ph_arr)
    return _mag_arrs, _ph_arrs 


def get_parameters(_mag_arrs, _ph_arrs, _ang_iters, _sp_iters):
    _angles, _spacings, _magnitudes, _phases = [], [], [], []
    for i in range(len(_mag_arrs)):
        _angle, _spacing, _magnitude, _phase = get_maximum(_mag_arrs[i], _ph_arrs[i], _ang_iters[i], _sp_iters[i])
        _angles.append(_angle)
        _spacings.append(_spacing)
        _magnitudes.append(_magnitude)
        _phases.append(_phase)
    return _angles, _spacings, _magnitudes, _phases


def get_maximum(_mag_arr, _ph_arr, _ang_iter, _sp_iter):
    _m = np.array(_mag_arr)
    _k, _l = np.unravel_index(np.argmax(_m, axis=None), _m.shape)
    return _ang_iter[_k], _sp_iter[_l], _mag_arr[_k][_l], _ph_arr[_k][_l]


@task(returns=1)
def apod(eta, nx, ny):
    return np.fft.fftshift(window(('kaiser', eta), (ny, nx)))


@task(returns=2)
def reconstruct_zero_order(_otfs, _imgfs, _nx, _ny):
    _numera = np.zeros((_nx, _ny), dtype=np.complex64)
    _denomi = np.zeros((_nx, _ny), dtype=np.complex64)
    for i in range(len(_otfs)):
        _numera += np.conj(_otfs[i]) * _imgfs[i]
        _denomi += abs(_otfs[i]) ** 2
    return _numera, _denomi


@task(returns=2)
def reconstruct_high_order(_otfs, _imgfs, _magnitudes, _phases, _nx, _ny):
    _numera = np.zeros((_nx, _ny), dtype=np.complex64)
    _denomi = np.zeros((_nx, _ny), dtype=np.complex64)
    for i in range(len(_magnitudes)):
        ph = _magnitudes[i] * np.exp(-1j * _phases[i])
        _numera += ph * np.conj(_otfs[i][0]) * _imgfs[i][0]
        _denomi += abs(_otfs[i][0]) ** 2
        _numera += np.conj(ph) * np.conj(_otfs[i][1]) * _imgfs[i][1]
        _denomi += abs(_otfs[i][1]) ** 2
    return _numera, _denomi


# def reconstruct_all(_apd, _mu, _nangs, _nords, _otfs, _imgfs, _magnitudes, _phases, _nx, _ny, zero_order=True):
#     numera = np.zeros((_nx, _ny), dtype=np.complex64)
#     denomi = np.zeros((_nx, _ny), dtype=np.complex64)
#     denomi += _mu ** 2
#     for i in range(_nangs):
#         if zero_order:
#             numera += np.conj(_otfs[i][0]) * _imgfs[i][0]
#             denomi += abs(_otfs[i][0]) ** 2
#         for od in range(_nords):
#             ph = _magnitudes[i][od + 1] * np.exp(-1j * _phases[i][od + 1])
#             numera += ph * np.conj(_otfs[i][od + 1]["positive"]) * _imgfs[i][od + 1]["positive"]
#             denomi += abs(_otfs[i][od + 1]["positive"]) ** 2
#             numera += np.conj(ph) * np.conj(_otfs[i][od + 1]["negative"]) * _imgfs[i][od + 1]["negative"]
#             denomi += abs(_otfs[i][od + 1]["negative"]) ** 2
#     temp = _apd * numera / denomi
#     imgrecon = np.fft.ifft2(temp).real.astype(np.float32)
#     imgfrecon = np.abs(np.fft.fftshift(temp)).astype(np.float32)
#     return imgrecon, imgfrecon


def save_result(_img, _imgf, _angles, _spacings, _magnitudes, _phases):
    tf.imwrite('nlsim2d_final_image.tif', _img)
    tf.imwrite('nlsim2d_effective_otf.tif', _imgf)
    df_angles = pd.DataFrame(_angles).T.sort_index()
    df_spacings = pd.DataFrame(_spacings).T.sort_index()
    df_magnitudes = pd.DataFrame(_magnitudes).T.sort_index()
    df_phases = pd.DataFrame(_phases).T.sort_index()
    with pd.ExcelWriter('nlsim2d_parameters.xlsx', engine='openpyxl') as writer:
        df_angles.to_excel(writer, sheet_name='Angles')
        df_spacings.to_excel(writer, sheet_name='Spacings')
        df_magnitudes.to_excel(writer, sheet_name='Magnitudes')
        df_phases.to_excel(writer, sheet_name='Phases')


if __name__ == "__main__":
    import time
    start_time = time.time()
    # imaging paramters
    na = 1.4
    wl = 0.5
    ps = 0.075
    nph = 7
    nang = 7
    ang_offset = 0
    angs = [ang_offset + 2 * i * np.pi / nang for i in range(nang)]
    sps = [0.24] * nang
    norders = 7
    tords = 2
    ords = 2 * tords + 1
    resolution_target = (wl / (2 * na)) / tords
    # load data
    fn = r"/home/ruizhe/codes/nlsim2d_simulation_data.tif"
    img, npx, npy = load_data(fn, nang, nph)
    ratio = int(ps / (resolution_target / 2)) + 1
    dx = ps / ratio
    nx = npx * ratio
    ny = npy * ratio
    eta = 2
    apd = apod(eta, nx, ny)
    # image space
    xv, yv = generate_coords(nx, ny)
    # generate point spread function
    psf, radius = get_psf(nx, dx, na, wl)
    # optimization paramters
    strength = 1.
    sigma = 4.
    alpha = 0.04
    cutoff = 0.01
    wd = window_function(alpha, nx)
    sep_mat = separate_matrix(norders, nph)
    cps, otfs_0, imgfs_0 = separate_orders(img, psf, nang, norders, sep_mat, ratio, wd)
    cps, otfs_0, imgfs_0 = compss_wait_on(cps, otfs_0, imgfs_0)
    # searching
    ang_iters, sp_iters = get_search(angs, sps, 10, 0.005, 0.005)
    mag_arrs, ph_arrs = map_overlap_w_zero(cps, 1, otfs_0, imgfs_0, ang_iters, sp_iters, psf, dx, xv, yv, cutoff)
    mag_arrs, ph_arrs = compss_wait_on(mag_arrs, ph_arrs)
    angles_1, spacings_1, magnitudes_1, phases_1 = get_parameters(mag_arrs, ph_arrs, ang_iters, sp_iters)
    # ang_iters, sp_iters = get_search(angles_1, spacings_1, 10, 0.005, 0.005)
    # mag_arrs, ph_arrs = map_overlap_w_zero(cps, 1, otf_0, imgf_0, ang_iters, sp_iters, psf, dx, xv, yv, cutoff)
    # mag_arrs, ph_arrs = compss_wait_on(mag_arrs, ph_arrs)
    # angles_1, spacings_1, magnitudes_1, phases_1 = get_parameters(mag_arrs, ph_arrs, ang_iters, sp_iters)
    otfs_1, imgfs_1 = shift_otfs_n_imgfs(cps, psf, angles_1, spacings_1, 1, nx, dx, xv, yv, strength, sigma)
    otfs_1, imgfs_1 = compss_wait_on(otfs_1, imgfs_1)
    ang_iters, sp_iters = get_search(angles_1, [x / 2 for x in spacings_1], 10, 0.005, 0.005)
    mag_arrs, ph_arrs = map_overlap(cps, 2, otfs_1, imgfs_1, ang_iters, sp_iters, psf, dx, xv, yv, cutoff)
    mag_arrs, ph_arrs = compss_wait_on(mag_arrs, ph_arrs)
    angles_2, spacings_2, magnitudes_2, phases_2 = get_parameters(mag_arrs, ph_arrs, ang_iters, sp_iters)
    otfs_2, imgfs_2 = shift_otfs_n_imgfs(cps, psf, angles_2, spacings_2, 2, nx, dx, xv, yv, strength, sigma)
    otfs_2, imgfs_2 = compss_wait_on(otfs_2, imgfs_2)
    mu = 0.08
    numera = np.zeros((nx, ny), dtype=np.complex64)
    denomi = np.zeros((nx, ny), dtype=np.complex64)
    denomi += mu ** 2
    _numera_0, _denomi_0 = reconstruct_zero_order(otfs_0, imgfs_0, nx, ny)
    _numera_1, _denomi_1 = reconstruct_high_order(otfs_1, imgfs_1, magnitudes_1, phases_1, nx, ny)
    _numera_2, _denomi_2 = reconstruct_high_order(otfs_2, imgfs_2, magnitudes_2, phases_2, nx, ny) 
    (apd, _numera_0, _numera_1, _numera_2, _denomi_0,  _denomi_1, _denomi_2) = compss_wait_on(apd, _numera_0, _numera_1, _numera_2, _denomi_0,  _denomi_1, _denomi_2)
    temp = apd * (numera + _numera_0 + _numera_1 + _numera_2) / (denomi + _denomi_0 + _denomi_1 + _denomi_2)
    imgrecon = np.fft.ifft2(temp).real.astype(np.float32)
    imgfrecon = np.abs(np.fft.fftshift(temp)).astype(np.float32)
    end_time = time.time()
    execution_time = end_time - start_time
    with open('execution_time.txt', 'w') as file:
        file.write(str(execution_time))
    # imgrecon, imgfrecon = compss_wait_on(imgrecon, imgfrecon)
    tf.imwrite('nlsim2d_final_image.tif', imgrecon)
    tf.imwrite('nlsim2d_effective_otf.tif', imgfrecon)
