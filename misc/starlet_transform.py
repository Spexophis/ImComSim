import numpy as np
from scipy.signal import convolve2d
import tifffile as tf


def b_spline_filter(order):
    """Generates B-spline coefficients for a given order."""
    if order == 1:
        # The B1-spline is a simple box filter
        return np.array([1, 1])
    else:
        # Convolve the B-spline of the previous order with the box filter to get the next order
        lower_order_spline = b_spline_filter(order - 1)
        box_filter = np.array([1, 1])
        return np.convolve(lower_order_spline, box_filter, mode='full')


def generate_bsplines_wavelets(order):
    b4_spline_coefficients = b_spline_filter(order)
    b4_spline_coefficients_normalized = b4_spline_coefficients / np.sum(b4_spline_coefficients)
    b4_spline_2d = np.outer(b4_spline_coefficients_normalized, b4_spline_coefficients_normalized)
    return b4_spline_2d


def upsample_filter(h):
    """Upsample a filter by inserting zeros between coefficients."""
    filter_size = len(h)
    # Create an empty array of zeros with spaces for upsampling
    upsampled_filter = np.zeros((2 * filter_size - 1, 2 * filter_size - 1))
    upsampled_filter[::2, ::2] = h
    return upsampled_filter


def starlet_transform(image, order, scales):
    """Apply the starlet (à trous) wavelet transform to a 2D image."""
    wavelet_coeffs = []
    # h = generate_bsplines_wavelets(order)
    # Initialize the filter coefficients for a B3-spline wavelet (a common choice)
    h = np.array([[1, 4, 6, 4, 1],
                  [4, 16, 24, 16, 4],
                  [6, 24, 36, 24, 6],
                  [4, 16, 24, 16, 4],
                  [1, 4, 6, 4, 1]]) / 256
    current_image = image
    for s in range(scales):
        if s > 0:
            # Upsample filter for next scale
            h = upsample_filter(h)
        # Convolve the image with the filter
        smoothed_image = convolve2d(current_image, h, mode='same', boundary='wrap')
        # Subtract the smoothed image from the original to get the wavelet plane
        wavelet_plane = current_image - smoothed_image
        wavelet_coeffs.append(wavelet_plane)
        current_image = smoothed_image

    return wavelet_coeffs, current_image


def soft_threshold(coefficients, threshold):
    """Apply soft thresholding to the wavelet coefficients."""
    return np.sign(coefficients) * np.maximum(np.abs(coefficients) - threshold, 0)


def denoise_image(image, wavelet_coeffs, sigma_noise):
    """Denoise the image using the wavelet coefficients."""
    # Estimate the universal threshold
    sz = image.size
    threshold = sigma_noise * np.sqrt(2 * np.log(sz))

    # Apply soft thresholding to each set of wavelet coefficients
    denoised_coeffs = [soft_threshold(c, threshold) for c in wavelet_coeffs]

    # Reconstruct the denoised image from the thresholded wavelet coefficients
    denoised_image = image.copy()  # Start with the residual image
    for c in denoised_coeffs:
        denoised_image += c

    return denoised_image


if __name__ == '__main__':
    data = tf.imread(r"C:\Users\ruizhe.lin\Desktop\vim_24pix_12phs-1.tif")
    scales = 4
    nz, ny, nx = data.shape
    coefficients = np.zeros((nz, ny, nx))
    # residuals = np.zeros((nz, ny, nx))
    # denoised_image = np.zeros((nz, ny, nx))
    for i in range(nz):
        coefficient, residual = starlet_transform(data[i], order=3, scales=scales)
        for j in range(scales):
            coefficients[i] += coefficient[j]
        # denoised_image[i] = denoise_image(residual, coefficient, 128.)
    tf.imwrite(r"C:\Users\ruizhe.lin\Desktop\vim_24pix_12phs-1_startlet_transform_coefficients.tif", coefficients.astype(np.float32))
    # tf.imwrite(r"C:\Users\ruizhe.lin\Desktop\startlet_transform_residual.tif", residuals)
    # tf.imwrite(r"C:\Users\ruizhe.lin\Desktop\startlet_transform_denoised_image.tif", denoised_image)