import numpy as np
from math import factorial
from functools import lru_cache


# --- Noll ordering (computed once at import) ---

def _build_noll_table(max_order):
    """Build Noll-ordered (n, m) lookup table."""
    entries = []
    for n in range(max_order):
        for m in range(-n, n + 1, 2):
            entries.append((n, m))
    entries.sort(key=lambda t: (t[0], abs(t[1]), t[1]))
    nm_array = np.array(entries, dtype=np.int32)
    nm_dict = {(n, m): j for j, (n, m) in enumerate(entries)}
    return nm_array, nm_dict


NOLL_NM, NOLL_INDEX = _build_noll_table(15)


# --- Cached radial polynomial coefficients ---

@lru_cache(maxsize=128)
def _radial_coeffs(m, n):
    """Pre-compute radial polynomial coefficients for given (|m|, n).
    Returns list of (coeff, power) tuples for non-zero terms."""
    m_abs = abs(m)
    half_diff = (n - m_abs) // 2
    terms = []
    for s in range(half_diff + 1):
        coeff = ((-1) ** s * factorial(n - s)
                 / (factorial(s) * factorial((n + m_abs) // 2 - s) * factorial(half_diff - s)))
        power = n - 2 * s
        if coeff != 0:
            terms.append((coeff, power))
    return tuple(terms)


# --- Core grid/coordinate helpers ---

class _GridCache:
    """Lazily computed, cached coordinate grids for a given (nx, rad, center)."""

    __slots__ = ('_key', '_rho', '_theta', '_mask')

    def __init__(self):
        self._key = None
        self._rho = None
        self._theta = None
        self._mask = None

    def get(self, rad, cx, cy, nx):
        key = (rad, cx, cy, nx)
        if self._key != key:
            x = np.arange(nx, dtype=np.float64) - cx
            y = np.arange(nx, dtype=np.float64) - cy
            # Use broadcasting instead of full meshgrid
            x2 = x[np.newaxis, :] ** 2
            y2 = y[:, np.newaxis] ** 2
            rho_abs = np.sqrt(x2 + y2)
            self._rho = rho_abs / rad
            self._theta = np.arctan2(y[:, np.newaxis], x[np.newaxis, :])
            self._mask = (self._rho <= 1.0)
            self._key = key
        return self._rho, self._theta, self._mask


_cache = _GridCache()


# --- Radial polynomial (vectorized, no per-pixel loop) ---

def _radial_poly(m_abs, n, rho, mask):
    """Evaluate radial polynomial R_n^|m| on rho, masked to unit disc."""
    coeffs = _radial_coeffs(m_abs, n)
    result = np.zeros_like(rho)
    # Evaluate polynomial using Horner-like accumulation
    # (sorted by descending power for numerical stability)
    for coeff, power in sorted(coeffs, key=lambda t: -t[1]):
        if power == 0:
            result += coeff
        else:
            result += coeff * np.power(rho, power)
    result *= mask
    return result


# --- Public API ---

def Z(m, n, rad=None, orig=None, nx=256):
    """Compute Zernike polynomial Z_n^m on an nx×nx grid.

    Parameters
    ----------
    m : int — azimuthal frequency
    n : int — radial order (n >= |m|, n - |m| even)
    rad : float — pupil radius in pixels (default nx/2)
    orig : (x, y) — center offset from grid center (default None)
    nx : int — grid size
    """
    if abs(m) > n:
        raise ValueError('|m| must be <= n')
    if (n - abs(m)) % 2 != 0:
        raise ValueError('n - |m| must be even')

    if rad is None:
        rad = nx / 2
    cx = cy = nx / 2 - 0.5

    rho, theta, mask = _cache.get(rad, cx, cy, nx)
    R = _radial_poly(abs(m), n, rho, mask)

    if m == 0:
        Z_out = np.sqrt(n + 1.0) * R
    elif m > 0:
        Z_out = np.sqrt(2.0 * (n + 1)) * R * np.cos(m * theta)
    else:
        Z_out = np.sqrt(2.0 * (n + 1)) * R * np.sin(abs(m) * theta)

    if orig is not None:
        shift = (int(orig[0] - nx / 2), int(orig[1] - nx / 2))
        Z_out = np.roll(Z_out, shift, axis=(0, 1))

    return Z_out


def Zm(j, rad=None, orig=None, nx=256):
    """Zernike polynomial by single Noll index j."""
    n, m = NOLL_NM[j]
    return Z(m, n, rad, orig, nx)


def zernike_sum(coeffs, rad=None, orig=None, nx=256):
    """Efficiently compute weighted sum of Zernike polynomials.

    Parameters
    ----------
    coeffs : array-like — Zernike coefficients indexed by Noll order
    rad, orig, nx : same as Z()

    Returns
    -------
    phase : (nx, nx) array
    """
    if rad is None:
        rad = nx / 2
    cx = cy = nx / 2 - 0.5

    rho, theta, mask = _cache.get(rad, cx, cy, nx)
    phase = np.zeros((nx, nx), dtype=np.float64)

    for j, c in enumerate(coeffs):
        if c == 0:
            continue
        n, m = NOLL_NM[j]
        R = _radial_poly(abs(m), n, rho, mask)

        if m == 0:
            phase += c * np.sqrt(n + 1.0) * R
        elif m > 0:
            phase += c * np.sqrt(2.0 * (n + 1)) * R * np.cos(m * theta)
        else:
            phase += c * np.sqrt(2.0 * (n + 1)) * R * np.sin(abs(m) * theta)

    if orig is not None:
        shift = (int(orig[0] - nx / 2), int(orig[1] - nx / 2))
        phase = np.roll(phase, shift, axis=(0, 1))

    return phase
