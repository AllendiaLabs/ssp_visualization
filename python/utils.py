import numpy as np

def custom_fft(x):
    """
    Custom implementation of 1D FFT using the Cooley-Tukey algorithm for powers of 2,
    and direct DFT computation for other lengths.
    
    Args:
        x: Input array (1D)
    
    Returns:
        Complex array representing the FFT of x
    """
    x = np.asarray(x, dtype=complex)
    N = len(x)
    
    if N <= 1:
        return x
    
    # If not a power of 2, use direct DFT computation
    if N & (N - 1) != 0:
        # Direct DFT computation for non-power-of-2 lengths
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.exp(-2j * np.pi * k * n / N)
        return np.dot(M, x)
    
    # Cooley-Tukey FFT for power-of-2 lengths
    # Divide
    even = custom_fft(x[0::2])
    odd = custom_fft(x[1::2])
    
    # Conquer
    T = np.exp(-2j * np.pi * np.arange(N // 2) / N)
    return np.concatenate([even + T * odd, even - T * odd])

def custom_conj(x):
    """
    Custom implementation of complex conjugate.
    
    Args:
        x: Input array (complex)
    
    Returns:
        Complex conjugate of x (real part unchanged, imaginary part negated)
    """
    x = np.asarray(x, dtype=complex)
    return x.real - 1j * x.imag

def custom_ifft(x):
    """
    Custom implementation of 1D inverse FFT.
    
    Uses the relationship: IFFT(x) = conj(FFT(conj(x))) / N
    
    Args:
        x: Input array (1D complex)
    
    Returns:
        Complex array representing the inverse FFT of x
    """
    x = np.asarray(x, dtype=complex)
    N = len(x)
    
    # Use the relationship: IFFT(x) = conj(FFT(conj(x))) / N
    return custom_conj(custom_fft(custom_conj(x))) / N

USE_CUSTOM_FFT = True
if USE_CUSTOM_FFT:
    fft = custom_fft
    ifft = custom_ifft
else:
    fft = np.fft.fft
    ifft = np.fft.ifft

def make_good_unitary(D, eps=1e-3, rng=np.random):
    a = rng.rand((D - 1) // 2)
    sign = rng.choice((-1, +1), len(a))
    phi = sign * np.pi * (eps + a * (1 - 2 * eps))
    assert np.all(np.abs(phi) >= np.pi * eps)
    assert np.all(np.abs(phi) <= np.pi * (1 - eps))

    fv = np.zeros(D, dtype='complex64')
    fv[0] = 1
    fv[1:(D + 1) // 2] = np.cos(phi) + 1j * np.sin(phi)
    fv[-1:D // 2:-1] = np.conj(fv[1:(D + 1) // 2])
    if D % 2 == 0:
        fv[D // 2] = 1

    assert np.allclose(np.abs(fv), 1)
    v = ifft(fv)
    # assert np.allclose(v.imag, 0, atol=1e-5)
    v = v.real
    assert np.allclose(fft(v), fv)
    assert np.allclose(np.linalg.norm(v), 1)
    return v

def power_ssp(ssp, exponent):
    """
    Raises the SSP to a given power/exponent in the Fourier domain.
    """
    ssp_fft = fft(ssp)
    ssp_pow = ifft(ssp_fft ** exponent).real
    return ssp_pow