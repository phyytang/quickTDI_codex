"""
Noise generation module for TDI simulations. TDI

This module provides functions to generate various types of noise with different
power spectral density (PSD) characteristics for use in gravitational wave
detector simulations. (PSD)


Notes 
-----
The global parameter alpha_para controls the IIR filter coefficient for 1/f^n
noise generation. Current value (0.9999) may cause convergence issues for very
long time series. alpha_para1/f^nIIR(0.9999)

"""

import numpy as np
from numpy.typing import NDArray

# IIR filter coefficient for power law noise generation IIR
# WARNING: Convergence depends on this value being close to but less than 1.0 1.0
alpha_para = 0.9999


def white(length: int,
          mean: float = 0.0,
          sigma: float = 1.0,
          fsample: float = 1.0) -> NDArray[np.float64]:
    """
    Generate white Gaussian noise with flat power spectral density. 

    Parameters
    ----------
    length : int
        Number of samples to generate.
    mean : float, optional
        Mean value of the noise. Default is 0.0.
    sigma : float, optional
        Standard deviation in phase units. Default is 1.0.
    fsample : float, optional
        Sampling frequency in Hz. Default is 1.0.

    Returns
    -------
    np.ndarray
        White noise time series of shape (length,).

    Mathematical Formula
    --------------------
    The output x[n] is drawn from a normal distribution:

        x[n] ~ N(μ, σ²·f_s/2)

    where:
        μ = mean
        σ = sigma (standard deviation)
        f_s = fsample (sampling frequency)

    Power Spectral Density (PSD):
        S(f) = σ² (constant, frequency-independent)

    Notes
    -----
    The noise is scaled by sqrt(fsample/2) to maintain consistent power
    spectral density regardless of sampling rate. For a given sigma, the
    PSD remains constant when fsample changes.

    Examples
    --------
    >>> noise = white(10000, mean=0.0, sigma=1e-13, fsample=10.0)
    >>> len(noise)
    10000
    """
    if length == 0:
        return np.array([])
    else:
        return np.random.normal(mean, sigma * np.sqrt(fsample / 2), length)


def power_2(length: int,
            mean: float = 0.0,
            sigma: float = 1.0,
            fsample: float = 1.0) -> NDArray[np.float64]:
    """
    Generate noise with f^2 power spectral density. f^2

    Applies first-order differencing to white noise to create PSD ~ f^2
    (approximately Sin^2(f) at low frequencies).

    Parameters
    ----------
    length : int
        Number of samples to generate.
    mean : float, optional
        Mean value of the underlying white noise. Default is 0.0.
    sigma : float, optional
        Standard deviation of the underlying white noise. Default is 1.0.
    fsample : float, optional
        Sampling frequency in Hz. Default is 1.0.

    Returns
    -------
    np.ndarray
        Colored noise time series with f^2 PSD, shape (length,).

    Mathematical Formula
    --------------------
    The first-order FIR filter applies discrete differentiation:

        y[n] = x[n] - x[n-1]

    where x[n] is white Gaussian noise.

    Transfer Function (Z-domain):
        H(z) = 1 - z⁻¹

    Frequency Response:
        |H(f)|² = 4·sin²(πf/f_s)

    At low frequencies (f << f_s):
        |H(f)|² ≈ (2πf/f_s)²

    Power Spectral Density:
        S(f) ∝ f²  (for f << f_s)

    The output is normalized by (f_s/(2π)) to ensure proper scaling.

    Notes
    -----
    Implements the discrete difference: output[i] = data[i] - data[i-1]
    Normalization factor (fsample/(2*pi)) ensures PSD is normalized at f=1Hz.

    Optimized implementation uses numpy's diff function for vectorization.
    """
    if length == 0:
        return np.array([])
    else:
        data = np.random.normal(mean, sigma * np.sqrt(fsample / 2), length)

        # Old slow implementation (list append):
        # data2 = []
        # for i in range(length):
        #     if i == 0:
        #         data2.append(data[0])
        #     else:
        #         data2.append(data[i] - data[i-1])
        # return np.array(data2) * (fsample / (2 * np.pi))

        # New vectorized implementation:
        data2 = np.empty(length, dtype=np.float64)
        data2[0] = data[0]
        data2[1:] = np.diff(data)  # Vectorized difference operation
        return data2 * (fsample / (2 * np.pi))


def power_m2(length: int,
             mean: float = 0.0,
             sigma: float = 1.0,
             fsample: float = 1.0) -> NDArray[np.float64]:
    """
    Generate noise with 1/f^2 power spectral density. 1/f^2

    Uses first-order IIR (infinite impulse response) filter to create
    PSD ~ 1/f^2 (approximately 1/Sin^2(f) at low frequencies).

    Parameters
    ----------
    length : int
        Number of samples to generate.
    mean : float, optional
        Mean value of the underlying white noise. Default is 0.0.
    sigma : float, optional
        Standard deviation of the underlying white noise. Default is 1.0.
    fsample : float, optional
        Sampling frequency in Hz. Default is 1.0.

    Returns
    -------
    np.ndarray
        Colored noise time series with 1/f^2 PSD, shape (length,).

    Mathematical Formula
    --------------------
    The first-order IIR filter implements recursive integration:

        y[n] = α·y[n-1] + x[n]

    where:
        α = alpha_para ≈ 0.9999 (global parameter)
        x[n] = white Gaussian noise

    Transfer Function (Z-domain):
        H(z) = 1/(1 - α·z⁻¹)

    Frequency Response (for α ≈ 1):
        |H(f)|² ≈ 1/(4·sin²(πf/f_s))

    At low frequencies (f << f_s):
        |H(f)|² ≈ (f_s/(2πf))²

    Power Spectral Density:
        S(f) ∝ 1/f²  (for f << f_s)

    The output is normalized by (f_s/(2π)) to ensure proper scaling.

    Notes
    -----
    Implements recursive filter: output[i] = alpha_para * output[i-1] + data[i]
    where alpha_para is a global parameter (currently 0.9999).
    Convergence and stability depend on alpha_para being close to but less than 1.

    Optimized implementation uses pre-allocated numpy array instead of Python list.
    """
    if length == 0:
        return np.array([])
    else:
        data = np.random.normal(mean, sigma * np.sqrt(fsample / 2), length)

        # Old slow implementation (list append):
        # data3 = []
        # for i in range(length):
        #     if i == 0:
        #         data3.append(data[0])
        #     else:
        #         data3.append(alpha_para * data3[i-1] + data[i])
        # return np.array(data3) / (fsample / (2 * np.pi))

        # New optimized implementation (pre-allocated array):
        data3 = np.empty(length, dtype=np.float64)
        data3[0] = data[0]
        for i in range(1, length):
            data3[i] = alpha_para * data3[i-1] + data[i]
        return data3 / (fsample / (2 * np.pi))


def power_4(length: int,
            mean: float = 0.0,
            sigma: float = 1.0,
            fsample: float = 1.0) -> NDArray[np.float64]:
    """
    Generate noise with f^4 power spectral density.

    Applies second-order differencing to white noise to create PSD ~ f^4
    (approximately Sin^4(f) at low frequencies).

    Parameters
    ----------
    length : int
        Number of samples to generate.
    mean : float, optional
        Mean value of the underlying white noise. Default is 0.0.
    sigma : float, optional
        Standard deviation of the underlying white noise. Default is 1.0.
    fsample : float, optional
        Sampling frequency in Hz. Default is 1.0.

    Returns
    -------
    np.ndarray
        Colored noise time series with f^4 PSD, shape (length,).

    Mathematical Formula
    --------------------
    The second-order FIR filter applies discrete second derivative:

        y[n] = x[n] - 2·x[n-1] + x[n-2]

    where x[n] is white Gaussian noise.

    Transfer Function (Z-domain):
        H(z) = (1 - z⁻¹)²

    Frequency Response:
        |H(f)|² = 16·sin⁴(πf/f_s)

    At low frequencies (f << f_s):
        |H(f)|² ≈ (2πf/f_s)⁴

    Power Spectral Density:
        S(f) ∝ f⁴  (for f << f_s)

    The output is normalized by (f_s/(2π))² to ensure proper scaling.

    Notes
    -----
    Implements second difference: output[i] = data[i] - 2*data[i-1] + data[i-2]
    Normalization factor (fsample/(2*pi))^2 ensures proper scaling.

    Optimized implementation uses vectorized array operations.
    """
    if length == 0:
        return np.array([])
    else:
        data = np.random.normal(mean, sigma * np.sqrt(fsample / 2), length)

        # Old slow implementation (list append):
        # data4 = []
        # for i in range(length):
        #     if i == 0:
        #         data4.append(data[0])
        #     elif i == 1:
        #         data4.append(data[1] - 2 * data[0])
        #     else:
        #         data4.append(data[i] - 2 * data[i-1] + data[i-2])
        # return np.array(data4) * (fsample / (2 * np.pi))**2

        # New vectorized implementation:
        data4 = np.empty(length, dtype=np.float64)
        data4[0] = data[0]
        data4[1] = data[1] - 2 * data[0]
        data4[2:] = data[2:] - 2 * data[1:-1] + data[:-2]  # Vectorized second difference
        return data4 * (fsample / (2 * np.pi))**2


def power_m4(length: int,
             mean: float = 0.0,
             sigma: float = 1.0,
             fsample: float = 1.0) -> NDArray[np.float64]:
    """
    Generate noise with 1/f^4 power spectral density (UNSTABLE).

    Uses second-order IIR filter to create PSD ~ 1/f^4.

    Parameters
    ----------
    length : int
        Number of samples to generate.
    mean : float, optional
        Mean value of the underlying white noise. Default is 0.0.
    sigma : float, optional
        Standard deviation of the underlying white noise. Default is 1.0.
    fsample : float, optional
        Sampling frequency in Hz. Default is 1.0.

    Returns
    -------
    np.ndarray
        Colored noise time series with 1/f^4 PSD, shape (length,).

    Mathematical Formula
    --------------------
    The second-order IIR filter implements double integration:

        y[n] = x[n] + 2·y[n-1] - y[n-2]

    where x[n] is white Gaussian noise.

    Transfer Function (Z-domain):
        H(z) = 1/(1 - z⁻¹)²

    Frequency Response (approximate):
        |H(f)|² ≈ 1/(16·sin⁴(πf/f_s))

    At low frequencies (f << f_s):
        |H(f)|² ≈ (f_s/(2πf))⁴

    Power Spectral Density:
        S(f) ∝ 1/f⁴  (for f << f_s)

    The output is normalized by (f_s/(2π))² to ensure proper scaling.

    Warnings
    --------
    This implementation is numerically unstable and may not converge for
    long time series. Consider using power_m4_2() instead, which uses
    cascaded first-order filters for better stability.

    Notes
    -----
    Implements recursive filter: output[i] = data[i] + 2*output[i-1] - output[i-2]
    This second-order recursion can exhibit numerical instability.

    Optimized implementation uses pre-allocated numpy array.
    """
    if length == 0:
        return np.array([])
    else:
        data = np.random.normal(mean, sigma * np.sqrt(fsample / 2), length)

        # Old slow implementation (list append):
        # data5 = []
        # for i in range(length):
        #     if i == 0:
        #         data5.append(data[0])
        #     elif i == 1:
        #         data5.append(data[1] + 2 * data5[0])
        #     else:
        #         data5.append(data[i] + (2 * data5[i-1] - data5[i-2]))
        # return np.array(data5) / ((fsample / (2 * np.pi))**2)

        # New optimized implementation (pre-allocated array):
        data5 = np.empty(length, dtype=np.float64)
        data5[0] = data[0]
        data5[1] = data[1] + 2 * data5[0]
        for i in range(2, length):
            data5[i] = data[i] + (2 * data5[i-1] - data5[i-2])
        return data5 / ((fsample / (2 * np.pi))**2)


def power_m4_2(length: int,
               mean: float = 0.0,
               sigma: float = 1.0,
               fsample: float = 1.0) -> NDArray[np.float64]:
    """
    Generate noise with 1/f^4 power spectral density (stable version).

    Uses two cascaded first-order IIR filters (1/f^2 * 1/f^2) to create
    PSD ~ 1/f^4. More stable than power_m4().

    Parameters
    ----------
    length : int
        Number of samples to generate.
    mean : float, optional
        Mean value of the underlying white noise. Default is 0.0.
    sigma : float, optional
        Standard deviation of the underlying white noise. Default is 1.0.
    fsample : float, optional
        Sampling frequency in Hz. Default is 1.0.

    Returns
    -------
    np.ndarray
        Colored noise time series with 1/f^4 PSD, shape (length,).

    Mathematical Formula
    --------------------
    Cascades two first-order IIR filters to achieve 1/f^4 behavior:

        Stage 1: y₁[n] = α·y₁[n-1] + x[n]
        Stage 2: y[n] = α·y[n-1] + y₁[n]

    where α = alpha_para ≈ 0.9999.

    Combined Transfer Function (Z-domain):
        H(z) = 1/(1 - α·z⁻¹)²

    This is equivalent to power_m4() but numerically more stable.

    Power Spectral Density:
        S(f) ∝ 1/f⁴

    The cascaded approach prevents numerical overflow that can occur
    in direct second-order recursion.

    Notes
    -----
    Cascades two power_m2() filters for better numerical stability compared
    to direct second-order recursion in power_m4().

    Optimized implementation uses pre-allocated numpy array.
    """
    if length == 0:
        return np.array([])
    else:
        data = power_m2(length, mean, sigma, fsample)

        # Old slow implementation (list append):
        # data5 = []
        # for i in range(length):
        #     if i == 0:
        #         data5.append(data[0])
        #     else:
        #         data5.append(alpha_para * data5[i-1] + data[i])
        # return np.array(data5) / (fsample / (2 * np.pi))

        # New optimized implementation (pre-allocated array):
        data5 = np.empty(length, dtype=np.float64)
        data5[0] = data[0]
        for i in range(1, length):
            data5[i] = alpha_para * data5[i-1] + data[i]
        return data5 / (fsample / (2 * np.pi))


def m2_trun(length: int,
            mean: float = 0.0,
            sigma: float = 1.0,
            fsample: float = 1.0,
            lowfz: float = 1e-9,
            highfz: float = 10.0) -> NDArray[np.float64]:
    """
    Generate truncated 1/f^2 noise with flat PSD outside cutoff frequencies.

    Implements the Plaszczynski (2007) algorithm for generating 1/f^alpha noise
    with controlled low and high frequency behavior.

    Parameters
    ----------
    length : int
        Number of samples to generate.
    mean : float, optional
        Mean value of the underlying white noise. Default is 0.0.
    sigma : float, optional
        Standard deviation of the underlying white noise. Default is 1.0.
    fsample : float, optional
        Sampling frequency in Hz. Default is 1.0.
    lowfz : float, optional
        Low frequency cutoff in Hz. PSD is white below this. Default is 1e-9.
    highfz : float, optional
        High frequency cutoff in Hz. PSD is white above this. Default is 10.0.

    Returns
    -------
    np.ndarray
        Truncated 1/f^2 noise time series, shape (length,).

    Mathematical Formula
    --------------------
    Uses a first-order IIR filter with frequency-dependent coefficients:

        y[n] = a₀·x[n] + a₁·x[n-1] + b₁·y[n-1]

    where:
        r₀ = π·f_low/f_s
        r₁ = π·f_high/f_s
        a₀ = (1 + r₁)/(1 + r₀)
        a₁ = -(1 - r₁)/(1 + r₀)
        b₁ = (1 - r₀)/(1 + r₀)

    Transfer Function:
        H(z) = (a₀ + a₁·z⁻¹)/(1 - b₁·z⁻¹)

    Power Spectral Density:
        S(f) ~ constant         for f < f_low
        S(f) ~ 1/f²            for f_low < f < f_high
        S(f) ~ constant         for f > f_high

    The output is scaled by √((1 + f_low²)/(1 + f_high²)) for normalization.

    Notes
    -----
    Uses first-order IIR filter with cutoff-dependent coefficients.
    The PSD transitions smoothly between white noise and 1/f^2 regions.

    Optimized implementation uses pre-allocated numpy array.

    References
    ----------
    Plaszczynski, S. (2007). "Generating long streams of 1/f^alpha noise."
    """
    if length == 0:
        return np.array([])
    else:
        r0 = np.pi * lowfz / fsample
        r1 = np.pi * highfz / fsample
        a0 = (1 + r1) / (1 + r0)
        a1 = -(1 - r1) / (1 + r0)
        b1 = (1 - r0) / (1 + r0)

        data = white(length, mean, sigma, fsample)

        # Old slow implementation (list append):
        # data5 = []
        # for i in range(length):
        #     if i == 0:
        #         data5.append(a0 * data[0])
        #     else:
        #         data5.append(a0 * data[i] + a1 * data[i-1] + b1 * data5[i-1])
        # return np.array(data5) * np.sqrt((1 + lowfz * lowfz) / (1 + highfz * highfz))

        # New optimized implementation (pre-allocated array):
        data5 = np.empty(length, dtype=np.float64)
        data5[0] = a0 * data[0]
        for i in range(1, length):
            data5[i] = a0 * data[i] + a1 * data[i-1] + b1 * data5[i-1]
        return data5 * np.sqrt((1 + lowfz * lowfz) / (1 + highfz * highfz))


def m4_trun(length: int,
            mean: float = 0.0,
            sigma: float = 1.0,
            fsample: float = 1.0,
            lowfz: float = 1e-9,
            highfz: float = 10.0) -> NDArray[np.float64]:
    """
    Generate truncated 1/f^4 noise with flat PSD outside cutoff frequencies.

    Cascades two m2_trun() filters to create 1/f^4 noise with controlled
    low and high frequency behavior.

    Parameters
    ----------
    length : int
        Number of samples to generate.
    mean : float, optional
        Mean value of the underlying white noise. Default is 0.0.
    sigma : float, optional
        Standard deviation of the underlying white noise. Default is 1.0.
    fsample : float, optional
        Sampling frequency in Hz. Default is 1.0.
    lowfz : float, optional
        Low frequency cutoff in Hz. PSD is white below this. Default is 1e-9.
    highfz : float, optional
        High frequency cutoff in Hz. PSD is white above this. Default is 10.0.

    Returns
    -------
    np.ndarray
        Truncated 1/f^4 noise time series, shape (length,).

    Mathematical Formula
    --------------------
    Cascades two m2_trun() filters to achieve truncated 1/f^4 behavior:

        Stage 1: y₁[n] = a₀·x[n] + a₁·x[n-1] + b₁·y₁[n-1]
        Stage 2: y[n] = a₀·y₁[n] + a₁·y₁[n-1] + b₁·y[n-1]

    where coefficients are the same as in m2_trun().

    Combined Transfer Function:
        H(z) = [(a₀ + a₁·z⁻¹)/(1 - b₁·z⁻¹)]²

    Power Spectral Density:
        S(f) ~ constant         for f < f_low
        S(f) ~ 1/f⁴            for f_low < f < f_high
        S(f) ~ constant         for f > f_high

    The cascaded approach provides better numerical stability and smooth
    transitions at cutoff frequencies.

    Notes
    -----
    Applies the Plaszczynski algorithm twice for 1/f^4 behavior.
    More stable than direct second-order recursion.

    Optimized implementation uses pre-allocated numpy array.

    References
    ----------
    Plaszczynski, S. (2007). "Generating long streams of 1/f^alpha noise."
    """
    if length == 0:
        return np.array([])
    else:
        r0 = np.pi * lowfz / fsample
        r1 = np.pi * highfz / fsample
        a0 = (1 + r1) / (1 + r0)
        a1 = -(1 - r1) / (1 + r0)
        b1 = (1 - r0) / (1 + r0)

        data = m2_trun(length, mean, sigma, fsample, lowfz, highfz)

        # Old slow implementation (list append):
        # data5 = []
        # for i in range(length):
        #     if i == 0:
        #         data5.append(a0 * data[0])
        #     else:
        #         data5.append(a0 * data[i] + a1 * data[i-1] + b1 * data5[i-1])
        # return np.array(data5) * np.sqrt((1 + lowfz * lowfz) / (1 + highfz * highfz))

        # New optimized implementation (pre-allocated array):
        data5 = np.empty(length, dtype=np.float64)
        data5[0] = a0 * data[0]
        for i in range(1, length):
            data5[i] = a0 * data[i] + a1 * data[i-1] + b1 * data5[i-1]
        return data5 * np.sqrt((1 + lowfz * lowfz) / (1 + highfz * highfz))


def oms(length: int,
        fsample: float = 1.0,
        lowfz: float = 1e-9,
        highfz: float = 10.0,
        strain: float = 8.0e-12) -> NDArray[np.float64]:
    """
    Generate optical metrology system (OMS) noise for Taiji/LISA. /LISA(OMS)

    Combines f^2 and 1/f^2 noise components to model the optical path length
    noise in Taiji/LISA's optical metrology system.

    Parameters
    ----------
    length : int
        Number of samples to generate.
    fsample : float, optional
        Sampling frequency in Hz. Default is 1.0.
    lowfz : float, optional
        Low frequency cutoff for 1/f^2 component. Default is 1e-9 Hz.
    highfz : float, optional
        High frequency cutoff for 1/f^2 component. Default is 10.0 Hz.
    strain : float, optional
        OMS noise strain amplitude. Default is 8.0e-12 (LISA specification).

    Returns
    -------
    np.ndarray
        OMS noise time series in phase units, shape (length,).

    Mathematical Formula
    --------------------
    The OMS noise is a combination of two components:

        y_OMS[n] = y_f²[n] + y_1/f²[n]

    Component 1 (shot noise, high frequency):
        y_f²[n] ~ power_2(σ₁)
        σ₁ = 2π·a_OMS
        S₁(f) ∝ f²

    Component 2 (drift, low frequency):
        y_1/f²[n] ~ m2_trun(σ₂)
        σ₂ = 2π·a_OMS·f₃²
        S₂(f) ∝ 1/f²  (for f_low < f < f_high)

    where:
        a_OMS = strain/c = strain/(3×10⁸ m/s)
        f₃ = 2.0 mHz (corner frequency)

    Combined Power Spectral Density:
        S_OMS(f) = σ₁²·f² + σ₂²/f²

    At the corner frequency f₃, both components contribute equally.

    Notes
    -----
    The noise model combines:
    - f^2 component (shot noise-like)
    - 1/f^2 component (low-frequency drift)

    The corner frequency f3 = 2.0 mHz sets the transition between regimes.
    Strain is converted to phase units by dividing by speed of light (3e8 m/s).

    References
    ----------
    LISA noise budget specifications (LISA-LCST-SGS-TN-001).
    """
    if length == 0:
        return np.array([])
    else:
        f3 = 2.0e-3  # Corner frequency in Hz
        a_oms = strain / (3.0e8)  # Convert strain to phase units
        sigma1 = 2 * np.pi * a_oms
        sigma2 = 2 * np.pi * a_oms * (f3 * f3)
        d1 = power_2(length, 0.0, sigma1, fsample)
        d2 = m2_trun(length, 0.0, sigma2, fsample, lowfz, highfz)
    return d1 + d2


def acc(length: int,
        fsample: float = 1.0,
        lowfz: float = 1e-9,
        highfz: float = 10.0,
        strain: float = 3.0e-15) -> NDArray[np.float64]:
    """
    Generate acceleration noise for LISA test masses. LISA

    Combines white, f^2, 1/f^2, and 1/f^4 noise components to model the
    residual acceleration noise of free-falling test masses in LISA.

    Parameters
    ----------
    length : int
        Number of samples to generate.
    fsample : float, optional
        Sampling frequency in Hz. Default is 1.0.
    lowfz : float, optional
        Low frequency cutoff for power law components. Default is 1e-9 Hz.
    highfz : float, optional
        High frequency cutoff for power law components. Default is 10.0 Hz.
    strain : float, optional
        Acceleration noise strain amplitude. Default is 3.0e-15 (LISA spec).

    Returns
    -------
    np.ndarray
        Acceleration noise time series in phase units, shape (length,).

    Mathematical Formula
    --------------------
    The acceleration noise combines four components:

        y_acc[n] = y_white[n] + y_f²[n] + y_1/f²[n] + y_1/f⁴[n]

    Component 1 (thermal, white noise):
        y_white[n] ~ white(σ₁)
        σ₁ = (1/2π)·a_TM·f₁/f₂²
        S₁(f) = σ₁²

    Component 2 (high frequency):
        y_f²[n] ~ power_2(σ₂)
        σ₂ = (1/2π)·a_TM/f₂²
        S₂(f) ∝ f²

    Component 3 (low frequency drift):
        y_1/f²[n] ~ m2_trun(σ₃)
        σ₃ = (1/2π)·a_TM
        S₃(f) ∝ 1/f²  (for f_low < f < f_high)

    Component 4 (very low frequency):
        y_1/f⁴[n] ~ m4_trun(σ₄)
        σ₄ = (1/2π)·a_TM·f₁
        S₄(f) ∝ 1/f⁴  (for f_low < f < f_high)

    where:
        a_TM = strain/c = strain/(3×10⁸ m/s)
        f₁ = 0.4 mHz (first corner frequency)
        f₂ = 8.0 mHz (second corner frequency)

    Combined Power Spectral Density:
        S_acc(f) = σ₁² + σ₂²·f² + σ₃²/f² + σ₄²/f⁴

    The corner frequencies determine transitions between different regimes:
    - f < f₁: 1/f⁴ dominates
    - f₁ < f < f₂: 1/f² dominates
    - f > f₂: f² and white noise dominate

    Notes
    -----
    The noise model combines four components:
    - White noise (thermal fluctuations)
    - f^2 component
    - 1/f^2 component (low-frequency drift)
    - 1/f^4 component (very low frequency effects)

    Corner frequencies:
    - f1 = 0.4 mHz
    - f2 = 8.0 mHz

    These set the transitions between different noise regimes.
    Strain is converted to phase units by dividing by speed of light.

    References
    ----------
    LISA noise budget specifications (LISA-LCST-SGS-TN-001).
    """
    if length == 0:
        return np.array([])
    else:
        a_tm = strain / (3.0e8)  # Convert strain to phase units
        f1 = 4e-4  # First corner frequency in Hz
        f2 = 8e-3  # Second corner frequency in Hz

        # Compute amplitudes for each noise component
        sigma1 = 1 / (2 * np.pi) * a_tm * f1 / (f2**2)
        sigma2 = 1 / (2 * np.pi) * a_tm * 1 / (f2**2)
        sigma3 = 1 / (2 * np.pi) * a_tm
        sigma4 = 1 / (2 * np.pi) * a_tm * f1

        # Generate and combine noise components
        d1 = white(length, 0.0, sigma1, fsample)
        d2 = power_2(length, 0.0, sigma2, fsample)
        d3 = m2_trun(length, 0.0, sigma3, fsample, lowfz, highfz)
        d4 = m4_trun(length, 0.0, sigma4, fsample, lowfz, highfz)
    return d1 + d2 + d3 + d4
