"""
Gravitational wave signal generation module for Taiji/LISA. /LISA

Generates gravitational wave strain responses in spacecraft arms, including: :
- Antenna pattern calculations for arbitrary sky positions 
- Polarization tensor computations 
- Light travel time effects 
- Support for single or multiple monochromatic sources 

The module computes the strain induced by GW on each of the 6 spacecraft arms
(3 forward + 3 reverse directions) and provides interpolators for time-delayed
access required in TDI combinations. 63+3
TDI

Author : TY
"""

import math
import numpy as np
from scipy.interpolate import interp1d

from TJ_constant import XVEC, YVEC, ZVEC, ARM_REC, ARM_SEN, RS_ARM, ARM_ARRAY

# Create lowercase aliases for backward compatibility
xvec = XVEC
yvec = YVEC
zvec = ZVEC
armRec = ARM_REC
armSen = ARM_SEN
RSarm = RS_ARM
arm_array = ARM_ARRAY

# Gravitational wave modulates arm length, affecting mainly laser phases
# References:
# - LISA Sensitivity and SNR Calculations, arXiv:2108.01176
# - Simulation and Data Analysis for LISA, Bayle

def uvec(beta=0, lamda=0):
    """
    Compute u basis vector for GW polarization.

    Parameters
    ----------
    beta : float
        Ecliptic latitude in radians
    lamda : float
        Ecliptic longitude in radians

    Returns
    -------
    ndarray
        u basis vector (3D)
    """
    return math.sin(lamda)*xvec - math.cos(lamda)*yvec

def vvec(beta=0, lamda=0):
    """
    Compute v basis vector for GW polarization.

    Parameters
    ----------
    beta : float
        Ecliptic latitude in radians
    lamda : float
        Ecliptic longitude in radians

    Returns
    -------
    ndarray
        v basis vector (3D)
    """
    return (-math.sin(beta)*math.cos(lamda)*xvec
            -math.sin(beta)*math.sin(lamda)*yvec+math.cos(beta)*zvec)

def kvec(beta=0, lamda=0):
    """
    Compute wave propagation direction vector.

    Parameters
    ----------
    beta : float
        Ecliptic latitude in radians
    lamda : float
        Ecliptic longitude in radians

    Returns
    -------
    ndarray
        Wave vector k (3D), pointing from source to detector
    """
    return -(math.cos(beta)*math.cos(lamda)*xvec
            +math.cos(beta)*math.sin(lamda)*yvec+math.sin(beta)*zvec)

def pvec(beta: float = 0, lamda: float = 0, psi: float = 0) -> np.ndarray:
    """
    Compute polarization vector p for gravitational wave.

    The p vector is one of the two orthogonal polarization basis vectors,
    rotated by angle psi from the u-v basis.

    Parameters
    ----------
    beta : float, optional
        Ecliptic latitude in radians. Default is 0.
    lamda : float, optional
        Ecliptic longitude in radians. Default is 0.
    psi : float, optional
        Polarization angle in radians. Default is 0.

    Returns
    -------
    np.ndarray
        Polarization vector p (3D array)

    Notes
    -----
    The p vector is defined as:
        p = cos(ψ)·u + sin(ψ)·v

    where u and v are the standard GW polarization basis vectors.
    """
    return math.cos(psi)*uvec(beta, lamda) + math.sin(psi)*vvec(beta, lamda)

def qvec(beta: float = 0, lamda: float = 0, psi: float = 0) -> np.ndarray:
    """
    Compute polarization vector q for gravitational wave.

    The q vector is the second orthogonal polarization basis vector,
    perpendicular to p, rotated by angle psi from the u-v basis.

    Parameters
    ----------
    beta : float, optional
        Ecliptic latitude in radians. Default is 0.
    lamda : float, optional
        Ecliptic longitude in radians. Default is 0.
    psi : float, optional
        Polarization angle in radians. Default is 0.

    Returns
    -------
    np.ndarray
        Polarization vector q (3D array)

    Notes
    -----
    The q vector is defined as:
        q = -sin(ψ)·u + cos(ψ)·v

    where u and v are the standard GW polarization basis vectors.
    q is orthogonal to p: p·q = 0
    """
    return -math.sin(psi)*uvec(beta, lamda) + math.cos(psi)*vvec(beta, lamda)

def eplus(beta: float = 0, lamda: float = 0) -> np.ndarray:
    """
    Compute plus polarization tensor for gravitational wave.

    Constructs the 3x3 polarization tensor for the plus (+) polarization mode
    of a gravitational wave propagating from direction (beta, lamda).

    Parameters
    ----------
    beta : float, optional
        Ecliptic latitude in radians. Default is 0.
    lamda : float, optional
        Ecliptic longitude in radians. Default is 0.

    Returns
    -------
    np.ndarray
        3x3 plus polarization tensor matrix

    Notes
    -----
    The plus polarization tensor is defined as:
        e₊ = u ⊗ u - v ⊗ v

    where u and v are the GW polarization basis vectors and ⊗ denotes
    the outer product. This tensor is symmetric and traceless.

    The plus polarization represents the "breathing" mode of GW strain.
    """
    u = uvec(beta, lamda)
    ut = u.reshape(3, 1)
    v = vvec(beta, lamda)
    vt = v.reshape(3, 1)
    return np.dot(ut, [u]) - np.dot(vt, [v])

def ecros(beta: float = 0, lamda: float = 0) -> np.ndarray:
    """
    Compute cross polarization tensor for gravitational wave.

    Constructs the 3x3 polarization tensor for the cross (×) polarization mode
    of a gravitational wave propagating from direction (beta, lamda).

    Parameters
    ----------
    beta : float, optional
        Ecliptic latitude in radians. Default is 0.
    lamda : float, optional
        Ecliptic longitude in radians. Default is 0.

    Returns
    -------
    np.ndarray
        3x3 cross polarization tensor matrix

    Notes
    -----
    The cross polarization tensor is defined as:
        e× = u ⊗ v + v ⊗ u

    where u and v are the GW polarization basis vectors and ⊗ denotes
    the outer product. This tensor is symmetric and traceless.

    The cross polarization represents the "shearing" mode of GW strain,
    rotated 45° from the plus polarization.
    """
    u = uvec(beta, lamda)
    ut = u.reshape(3, 1)
    v = vvec(beta, lamda)
    vt = v.reshape(3, 1)
    return np.dot(ut, [v]) + np.dot(vt, [u])

def xipls(beta=0, lamda=0, nij=[1,1,1]):
    """
    Antenna pattern function for plus polarization.

    Parameters
    ----------
    beta : float
        Ecliptic latitude in radians
    lamda : float
        Ecliptic longitude in radians
    nij : array_like
        Normalized arm direction vector

    Returns
    -------
    float
        Plus polarization antenna pattern coefficient
    """
    u = uvec(beta, lamda)
    v = vvec(beta, lamda)
    return (np.dot(nij,u))**2-(np.dot(nij,v))**2

def xicrs(beta=0, lamda=0, nij=[1,1,1]):
    """
    Antenna pattern function for cross polarization.

    Parameters
    ----------
    beta : float
        Ecliptic latitude in radians
    lamda : float
        Ecliptic longitude in radians
    nij : array_like
        Normalized arm direction vector

    Returns
    -------
    float
        Cross polarization antenna pattern coefficient
    """
    u = uvec(beta, lamda)
    v = vvec(beta, lamda)
    return 2*(np.dot(nij,u))*(np.dot(nij,v))

def ygwsr(arm_num, t, orbit_data, fgw=0.001, strain=1e-20, beta=0, lamda=0, psi=0):
    """
    Compute GW strain response for one arm at one time point.

    This function calculates the strain induced by a monochromatic GW on a
    spacecraft arm, including light travel time effects and proper antenna
    pattern response.

    Parameters
    ----------
    arm_num : int
        Arm number (1, 2, 3 for forward; -1, -2, -3 for reverse)
    t : float
        Current time in seconds
    orbit_data : orbit
        Orbit object providing spacecraft positions
    fgw : float, optional
        GW frequency in Hz. Default is 0.001 Hz
    strain : float, optional
        GW strain amplitude. Default is 1e-20
    beta : float, optional
        Ecliptic latitude in radians. Default is 0
    lamda : float, optional
        Ecliptic longitude in radians. Default is 0
    psi : float, optional
        Polarization angle in radians. Default is 0

    Returns
    -------
    float
        GW-induced strain on the specified arm at time t

    Notes
    -----
    The calculation includes:
    - Light travel time from sender to receiver spacecraft
    - Wave propagation time from GW source
    - Antenna pattern response for arm direction
    - Both plus and cross polarization contributions
    """
    recv = armRec[arm_num]
    send = armSen[arm_num]

    precv = orbit_data.position(recv, t)
    
    tsend = t - orbit_data.dij(arm_num,t)
    
    psend = orbit_data.position(send, t)
    
    vsr = precv - psend
    
    nrs = np.array(vsr)/(math.sqrt(np.dot(vsr,vsr)))
    
    k = kvec(beta,lamda)
    
    hp = (math.cos(2*math.pi*fgw*(tsend-np.dot(k,psend)))-math.cos(2*math.pi*fgw*(t-np.dot(k,precv))))
    hc = (math.cos(2*math.pi*fgw*(tsend-np.dot(k,psend)))-math.cos(2*math.pi*fgw*(t-np.dot(k,precv))))
    
    return (strain)/(2*(1-np.dot(k,nrs)))*(
            (hp*math.cos(2*psi)-hc*math.sin(2*psi))*xipls(beta,lamda,nrs)+(hp*math.sin(2*psi)+hc*math.cos(2*psi))*xicrs(beta,lamda,nrs) )

class gw():
    """
    Gravitational wave signal generator for Taiji/LISA. /LISA

    Computes GW strain response in all 6 spacecraft arms for single or multiple
    monochromatic sources. Pre-computes strain time series and creates interpolators
    for efficient time-delayed access in TDI combinations. 6
    TDI

    Parameters 
    ----------
    orbits : orbit
        Orbit object providing spacecraft positions and arm geometry 
    t_start : float, optional
        Start time in seconds. Default is 0.0 0.0
    t_end : float, optional
        End time in seconds. Default is 10000.0 10000.0
    fsample : float, optional
        Sampling frequency in Hz. Default is 2 Hz Hz2 Hz
    fgw : float, optional
        GW frequency in Hz (for single source). Default is 0.001 Hz Hz0.001 Hz
    strain : float, optional
        GW strain amplitude (for single source). Default is 1e-20 1e-20
    hasGW : bool, optional
        Enable GW signal generation. Default is False False
    beta : float, optional
        Ecliptic latitude in radians (for single source). Default is 0 0
    lamda : float, optional
        Ecliptic longitude in radians (for single source). Default is 0 0
    psi : float, optional
        Polarization angle in radians (for single source). Default is 0 0
    source_list : list of dict, optional
        List of multiple GW sources. Each dict contains keys :
        'fgw', 'strain', 'beta', 'lamda', 'psi'
        If provided, overrides single source parameters. 
        Example : [
            {'fgw': 0.001, 'strain': 1e-20, 'beta': 0, 'lamda': 0, 'psi': 0},
            {'fgw': 0.002, 'strain': 5e-21, 'beta': 1.0, 'lamda': 2.0, 'psi': 0.5}
        ]

    Attributes 
    ----------
    sources : list of dict
        List of all GW source parameters 
    n_sources : int
        Number of GW sources in the superposition 
    _gw_arr : ndarray, shape (6, length)
        Pre-computed GW strain for all 6 arms 6
    gw_int : list of interp1d
        List of 6 interpolators (one per arm) 6
    gwint : list of list
        3x3 matrix of interpolators indexed by [receiver-1][sender-1] [-1][-1]3x3
    gw : list
        Interpolators indexed as [None, 1, 2, 3, 3p, 2p, 1p] for arm numbers 

    Examples 
    --------
    Single monochromatic source :
    >>> gws = gw(orbits, t_start=0, t_end=10000, fsample=2,
    ...          fgw=0.003, strain=1e-20, beta=0.5, lamda=1.2, psi=0,
    ...          hasGW=True)

    Multiple sources with different sky directions :
    >>> sources = [
    ...     {'fgw': 0.001, 'strain': 1e-20, 'beta': 0, 'lamda': 0, 'psi': 0},
    ...     {'fgw': 0.002, 'strain': 5e-21, 'beta': 1.0, 'lamda': 2.0, 'psi': 0.5},
    ...     {'fgw': 0.003, 'strain': 2e-21, 'beta': -0.5, 'lamda': 3.0, 'psi': 1.0}
    ... ]
    >>> gws = gw(orbits, t_start=0, t_end=10000, fsample=2,
    ...          source_list=sources, hasGW=True)

    Access GW strain at specific time for arm 1 1:
    >>> strain_value = gws.gw[1](t=5000.0)

    Notes 
    -----
    - The strain superposition is linear: total strain = sum of individual strains =
    - All sources are evaluated at the same time samples 
    - Interpolation uses cubic splines (kind=3) for smooth time-delayed access kind=3
    - Sky coordinates: beta (ecliptic latitude), lamda (ecliptic longitude) betalamda
    """
    def __init__(self, orbits, t_start=0.0, t_end=10000.0, fsample=2,
                 fgw=0.001, strain=1e-20, hasGW=False, beta=0, lamda=0, psi=0,
                 source_list=None):
        self._fsample = fsample
        self._tarray = np.arange(t_start, t_end, 1./fsample)
        self._length = len(self._tarray)
        self.i_ord = 3

        self.armlen = orbits._armlen
        self.orbit = orbits
        self.hasGW = hasGW  # Store hasGW flag for synthesis optimization hasGW

        # Handle source parameters: either source_list or individual parameters
        if source_list is not None:
            # Multiple sources mode
            self.sources = source_list
            self.n_sources = len(source_list)
        else:
            # Single source mode (backward compatibility)
            self.sources = [{
                'fgw': fgw,
                'strain': strain,
                'beta': beta,
                'lamda': lamda,
                'psi': psi
            }]
            self.n_sources = 1
            # Store as attributes for backward compatibility
            self.fgw = fgw
            self.strain = strain
            self.beta = beta
            self.lamda = lamda
            self.psi = psi

        # Initialize GW array
        self._gw_arr = np.zeros((6, self._length))

        # Old implementation (very slow - nested triple loop with 6 * length * n_sources function calls):
        # if hasGW == 'True':
        #     for src_idx in range(self.n_sources):
        #         src = self.sources[src_idx]
        #         for i in range(6):
        #             for j in range(self._length):
        #                 self._gw_arr[i][j] += ygwsr(
        #                     arm_array[i], self._tarray[j], self.orbit,
        #                     src['fgw'], src['strain'], src['beta'],
        #                     src['lamda'], src['psi']
        #                 )

        # Optimized implementation (pre-compute orbit data and vectorize time loop):
        if hasGW:
            # Pre-compute spacecraft positions and arm vectors for all time points
            # This avoids redundant orbit lookups inside the inner loops
            positions = {}  # positions[spacecraft][time_idx] = position vector
            for sc in [1, 2, 3]:
                positions[sc] = np.array([self.orbit.position(sc, t) for t in self._tarray])

            # Pre-compute arm distances at all time points
            arm_distances = {}  # arm_distances[arm_num][time_idx] = distance
            for arm_num in arm_array:
                arm_distances[arm_num] = np.array([self.orbit.dij(arm_num, t) for t in self._tarray])

            # Compute GW strain for each source and arm
            for src_idx in range(self.n_sources):
                src = self.sources[src_idx]

                # Pre-compute source direction vectors (constant for each source)
                k = kvec(src['beta'], src['lamda'])
                u = uvec(src['beta'], src['lamda'])
                v = vvec(src['beta'], src['lamda'])

                for i in range(6):
                    arm_num = arm_array[i]
                    recv = armRec[arm_num]
                    send = armSen[arm_num]

                    # Get pre-computed positions for this arm
                    precv_arr = positions[recv]  # shape: (length, 3)
                    psend_arr = positions[send]  # shape: (length, 3)

                    # Compute send times (vectorized)
                    tsend_arr = self._tarray - arm_distances[arm_num]

                    # Get sender positions at delayed times (requires interpolation)
                    psend_delayed = np.array([self.orbit.position(send, tsend_arr[j])
                                             for j in range(self._length)])

                    # Compute arm direction vectors at all times (vectorized)
                    vsr_arr = precv_arr - psend_delayed  # shape: (length, 3)
                    vsr_norm = np.sqrt(np.sum(vsr_arr**2, axis=1))  # shape: (length,)
                    nrs_arr = vsr_arr / vsr_norm[:, np.newaxis]  # shape: (length, 3)

                    # Compute k·p dot products (vectorized)
                    k_dot_precv = np.dot(precv_arr, k)  # shape: (length,)
                    k_dot_psend = np.dot(psend_delayed, k)  # shape: (length,)

                    # Compute GW phase terms (vectorized)
                    omega = 2 * np.pi * src['fgw']
                    phase_recv = omega * (self._tarray - k_dot_precv)
                    phase_send = omega * (tsend_arr - k_dot_psend)

                    hp = np.cos(phase_send) - np.cos(phase_recv)
                    hc = np.sin(phase_send) - np.sin(phase_recv)

                    # Compute antenna patterns (vectorized)
                    nrs_dot_u = np.dot(nrs_arr, u)  # shape: (length,)
                    nrs_dot_v = np.dot(nrs_arr, v)  # shape: (length,)

                    xipls_arr = nrs_dot_u**2 - nrs_dot_v**2  # shape: (length,)
                    xicrs_arr = 2 * nrs_dot_u * nrs_dot_v    # shape: (length,)

                    # Compute k·n dot product
                    k_dot_nrs = np.dot(nrs_arr, k)  # shape: (length,)

                    # Final strain calculation (vectorized)
                    prefactor = src['strain'] / (2 * (1 - k_dot_nrs))
                    polarization_term = (
                        (hp * np.cos(2*src['psi']) - hc * np.sin(2*src['psi'])) * xipls_arr +
                        (hp * np.sin(2*src['psi']) + hc * np.cos(2*src['psi'])) * xicrs_arr
                    )

                    self._gw_arr[i] += prefactor * polarization_term

        # Create interpolators
        self.gw_int = [interp1d(self._tarray, self._gw_arr[i], kind=self.i_ord,
                                fill_value=(1e-99, 1e-99), bounds_error=False)
                       for i in range(6)]
        self.gwint = [[None, self.gw_int[0], self.gw_int[1]],
                      [self.gw_int[2], None, self.gw_int[3]],
                      [self.gw_int[4], self.gw_int[5], None]]

        self.gw = [None, self.gw_int[0], self.gw_int[3], self.gw_int[4],
                   self.gw_int[5], self.gw_int[2], self.gw_int[1]]
        # none, gw[1], gw[2], gw[3], gw[3p], gw[2p], gw[1p]
        
    def ygw(self, arm_num, t):
        """
        Get GW strain for a specific arm at a specific time.

        Parameters
        ----------
        arm_num : int
            Arm number (1, 2, 3 for forward arms; -1, -2, -3 for reverse)
        t : float or ndarray
            Time value(s) in seconds

        Returns
        -------
        float or ndarray
            Interpolated GW strain value(s) at time t
        """
        sc_rev = armRec[arm_num]
        sc_sed = armSen[arm_num]
        return self.gwint[sc_rev-1][sc_sed-1](t)