"""
Optical bench module for Taiji/LISA. /LISA

Simulates onboard instrumentation at each spacecraft, including laser noise,
acceleration noise, optical metrology noise, and phasemeter outputs. 


Key Features :
- Laser noise generation (free-running and locked) 
- Acceleration noise (test mass disturbances) 
- Optical metrology system (OMS) noise (OMS)
- Phasemeter outputs: science (sci), test (tes), reference (ref) (sci)(tes)(ref)
- High-order interpolation for time-delayed signal access 
- Support for all six arm directions (forward and reverse) 

The optical bench objects store both raw time series data and interpolators
for efficient time-delayed access needed in TDI combinations. 
TDI

Author : TY
"""

import numpy as np
from scipy.interpolate import interp1d

import TJ_noise as noise
from TJ_constant import ARM_REC, ARM_SEN, RS_ARM, ARM_ARRAY

# Create lowercase aliases for backward compatibility
armRec = ARM_REC
armSen = ARM_SEN
RSarm = RS_ARM
arm_array = ARM_ARRAY

class ob():
    """
    Optical bench class for simulating onboard instrumentation. 

    This class models the optical bench at each spacecraft, which measures
    phase differences between local and incoming laser beams. It generates
    various noise sources and provides phasemeter outputs used in TDI. 
    TDI

    Parameters 
    ----------
    t_start : float, optional
        Start time in seconds 
        Default is 0.0 0.0
    t_end : float, optional
        End time in seconds 
        Default is 10000.0 10000.0
    fsample : float, optional
        Sampling frequency in Hz Hz
        Default is 2 Hz 2 Hz
    hasLaser : bool, optional
        Enable laser frequency noise generation 
        Default is True True
    hasAcc : bool, optional
        Enable acceleration noise (test mass) 
        Default is False False
    hasOms : bool, optional
        Enable optical metrology system noise 
        Default is False False
    lowfz : float, optional
        Low frequency cutoff for noise generation 
        Default is 1e-5 Hz 1e-5 Hz

    Attributes
    ----------
    _fsample : float
        Sampling frequency in Hz
    _duration : float
        Total duration of simulation in seconds
    _tarray : ndarray
        Time array for the simulation
    _length : int
        Number of time samples
    _lowfz : float
        Low frequency cutoff
    i_ord : int
        Interpolation order (31 for high-order interpolation)
    _laser_noise : ndarray
        Free-running laser frequency noise
    _lock_laser : ndarray
        Locked laser frequency noise (after arm-locking)
    _laser : ndarray
        Current laser noise (either free-running or locked)
    _acc : ndarray
        Acceleration noise time series
    _oms : ndarray
        Optical metrology system noise time series
    _clock : ndarray
        Clock noise time series (currently zero)
    _sci : ndarray
        Science phasemeter output
    _tes : ndarray
        Test phasemeter output
    _ref : ndarray
        Reference phasemeter output
    _xi : ndarray
        Xi signal (intermediate TDI quantity)
    _eta : ndarray
        Eta signal (input to TDI combinations)
    laser : interp1d
        Interpolator for laser noise
    lock : interp1d
        Interpolator for locked laser
    acc : interp1d
        Interpolator for acceleration noise
    oms : interp1d
        Interpolator for OMS noise
    clock : interp1d
        Interpolator for clock noise
    sci : interp1d
        Interpolator for science phasemeter
    tes : interp1d
        Interpolator for test phasemeter
    ref : interp1d
        Interpolator for reference phasemeter
    xi : interp1d
        Interpolator for xi signal
    eta : interp1d
        Interpolator for eta signal

    Examples
    --------
    Create optical bench with only laser noise:
    >>> ob1 = ob(t_start=0, t_end=10000, fsample=10, hasLaser=True)

    Create optical bench with all noise sources:
    >>> ob2 = ob(t_start=0, t_end=10000, fsample=10,
    ...          hasLaser=True, hasAcc=True, hasOms=True)

    Access laser noise at specific time:
    >>> laser_value = ob1.laser(100.5)  # Interpolated value at t=100.5s

    Notes
    -----
    - All time series arrays have prefix '_' (e.g., _laser, _acc)
    - All interpolators are accessed without prefix (e.g., laser, acc)
    - Uses high-order interpolation (kind=31) for accurate time delays
    - Interpolators return 1e-99 outside time bounds (effectively zero)
    """
    # Type annotations for dynamically created interpolator attributes
    sci: interp1d
    tes: interp1d
    ref: interp1d
    eta: interp1d
    xi: interp1d
    laser: interp1d
    lock: interp1d
    acc: interp1d
    oms: interp1d
    clock: interp1d

    def __init__(self,
                 t_start: float = 0.0,
                 t_end: float = 10000.0,
                 fsample: float = 2,
                 hasLaser: bool = True,
                 LNamp: float = 1.0e-13,
                 hasAcc: bool = False,
                 hasOms: bool = False,
                 lowfz: float = 1e-5) -> None:
        self._fsample = fsample
        self._duration= t_end - t_start
        self._tarray = np.arange(t_start, t_end, 1./fsample)
        self._length =len(self._tarray)
        self._lowfz =lowfz
        self.i_ord =31

        # Store noise configuration flags for synthesis optimization 
        self.hasLaser = hasLaser
        self.hasAcc = hasAcc
        self.hasOms = hasOms

        self._laser_noise = noise.white(self._length, 0.0, LNamp, self._fsample) if hasLaser else np.zeros(self._length)

        self._lock_laser  = np.zeros(self._length)

        self._acc = noise.acc(self._length, self._fsample, lowfz) if hasAcc else np.zeros(self._length)

        self._oms = noise.oms(self._length, self._fsample, lowfz) if hasOms else np.zeros(self._length)

        self._clock       = np.zeros(self._length)

        self._sci         = np.zeros(self._length)
        self._tes         = np.zeros(self._length)
        self._ref         = np.zeros(self._length)
        self._xi          = np.zeros(self._length)
        self._eta         = np.zeros(self._length)

        self._laser = self._laser_noise

        # Create all interpolators in batch
        signal_data = {
            'laser': self._laser_noise,
            'lock':  self._lock_laser,
            'acc':   self._acc,
            'oms':   self._oms,
            'clock': self._clock,
            'sci':   self._sci,
            'tes':   self._tes,
            'ref':   self._ref,
            'xi':    self._xi,
            'eta':   self._eta
        }
        self._create_interpolators(signal_data)

    def _interp(self, data: np.ndarray) -> interp1d:
        """
        Create standard interpolator for signal data.

        Uses high-order interpolation (kind=31) for accurate time-delayed
        signal access required in TDI combinations.

        Parameters
        ----------
        data : ndarray
            Time series data to interpolate

        Returns
        -------
        interp1d
            Interpolator function that accepts time values and returns
            interpolated signal values. Returns 1e-99 outside bounds.
        """
        return interp1d(self._tarray, data,
                       kind=self.i_ord,
                       fill_value=(1e-99, 1e-99),
                       bounds_error=False)

    def _create_interpolators(self, signal_dict: dict) -> None:
        """
        Batch create interpolators from dictionary.

        Efficiently creates multiple interpolators at initialization time
        by iterating through a dictionary of signal names and data.

        Parameters
        ----------
        signal_dict : dict
            Dictionary mapping signal names to their time series data.
            Keys become attribute names for the interpolators.
        """
        for name, data in signal_dict.items():
            setattr(self, name, self._interp(data))

    def set_noise(self, parameters: float, noise_type: str = 'laser') -> None:
        """
        Update noise sources and refresh their interpolators.

        Regenerates the specified noise type with new parameters and updates
        the corresponding interpolator.

        Parameters
        ----------
        parameters : float
            For 'laser': sigma value for white noise (e.g., 1e-13)
            For 'acc' and 'oms': not used (uses internal _lowfz)
        noise_type : {'laser', 'acc', 'oms'}, optional
            Type of noise to update:
            - 'laser': Laser frequency noise (white noise)
            - 'acc': Acceleration noise (f^-2 spectrum)
            - 'oms': Optical metrology system noise
            Default is 'laser'

        Examples
        --------
        >>> ob1 = ob(t_start=0, t_end=1000, fsample=10)
        >>> ob1.set_noise(5e-14, noise_type='laser')  # Change laser noise level
        >>> ob1.set_noise(None, noise_type='acc')     # Enable acceleration noise
        """
        if noise_type == 'laser':
            self._laser_noise = noise.white(self._length, 0.0, parameters, self._fsample)
            self.laser = self._interp(self._laser_noise)
        elif noise_type == 'acc':
            self._acc = noise.acc(self._length, self._fsample, self._lowfz)
            self.acc = self._interp(self._acc)
        elif noise_type == 'oms':
            self._oms = noise.oms(self._length, self._fsample, self._lowfz)
            self.oms = self._interp(self._oms)
        else:
            raise ValueError(f"Invalid noise_type '{noise_type}'. Must be 'laser', 'acc', or 'oms'")

    def update_itfmeter(self) -> None:
        """
        Update phasemeter interpolators.

        Refreshes the interpolators for all three phasemeter outputs (sci, tes, ref)
        after their underlying time series data has been modified by synthesis.

        This method should be called after TJ_synthesis updates the _sci, _tes,
        and _ref arrays to ensure the interpolators reflect the new data.

        Examples
        --------
        >>> ob1._sci = new_science_data  # Update raw data
        >>> ob1.update_itfmeter()        # Refresh interpolators
        """
        for signal in ['sci', 'tes', 'ref']:
            setattr(self, signal, self._interp(getattr(self, f'_{signal}')))

    def update_xi(self) -> None:
        """
        Update xi signal interpolator.

        Refreshes the interpolator for the xi signal (intermediate TDI quantity)
        after the _xi array has been modified.

        Examples
        --------
        >>> ob1._xi = new_xi_data  # Update raw data
        >>> ob1.update_xi()        # Refresh interpolator
        """
        self.xi = self._interp(self._xi)

    def update_eta(self) -> None:
        """
        Update eta signal interpolator.

        Refreshes the interpolator for the eta signal (input to TDI combinations)
        after the _eta array has been modified by synthesis.

        The eta signals are the primary inputs to TDI combinations and represent
        the combined measurement from all noise sources and GW signals.

        Examples
        --------
        >>> ob1._eta = new_eta_data  # Update raw data
        >>> ob1.update_eta()         # Refresh interpolator
        """
        self.eta = self._interp(self._eta)

    def update_laser(self) -> None:
        """
        Update laser noise interpolator.

        Refreshes the interpolator for the laser noise after the _laser array
        has been modified. This is typically called after arm-locking updates
        the laser noise from free-running to locked state.

        Examples
        --------
        >>> ob1._laser = locked_laser_data  # Update to locked laser
        >>> ob1.update_laser()              # Refresh interpolator
        """
        self.laser = self._interp(self._laser)