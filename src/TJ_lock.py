"""
Laser frequency stabilization module for Taiji/LISA. /LISA

Implements arm-locking algorithms to suppress laser frequency noise by using
delayed feedback from spacecraft arm measurements. 

Key Features :
- Single-arm locking: Uses one arm's delayed signal for feedback 
- Dual-arm locking: Uses two arms with common-mode rejection 
- Common-arm locking: Uses two arms with backforward rule 
- Automatic low-pass filtering of laser noise 
- Automatic propagation of locked laser to all optical bench objects 
- Integration with TJ_ob optical bench objects and TJ_orbit orbit objects TJ_obTJ_orbit

Locking Algorithms :
- Single-arm : Backward rule or Trapezoidal rule 
- Dual-arm : Bilinear (Tustin) transform for stability (Tustin)
- Common-arm : Backforward rule for discrete-time filtering 

Author : TY
"""

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from typing import Literal

from TJ_ob import ob


class lock:
    """
    Laser frequency stabilization using arm-locking. 

    This class implements arm-locking algorithms that use time-delayed feedback
    from spacecraft arm measurements to suppress laser frequency noise. The run()
    method automatically updates all optical bench objects in the obs list with
    the locked laser noise. 
    run()obs

    Parameters
    ----------
    orbits : orbit
        Orbit object providing spacecraft positions and time delays
    obs : list of ob
        List of optical bench objects [None, ob1, ob2, ob3, ob4, ob5, ob6]
        The master laser (obs[-1]) will be locked and propagated to all others
    gws : gw
        Gravitational wave object containing GW signal information
    lock_type : {'single', 'dual'}, optional
        Type of arm locking:
        - 'single': Single-arm locking using one delayed arm
        - 'dual': Dual-arm locking using two arms with different delays
        Default is 'single'
    method : str, optional
        Integration method for single-arm: 'backward' or 'trapezoidal'
        For dual-arm, always uses bilinear (Tustin) transform
        Default is 'trapezoidal'
    arm_delay : float, optional
        Time delay in seconds for single-arm locking
        Default is 20.0 seconds
    arm1_delay : float, optional
        First arm delay in seconds for dual-arm locking
        Default is 20.1 seconds
    arm2_delay : float, optional
        Second arm delay in seconds for dual-arm locking
        Default is 19.9 seconds
    gfactor : float, optional
        Loop gain for the feedback system
        Default is 10000.0
    afactor : float, optional
        Proportional gain factor
        Default is 100.0
    w0 : float, optional
        Corner frequency (rad/s) for dual-arm locking filters
        Default is 2*π*0.5 rad/s
    lowpass_cutoff : float, optional
        Low-pass filter cutoff frequency in Hz (applied to free-running laser)
        Default is 1.0 Hz
    lowpass_order : int, optional
        Order of Butterworth low-pass filter
        Default is 6

    Attributes
    ----------
    locked_laser : ndarray
        The locked laser noise time series
    free_laser : ndarray
        The original free-running laser noise (after low-pass filtering)
    master_ob : ob
        Reference to the master optical bench (obs[-1])
    obs : list
        Reference to the full optical bench list
    orb : orbit
        Reference to the orbit object
    gws : gw
        Reference to the gravitational wave object

    Examples
    --------
    Single-arm locking:
    >>> lk = lock(orbits, obs, gws, lock_type='single', arm_delay=20.0,
    ...           gfactor=10000.0, afactor=100.0)
    >>> lk.run()  # Automatically updates all obs objects

    Dual-arm locking:
    >>> lk = lock(orbits, obs, gws, lock_type='dual', arm1_delay=20.1, arm2_delay=19.9,
    ...           gfactor=10000.0, afactor=100.0, w0=2*np.pi*0.5)
    >>> locked_laser = lk.run()  # Returns locked laser and updates all obs objects
    """

    def __init__(self,
                 orbits,
                 obs,
                 gws,
                 lock_type: Literal['single', 'dual', 'common'] = 'single',
                 method: str = 'trapezoidal',
                 arm_delay: float = 20.0,
                 arm1_delay: float = 20.1,
                 arm2_delay: float = 19.9,
                 gfactor: float = 10000.0,
                 afactor: float = 100.0,
                 w0: float = 2 * np.pi * 0.4,
                 lowpass_cutoff: float = 1.0,
                 lowpass_order: int = 6):
        
        self.obs = obs  # Store the full obs list
        self.orb = orbits # Store the orbit object
        self.gws = gws # Store the gw object

        master_ob = obs[-1]
        self.master_ob = master_ob
        self.lock_type = lock_type
        self.method = method

        # OPTIMIZATION: Check if laser noise is enabled 
        # If no laser noise, skip locking (no point locking zero noise) 
        self.has_laser = any(obs[i].hasLaser for i in range(1, 7) if obs[i] is not None)

        # Get time array and sampling parameters from optical bench
        self._tarray = master_ob._tarray
        self._fsample = master_ob._fsample
        self._length = master_ob._length
        self._duration = master_ob._duration
        self.t_s = 1.0 / self._fsample  # sampling time step

        # Locking parameters
        self.gfactor = gfactor
        self.afactor = afactor
        self.w0 = w0

        # Arm delay parameters
        self.arm_delay = arm_delay
        self.arm1_delay = arm1_delay  # for dual-arm (longer delay)
        self.arm2_delay = arm2_delay  # for dual-arm (shorter delay)

        # Low-pass filter parameters
        self.lowpass_cutoff = lowpass_cutoff
        self.lowpass_order = lowpass_order

        # Storage for laser time series
        self.free_laser = np.zeros(self._length)
        self.locked_laser = np.zeros(self._length)
        
        self.free_laser = self.master_ob._laser_noise

        # Apply low-pass filter to original laser noise
        # self._apply_lowpass_filter()

    def _apply_lowpass_filter(self):
        """Apply Butterworth low-pass filter to free-running laser noise."""
        nyquist = 0.5 * self._fsample
        normal_cutoff = self.lowpass_cutoff / nyquist

        b, a = signal.butter(self.lowpass_order, normal_cutoff, btype='low', analog=False)

        # Apply zero-phase filtering
        self.free_laser = signal.filtfilt(b, a, self.master_ob._laser_noise)

    def run(self):
        """
        Execute the arm-locking algorithm.

        This method runs the selected locking algorithm and automatically updates
        all optical bench objects in the obs list with the locked laser noise:
        - obs[-1]: Master optical bench (directly locked)
        - obs[1], obs[2], obs[3], obs[-2], obs[-3]: All updated with locked laser

        Returns
        -------
        locked_laser : ndarray
            The locked laser noise time series (also stored in all obs objects)
        """
        # OPTIMIZATION: Skip locking if no laser noise is present 
        if not self.has_laser:
            print("Laser locking skipped: No laser noise enabled ")
            self.locked_laser = self.free_laser.copy()
            self._update_master_ob()
            self._phase_lock()
            return self.locked_laser

        if self.lock_type == 'single':
            if self.method == 'backward':
                self._run_single_arm_backward()
            elif self.method == 'trapezoidal':
                self._run_single_arm_trapezoidal()
            else:
                raise ValueError(f"Unknown method '{self.method}' for single-arm locking. "
                               "Use 'backward' or 'trapezoidal'.")
        elif self.lock_type == 'dual':
            self._run_dual_arm_bilinear()
        elif self.lock_type == 'common':
            self._run_common_arm()            
        # The unfinished modified-dual algorithm is intentionally disabled.
        # elif self.lock_type == 'modified_dual':
        #     self._run_modified_dual_arm()
        else:
            raise ValueError(f"Unknown lock_type '{self.lock_type}'. Use 'single' or 'dual' or 'common'.")

        # Update the master optical bench
        self._update_master_ob()

        # Update all other laser noises
        # self._update_laser()
        self._phase_lock()

        return self.locked_laser

    def _run_single_arm_backward(self):
        """Single-arm locking using backward Euler integration."""
        delay_samples = int(self.arm_delay * self._fsample)

        for i in range(self._length - 1):
            if i < delay_samples + 1:
                self.locked_laser[i] = self.free_laser[i]
            else:
                # Backward rule integration
                self.locked_laser[i] = (
                    (1 + self.gfactor) * self.locked_laser[i - 1]
                    + (1 + self.gfactor + self.t_s * self.gfactor * self.afactor) * self.locked_laser[i - delay_samples]
                    - (1 + self.gfactor) * self.locked_laser[i - delay_samples - 1]
                    + (self.free_laser[i] - self.free_laser[i - 1])
                ) / ((1 + self.gfactor) + self.t_s * self.gfactor * self.afactor)

    def _run_single_arm_trapezoidal(self):
        """Single-arm locking using trapezoidal integration."""
        delay_samples = int(self.arm_delay * self._fsample)

        for i in range(self._length - 1):
            if i < delay_samples + 1:
                self.locked_laser[i] = self.free_laser[i]
            else:
                # Trapezoidal rule integration
                self.locked_laser[i] = (
                    (1 + self.gfactor - self.t_s * self.gfactor * self.afactor / 2) * self.locked_laser[i - 1]
                    + (1 + self.gfactor + self.t_s * self.gfactor * self.afactor / 2) * self.locked_laser[i - delay_samples]
                    - (1 + self.gfactor - self.t_s * self.gfactor * self.afactor / 2) * self.locked_laser[i - delay_samples - 1]
                    + (self.free_laser[i] - self.free_laser[i - 1])
                ) / ((1 + self.gfactor) + self.t_s * self.gfactor * self.afactor / 2)

    def _run_common_arm(self):
        """Common-arm locking using backforward rule."""
        # Calculate arm parameters
        delay1_samples = int(self.arm1_delay * self._fsample)
        delay2_samples = int(self.arm2_delay * self._fsample)

        max_delay = max(delay1_samples, delay2_samples)

        # Compute filter coefficients using backforward rule
        # These implement discrete-time approximations of the continuous filters

        # Locked laser coefficients (denominator)
        l_factor_0 = (1 + 2 * self.gfactor + 2 * self.gfactor * self.afactor * self.t_s)
        l_factor_1 = (1 + 2 * self.gfactor)

        # Free laser pass-through coefficients (numerator for laser input)
        p_factor_0 =  1
        p_factor_1 = -1

        # Delayed arm1 coefficients (numerator for first delayed arm)
        d_factor_0 = (1+self.afactor*self.t_s)
        d_factor_1 = -1

        # Delayed arm2 coefficients (numerator for second delayed arm)
        f_factor_0 = (1+self.afactor*self.t_s)
        f_factor_1 = -1


        # Main iteration loop
        for i in range(self._length):
            # Initial settling period: use free-running laser
            if i < max_delay + 25 * self._fsample:
                self.locked_laser[i] = self.free_laser[i]
            else:
                # Calculate delayed indices
                j = int(i - delay1_samples)  # arm1 delay index
                k = int(i - delay2_samples)  # arm2 delay index

                # Compute locked laser using bilinear transform
                self.locked_laser[i] = (
                    # Previous locked laser states (IIR part)
                    (l_factor_1 * self.locked_laser[i - 1]) / l_factor_0

                    # Free-running laser contribution
                    + (p_factor_0 * self.free_laser[i]
                       + p_factor_1 * self.free_laser[i - 1]) / l_factor_0

                    # Delayed arm1 feedback
                    + (d_factor_0 * self.locked_laser[j]
                       + d_factor_1 * self.locked_laser[j - 1]) * (self.gfactor / l_factor_0)

                    # Delayed arm2 feedback
                    + (f_factor_0 * self.locked_laser[k]
                       + f_factor_1 * self.locked_laser[k - 1]) * (self.gfactor / l_factor_0)
                )


    def _run_dual_arm_bilinear(self):
        """Dual-arm locking using bilinear (Tustin) transform."""
        # Calculate arm parameters
        dt = (self.arm1_delay - self.arm2_delay) / 2.0  # half delay difference
        tb = (self.arm1_delay + self.arm2_delay) / 2.0  # average delay

        delay1_samples = int(self.arm1_delay * self._fsample)
        delay2_samples = int(self.arm2_delay * self._fsample)
        max_delay = max(delay1_samples, delay2_samples)

        # Compute filter coefficients using bilinear transform
        # These implement discrete-time approximations of the continuous filters

        # Locked laser coefficients (denominator)
        l_factor_0 = (8 * (1 + 2 * self.gfactor)
                     + 4 * self.t_s * ((1 + 2 * self.gfactor) * self.w0 + 2 * self.gfactor * self.afactor)
                     + 2 * self.t_s**2 * (2 * self.gfactor * self.afactor * self.w0))

        l_factor_1 = (-24 * (1 + 2 * self.gfactor)
                     - 4 * self.t_s * ((1 + 2 * self.gfactor) * self.w0 + 2 * self.gfactor * self.afactor)
                     + 2 * self.t_s**2 * (2 * self.gfactor * self.afactor * self.w0))

        l_factor_2 = (24 * (1 + 2 * self.gfactor)
                     - 4 * self.t_s * ((1 + 2 * self.gfactor) * self.w0 + 2 * self.gfactor * self.afactor)
                     - 2 * self.t_s**2 * (2 * self.gfactor * self.afactor * self.w0))

        l_factor_3 = (-8 * (1 + 2 * self.gfactor)
                     + 4 * self.t_s * ((1 + 2 * self.gfactor) * self.w0 + 2 * self.gfactor * self.afactor)
                     - 2 * self.t_s**2 * (2 * self.gfactor * self.afactor * self.w0))

        # Free laser pass-through coefficients (numerator for laser input)
        p_factor_0 =   8 + 4 * self.t_s * self.w0
        p_factor_1 = -24 - 4 * self.t_s * self.w0
        p_factor_2 =  24 - 4 * self.t_s * self.w0
        p_factor_3 =  -8 + 4 * self.t_s * self.w0

        # Delayed arm1 coefficients (numerator for first delayed arm)
        d_factor_0 = (8
                     + 4 * self.t_s * (self.w0 + self.afactor)
                     + 2 * self.t_s**2 * (self.afactor * self.w0 + self.w0 / dt)
                     + self.t_s**3 * self.afactor * self.w0 / dt)

        d_factor_1 = (-24
                     - 4 * self.t_s * (self.w0 + self.afactor)
                     + 2 * self.t_s**2 * (self.afactor * self.w0 + self.w0 / dt)
                     + 3 * self.t_s**3 * self.afactor * self.w0 / dt)

        d_factor_2 = (24
                     - 4 * self.t_s * (self.w0 + self.afactor)
                     - 2 * self.t_s**2 * (self.afactor * self.w0 + self.w0 / dt)
                     + 3 * self.t_s**3 * self.afactor * self.w0 / dt)

        d_factor_3 = (-8
                     + 4 * self.t_s * (self.w0 + self.afactor)
                     - 2 * self.t_s**2 * (self.afactor * self.w0 + self.w0 / dt)
                     + self.t_s**3 * self.afactor * self.w0 / dt)

        # Delayed arm2 coefficients (numerator for second delayed arm)
        f_factor_0 = (8
                     + 4 * self.t_s * (self.w0 + self.afactor)
                     + 2 * self.t_s**2 * (self.afactor * self.w0 - self.w0 / dt)
                     - self.t_s**3 * self.afactor * self.w0 / dt)

        f_factor_1 = (-24
                     - 4 * self.t_s * (self.w0 + self.afactor)
                     + 2 * self.t_s**2 * (self.afactor * self.w0 - self.w0 / dt)
                     - 3 * self.t_s**3 * self.afactor * self.w0 / dt)

        f_factor_2 = (24
                     - 4 * self.t_s * (self.w0 + self.afactor)
                     - 2 * self.t_s**2 * (self.afactor * self.w0 - self.w0 / dt)
                     - 3 * self.t_s**3 * self.afactor * self.w0 / dt)

        f_factor_3 = (-8
                     + 4 * self.t_s * (self.w0 + self.afactor)
                     - 2 * self.t_s**2 * (self.afactor * self.w0 - self.w0 / dt)
                     - self.t_s**3 * self.afactor * self.w0 / dt)

        # Main iteration loop
        for i in range(self._length):
            # Initial settling period: use free-running laser
            if i < max_delay + 25 * self._fsample:
                self.locked_laser[i] = self.free_laser[i]
            else:
                # Calculate delayed indices
                j = int(i - delay1_samples)  # arm1 delay index
                k = int(i - delay2_samples)  # arm2 delay index

                # Compute locked laser using bilinear transform
                self.locked_laser[i] = (
                    # Previous locked laser states (IIR part)
                    -(l_factor_1 * self.locked_laser[i - 1]
                      + l_factor_2 * self.locked_laser[i - 2]
                      + l_factor_3 * self.locked_laser[i - 3]) / l_factor_0

                    # Free-running laser contribution
                    + (p_factor_0 * self.free_laser[i]
                       + p_factor_1 * self.free_laser[i - 1]
                       + p_factor_2 * self.free_laser[i - 2]
                       + p_factor_3 * self.free_laser[i - 3]) / l_factor_0

                    # Delayed arm1 feedback
                    + (d_factor_0 * self.locked_laser[j]
                       + d_factor_1 * self.locked_laser[j - 1]
                       + d_factor_2 * self.locked_laser[j - 2]
                       + d_factor_3 * self.locked_laser[j - 3]) * (self.gfactor / l_factor_0)

                    # Delayed arm2 feedback
                    + (f_factor_0 * self.locked_laser[k]
                       + f_factor_1 * self.locked_laser[k - 1]
                       + f_factor_2 * self.locked_laser[k - 2]
                       + f_factor_3 * self.locked_laser[k - 3]) * (self.gfactor / l_factor_0)
                )

    # def _run_modified_dual_arm(self):
        # """Modified dual-arm locking algorithm (not implemented yet)."""
        # g=self.gfactor
        # a=self.afactor
        # w0=self.w0
        # T=self.t_s

        # dt=(self.arm1_delay - self.arm2_delay)/2.0  # half delay difference
        # dt1 = 1.0/dt
        # tb=(self.arm1_delay + self.arm2_delay)/2.0  # average
        # delay1_samples = int(self.arm1_delay * self._fsample)
        # delay2_samples = int(self.arm2_delay * self._fsample)
        # max_delay = max(delay1_samples, delay2_samples)

        # ga = 3.3
        # gb = 1.
        # gc = 1.
        # gd = 1.
        # ge = 1.
        # gf = gb*gc*gd*ge

        # wa = 0.
        # wb = 2*np.pi*0.9e-3
        # wc = 2*np.pi*0.9e-3
        # wd = 2*np.pi*0.9e-3
        # we = 2*np.pi*0.9e-3

        # l_factor = [1,0,0,0,0,0,0,0]
        
        # p_factor = [1,0,0,0,0,0,0,0]

        # d_factor = [1,0,0,0,0,0,0,0]

        # f_factor = [1,0,0,0,0,0,0,0]


        # for i in range(self._length):
            # if i < max_delay + 25 * self._fsample:
                # self.locked_laser[i] = self.free_laser[i]
            # else:
                # j = int(i - delay1_samples)  # arm1 delay index
                # k = int(i - delay2_samples)  # arm2 delay index

                 # # Compute locked laser using bilinear transform
                # self.locked_laser[i] = (
                    # # Previous locked laser states (IIR part)
                    # -(l_factor[1] * self.locked_laser[i - 1]+
                      # l_factor[2] * self.locked_laser[i - 2]+
                      # l_factor[3] * self.locked_laser[i - 3]+
                      # l_factor[4] * self.locked_laser[i - 4]+
                      # l_factor[5] * self.locked_laser[i - 5]+
                      # l_factor[6] * self.locked_laser[i - 6]+
                      # l_factor[7] * self.locked_laser[i - 7]) / l_factor[0]

                    # # Free-running laser contribution
                    # + (p_factor[0] * self.free_laser[i]+
                       # p_factor[1] * self.free_laser[i-1]+
                       # p_factor[2] * self.free_laser[i-2]+
                       # p_factor[3] * self.free_laser[i-3]+
                       # p_factor[4] * self.free_laser[i-4]+
                       # p_factor[5] * self.free_laser[i-5]+
                       # p_factor[6] * self.free_laser[i-6]+
                       # p_factor[7] * self.free_laser[i-7]) / l_factor[0]

                    # # Delayed arm1 feedback
                    # + (d_factor[0] * self.locked_laser[j]+
                       # d_factor[1] * self.locked_laser[j - 1]+
                       # d_factor[2] * self.locked_laser[j - 2]+
                       # d_factor[3] * self.locked_laser[j - 3]+
                       # d_factor[4] * self.locked_laser[j - 4]+
                       # d_factor[5] * self.locked_laser[j - 5]+
                       # d_factor[6] * self.locked_laser[j - 6]+
                       # d_factor[7] * self.locked_laser[j - 7] ) / l_factor[0]

                    # # Delayed arm2 feedback
                    # + (f_factor[0] * self.locked_laser[k]+
                       # f_factor[1] * self.locked_laser[k - 1]+
                       # f_factor[2] * self.locked_laser[k - 2]+
                       # f_factor[3] * self.locked_laser[k - 3]+
                       # f_factor[4] * self.locked_laser[k - 4]+
                       # f_factor[5] * self.locked_laser[k - 5]+
                       # f_factor[6] * self.locked_laser[k - 6]+
                       # f_factor[7] * self.locked_laser[k - 7]) / l_factor[0]
                # ) 


    # def _run_modified_dual_arm_backward(self):
        # """Modified dual-arm locking algorithm (not implemented yet)."""
        # g=self.gfactor
        # a=self.afactor
        # w0=self.w0
        # T=self.t_s

        # dt=(self.arm1_delay - self.arm2_delay)/2.0  # half delay difference
        # dt1 = 1.0/dt
        # tb=(self.arm1_delay + self.arm2_delay)/2.0  # average
        # delay1_samples = int(self.arm1_delay * self._fsample)
        # delay2_samples = int(self.arm2_delay * self._fsample)
        # max_delay = max(delay1_samples, delay2_samples)

        # ga = 3.3
        # gb = 1.
        # gc = 1.
        # gd = 1.
        # ge = 1.
        # gf = gb*gc*gd*ge

        # wa = 0.
        # wb = 2*np.pi*0.9e-6
        # wc = 2*np.pi*0.9e-6
        # wd = 2*np.pi*0.9e-6
        # we = 2*np.pi*0.9e-6

        # l_factor = [1,0,0,0,0,0,0,0]
        
        # p_factor = [1,0,0,0,0,0,0,0]

        # d_factor = [1,0,0,0,0,0,0,0]

        # f_factor = [1,0,0,0,0,0,0,0]

        # for i in range(self._length):
            # if i < max_delay + 25 * self._fsample:
                # self.locked_laser[i] = self.free_laser[i]
            # else:
                # j = int(i - delay1_samples)  # arm1 delay index
                # k = int(i - delay2_samples)  # arm2 delay index

                 # # Compute locked laser using bilinear transform
                # self.locked_laser[i] = (
                    # # Previous locked laser states (IIR part)
                    # -(l_factor[1] * self.locked_laser[i - 1]+
                      # l_factor[2] * self.locked_laser[i - 2]+
                      # l_factor[3] * self.locked_laser[i - 3]+
                      # l_factor[4] * self.locked_laser[i - 4]+
                      # l_factor[5] * self.locked_laser[i - 5]+
                      # l_factor[6] * self.locked_laser[i - 6]+
                      # l_factor[7] * self.locked_laser[i - 7]) / l_factor[0]

                    # # Free-running laser contribution
                    # + (p_factor[0] * self.free_laser[i]+
                       # p_factor[1] * self.free_laser[i-1]+
                       # p_factor[2] * self.free_laser[i-2]+
                       # p_factor[3] * self.free_laser[i-3]+
                       # p_factor[4] * self.free_laser[i-4]+
                       # p_factor[5] * self.free_laser[i-5]+
                       # p_factor[6] * self.free_laser[i-6]+
                       # p_factor[7] * self.free_laser[i-7]) / l_factor[0]

                    # # Delayed arm1 feedback
                    # + (d_factor[0] * self.locked_laser[j]+
                       # d_factor[1] * self.locked_laser[j - 1]+
                       # d_factor[2] * self.locked_laser[j - 2]+
                       # d_factor[3] * self.locked_laser[j - 3]+
                       # d_factor[4] * self.locked_laser[j - 4]+
                       # d_factor[5] * self.locked_laser[j - 5]+
                       # d_factor[6] * self.locked_laser[j - 6]+
                       # d_factor[7] * self.locked_laser[j - 7] ) / l_factor[0]

                    # # Delayed arm2 feedback
                    # + (f_factor[0] * self.locked_laser[k]+
                       # f_factor[1] * self.locked_laser[k - 1]+
                       # f_factor[2] * self.locked_laser[k - 2]+
                       # f_factor[3] * self.locked_laser[k - 3]+
                       # f_factor[4] * self.locked_laser[k - 4]+
                       # f_factor[5] * self.locked_laser[k - 5]+
                       # f_factor[6] * self.locked_laser[k - 6]+
                       # f_factor[7] * self.locked_laser[k - 7]) / l_factor[0]
                # ) 





    def _update_master_ob(self):
        """Update the master optical bench with locked laser."""
        # Update the internal arrays
        self.master_ob._lock_laser = self.locked_laser.copy()
        self.master_ob._laser = self.locked_laser.copy()

        # Update the interpolator
        self.master_ob.laser = interp1d(
            self._tarray,
            self.locked_laser,
            kind=self.master_ob.i_ord,
            fill_value=(1e-99, 1e-99),
            bounds_error=False
        )

    def _update_laser(self):
        """
        Update all optical bench objects with the locked laser noise.

        Propagates the locked laser from obs[-1] to all other optical benches:
        obs[1], obs[2], obs[3] (forward arms) and obs[-2], obs[-3] (reverse arms).
        Calls update_laser() on each to refresh their interpolators.
        """
        for idx in [1, 2, 3, -2, -3]:
            if self.obs[idx] is not None:
                self.obs[idx]._laser = self.locked_laser.copy()
                self.obs[idx].update_laser()

    def _phase_lock(self, phase_lock='N1', delay_level='1'):
        """
        set the laser noise for phase lock.

        Parameters
        ----------
        phase_lock : {'N1', 'N2'}, optional
            Type of phase locking scheme:
            - 'N1': Locking 
        """
        if delay_level=='1':
            t_delay= self.orb.delay
        elif delay_level=='0':
            t_delay= self.orb.delay_0

        obs=self.obs
        GWs=self.gws

        # N1, laser_1p' as the master laser, obs[-1].laser
        if phase_lock == 'N1':
            obs[1]._laser = obs[-1]._laser - 2*obs[1]._acc
            obs[1].update_laser()
            
            obs[-2]._laser = obs[1].laser(t_delay(self._tarray,[-3])) + GWs.gw[-2](self._tarray) + obs[-2]._oms
            obs[-2].update_laser()
            obs[2]._laser = obs[-2]._laser - 2*obs[2]._acc
            obs[2].update_laser() 

            obs[3]._laser = obs[-1].laser(t_delay(self._tarray,[2])) + GWs.gw[3](self._tarray) + obs[3]._oms
            obs[3].update_laser()
            obs[-3]._laser = obs[3]._laser - 2*obs[-3]._acc
            obs[-3].update_laser()
