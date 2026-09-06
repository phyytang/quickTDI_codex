"""
Time Delay Interferometry (TDI) module for laser noise cancellation. (TDI)

This module implements TDI combinations that suppress laser frequency noise
in space-based gravitational wave detectors by combining time-delayed
measurements from multiple arms of the detector constellation. TDI


Key Features :
- Pre-defined TDI channels: X, Y, Z (Michelson), A, E, T (Sagnac) TDIX, Y, Z(), A, E, T()
- Support for first-generation (X1, Y1, Z1) and second-generation (X2, Y2, Z2) (X1, Y1, Z1)(X2, Y2, Z2)
- Flexible and constant arm length modes (delay_level '1' or '0') (delay_level '1''0')
- Helper functions for generating TDI channel definitions TDI

TDI Channel Format TDI:
Each TDI channel is a list of [eta_label, arm_delays, sign]: TDI[eta, , ]:
- eta_label: string identifying optical bench ('1', '2', '3', '1p', '2p', '3p') eta
- arm_delays: list of arms to delay through (e.g., [2, -2] means arm 2 then arm -2) arm
- sign: '+' or '-' for addition/subtraction in the combination 

Available Channels :
- X1, Y1, Z1: First-generation Michelson combinations 
- X2, Y2, Z2: Second-generation Michelson combinations 
- z1, z15, z2: Sagnac-based combinations 
- a1, a15, a2, b1, b15, b2, g1, g15, g2: Monitor and null combinations 

Author : TJ
"""

import numpy as np
from scipy.interpolate import interp1d

from TJ_orbit import orbit

sign_map = {1:'+', -1:'-'}
pm_sign  = {'+':1, '-':-1}
cycle_eta= {'1':'2', '2':'3', '3':'1', '1p':'2p', '2p':'3p', '3p':'1p'}
cycle_arm= {1:2, 2:3, 3:1, -1:-2, -2:-3, -3:-1}

def delay_tdi(delay_op, tdi_channel):
    """
    Apply additional time delays to a TDI channel definition.

    This function prepends arm delay sequences to an existing TDI channel,
    optionally flipping the sign. Used to construct higher-generation TDI
    combinations from simpler building blocks.

    Parameters
    ----------
    delay_op : list of [list, str]
        Delay operations to apply. Each element is [arm_list, sign] where:
        - arm_list: list of arms to prepend (e.g., [2, -2])
        - sign: '+' (keep sign) or '-' (flip sign)
    tdi_channel : list of [str, list, str]
        TDI channel in standard format [eta_label, arms, sign]

    Returns
    -------
    list
        New TDI channel with additional delays applied

    Examples
    --------
    >>> # Add delay through arms [2, -2] to a simple channel
    >>> simple = [['1', [], '+']]
    >>> delayed = delay_tdi([[[2, -2], '+']], simple)
    >>> # Result: [['1', [2, -2], '+']]
    """
    channel=[]
    for delay_arms, pm in delay_op:
        for etas, arms, sign in tdi_channel:
            channel += [[etas, arms+delay_arms, sign_map[pm_sign[sign]*pm_sign[pm]]]]

    return channel

def cycle_tdi(tdi_channel):
    """
    Cyclically permute a TDI channel definition.

    Rotates the spacecraft constellation indices (1→2→3→1, 1p→2p→3p→1p)
    to generate equivalent TDI channels for different arms. Used to construct
    Y and Z channels from X channels through cyclic permutation.

    Parameters
    ----------
    tdi_channel : list of [str, list, str]
        TDI channel in standard format [eta_label, arms, sign]

    Returns
    -------
    list
        New TDI channel with cyclic permutation applied

    Notes
    -----
    The cyclic permutation follows the mapping:
    - Eta signals: '1'→'2'→'3'→'1', '1p'→'2p'→'3p'→'1p'
    - Arms: 1→2→3→1, -1→-2→-3→-1

    Examples
    --------
    >>> # Generate Y1 from X1 through cyclic permutation
    >>> Y1 = cycle_tdi(X1)
    >>> # Generate Z1 from Y1 through another cyclic permutation
    >>> Z1 = cycle_tdi(Y1)
    """
    channel=[]

    for etas, arms, sign in tdi_channel:
        new_arms=[]
        for arm in arms:
            new_arms.append(cycle_arm[arm])

        channel += [[cycle_eta[etas], new_arms, sign]]

    return channel

X1 =[['1p',[],'+'],['3',[-2],'+'],['1',[2,-2],'+'],['2p',[3,2,-2],'+'],
     ['1',[],'-'],['2p',[3],'-'],['1p',[-3,3],'-'],['3',[-2,-3,3],'-']]
Xup =[['1',[],'+'],['2p',[3],'+']]
Xdo =[['1p',[],'+'],['3',[-2],'+']]
Xup2 = delay_tdi([[[2,-2],'+']], Xup) + Xdo
Xdo2 = delay_tdi([[[-3,3],'+']], Xdo) + Xup
X2 = delay_tdi([[[],'-'],[[2,-2,-3,3],'+']], Xup2) + delay_tdi([[[],'+'], [[-3,3,2,-2],'-']], Xdo2)

Y1 = cycle_tdi(X1)
Y2 = cycle_tdi(X2)
Z1 = cycle_tdi(Y1)
Z2 = cycle_tdi(Y2)

z1 =[['1p',[-1],'+'],['2p',[-2],'+'],['3p',[-3],'+'],['1',[1],'-'],['2',[2],'-'],['3',[3],'-']]
a1 =[['1p',[],'+'],['3p',[-2],'+'],['2p',[-1,-2],'+'],['1',[],'-'],['2',[3],'-'],['3',[1,3],'-']]
b1 = cycle_tdi(a1)
g1 = cycle_tdi(b1)

z15 =[['1',[1,-1],'+'],['2',[2,1],'+'],['3',[-3,-1],'+'],['1p',[-1,-2,-3],'+'],['2p',[2,-2,-3],'+'],['3p',[-3,2,3],'+'],
     ['1p',[-1,1],'-'],['2p',[2,1],'-'],['3p',[-3,-1],'-'],['1',[1,2,3],'-'],['2',[2,-2,-3],'-'],['3',[-3,2,3],'-']]
a15 =a1+[['1',[-3,-1,-2],'+'],['2',[3,-3,-1,-2],'+'],['3',[1,3,-3,-1,-2],'+'],
        ['1p',[2,1,3],'-'],['3p',[-2,2,1,3],'-'],['2p',[-1,-2,2,1,3],'-']]
b15 = cycle_tdi(a15)
g15 = cycle_tdi(b15)

aup = [['1',[],'+'],['2',[3],'+'],['3',[1,3],'+']]
ado = [['1p',[],'+'],['3p',[-2],'+'],['2p',[-1,-2],'+']]
aup2= delay_tdi([[[-3,-1,-2],'+']], aup) + ado
ado2= delay_tdi([[[2,1,3],'+']], ado) + aup
a2 = delay_tdi([[[-3,-1,-2,2,1,3],'+'],[[],'-']],aup2) + delay_tdi([[[],'+'],[[2,1,3,-3,-1,-2],'-']],ado2)


b2 = cycle_tdi(a2)
g2 = cycle_tdi(b2)
z2 = ( delay_tdi([[[-3,-1,-2,2,1,3],'+'],[[],'-']], delay_tdi([[[-3,-1,-2],'+'],[[],'-']], delay_tdi([[[1],'+']], X2)))
      +delay_tdi([[[],'+'],[[2,-2,-3,3],'-']], delay_tdi([[[3],'+']],g2)+delay_tdi([[[2],'+']],b2)+delay_tdi([[[3,2],'-']],a2))
      )

U1 = [['2',[],'+'],['3p',[1],'+'],['2p',[-1,1],'+'],['1p',[-3,-1,1],'+'],
      ['2p',[],'-'],['1p',[-3],'-'],['3p',[-2,-3],'-'],['2',[-1,-2,-3],'-']]
V1 = cycle_tdi(U1)
W1 = cycle_tdi(V1)

E1 = [['2',[1,2],'+'],['3p',[2],'+'],['3p',[-1,-3],'-'],['2',[-3],'-'],
      ['1p',[1,-1],'-'],['1', [-1,1],'+'],['1p',[],'+'],['1',[],'-']]
F1 = cycle_tdi(E1)
G1 = cycle_tdi(F1)

P1 = [['2',[1,3],'+'],['3p',[3],'+'],['3p',[-1,-2],'-'],['2',[-2],'-'],
      ['2p',[-2,1,-1],'-'],['3',[3,1,-1],'+'],['2p',[-2],'+'],['3',[3],'-']]
Q1 = cycle_tdi(P1)
R1 = cycle_tdi(Q1)

PL4L1 = E1+P1
PL4L2 = cycle_tdi(PL4L1)
PL4L3 = cycle_tdi(PL4L2)


class tdi():
    """
    Time Delay Interferometry (TDI) class for laser noise suppression.

    Implements TDI combinations that cancel laser frequency noise by combining
    time-delayed phasemeter measurements (eta signals) from multiple arms.
    Supports both first-generation (e.g., X1, Y1, Z1) and second-generation
    (e.g., X2, Y2, Z2) TDI observables.

    Parameters
    ----------
    orbits : orbit
        Orbit object providing time delay functions for signal propagation
    obs : list of ob
        List of optical bench objects [None, ob1, ob2, ob3, ob4, ob5, ob6]
        containing synthesized eta signals for all six arms

    Attributes
    ----------
    _fsample : float
        Sampling frequency in Hz
    _tarray : ndarray
        Time array for simulation
    _length : int
        Number of time samples
    i_ord : int
        Interpolation order (default: 31)
    tdi_arr : ndarray
        Output array for TDI combination result
    orbit : orbit
        Orbit object for computing time delays
    _obs : list
        List of optical bench objects

    Notes
    -----
    TDI works by exploiting the time-delay structure of the detector to create
    combinations where laser noise terms cancel while preserving gravitational
    wave signals. The effectiveness depends on:
    - Accurate time delay calculations
    - High-order interpolation of delayed signals
    - Proper arm length modeling (flexible or constant)

    The run() method is the main interface, accepting any TDI channel definition
    and returning the combined output array.

    Examples
    --------
    >>> # Create TDI object
    >>> td = tdi(orbits, obs)
    >>>
    >>> # Compute first-generation Michelson X channel
    >>> X1_output = td.run(TDI_channel=X1, delay_level='1')
    >>>
    >>> # Compute second-generation Y channel
    >>> Y2_output = td.run(TDI_channel=Y2, delay_level='1')
    >>>
    >>> # Compute Sagnac combination with constant arm length
    >>> a1_output = td.run(TDI_channel=a1, delay_level='0')
    """
    def __init__(self, orbits, obs):
        self._fsample = obs[1]._fsample
        self._tarray =  obs[1]._tarray
        self._length =  len(self._tarray)
        self.i_ord = 31

        self.tdi_arr = np.zeros(self._length)
        self.orbit = orbits
        self._obs = obs
    
    def run(self, TDI_channel=X1, delay_level='1'):
        """
        Compute a TDI combination by summing time-delayed eta signals.

        This is the main method for computing TDI observables. It takes a TDI
        channel definition (list of eta signals with arm delays and signs) and
        combines them to produce the final TDI output that suppresses laser noise.

        Parameters
        ----------
        TDI_channel : list of [str, list, str], optional
            TDI channel definition. Each element is [eta_label, arms, sign]:
            - eta_label: optical bench identifier ('1', '2', '3', '1p', '2p', '3p')
            - arms: list of arms to delay through (e.g., [2, -2])
            - sign: '+' for addition, '-' for subtraction
            Default is X1 (first-generation Michelson X)
        delay_level : {'1', '0'}, optional
            Time delay calculation mode:
            - '1': Flexible arm length (time-varying, uses orbit.delay)
            - '0': Constant arm length (uses orbit.delay_0)
            Default is '1'

        Returns
        -------
        ndarray
            TDI combination output array with laser noise suppressed

        Notes
        -----
        The method implements the general TDI formula:
            TDI = Σᵢ (±1) × ηᵢ(t - Lⱼ(t) - Lₖ(t - Lⱼ) - ...)

        where ηᵢ are the eta signals from each optical bench and Lⱼ are the
        light travel times through arms j.

        Time delays are computed by the orbit object and applied through
        high-order interpolation. The choice of delay_level affects:
        - '1': More accurate for realistic orbits with breathing modes
        - '0': Faster computation, suitable for testing or constant arm analysis

        Examples
        --------
        >>> td = tdi(orbits, obs)
        >>>
        >>> # First-generation Michelson channels
        >>> X1_data = td.run(TDI_channel=X1)
        >>> Y1_data = td.run(TDI_channel=Y1)
        >>> Z1_data = td.run(TDI_channel=Z1)
        >>>
        >>> # Second-generation with flexible arms
        >>> X2_data = td.run(TDI_channel=X2, delay_level='1')
        >>>
        >>> # Sagnac combination with constant arms
        >>> a1_data = td.run(TDI_channel=a1, delay_level='0')
        """
        eta1 = self._obs[1].eta
        eta1p= self._obs[-1].eta

        eta2 = self._obs[2].eta
        eta2p= self._obs[-2].eta

        eta3 = self._obs[3].eta
        eta3p= self._obs[-3].eta

        eta_map={'1':eta1,'1p':eta1p,'2':eta2,'2p':eta2p,'3':eta3,'3p':eta3p}
        pow_map={'+':0,'-':1}

        t_samp = self._tarray

        if delay_level=='1':
            t_delay= self.orbit.delay
        elif delay_level=='0':
            t_delay= self.orbit.delay_0

        self.tdi_arr = np.zeros(self._length)

        #unified function to deal with TDI channels
        for etas, arms, sign in TDI_channel:
            self.tdi_arr += (-1)**pow_map[sign]*eta_map[etas](t_delay(t_samp,arms))

        return self.tdi_arr