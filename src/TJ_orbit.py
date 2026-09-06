"""
Orbit modeling for Taiji spacecraft constellation. 

This module provides analytical and file-based orbit modeling for the three Taiji spacecraft,
including position, velocity, and inter-spacecraft distance calculations. Supports both
heliocentric and geocentric orbit types. 


Key Features :
- Analytical orbit computation based on Kepler equations (heliocentric and geocentric) 
- File-based orbit data loading from CSV files CSV
- Cubic spline interpolation for smooth time-delayed access 
- Time Delay Interferometry (TDI) delay calculations (TDI)

Spacecraft Numbering Convention :
- Spacecraft are numbered 1, 2, 3 1, 2, 3
- Arms connect spacecraft pairs (receiver-sender) -

Arm Numbering Convention :
- Positive arms (1, 2, 3): Forward links (1, 2, 3)
- Negative arms (-1, -2, -3): Reverse links (-1, -2, -3)
- arm_array = [3, -2, -3, 1, 2, -1] corresponds to receiver-sender pairs [12, 13, 21, 23, 31, 32] -

Units :
- Time : seconds 
- Position : light-seconds (natural units where c=1) c=1
- Velocity : dimensionless (v/c) (v/c)
- Distance : light-seconds 

Author : TY
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from TJ_constant import (
    AU, FREQ_ORBIT, R_ORB, DAY,
    XVEC, YVEC, ZVEC,
    ARM_REC, ARM_SEN, RS_ARM, ARM_ARRAY
)

# Create lowercase aliases for backward compatibility with existing code
xvec = XVEC
yvec = YVEC
zvec = ZVEC
armRec = ARM_REC
armSen = ARM_SEN
RSarm = RS_ARM
arm_array = ARM_ARRAY

def sc_pos_analytic(sc_i: int, t: float, armlen: float, orbit_type: str = 'heliocentric') -> NDArray[np.float64]:
    """
    Compute analytical position of spacecraft from Kepler orbit equations. 

    Uses second-order eccentricity expansion for heliocentric orbits or
    geocentric orbit model for Earth-centered reference frame. 
    

    Parameters 
    ----------
    sc_i : int
        Spacecraft number (1, 2, or 3) 123
    t : float
        Time in seconds 
    armlen : float
        LISA arm length in light-seconds (used to compute orbit eccentricity) LISA
    orbit_type : str, optional
        Type of orbit model :
        - 'heliocentric': Sun-centered orbit (default) 
        - 'geocentric': Earth-centered orbit 

    Returns 
    -------
    NDArray[np.float64]
        3D position vector [x, y, z] in light-seconds [x, y, z]

    Notes 
    -----
    Heliocentric orbit uses eccentricity ecc = armlen / (2*sqrt(3)*AU).
    The position is computed to second order in eccentricity using
    Kepler orbit expansion with proper phasing for the three spacecraft.
    ecc = armlen / (2*sqrt(3)*AU)
    

    Geocentric orbit uses Earth orbital parameters with e=0.0167 and
    spacecraft orbital eccentricity e1=0.05.
    e=0.0167e1=0.05
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    if orbit_type == 'heliocentric':
        ecc = armlen/(2*math.sqrt(3)*AU); #print(ecc)

        beta = 2/3*math.pi*(sc_i-1);

        alpha = 2*math.pi*FREQ_ORBIT*t; # np.array(t)

        x = ( R_ORB*math.cos(alpha)+1/2*ecc*R_ORB*(math.cos(2*alpha-beta)-3*math.cos(beta))
            +1/8*ecc**2*R_ORB*( 3*math.cos(3*alpha-2*beta)-10*math.cos(alpha)
                            -5*math.cos(alpha - 2*beta) ) )

        y = ( R_ORB*math.sin(alpha)+1/2*ecc*R_ORB*(math.sin(2*alpha-beta)-3*math.sin(beta))
            +1/8*ecc**2*R_ORB*( 3*math.sin(3*alpha-2*beta)-10*math.sin(alpha)
                            +5*math.sin(alpha - 2*beta) ) )

        z = ( -math.sqrt(3)*ecc*R_ORB*math.cos(alpha-beta)
            +math.sqrt(3)*ecc**2*R_ORB*( (math.cos(alpha-beta))**2
                                        +2*(math.sin(alpha - beta))**2 ) )
    elif orbit_type == 'geocentric':
        cosfs = math.cos(120.5/math.pi)
        sinfs = math.sin(120.5/math.pi)
        cosths= math.cos(-4.7/math.pi)
        sinths= math.sin(-4.7/math.pi)

        alphn = 2*math.pi*(t/(3.65*DAY) + (sc_i-1)/3.)
        betap = 0
        cosalpn = math.cos(alphn - betap)
        sinalpn = math.sin(alphn - betap)
        cos2alpn= math.cos(2*(alphn - betap))

        amb = 2*math.pi*(3.14e-8)*t
        cosamb = math.cos(amb)
        sinamb = math.sin(amb)
        cos2amb= math.cos(2*amb)
        sin2amb= math.sin(2*amb)

        R = AU
        R1 = armlen
        e =0.0167
        e1=0.05


        x = ( R1*(cosfs*sinths*sinalpn + cosalpn*sinfs) + R1*e1*(0.5*(cos2alpn-3)*sinfs+cosalpn*cosfs*sinths*sinalpn) 
            +e1*e1*R1/4.*sinalpn*((3*cos2alpn-1)*cosfs*sinths - 6*cosalpn*sinalpn*sinfs) + R*cosamb + R*e*(cos2amb-3)/2
            -3*R*e*e/2.*cos2amb*sin2amb*sin2amb )

        y = ( R1*(sinfs*sinths*sinalpn - cosalpn*cosfs) - R1*e1*(0.5*(cos2alpn-3)*cosfs-cosalpn*sinfs*sinths*sinalpn) 
            +e1*e1*R1/4.*sinalpn*((3*cos2alpn-1)*sinfs*sinths + 6*cosalpn*sinalpn*cosfs) + R*sinamb + R*e*sin2amb/2
            +R*e*e/4.*(3*cos2amb -1)*sinamb )

        z = (-R1*sinalpn*cosths - R1*e1*cosalpn*sinalpn*cosths - 1/4.*e1*e1*R1*(3*cos2alpn-1)*sinalpn*cosths)
                                            
    return np.array([x,y,z])

def sc_vel_analytic(sc_i: int, t: float, armlen: float, orbit_type: str = 'heliocentric') -> NDArray[np.float64]:
    """
    Compute analytical velocity of spacecraft using numerical differentiation.

    Velocity is computed using centered finite difference of positions
    with step size dt = 1/20 seconds.

    Parameters
    ----------
    sc_i : int
        Spacecraft number (1, 2, or 3)
    t : float
        Time in seconds
    armlen : float
        LISA arm length in light-seconds
    orbit_type : str, optional
        Type of orbit model ('heliocentric' or 'geocentric'), default 'heliocentric'

    Returns
    -------
    NDArray[np.float64]
        3D velocity vector [vx, vy, vz] in dimensionless units (v/c)

    Notes
    -----
    Uses centered difference: v(t) ≈ [pos(t+dt) - pos(t-dt)] / (2*dt)
    with dt = 1/20 seconds.
    """
    inv: float = 20.0
    tplus: float = t + 1/inv
    tminus: float = t - 1/inv
    return (sc_pos_analytic(sc_i, tplus, armlen, orbit_type) -
            sc_pos_analytic(sc_i, tminus, armlen, orbit_type)) * inv / 2

def sc_dis_analytic(sc_i: int, sc_j: int, t: float, armlen: float, orbit_type: str = 'heliocentric') -> float:
    """
    Compute distance between two spacecraft at given time.

    Parameters
    ----------
    sc_i : int
        First spacecraft number (1, 2, or 3)
    sc_j : int
        Second spacecraft number (1, 2, or 3)
    t : float
        Time in seconds
    armlen : float
        LISA arm length in light-seconds
    orbit_type : str, optional
        Type of orbit model ('heliocentric' or 'geocentric'), default 'heliocentric'

    Returns
    -------
    float
        Distance between spacecraft i and j in light-seconds

    Notes
    -----
    Computed as Euclidean norm of position difference: ||pos_i - pos_j||
    """
    dis: NDArray[np.float64] = (sc_pos_analytic(sc_i, t, armlen, orbit_type) -
                                 sc_pos_analytic(sc_j, t, armlen, orbit_type))
    return math.sqrt(np.dot(dis, dis))


def sc_dis_arm_analytic(arm_num: int, t: float, armlen: float, orbit_type: str = 'heliocentric') -> float:
    """
    Compute distance along a specific arm at given time.

    Parameters
    ----------
    arm_num : int
        Arm number (1, 2, 3, -1, -2, or -3)
    t : float
        Time in seconds
    armlen : float
        LISA arm length in light-seconds
    orbit_type : str, optional
        Type of orbit model ('heliocentric' or 'geocentric'), default 'heliocentric'

    Returns
    -------
    float
        Distance from sender to receiver along the arm in light-seconds

    Notes
    -----
    Uses armRec and armSen dictionaries to map arm number to spacecraft pair.
    Positive arms (1,2,3) go in forward direction, negative arms (-1,-2,-3) in reverse.
    """
    sc_i: int = armRec[arm_num]
    sc_j: int = armSen[arm_num]
    dis: NDArray[np.float64] = (sc_pos_analytic(sc_i, t, armlen, orbit_type) -
                                 sc_pos_analytic(sc_j, t, armlen, orbit_type))
    return math.sqrt(np.dot(dis, dis))

def nij(sc_i: int, sc_j: int, t: float, armlen: float,
        orbit_type: str = 'heliocentric', level: int = 0) -> NDArray[np.float64]:
    """
    Compute unit vector along arm from sender j to receiver i.

    Parameters
    ----------
    sc_i : int
        Receiver spacecraft number (1, 2, or 3)
    sc_j : int
        Sender spacecraft number (1, 2, or 3)
    t : float
        Time in seconds at receiver
    armlen : float
        LISA arm length in light-seconds
    orbit_type : str, optional
        Type of orbit model ('heliocentric' or 'geocentric'), default 'heliocentric'
    level : int, optional
        Level of approximation, default 0
        - level=0: Both spacecraft evaluated at same time t
        - level=1: Sender evaluated at retarded time t-armlen

    Returns
    -------
    NDArray[np.float64]
        Unit vector [nx, ny, nz] pointing from sender to receiver

    Notes
    -----
    The level parameter controls whether light travel time is accounted for:
    - level=0: n = (pos_i(t) - pos_j(t)) / ||pos_i(t) - pos_j(t)||
    - level=1: n = (pos_i(t) - pos_j(t-armlen)) / ||pos_i(t) - pos_j(t-armlen)||
    """
    if level == 0:
        nv: NDArray[np.float64] = (sc_pos_analytic(sc_i, t, armlen, orbit_type) -
                                    sc_pos_analytic(sc_j, t, armlen, orbit_type))
    elif level == 1:
        nv = (sc_pos_analytic(sc_i, t, armlen, orbit_type) -
              sc_pos_analytic(sc_j, t - armlen, armlen, orbit_type))
    else:
        raise ValueError(f"Invalid level={level}. Must be 0 or 1")
    return nv / math.sqrt(np.dot(nv, nv))


def vij(sc_i: int, sc_j: int, t: float, armlen: float,
        orbit_type: str = 'heliocentric', level: int = 0) -> NDArray[np.float64]:
    """
    Compute velocity difference between two spacecraft.

    Parameters
    ----------
    sc_i : int
        First spacecraft number (1, 2, or 3)
    sc_j : int
        Second spacecraft number (1, 2, or 3)
    t : float
        Time in seconds
    armlen : float
        LISA arm length in light-seconds
    orbit_type : str, optional
        Type of orbit model ('heliocentric' or 'geocentric'), default 'heliocentric'
    level : int, optional
        Level of approximation, default 0
        - level=0: Both spacecraft evaluated at same time t
        - level=1: Second spacecraft evaluated at retarded time t-armlen

    Returns
    -------
    NDArray[np.float64]
        Velocity difference vector [dvx, dvy, dvz] in dimensionless units (v/c)

    Notes
    -----
    Returns v_i - v_j where:
    - level=0: v_ij = vel_i(t) - vel_j(t)
    - level=1: v_ij = vel_i(t) - vel_j(t-armlen)
    """
    if level == 0:
        nvel: NDArray[np.float64] = (sc_vel_analytic(sc_i, t, armlen, orbit_type) -
                                      sc_vel_analytic(sc_j, t, armlen, orbit_type))
    elif level == 1:
        nvel = (sc_vel_analytic(sc_i, t, armlen, orbit_type) -
                sc_vel_analytic(sc_j, t - armlen, armlen, orbit_type))
    else:
        raise ValueError(f"Invalid level={level}. Must be 0 or 1")
    return nvel

class orbit:
    """
    LISA spacecraft constellation orbit model. LISA

    Provides both analytical (Kepler-based) and file-based orbit data
    with interpolation for arbitrary time queries. Supports heliocentric
    and geocentric reference frames. 
    

    Attributes 
    ----------
    orb_type : str
        Orbit type ('heliocentric' or 'geocentric') ''''
    data_source : str
        Data source ('analytical' or 'file') ''''
    d_int : List[interp1d]
        List of 6 interpolators for inter-spacecraft distances 6
    dint : List[List[Optional[interp1d]]]
        3x3 matrix of distance interpolators indexed by spacecraft numbers 3x3
    di : List[Optional[interp1d]]
        Distance interpolators indexed by arm number [None, 1, 2, 3, -3, -2, -1] 
    pos_int : Optional[List[Tuple[interp1d, interp1d, interp1d]]]
        Position interpolators (only for file-based orbits) 
    vel_int : Optional[List[Tuple[interp1d, interp1d, interp1d]]]
        Velocity interpolators (only for file-based orbits) 

    Examples 
    --------
    Create analytical heliocentric orbit :

    >>> orb = orbit(t_start=0.0, t_end=1000.0, t_step=1.0, tri_arm=10.0)
    >>> pos = orb.position(1, 500.0)  # Position of spacecraft 1 at t=500s t=500s1

    Load orbit from CSV files CSV:

    >>> data_files = {'positions': 'orbit_pos.csv'}
    >>> orb = orbit(tri_arm=10.0, data_source='file', data_files=data_files)
    """

    # Type annotations for attributes that can be None or lists
    pos_int: Optional[List[Tuple[interp1d, interp1d, interp1d]]]
    vel_int: Optional[List[Tuple[interp1d, interp1d, interp1d]]]

    def __init__(self,
                 t_start: float = 0.0,
                 t_end: float = 10000.0,
                 t_step: float = 1.0,
                 tri_arm: float = 10.0,
                 orbit_type: str = 'heliocentric',
                 data_source: str = 'analytical',
                 data_files: Optional[Dict[str, str]] = None,
                 distance_noise_sigma: float = 0.0) -> None:
        """
        Initialize orbit object with either analytical or file-based orbit data. 

        Parameters :
        -----------
        t_start, t_end, t_step : float
            Time range parameters (only used for analytical orbits) 
        tri_arm : float
            Triangle arm length 
        orbit_type : str
            'heliocentric' or 'geocentric' (for analytical orbits) ''''
        data_source : str
            'analytical' - compute from Kepler orbit equations '' - 
            'file' - load from external data files '' - 
        data_files : dict or None
            Dictionary with keys: 'positions', 'velocities', 'distances' 'positions', 'velocities', 'distances'
            Each value is a file path to CSV file. CSV
            Required format for CSV files CSV:
            - positions.csv: time, sc1_x, sc1_y, sc1_z, sc2_x, sc2_y, sc2_z, sc3_x, sc3_y, sc3_z
            - velocities.csv: time, sc1_vx, sc1_vy, sc1_vz, sc2_vx, sc2_vy, sc2_vz, sc3_vx, sc3_vy, sc3_vz
            - distances.csv: time, d_arm3, d_arm-2, d_arm-3, d_arm1, d_arm2, d_arm-1
                            (following arm_array=[3,-2,-3,1,2,-1] ordering) arm_array=[3,-2,-3,1,2,-1]
        distance_noise_sigma : float, optional
            Standard deviation of Gaussian noise to add to inter-spacecraft distances (in light-seconds). 
            Default is 0.0 (no noise). Use this to simulate ranging measurement uncertainties. 0.0
        """
        self.orb_type = orbit_type
        self.data_source = data_source
        self._armlen = tri_arm
        self._ecc = tri_arm/(2*math.sqrt(3)*AU)
        self._distance_noise_sigma = distance_noise_sigma

        if data_source == 'analytical':
            # Original analytical orbit initialization
            self._t_start = t_start
            self._t_end = t_end
            self._t_step= t_step
            self._tarray = np.arange(t_start, t_end, t_step)
            self._tlen = len(self._tarray)

            self._p_arr = np.zeros((3, self._tlen))
            self._v_arr = np.zeros((3, self._tlen))
            self._n_arr = np.zeros((6, self._tlen))
            self._d_arr = np.zeros((6, self._tlen))

            for i in range(6):
                for j in range(self._tlen):
                    self._d_arr[i][j] = sc_dis_arm_analytic(arm_array[i],self._tarray[j],self._armlen, self.orb_type)

            # Add Gaussian noise to distances if requested
            if distance_noise_sigma > 0.0:
                for i in range(6):
                    noise = np.random.normal(0.0, distance_noise_sigma, self._tlen)
                    self._d_arr[i] += noise

            self.d_int=[interp1d(self._tarray,self._d_arr[i],kind=3,fill_value=(self._armlen,self._armlen),bounds_error=False) for i in range(6)]

            # Position and velocity interpolators not created for analytical mode
            self.pos_int = None
            self.vel_int = None

        elif data_source == 'file':
            # Load orbit data from files
            if data_files is None:
                raise ValueError("data_files must be provided when data_source='file'")
            self._load_from_files(data_files)

        else:
            raise ValueError(f"Unknown data_source: {data_source}. Must be 'analytical' or 'file'")

        # Create convenience access structures (same for both modes)
        self.dint = [[None, self.d_int[0], self.d_int[1]],[self.d_int[2], None, self.d_int[3]],[self.d_int[4], self.d_int[5], None]]
        self.di = [None, self.d_int[3], self.d_int[4], self.d_int[0], self.d_int[2], self.d_int[1], self.d_int[5]]
        #di[1][2][3][-3][-2][-1]
        
        # self._dij = np.zeros((3,3,self._tlen))
        # for sc_i in range(3):
        #     for sc_j in range(3):
        #         if sc_i!=sc_j :
        #             for t in range(self._tlen):
        #                 self._dij[sc_i,sc_j,t] = sc_dis_analytic(sc_i+1,sc_j+1,self._tarray[t],self._armlen)

        # self._dij_f =[[0,0,0],[0,0,0],[0,0,0]]
        # for sc_i in range(3):
        #     for sc_j in range(3):
        #         if sc_i!=sc_j :
        #             self._dij_f[sc_i][sc_j] = interp1d(list(self._tarray), list(self._dij[sc_i,sc_j]))


    def _load_from_files(self, data_files: Dict[str, str]) -> None:
        """
        Load orbit data from CSV files and create cubic spline interpolators.

        Parameters
        ----------
        data_files : Dict[str, str]
            Dictionary mapping data type to file path.
            Required keys: 'positions'
            Optional keys: 'velocities', 'distances'

        Raises
        ------
        ValueError
            If 'positions' key is missing from data_files

        Notes
        -----
        CSV File Formats:
        - positions.csv: time, sc1_x, sc1_y, sc1_z, sc2_x, sc2_y, sc2_z, sc3_x, sc3_y, sc3_z
        - velocities.csv: time, sc1_vx, sc1_vy, sc1_vz, sc2_vx, sc2_vy, sc2_vz, sc3_vx, sc3_vy, sc3_vz
        - distances.csv: time, d_arm3, d_arm-2, d_arm-3, d_arm1, d_arm2, d_arm-1

        If velocities are not provided, they will be computed numerically from positions.
        If distances are not provided, they will be computed from positions.
        All interpolators use cubic splines (kind=3).
        """
        # Load positions
        if 'positions' not in data_files:
            raise ValueError("data_files must contain 'positions' key")

        pos_data = np.loadtxt(data_files['positions'], delimiter=',', skiprows=1)
        self._tarray = pos_data[:, 0]
        self._tlen = len(self._tarray)
        self._t_start = self._tarray[0]
        self._t_end = self._tarray[-1]
        self._t_step = self._tarray[1] - self._tarray[0] if self._tlen > 1 else 1.0

        # Reshape positions: columns are [time, sc1_x, sc1_y, sc1_z, sc2_x, sc2_y, sc2_z, sc3_x, sc3_y, sc3_z]
        self._p_arr = np.zeros((3, self._tlen, 3))  # [spacecraft, time, xyz]
        for sc in range(3):
            for coord in range(3):
                self._p_arr[sc, :, coord] = pos_data[:, 1 + sc*3 + coord]

        # Create position interpolators for each spacecraft
        self.pos_int = []
        for sc in range(3):
            # Create 3D interpolator (returns [x, y, z] for given time)
            interp_x = interp1d(self._tarray, self._p_arr[sc, :, 0], kind=3, fill_value='extrapolate', bounds_error=False)
            interp_y = interp1d(self._tarray, self._p_arr[sc, :, 1], kind=3, fill_value='extrapolate', bounds_error=False)
            interp_z = interp1d(self._tarray, self._p_arr[sc, :, 2], kind=3, fill_value='extrapolate', bounds_error=False)
            self.pos_int.append((interp_x, interp_y, interp_z))

        # Load velocities
        if 'velocities' in data_files:
            vel_data = np.loadtxt(data_files['velocities'], delimiter=',', skiprows=1)
            self._v_arr = np.zeros((3, self._tlen, 3))
            for sc in range(3):
                for coord in range(3):
                    self._v_arr[sc, :, coord] = vel_data[:, 1 + sc*3 + coord]

            # Create velocity interpolators
            self.vel_int = []
            for sc in range(3):
                interp_vx = interp1d(self._tarray, self._v_arr[sc, :, 0], kind=3, fill_value='extrapolate', bounds_error=False)
                interp_vy = interp1d(self._tarray, self._v_arr[sc, :, 1], kind=3, fill_value='extrapolate', bounds_error=False)
                interp_vz = interp1d(self._tarray, self._v_arr[sc, :, 2], kind=3, fill_value='extrapolate', bounds_error=False)
                self.vel_int.append((interp_vx, interp_vy, interp_vz))
        else:
            # Compute velocities numerically from positions if not provided
            self.vel_int = None
            print("Warning: Velocities not provided, will be computed numerically from positions")

        # Load distances
        if 'distances' in data_files:
            dist_data = np.loadtxt(data_files['distances'], delimiter=',', skiprows=1)
            self._d_arr = np.zeros((6, self._tlen))
            for i in range(6):
                self._d_arr[i, :] = dist_data[:, 1 + i]

            self.d_int = [interp1d(self._tarray, self._d_arr[i], kind=3, fill_value=(self._armlen, self._armlen), bounds_error=False) for i in range(6)]
        else:
            # Compute distances from positions if not provided
            print("Computing inter-spacecraft distances from position data...")
            self._d_arr = np.zeros((6, self._tlen))
            for i in range(6):
                arm_num = arm_array[i]
                sc_rec = armRec[arm_num]
                sc_sen = armSen[arm_num]
                for j in range(self._tlen):
                    pos_rec = self._p_arr[sc_rec-1, j, :]
                    pos_sen = self._p_arr[sc_sen-1, j, :]
                    diff = pos_rec - pos_sen
                    self._d_arr[i, j] = np.sqrt(np.dot(diff, diff))

            self.d_int = [interp1d(self._tarray, self._d_arr[i], kind=3, fill_value=(self._armlen, self._armlen), bounds_error=False) for i in range(6)]

    def position(self, sc_i: int, t: float) -> NDArray[np.float64]:
        """
        Get position of spacecraft at given time.

        Parameters
        ----------
        sc_i : int
            Spacecraft number (1, 2, or 3)
        t : float
            Time in seconds

        Returns
        -------
        NDArray[np.float64]
            3D position vector [x, y, z] in light-seconds

        Notes
        -----
        For analytical orbits, calls sc_pos_analytic().
        For file-based orbits, uses cubic spline interpolation.
        """
        if self.data_source == 'analytical':
            return sc_pos_analytic(sc_i, t, self._armlen, self.orb_type)
        else:  # file-based
            if self.pos_int is None:
                raise ValueError("Position interpolators not initialized")
            interp_x, interp_y, interp_z = self.pos_int[sc_i - 1]
            return np.array([float(interp_x(t)), float(interp_y(t)), float(interp_z(t))])

    def velocity(self, sc_i: int, t: float) -> NDArray[np.float64]:
        """
        Get velocity of spacecraft at given time.

        Parameters
        ----------
        sc_i : int
            Spacecraft number (1, 2, or 3)
        t : float
            Time in seconds

        Returns
        -------
        NDArray[np.float64]
            3D velocity vector [vx, vy, vz] in dimensionless units (v/c)

        Notes
        -----
        For analytical orbits, calls sc_vel_analytic().
        For file-based orbits:
        - Uses cubic spline interpolation if velocity data was provided
        - Otherwise computes numerically from positions using centered difference
          with dt = 1/20 seconds
        """
        if self.data_source == 'analytical':
            return sc_vel_analytic(sc_i, t, self._armlen, self.orb_type)
        else:  # file-based
            if self.vel_int is not None:
                interp_vx, interp_vy, interp_vz = self.vel_int[sc_i - 1]
                return np.array([float(interp_vx(t)), float(interp_vy(t)), float(interp_vz(t))])
            else:
                # Compute numerically from positions
                dt = 20.0  # time step for numerical derivative
                pos_plus = self.position(sc_i, t + 1/dt)
                pos_minus = self.position(sc_i, t - 1/dt)
                return (pos_plus - pos_minus) * dt / 2

#     def nij(self, sc_rev, sc_sed, t):
#         return nij(sc_rev, sc_sed, t, self._armlen)
    
#     def dij(self, sc_rev, sc_sed, t):
#         dij_f = interp1d(self._tarray, self._dij[sc_rev-1,sc_sed-1])
#         return sc_dis_analytic(sc_rev, sc_sed, t, self._armlen)

#     def nij(self, arm_num, t):
#         sc_rev = armRec[arm_num]
#         sc_sed = armSen[arm_num]
#         return nij(sc_rev, sc_sed, t, self._armlen)
    
#     def dij(self, arm_num, t):
# #       will be updated for general orbit data interpolation
#         sc_rev = armRec[arm_num]
#         sc_sed = armSen[arm_num]
#         return sc_dis_analytic(sc_rev, sc_sed, t, self._armlen) 

    def dij(self, arm_num: int, t: float) -> float:
        """
        Get inter-spacecraft distance along specified arm at given time.

        Parameters
        ----------
        arm_num : int
            Arm number (1, 2, 3, -1, -2, or -3)
        t : float
            Time in seconds

        Returns
        -------
        float
            Distance from sender to receiver in light-seconds

        Notes
        -----
        Uses pre-computed cubic spline interpolators for efficient access.
        Maps arm number to receiver-sender pair using armRec and armSen dictionaries.
        """
        sc_rev: int = armRec[arm_num]
        sc_sed: int = armSen[arm_num]
        d_interp = self.dint[sc_rev - 1][sc_sed - 1]
        assert d_interp is not None, f"No interpolator for arm {arm_num}"
        return d_interp(t)  # type: ignore[return-value]

    def delay_0(self, t_arr: NDArray[np.float64], arm_array: List[int]) -> NDArray[np.float64]:
        """
        Compute time delays using constant arm length approximation.

        Parameters
        ----------
        t_arr : NDArray[np.float64]
            Array of times in seconds
        arm_array : List[int]
            List of arm numbers to traverse (e.g., [2, -2, 3])

        Returns
        -------
        NDArray[np.float64]
            Time-delayed array: t_arr - n_arms * armlen

        Notes
        -----
        This is a simplified delay calculation assuming constant arm length.
        For accurate TDI, use delay() which accounts for time-varying distances.
        """
        temp_arr: NDArray[np.float64] = t_arr.copy()
        for j in range(len(arm_array)):
            temp_arr -= self._armlen
        return temp_arr

    def delay(self, t_arr: NDArray[np.float64], arm_array: List[int]) -> NDArray[np.float64]:
        """
        Compute time-delayed array accounting for time-varying arm lengths.

        This is the key method for Time Delay Interferometry (TDI) calculations.
        It recursively applies light travel time delays through a chain of arms.

        Parameters
        ----------
        t_arr : NDArray[np.float64]
            Array of times in seconds at which to evaluate the delay
        arm_array : List[int]
            Ordered list of arms to traverse (e.g., [2, -2, 3] for TDI combinations)
            Arms are applied in reverse order (rightmost first)

        Returns
        -------
        NDArray[np.float64]
            Time-delayed array where each element is reduced by cumulative light travel times

        Notes
        -----
        The algorithm works backward through arm_array:
        - For arm_array = [a1, a2, a3], applies delays in order a3, a2, a1
        - At each step, queries interpolated distance dij(arm, t)
        - Delays accumulate: t_final = t - L(a3,t) - L(a2,t-L(a3,t)) - L(a1,...)

        This implements the TDI delay operator crucial for laser noise cancellation.

        Implementation uses vectorized interpolation for performance:
        - Calls interpolators on entire arrays rather than element-by-element
        - Significantly faster for large t_arr (100x+ speedup for 100k elements)

        Examples
        --------
        >>> orb = orbit(t_start=0, t_end=1000, t_step=1, tri_arm=10.0)
        >>> t = np.array([100.0, 200.0, 300.0])
        >>> delayed_t = orb.delay(t, [2, -2])  # Two-way delay along arm 2
        """
        temp_arr: NDArray[np.float64] = t_arr.copy()

        # Old implementation (element-by-element, slow for large arrays):
        # for j in range(len(arm_array)):
        #     for arr_i in range(len(temp_arr)):
        #         temp_arr[arr_i] -= self.dij(arm_array[-(j + 1)], temp_arr[arr_i])

        # New vectorized implementation (much faster):
        for j in range(len(arm_array)):
            arm_num = arm_array[-(j + 1)]
            sc_rev: int = armRec[arm_num]
            sc_sed: int = armSen[arm_num]
            d_interp = self.dint[sc_rev - 1][sc_sed - 1]
            assert d_interp is not None, f"No interpolator for arm {arm_num}"
            # Vectorized interpolation call - operates on entire array at once
            temp_arr -= d_interp(temp_arr)  # type: ignore[arg-type]

        return temp_arr