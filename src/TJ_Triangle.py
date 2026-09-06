"""
Triangle constellation simulation class for Taiji/LISA. /LISA

Provides a high-level interface that encapsulates the complete simulation pipeline:
orbit → gravitational waves → optical benches → laser locking → signal synthesis → TDI.
 →  →  →  →  → TDI

This class simplifies the workflow by managing all components in a single object,
making it easy to configure parameters, run simulations, and access results.


Author : Based on test.ipynb workflow
"""

from typing import List, Dict, Optional, Literal
import numpy as np
from numpy.typing import NDArray

import TJ_orbit
import TJ_gw
import TJ_ob
import TJ_lock
import TJ_synthesis
import TJ_tdi


class Triangle:
    """
    Triangle constellation simulation for Taiji/LISA. /LISA

    This class provides a unified interface for setting up and running complete
    TDI simulations, including orbit modeling, GW sources, noise generation,
    laser locking, signal synthesis, and TDI processing. 
    TDITDI

    Parameters 
    ----------
    t_start : float, optional
        Start time in seconds. Default is 0.0. 0.0
    t_end : float, optional
        End time in seconds. Default is 40000.0. 40000.0
    tri_arm : float, optional
        Triangle arm length in light-seconds. Default is 10.0 (≈3×10⁶ km).
        10.0≈3×10⁶
    orbit_type : {'heliocentric', 'geocentric'}, optional
        Type of orbit model. Default is 'heliocentric'. ''
    fsample_ob : float, optional
        Sampling frequency for optical benches in Hz. Default is 5.0 Hz.
        Hz5.0 Hz
    fsample_gw : float, optional
        Sampling frequency for GW signal in Hz. Default is 0.1 Hz.
        Hz0.1 Hz

    Attributes 
    ----------
    orbit : TJ_orbit.orbit
        Orbit object for spacecraft constellation. 
    gw : TJ_gw.gw
        Gravitational wave signal object. 
    obs : list of TJ_ob.ob
        List of 6 optical bench objects [None, ob1, ob2, ob3, ob-3, ob-2, ob-1].
        6
    lock : TJ_lock.lock or None
        Laser locking object (None if locking not applied). None
    synthesis : TJ_synthesis.synthesis or None
        Signal synthesis object (None if not synthesized). None
    tdi : TJ_tdi.tdi or None
        TDI processing object (None if not initialized). TDINone
    tdi_results : dict
        Dictionary storing TDI channel results. TDI

    Examples 
    --------
    Basic usage with single GW source :

    >>> tri = Triangle(t_start=0, t_end=40000, tri_arm=10.0)
    >>> tri.add_gw_source(fgw=0.007, strain=1e-23, beta=0, lamda=0, psi=0)
    >>> tri.setup_optical_benches(hasLaser=True, hasAcc=False, hasOms=False)
    >>> tri.apply_laser_lock(lock_type='dual', arm1_delay=20.1, arm2_delay=19.9)
    >>> tri.synthesize_signals(delay_level='1')
    >>> X1 = tri.run_tdi(TJ_tdi.X1, delay_level='1')

    Multiple GW sources :

    >>> tri = Triangle(t_start=0, t_end=40000)
    >>> sources = [
    ...     {'fgw': 0.003, 'strain': 1e-23, 'beta': 0, 'lamda': 0, 'psi': 0},
    ...     {'fgw': 0.007, 'strain': 8e-24, 'beta': np.pi/3, 'lamda': np.pi/2, 'psi': np.pi/4}
    ... ]
    >>> tri.set_gw_sources(sources)
    >>> tri.setup_optical_benches()
    >>> tri.run_full_pipeline()  # Runs locking, synthesis, and TDI

    Frequency sweep :

    >>> tri = Triangle()
    >>> freq_array = np.logspace(-4, -1, 30)
    >>> tri.add_gw_frequency_series(freq_array, strain=1e-23)
    >>> tri.run_full_pipeline()
    >>> X1 = tri.get_tdi_result('X1')
    """

    def __init__(self,
                 t_start: float = 0.0,
                 t_end: float = 40000.0,
                 tri_arm: float = 10.0,
                 orbit_type: Literal['heliocentric', 'geocentric'] = 'heliocentric',
                 fsample_ob: float = 5.0,
                 fsample_gw: float = 0.1) -> None:
        """Initialize Triangle constellation with time range and orbit parameters."""
        # Store simulation parameters
        self.t_start = t_start
        self.t_end = t_end
        self.tri_arm = tri_arm
        self.orbit_type = orbit_type
        self.fsample_ob = fsample_ob
        self.fsample_gw = fsample_gw

        # Create orbit
        self.orbit = TJ_orbit.orbit(
            t_start=t_start,
            t_end=t_end,
            tri_arm=tri_arm,
            orbit_type=orbit_type
        )

        # Initialize component placeholders
        self.gw: Optional[TJ_gw.gw] = None
        self.obs: List[Optional[TJ_ob.ob]] = [None] * 7  # [None, ob1, ..., ob6]
        self.lock: Optional[TJ_lock.lock] = None
        self.synthesis: Optional[TJ_synthesis.synthesis] = None
        self.tdi: Optional[TJ_tdi.tdi] = None

        # Storage for GW sources and TDI results
        self._gw_sources: List[Dict] = []
        self.tdi_results: Dict[str, NDArray] = {}

        print(f"Triangle constellation initialized:")
        print(f"  Time range: {t_start:.0f} - {t_end:.0f} s")
        print(f"  Arm length: {tri_arm:.2f} light-seconds")
        print(f"  Orbit type: {orbit_type}")

    def add_gw_source(self,
                     fgw: float,
                     strain: float,
                     beta: float = 0.0,
                     lamda: float = 0.0,
                     psi: float = 0.0) -> None:
        """
        Add a single gravitational wave source. 

        Parameters 
        ----------
        fgw : float
            GW frequency in Hz. Hz
        strain : float
            GW strain amplitude. 
        beta : float, optional
            Ecliptic latitude in radians. Default is 0. 0
        lamda : float, optional
            Ecliptic longitude in radians. Default is 0. 0
        psi : float, optional
            Polarization angle in radians. Default is 0. 0
        """
        source = {
            'fgw': fgw,
            'strain': strain,
            'beta': beta,
            'lamda': lamda,
            'psi': psi
        }
        self._gw_sources.append(source)
        print(f"Added GW source: f={fgw:.4f} Hz, h0={strain:.2e}")

    def add_gw_frequency_series(self,
                               frequencies: NDArray[np.float64],
                               strain: float = 1e-23,
                               beta: float = 0.0,
                               lamda: float = 0.0,
                               psi: float = 0.0) -> None:
        """
        Add multiple GW sources with a series of frequencies. 

        Parameters 
        ----------
        frequencies : array_like
            Array of frequencies in Hz. Hz
        strain : float, optional
            Common strain amplitude for all sources. Default is 1e-23.
            1e-23
        beta, lamda, psi : float, optional
            Common sky position and polarization for all sources.
            

        Examples 
        --------
        >>> freq_array = np.logspace(-4, -1, 30)  # 30 frequencies from 1e-4 to 0.1 Hz
        >>> tri.add_gw_frequency_series(freq_array, strain=1e-23)
        """
        for freq in frequencies:
            self.add_gw_source(freq, strain, beta, lamda, psi)
        print(f"Added frequency series: {len(frequencies)} sources from {frequencies[0]:.2e} to {frequencies[-1]:.2e} Hz")

    def set_gw_sources(self, source_list: List[Dict]) -> None:
        """
        Set GW sources from a list of dictionaries. 

        Parameters 
        ----------
        source_list : list of dict
            List of source dictionaries with keys: 'fgw', 'strain', 'beta', 'lamda', 'psi'.
            'fgw', 'strain', 'beta', 'lamda', 'psi'

        Examples 
        --------
        >>> sources = [
        ...     {'fgw': 0.003, 'strain': 1e-23, 'beta': 0, 'lamda': 0, 'psi': 0},
        ...     {'fgw': 0.007, 'strain': 8e-24, 'beta': np.pi/3, 'lamda': np.pi/2, 'psi': np.pi/4}
        ... ]
        >>> tri.set_gw_sources(sources)
        """
        self._gw_sources = source_list.copy()
        print(f"Set {len(source_list)} GW sources")

    def create_gw_object(self, hasGW: bool = True) -> None:
        """
        Create the GW object with configured sources. 

        Parameters 
        ----------
        hasGW : bool, optional
            Enable GW signal generation. Default is True. True
        """
        if not self._gw_sources and hasGW:
            print("Warning: No GW sources defined. Adding default source at 0.007 Hz.")
            self.add_gw_source(fgw=0.007, strain=1e-23)

        self.gw = TJ_gw.gw(
            self.orbit,
            t_start=self.t_start,
            t_end=self.t_end,
            fsample=self.fsample_gw,
            source_list=self._gw_sources if self._gw_sources else None,
            hasGW=hasGW
        )
        print(f"GW object created with {self.gw.n_sources} source(s)")

    def setup_optical_benches(self,
                             hasLaser: bool = True,
                             hasAcc: bool = False,
                             hasOms: bool = False,
                             LNamp: float = 1.0e-13) -> None:
        """
        Create all 6 optical bench objects. 6

        Parameters 
        ----------
        hasLaser : bool, optional
            Enable laser frequency noise. Default is True. True
        hasAcc : bool, optional
            Enable acceleration noise. Default is False. False
        hasOms : bool, optional
            Enable optical metrology noise. Default is False. False
        LNamp : float, optional
            Laser noise amplitude. Default is 1e-13. 1e-13
        """
        for ii in range(1, 7):
            self.obs[ii] = TJ_ob.ob(
                t_start=self.t_start,
                t_end=self.t_end,
                fsample=self.fsample_ob,
                hasLaser=hasLaser,
                LNamp=LNamp,
                hasAcc=hasAcc,
                hasOms=hasOms
            )
        print(f"Created 6 optical benches (hasLaser={hasLaser}, hasAcc={hasAcc}, hasOms={hasOms})")

    def apply_laser_lock(self,
                        lock_type: Literal['single', 'dual', 'common'] = 'dual',
                        method: str = 'trapezoidal',
                        arm_delay: float = 20.0,
                        arm1_delay: float = 20.1,
                        arm2_delay: float = 19.9,
                        gfactor: float = 10000.0,
                        afactor: float = 100.0,
                        w0: float = 2 * np.pi * 0.4) -> None:
        """
        Apply laser frequency locking. 

        Parameters 
        ----------
        lock_type : {'single', 'dual', 'common'}, optional
            Type of arm locking. Default is 'dual'. 'dual'
        method : str, optional
            Integration method for single-arm. Default is 'trapezoidal'.
            'trapezoidal'
        arm_delay : float, optional
            Arm delay for single-arm locking (seconds). Default is 20.0.
            20.0
        arm1_delay : float, optional
            First arm delay for dual/common-arm (seconds). Default is 20.1.
            /20.1
        arm2_delay : float, optional
            Second arm delay for dual/common-arm (seconds). Default is 19.9.
            /19.9
        gfactor : float, optional
            Loop gain. Default is 10000.0. 10000.0
        afactor : float, optional
            Proportional factor. Default is 100.0. 100.0
        w0 : float, optional
            Corner frequency (rad/s). Default is 2π×0.4 rad/s.
            /2π×0.4/
        """
        if self.gw is None:
            raise RuntimeError("GW object not created. Call create_gw_object() first.")
        if self.obs[1] is None:
            raise RuntimeError("Optical benches not set up. Call setup_optical_benches() first.")

        self.lock = TJ_lock.lock(
            self.orbit,
            self.obs,
            self.gw,
            lock_type=lock_type,
            method=method,
            arm_delay=arm_delay,
            arm1_delay=arm1_delay,
            arm2_delay=arm2_delay,
            gfactor=gfactor,
            afactor=afactor,
            w0=w0
        )
        self.lock.run()
        print(f"Laser locking applied: {lock_type} mode")

    def synthesize_signals(self, delay_level: Literal['0', '1'] = '1') -> None:
        """
        Synthesize phasemeter signals and eta signals. eta

        Parameters 
        ----------
        delay_level : {'0', '1'}, optional
            Delay calculation mode. '1' for flexible arm length (default), '0' for constant.
            '1''0'
        """
        if self.gw is None:
            raise RuntimeError("GW object not created. Call create_gw_object() first.")
        if self.obs[1] is None:
            raise RuntimeError("Optical benches not set up. Call setup_optical_benches() first.")

        self.synthesis = TJ_synthesis.synthesis(
            self.orbit,
            self.obs,
            self.gw,
            delay_level=delay_level
        )
        print(f"Signal synthesis completed (delay_level='{delay_level}')")

    def initialize_tdi(self) -> None:
        """Initialize TDI processing object. TDI"""
        if self.obs[1] is None:
            raise RuntimeError("Optical benches not set up. Call setup_optical_benches() first.")

        self.tdi = TJ_tdi.tdi(self.orbit, self.obs)
        print("TDI object initialized")

    def run_tdi(self,
                TDI_channel,
                delay_level: Literal['0', '1'] = '1',
                store_name: Optional[str] = None) -> NDArray[np.float64]:
        """
        Run a TDI combination and optionally store result. TDI

        Parameters 
        ----------
        TDI_channel : list
            TDI channel definition (e.g., TJ_tdi.X1, TJ_tdi.Y1).
            TDITJ_tdi.X1, TJ_tdi.Y1
        delay_level : {'0', '1'}, optional
            Delay calculation mode. Default is '1'. '1'
        store_name : str, optional
            Name to store result in tdi_results dict. If None, auto-generate.
            tdi_resultsNone

        Returns 
        -------
        ndarray
            TDI combination output array. TDI
        """
        if self.tdi is None:
            self.initialize_tdi()

        result = self.tdi.run(TDI_channel=TDI_channel, delay_level=delay_level)

        # Store result
        if store_name is None:
            store_name = f"tdi_{len(self.tdi_results)}"
        self.tdi_results[store_name] = result

        print(f"TDI combination computed: {store_name}")
        return result

    def get_tdi_result(self, name: str) -> NDArray[np.float64]:
        """
        Retrieve stored TDI result by name. TDI

        Parameters 
        ----------
        name : str
            Name of stored TDI result. TDI

        Returns 
        -------
        ndarray
            TDI result array. TDI
        """
        if name not in self.tdi_results:
            raise KeyError(f"TDI result '{name}' not found. Available: {list(self.tdi_results.keys())}")
        return self.tdi_results[name]

    def run_full_pipeline(self,
                         delay_level: Literal['0', '1'] = '1',
                         lock_type: Literal['single', 'dual', 'common'] = 'dual',
                         tdi_channels: Optional[List] = None) -> None:
        """
        Run the complete simulation pipeline. 

        Executes: GW creation → optical bench setup → laser locking → synthesis → TDI.
         →  →  →  → TDI

        Parameters 
        ----------
        delay_level : {'0', '1'}, optional
            Delay calculation mode. Default is '1'. '1'
        lock_type : {'single', 'dual', 'common'}, optional
            Laser locking type. Default is 'dual'. 'dual'
        tdi_channels : list, optional
            List of TDI channels to compute. If None, compute X1, X2, Y1, Z1.
            TDINoneX1X2Y1Z1
        """
        print("\n" + "="*60)
        print("Running full simulation pipeline ")
        print("="*60)

        # Step 1: Create GW object
        if self.gw is None:
            self.create_gw_object(hasGW=True)

        # Step 2: Setup optical benches
        if self.obs[1] is None:
            self.setup_optical_benches()

        # Step 3: Apply laser locking
        if self.lock is None:
            self.apply_laser_lock(lock_type=lock_type)

        # Step 4: Synthesize signals
        if self.synthesis is None:
            self.synthesize_signals(delay_level=delay_level)

        # Step 5: Run TDI combinations
        if tdi_channels is None:
            tdi_channels = [
                (TJ_tdi.X1, 'X1'),
                (TJ_tdi.X2, 'X2'),
                (TJ_tdi.Y1, 'Y1'),
                (TJ_tdi.Z1, 'Z1')
            ]

        for channel, name in tdi_channels:
            self.run_tdi(channel, delay_level=delay_level, store_name=name)

        print("="*60)
        print("Pipeline completed successfully! !")
        print(f"Available TDI results: {list(self.tdi_results.keys())}")
        print("="*60 + "\n")

    def get_laser_data(self) -> tuple:
        """
        Get free-running and locked laser noise data. 

        Returns 
        -------
        tuple of (free_laser, locked_laser, time_array, fsample)
            Free-running laser, locked laser, time array, and sampling frequency.
            
        """
        if self.obs[-1] is None:
            raise RuntimeError("Optical benches not set up.")

        return (
            self.obs[-1]._laser_noise,
            self.obs[-1]._laser,
            self.obs[-1]._tarray,
            self.obs[-1]._fsample
        )

    def summary(self) -> None:
        """Print summary of Triangle configuration. Triangle"""
        print("\n" + "="*60)
        print("Triangle Constellation Summary ")
        print("="*60)
        print(f"Time range: {self.t_start:.0f} - {self.t_end:.0f} s ({self.t_end - self.t_start:.0f} s duration)")
        print(f"Arm length: {self.tri_arm:.2f} light-seconds")
        print(f"Orbit type: {self.orbit_type}")
        print(f"Sampling frequencies: OB={self.fsample_ob} Hz, GW={self.fsample_gw} Hz")
        print(f"\nComponents initialized:")
        print(f"  Orbit: ✓")
        print(f"  GW: {'✓' if self.gw is not None else '✗'} ({len(self._gw_sources)} sources)")
        print(f"  Optical benches: {'✓' if self.obs[1] is not None else '✗'}")
        print(f"  Laser lock: {'✓' if self.lock is not None else '✗'}")
        print(f"  Synthesis: {'✓' if self.synthesis is not None else '✗'}")
        print(f"  TDI: {'✓' if self.tdi is not None else '✗'}")
        print(f"\nTDI results stored: {len(self.tdi_results)}")
        if self.tdi_results:
            print(f"  Channels: {list(self.tdi_results.keys())}")
        print("="*60 + "\n")
