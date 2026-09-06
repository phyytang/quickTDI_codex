"""Bayesian inference utilities for TDI data using dynesty.

This module provides helpers to:
- Define parameter priors
- Simulate TDI channels from the existing Triangle pipeline
- Compute PSD-weighted frequency-domain likelihoods
- Run dynesty nested sampling
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from pathlib import Path
import sys

import numpy as np
from numpy.typing import NDArray
from scipy import signal, stats


@dataclass
class Parameter:
    """Parameter definition for inference.

    Attributes
    ----------
    name : str
        Parameter name.
    prior : str
        Prior type: 'uniform', 'log_uniform', 'normal', or 'fixed'.
    bounds : tuple
        Bounds or (mu, sigma) for 'normal'.
    transform : callable, optional
        Custom transform from unit interval to parameter value.
    default : float, optional
        Default value for fixed parameters.
    """

    name: str
    prior: str = "uniform"
    bounds: Tuple[float, float] = (0.0, 1.0)
    transform: Optional[Callable[[float], float]] = None
    default: Optional[float] = None

    def sample_from_unit(self, u: float) -> float:
        """Map a unit interval value to the parameter domain."""
        if self.transform is not None:
            return float(self.transform(u))
        if self.prior == "uniform":
            lo, hi = self.bounds
            return float(lo + u * (hi - lo))
        if self.prior == "log_uniform":
            lo, hi = self.bounds
            if lo <= 0 or hi <= 0:
                raise ValueError(f"log_uniform bounds must be > 0 for {self.name}")
            return float(10 ** (np.log10(lo) + u * (np.log10(hi) - np.log10(lo))))
        if self.prior == "normal":
            mu, sigma = self.bounds
            return float(stats.norm.ppf(u, loc=mu, scale=sigma))
        if self.prior == "fixed":
            if self.default is not None:
                return float(self.default)
            return float(self.bounds[0])
        raise ValueError(f"Unknown prior type '{self.prior}' for {self.name}")


@dataclass
class InferenceConfig:
    """Configuration for TDI inference."""

    channels: List[str] = field(default_factory=lambda: ["X1"])
    fsample: float = 1.0
    t_start: float = 0.0
    t_end: float = 40000.0
    tri_arm: float = 10.0
    orbit_type: str = "heliocentric"
    fsample_ob: float = 5.0
    fsample_gw: float = 0.1
    delay_level: str = "1"

    # Optical bench noise toggles
    has_laser: bool = False
    has_acc: bool = False
    has_oms: bool = False
    ln_amp: float = 1.0e-13

    # Laser locking
    apply_lock: bool = False
    lock_type: str = "dual"
    lock_kwargs: Dict[str, float] = field(default_factory=dict)

    # GW source defaults
    gw_defaults: Dict[str, float] = field(default_factory=lambda: {
        "fgw": 0.007,
        "strain": 1.0e-23,
        "beta": 0.0,
        "lamda": 0.0,
        "psi": 0.0,
    })
    source_builder: Optional[Callable[[Dict[str, float]], List[Dict[str, float]]]] = None

    # PSD configuration
    psd_window: str = "hann"
    psd_nperseg: Optional[int] = None
    psd_noverlap: Optional[int] = None
    psd_detrend: str = "constant"

    # Noise PSD model configuration
    noise_f1: float = 4.0e-4
    noise_f2: float = 8.0e-3
    noise_f3: float = 2.0e-3
    noise_c: float = 3.0e8
    noise_floor: float = 1.0e-40
    acc_default: float = 0.0
    oms_default: float = 0.0


def ensure_src_on_path(src_dir: str = "src") -> str:
    """Ensure the local src directory is available for imports."""
    root = Path(__file__).resolve().parents[1]
    src_path = (root / src_dir).resolve()
    src_str = str(src_path)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return src_str


def prior_transform(u: Sequence[float], params: Sequence[Parameter]) -> NDArray[np.float64]:
    """Transform unit-cube samples to parameter values."""
    if len(u) != len(params):
        raise ValueError("Unit vector length does not match parameter list")
    theta = [p.sample_from_unit(ui) for ui, p in zip(u, params)]
    return np.array(theta, dtype=np.float64)


def theta_to_dict(theta: Sequence[float], params: Sequence[Parameter]) -> Dict[str, float]:
    """Convert a parameter vector to a name->value dict."""
    return {p.name: float(v) for p, v in zip(params, theta)}


def compute_psd(data: NDArray[np.float64], fsample: float, config: InferenceConfig) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute Welch PSD for a time series."""
    nperseg = config.psd_nperseg
    if nperseg is None:
        nperseg = min(1024, len(data))
    noverlap = config.psd_noverlap
    if noverlap is None:
        noverlap = nperseg // 2
    freqs, psd = signal.welch(
        data,
        fs=fsample,
        window=config.psd_window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=config.psd_detrend,
        return_onesided=True,
        scaling="density",
    )
    return freqs, psd


def noise_psd_model(freqs: NDArray[np.float64], acc: float, oms: float, config: InferenceConfig) -> NDArray[np.float64]:
    """Compute additive noise PSD from acc/oms parameters.

    The formulas follow the docstrings in TJ_noise.acc and TJ_noise.oms.
    """
    if acc <= 0 and oms <= 0:
        return np.zeros_like(freqs)

    f = freqs.copy()
    if len(f) == 0:
        return f
    f_min = f[1] if len(f) > 1 else 1.0e-6
    f = np.where(f > 0, f, f_min)

    s_acc = np.zeros_like(f)
    if acc > 0:
        a_tm = acc / config.noise_c
        f1 = config.noise_f1
        f2 = config.noise_f2
        sigma1 = 1.0 / (2 * np.pi) * a_tm * f1 / (f2**2)
        sigma2 = 1.0 / (2 * np.pi) * a_tm * 1.0 / (f2**2)
        sigma3 = 1.0 / (2 * np.pi) * a_tm
        sigma4 = 1.0 / (2 * np.pi) * a_tm * f1
        s_acc = (sigma1**2) + (sigma2**2) * (f**2) + (sigma3**2) / (f**2) + (sigma4**2) / (f**4)

    s_oms = np.zeros_like(f)
    if oms > 0:
        f3 = config.noise_f3
        a_oms = oms / config.noise_c
        sigma1 = 2 * np.pi * a_oms
        sigma2 = 2 * np.pi * a_oms * (f3**2)
        s_oms = (sigma1**2) * (f**2) + (sigma2**2) / (f**2)

    return s_acc + s_oms


def _resolve_tdi_channel(name: str):
    ensure_src_on_path()
    import TJ_tdi

    mapping = {
        "X1": TJ_tdi.X1,
        "X2": TJ_tdi.X2,
        "Y1": TJ_tdi.Y1,
        "Y2": TJ_tdi.Y2,
        "Z1": TJ_tdi.Z1,
        "Z2": TJ_tdi.Z2,
        "a1": TJ_tdi.a1,
        "a15": TJ_tdi.a15,
        "z1": TJ_tdi.z1,
        "z2": TJ_tdi.z2,
    }
    if name not in mapping:
        raise ValueError(f"Unknown TDI channel '{name}'. Available: {sorted(mapping.keys())}")
    return mapping[name]


class TriangleSimulator:
    """Cached simulator for repeated likelihood evaluations."""

    def __init__(self, config: InferenceConfig) -> None:
        ensure_src_on_path()
        from TJ_Triangle import Triangle

        self.config = config
        self.triangle = Triangle(
            t_start=config.t_start,
            t_end=config.t_end,
            tri_arm=config.tri_arm,
            orbit_type=config.orbit_type,
            fsample_ob=config.fsample_ob,
            fsample_gw=config.fsample_gw,
        )

        self.triangle.setup_optical_benches(
            hasLaser=config.has_laser,
            hasAcc=config.has_acc,
            hasOms=config.has_oms,
            LNamp=config.ln_amp,
        )
        if config.apply_lock:
            self.triangle.apply_laser_lock(
                lock_type=config.lock_type,
                **config.lock_kwargs,
            )

    def _build_sources(self, param_dict: Dict[str, float]) -> List[Dict[str, float]]:
        if self.config.source_builder is not None:
            return self.config.source_builder(param_dict)
        source = dict(self.config.gw_defaults)
        for key in source:
            if key in param_dict:
                source[key] = float(param_dict[key])
        return [source]

    def simulate(self, param_dict: Dict[str, float]) -> Dict[str, NDArray[np.float64]]:
        sources = self._build_sources(param_dict)
        self.triangle._gw_sources = sources
        self.triangle.create_gw_object(hasGW=True)

        if self.config.apply_lock:
            self.triangle.apply_laser_lock(
                lock_type=self.config.lock_type,
                **self.config.lock_kwargs,
            )

        self.triangle.synthesize_signals(delay_level=self.config.delay_level)
        if self.triangle.tdi is None:
            self.triangle.initialize_tdi()

        results: Dict[str, NDArray[np.float64]] = {}
        for ch in self.config.channels:
            channel_def = _resolve_tdi_channel(ch) if isinstance(ch, str) else ch
            results[str(ch)] = self.triangle.tdi.run(
                TDI_channel=channel_def,
                delay_level=self.config.delay_level,
            )
        return results


def prepare_data(data: Dict, config: InferenceConfig) -> Dict:
    """Prepare data for likelihood evaluation.

    Expected input format:
    data = {
        "t": array,
        "y": {"X1": series, ...},
        "psd": {"X1": (freqs, psd), ...}  # optional
    }
    """
    y = data["y"]
    channels = list(y.keys())
    n = len(next(iter(y.values())))

    if "t" in data and data["t"] is not None:
        t = np.asarray(data["t"], dtype=np.float64)
        dt = float(np.median(np.diff(t)))
    else:
        t = None
        dt = 1.0 / config.fsample
    fsample_eff = 1.0 / dt

    freqs = np.fft.rfftfreq(n, d=dt)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 0.0

    fft_data: Dict[str, NDArray[np.complex128]] = {}
    psd_base: Dict[str, NDArray[np.float64]] = {}

    for ch in channels:
        series = np.asarray(y[ch], dtype=np.float64)
        if len(series) != n:
            raise ValueError("All channels must have the same length")
        fft_data[ch] = np.fft.rfft(series) * dt

        if "psd" in data and ch in data["psd"]:
            f_psd, p_psd = data["psd"][ch]
        else:
            f_psd, p_psd = compute_psd(series, fsample_eff, config)
        psd_base[ch] = np.interp(freqs, f_psd, p_psd, left=p_psd[0], right=p_psd[-1])

    return {
        "t": t,
        "y": y,
        "freqs": freqs,
        "df": df,
        "dt": dt,
        "fft": fft_data,
        "psd": psd_base,
    }


def log_likelihood(
    theta: Sequence[float],
    prepared_data: Dict,
    config: InferenceConfig,
    params: Sequence[Parameter],
    simulator: TriangleSimulator,
) -> float:
    """Compute PSD-weighted frequency-domain log-likelihood."""
    param_dict = theta_to_dict(theta, params)
    acc_val = float(param_dict.get("acc", config.acc_default))
    oms_val = float(param_dict.get("oms", config.oms_default))

    model = simulator.simulate(param_dict)

    freqs = prepared_data["freqs"]
    df = prepared_data["df"]
    dt = prepared_data["dt"]

    logl = 0.0
    for ch, data_fft in prepared_data["fft"].items():
        model_series = np.asarray(model[ch], dtype=np.float64)
        if len(model_series) != len(prepared_data["y"][ch]):
            raise ValueError(f"Model length mismatch for channel {ch}")
        model_fft = np.fft.rfft(model_series) * dt
        resid = data_fft - model_fft

        s_eff = prepared_data["psd"][ch] + noise_psd_model(freqs, acc_val, oms_val, config)
        s_eff = np.maximum(s_eff, config.noise_floor)

        term = 4.0 * df * (np.abs(resid) ** 2) / s_eff
        logl += -0.5 * np.sum(term + np.log(s_eff))

    return float(logl)


def run_dynesty(
    data: Dict,
    params: Sequence[Parameter],
    config: InferenceConfig,
    dynesty_opts: Optional[Dict] = None,
):
    """Run dynesty nested sampling and return results."""
    from dynesty import NestedSampler

    prepared = prepare_data(data, config)
    simulator = TriangleSimulator(config)

    def _loglike(theta: Sequence[float]) -> float:
        return log_likelihood(theta, prepared, config, params, simulator)

    def _ptform(u: Sequence[float]) -> NDArray[np.float64]:
        return prior_transform(u, params)

    dynesty_opts = dynesty_opts or {}
    sampler_opts = dynesty_opts.get("sampler", {})
    run_opts = dynesty_opts.get("run", {})

    sampler = NestedSampler(_loglike, _ptform, ndim=len(params), **sampler_opts)
    sampler.run_nested(**run_opts)
    return sampler.results
