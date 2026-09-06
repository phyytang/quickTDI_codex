"""Reproduce the bounded checks used in the paper-to-code proposal.

Run from the repository root: python3 output/tdi_model_review/run_checks.py
This is a diagnostic, not a certification of the complete simulation.
"""

import contextlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from TJ_Triangle import Triangle
from TJ_constant import ARM_REC, ARM_SEN, SPEED_OF_LIGHT
from TJ_gw import gw, ygwsr
from TJ_ob import ob
from TJ_synthesis import synthesis
from TJ_tdi import X1, X2, tdi


def run_checks():
    results = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        triangle = Triangle(t_end=512, tri_arm=10.123)
        orbit = triangle.orbit
        benches = [None] + [
            ob(t_end=512, fsample=5, hasLaser=False) for _ in range(6)
        ]
        for index in range(1, 7):
            bench = benches[index]
            bench.hasLaser = True
            bench._laser = 1e-13 * np.sin(
                2 * np.pi * (0.009 + 0.001 * index) * bench._tarray
                + 0.23 * index
            )
            bench.update_laser()
        source = gw(orbit, t_end=512, fsample=1, hasGW=False)
        synthesis(orbit, benches, source, delay_level="0")
        processor = tdi(orbit, benches)
        interior = (processor._tarray > 180) & (processor._tarray < 480)
        results["static_equal_arm_sinusoidal_laser"] = {}
        for name, channel in [("X1", X1), ("X2", X2)]:
            data = processor.run(channel, delay_level="0")
            results["static_equal_arm_sinusoidal_laser"][name] = {
                "max_abs_interior": float(np.max(np.abs(data[interior]))),
                "max_abs_full": float(np.max(np.abs(data))),
            }

        source_options = dict(fgw=0.007, strain=1e-20, beta=0.3, lamda=0.7, psi=0.2)
        source = gw(orbit, t_end=512, fsample=1, hasGW=True, **source_options)
        time = 100.0
        results["gw_helper_vs_main"] = {
            "helper": float(ygwsr(3, time, orbit, **source_options)),
            "main": float(source.ygw(3, time)),
        }
        results["instantaneous_vs_retarded_seconds"] = {}
        for arm in [3, -3]:
            receiver, sender = ARM_REC[arm], ARM_SEN[arm]
            instantaneous = float(orbit.dij(arm, time))
            retarded = instantaneous
            for _ in range(10):
                retarded = float(np.linalg.norm(
                    orbit.position(receiver, time)
                    - orbit.position(sender, time - retarded)
                ))
            results["instantaneous_vs_retarded_seconds"][str(arm)] = {
                "instantaneous": instantaneous,
                "retarded_flat_spacetime": retarded,
                "difference_seconds": retarded - instantaneous,
                "difference_meters": (retarded - instantaneous) * SPEED_OF_LIGHT,
            }
    return results


if __name__ == "__main__":
    result = run_checks()
    serialized = json.dumps(result, indent=2)
    Path(__file__).with_name("checks.json").write_text(serialized + "\n")
    print(serialized)
