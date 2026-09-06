"""Run dynesty inference on TDI data stored in an .npz file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from utils.TJ_infer import (
    InferenceConfig,
    Parameter,
    run_dynesty,
)


def load_data(npz_path: Path, channels: list[str]) -> dict:
    npz = np.load(npz_path)
    t = npz["t"] if "t" in npz.files else None
    y = {}
    psd = {}

    for ch in channels:
        if ch not in npz.files:
            raise KeyError(f"Channel '{ch}' not found in {npz_path}")
        y[ch] = npz[ch]

        f_key = f"psd_freqs_{ch}"
        p_key = f"psd_vals_{ch}"
        if f_key in npz.files and p_key in npz.files:
            psd[ch] = (npz[f_key], npz[p_key])

    data = {"t": t, "y": y}
    if psd:
        data["psd"] = psd
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dynesty on TDI data")
    parser.add_argument("--data", required=True, help="Path to .npz file with TDI data")
    parser.add_argument("--channels", default="X1", help="Comma-separated channels, e.g. X1,Y1")
    parser.add_argument("--fsample", type=float, default=5.0, help="Sampling frequency of TDI data")
    parser.add_argument("--nlive", type=int, default=200, help="Dynesty nlive")
    parser.add_argument("--dlogz", type=float, default=0.1, help="Dynesty dlogz stopping criterion")
    parser.add_argument("--out", default="dynesty_results.npz", help="Output .npz for results")
    args = parser.parse_args()

    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    data = load_data(Path(args.data), channels)

    config = InferenceConfig(
        channels=channels,
        fsample=args.fsample,
        t_start=0.0,
        t_end=40000.0,
        fsample_ob=args.fsample,
        fsample_gw=0.1,
        has_laser=False,
        has_acc=False,
        has_oms=False,
        apply_lock=False,
    )

    params = [
        Parameter("fgw", prior="log_uniform", bounds=(1.0e-4, 1.0e-1)),
        Parameter("strain", prior="log_uniform", bounds=(1.0e-25, 1.0e-20)),
        Parameter("beta", prior="uniform", bounds=(-np.pi / 2, np.pi / 2)),
        Parameter("lamda", prior="uniform", bounds=(0.0, 2 * np.pi)),
        Parameter("psi", prior="uniform", bounds=(0.0, np.pi)),
        Parameter("acc", prior="log_uniform", bounds=(1.0e-16, 1.0e-12)),
        Parameter("oms", prior="log_uniform", bounds=(1.0e-16, 1.0e-12)),
    ]

    results = run_dynesty(
        data,
        params,
        config,
        dynesty_opts={
            "sampler": {"nlive": args.nlive},
            "run": {"dlogz": args.dlogz},
        },
    )

    np.savez(
        args.out,
        samples=results.samples,
        logz=results.logz,
        logzerr=results.logzerr,
    )
    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
