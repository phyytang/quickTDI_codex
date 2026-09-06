# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

quickTDI is a Python simulation framework for Time Delay Interferometry (TDI) in space-based gravitational wave detectors (Taiji/LISA). It models the full measurement chain: spacecraft orbits, gravitational wave signals, optical bench instrumentation, laser frequency locking, signal synthesis, and TDI combinations for laser noise cancellation.

## Dependencies & Setup

Runtime dependencies: `numpy`, `scipy`. Optional: `dynesty` (for Bayesian inference), `matplotlib`/`jupyter` (for visualization).

```bash
pip install numpy scipy
```

There is no `setup.py` or `requirements.txt`. Modules in `src/` import each other directly (not as a package), so `src/` must be on `PYTHONPATH`:

```bash
PYTHONPATH=src python -c "from TJ_Triangle import Triangle; Triangle()"
```

## Testing

No automated test suite. Validation is done via `test.ipynb` (Jupyter notebook). If adding tests, use `pytest` in a `tests/` directory.

## Architecture — Simulation Pipeline

The simulation follows a strict sequential pipeline, orchestrated by `TJ_Triangle.Triangle`:

```
TJ_orbit → TJ_gw → TJ_ob → TJ_lock → TJ_synthesis → TJ_tdi
```

1. **`TJ_constant`** — Physical constants and unit conversions. All quantities in natural units (time in seconds, distances in light-seconds, velocities dimensionless v/c). Provides both module-level `UPPER_SNAKE_CASE` constants and a deprecated `constant` class for backward compatibility.

2. **`TJ_orbit`** — Spacecraft constellation orbits. Computes positions, velocities, and inter-spacecraft distances via analytical Kepler equations (heliocentric/geocentric) or from CSV files. Creates cubic spline interpolators for all 6 arm distances. Key methods: `dij(arm_num, t)` for distance lookup, `delay(t_arr, arm_array)` for TDI time-delay operator (vectorized), `delay_0()` for constant-arm approximation.

3. **`TJ_gw`** — Gravitational wave signal generation. Computes strain response in all 6 arms for single or multiple monochromatic sources. Pre-computes strain arrays and creates interpolators (`gw[arm_num](t)`). Supports source superposition.

4. **`TJ_ob`** — Optical bench per spacecraft arm. Generates laser noise (`TJ_noise.white`), acceleration noise, and OMS noise. Stores time series with `_` prefix (e.g., `_laser`, `_sci`, `_eta`) and corresponding interpolators without prefix (e.g., `.laser`, `.sci`, `.eta`). Uses **order-31 interpolation** (`kind=31`) for accurate time-delayed access.

5. **`TJ_noise`** — Noise generators with specific PSDs: `white`, `power_2` (f²), `power_m2` (1/f²), `power_4` (f⁴), `power_m4`/`power_m4_2` (1/f⁴), truncated variants (`m2_trun`, `m4_trun`), and composite `acc`/`oms` noise models.

6. **`TJ_lock`** — Laser frequency stabilization via arm-locking. Implements single-arm (backward/trapezoidal), dual-arm (bilinear/Tustin), and common-arm locking. The `run()` method updates all optical bench objects with locked laser noise and applies phase-locking across the constellation.

7. **`TJ_synthesis`** — Synthesizes phasemeter outputs (`sci`, `tes`, `ref`) and `eta` signals for all 6 optical benches. Combines laser noise, GW signals, acceleration noise, and OMS noise with proper time delays. Pre-computes delayed time arrays for performance.

8. **`TJ_tdi`** — TDI combinations for laser noise cancellation. Defines channel specifications as lists of `[eta_label, arm_delays, sign]`. Pre-defined channels: X1/X2, Y1/Y2, Z1/Z2 (Michelson), a1/a2, z1/z2 (Sagnac), and others. `cycle_tdi()` generates Y/Z from X via cyclic permutation. `delay_tdi()` builds higher-generation combinations.

9. **`TJ_Triangle`** — High-level pipeline wrapper. The `Triangle` class manages the full workflow with methods like `add_gw_source()`, `setup_optical_benches()`, `apply_laser_lock()`, `synthesize_signals()`, `run_tdi()`, and `run_full_pipeline()`.

## Inference Utilities

- **`utils/TJ_infer`** — Bayesian inference for TDI data using dynesty nested sampling. Provides `Parameter` (prior definitions), `InferenceConfig`, `TriangleSimulator` (cached simulator for likelihood evaluation), and `log_likelihood` (PSD-weighted frequency-domain).
- **`scripts/run_dynesty.py`** — CLI for running dynesty on `.npz` TDI data files: `python scripts/run_dynesty.py --data data.npz --channels X1 --nlive 200`.

## Key Conventions

- **Spacecraft numbering**: 1, 2, 3. Arms: positive (1, 2, 3) forward, negative (-1, -2, -3) reverse.
- **Arm-to-spacecraft mapping**: `ARM_REC` (arm → receiver), `ARM_SEN` (arm → sender), `RS_ARM` (receiver-sender pair → arm), `ARM_ARRAY = [3, -2, -3, 1, 2, -1]`.
- **Optical bench list**: `obs = [None, ob1, ob2, ob3, ob_3p, ob_2p, ob_1p]` — 7 elements, index 0 unused. Negative indexing maps to primed benches (e.g., `obs[-1]` = ob_1p = master laser).
- **`delay_level`**: `'1'` = flexible/time-varying arm lengths; `'0'` = constant arm length approximation. Passed as string, not int.
- **Module naming**: All source modules use `TJ_` prefix. Classes are lowercase (e.g., `orbit`, `gw`, `ob`, `lock`, `synthesis`, `tdi`) except `Triangle`.
- **Docstrings**: NumPy-style with `Parameters`, `Returns`, `Examples` sections.
