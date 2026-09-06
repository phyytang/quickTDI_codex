# Bayle-Hartwig measurement model: digest, code comparison, and implementation plan

Prepared 2026-09-06 for the current quickTDI source snapshot. This is a design proposal; the simulation source has not been modified.

Source: Jean-Baptiste Bayle and Olaf Hartwig, *Unified model for the LISA measurements and instrument simulations*, Physical Review D **107**, 083019 (2023), DOI 10.1103/PhysRevD.107.083019. Equation and page references below refer to the 31-page PDF in `refs`. Mission numbers quoted here reproduce this paper's configuration, not current LISA requirements or validated Taiji requirements.

**Recommendation:** retain the existing Python pipeline and TDI channel definitions, establish a trustworthy delay/observable foundation, and insert an explicit instrument-to-telemetry layer before ground processing. The main change is the data model and measurement chain, not a new collection of TDI polynomials.

## 1. What the paper actually designs

The paper models the data a spacecraft would transmit, including the imperfections that ground processing must remove. It intentionally avoids simulating optical electric fields, GHz modulation waveforms, or 80 MHz digital phase-locked loops sample by sample.

Its important design choices are:

1. **Three kinds of time.** Orbits and source waveforms naturally use barycentric coordinate time, local instruments use each spacecraft's proper time, and measured data use each spacecraft's imperfect clock time (§II B). Proper pseudorange includes both propagation and conversion between the two spacecraft proper times; it is not simply distance divided by light speed.
2. **Frequency variables with separated dynamic ranges.** A laser has a common optical reference frequency plus a large offset and a small fluctuation: `nu = nu0 + offset + fluctuation` [Eq. (5)]. Store the last two independently. This retains tiny GW signals while handling MHz beatnotes and GHz sideband offsets. Each modulated laser has four arrays: carrier offset/fluctuation and upper-sideband offset/fluctuation [Eq. (12)]. The huge common optical frequency is a constant, not a sampled waveform.
3. **Explicit optical paths.** Six optical benches each have interspacecraft (ISI), test-mass (TMI), and reference (RFI) measurements. Beams acquire path, test-mass, backlink, and readout noise at specific places. Sharing an underlying disturbance across paths preserves physical correlations.
4. **Frequency propagation includes a Jacobian.** Delaying a phase and delaying its derivative are different operations. The frequency of a received beam includes the factor `1 - d_dot` as well as the optical Doppler term and GW term [Eqs. (30), (80)-(81)].
5. **Clocks affect values and timestamps.** A frequency measured against an imperfect clock is resampled and rescaled. Its large beatnote offset couples to fractional clock noise [Eqs. (68), (75)-(77)]. Adding a generic noise array to a science channel does not reproduce this behavior.
6. **Offset phase locking and frequency planning.** Five lasers are ideally locked, directly or indirectly, to one primary laser through ISI or RFI beatnotes. The controlled beatnote follows a frequency plan in local clock time, before telemetry filtering. The paper treats frequency plans as external inputs and describes six topologies with six possible primary lasers, giving 36 configurations (§VI).
7. **Auxiliary clock/ranging measurements.** Upper-sideband ISI/RFI beatnotes constrain clock and modulation noise. Measured pseudoranges contain directed proper pseudorange, clock desynchronization, and ranging error (§VII). The model does not generate TMI sideband telemetry.
8. **Separate physics and telemetry sampling.** The paper uses 16 Hz physics data, clock resampling, then an antialias FIR filter and decimation to 4 Hz (§V). These are configuration choices, not reasons to hard-code LISA values into Taiji.

The resulting products are 30 beatnote streams: 6 ISI carrier, 6 ISI sideband, 6 RFI carrier, 6 RFI sideband, and 6 TMI carrier; plus 6 measured pseudoranges. Offsets/fluctuations can also be exported for diagnostics. Total beatnote frequency is the telemetry-like output.

The paper then demonstrates ground processing with PyTDI and the algorithm in its Ref. [32]; it does not derive the full clock-correction pipeline itself. Its two implementations are a vectorized, stage-by-stage Python simulator (LISA Instrument) and a streaming node-based simulator (LISANode). quickTDI is structurally closer to the former.

The paper does **not** include a complete spacecraft dynamics/DFACS simulation, tilt-to-length/DWS model, explicit ADC/DPLL dynamics, PRN correlator or ambiguity resolution, or ground orbit determination. These should not be prerequisites for reproducing its model.

## 2. Where the current code matches, and where it differs

The physically useful interpretation of the current synthesized signals is dimensionless fractional frequency. The GW response is a one-way relative frequency response; the OMS and acceleration noise formulas have the corresponding frequency-domain scaling. Several docstrings instead call these quantities “phase” or “strain,” so this convention needs to be made explicit and verified before conversion to Hz.

| Paper component | Current implementation | Assessment and consequence |
|---|---|---|
| Stage-by-stage orchestration | `src/TJ_Triangle.py:122`, `:399` | Good foundation. The full pipeline always invokes arm locking and has no explicit telemetry/ground-processing boundary. Add independently selectable stages and invalidation of downstream results when inputs change. |
| Six MOSAs and link indexing | `src/TJ_constant.py:53`; synthesis uses signed bench indices | Existing mappings are sufficient, but bench labels and arm numbers are different. Introduce a canonical receiver/sender key with legacy adapters. |
| Directed proper pseudorange | `src/TJ_orbit.py:215`, `:376`, `:708` | Analytical distances are simultaneous Euclidean separations. Delay chains do evaluate nested times, but the underlying distances do not solve the moving-endpoint light-time problem or transform proper times. File mode could supply improved distances, but needs an explicit contract. |
| Delay versus Doppler-delay | `src/TJ_synthesis.py:105`, `src/TJ_tdi.py:264` | Signals are evaluated at delayed times without multiplying by `1 - d_dot`. This is an approximation for a fractional-frequency simulator; it does not implement the paper's complete frequency measurement model. |
| GW injection | `src/TJ_gw.py:270`, `:412` | Main path supports multiple monochromatic sources and evaluates the sender at a delayed time. The standalone `ygwsr` helper uses simultaneous sender position and different cross-polarization phase. The two paths disagree for identical inputs. Neither establishes a proper-time response interface. |
| ISI/TMI/RFI carriers | `src/TJ_synthesis.py:117` | `sci`, `tes`, `ref` already represent the appropriate three measurement families with distant/adjacent-minus-local polarity. Good reduced model, but no offsets, clock readout, or distinct local path/backlink/readout terms. |
| Intermediate ground combinations | `src/TJ_synthesis.py:169` | Directly constructs eta from ideal common-time readouts. `_xi` exists on benches but is not populated here. Ground preprocessing should become a separate stage. |
| Laser frequency state | `src/TJ_ob.py:178` | Stores laser fluctuations only. No carrier reference frequency or offset/fluctuation separation. |
| Phase locks and frequency plans | `src/TJ_lock.py:180`, `:634`; empty `src/TJ_freplan.py` | Existing arm servos are useful additional capability. `_phase_lock` hard-codes an N1-labelled chain with master bench `-1` and local test-mass terms. It is not the paper's clock-aware, frequency-planned ISI/RFI locking model. Do not equate the names without mapping the graph. |
| Clocks and sidebands | `src/TJ_ob.py:188` | `_clock` is a zero placeholder per bench; no spacecraft clock model, modulation signals, or sideband measurement channels. Two benches on the same spacecraft must share one clock realization. |
| Ranging | `src/TJ_orbit.py:405`, `:447` | Optional distance perturbations modify the orbit distance arrays. That is not measured pseudorange, and using the same disturbed distances for propagation and TDI is not a clean ranging-error experiment. Separate truth from estimates. |
| Instrument noise | `src/TJ_noise.py:25`, `:660`, `:739` | Useful power-law generators and effective laser/OMS/acceleration models. Need explicit physical units, PSD conventions, seed ownership, additional sources, and correlation assignments. Existing amplitudes are not the paper's benchmark noise budget. |
| Onboard decimation | `src/TJ_Triangle.py:127`; `src/TJ_lock.py:170` | Separate OB/GW sample rates exist, but no measurement antialias/decimation chain. The arm-locking Butterworth `filtfilt` on the primary laser is a different operation. |
| TDI | `src/TJ_tdi.py:109`, `:229` | Existing X1/X2 and cyclic channels are reusable as a structural starting point. Clock calibration, measured-delay preprocessing, frequency-consistent operators, and valid-data masks are missing. |
| Numerical support | `src/TJ_ob.py:210`; `src/TJ_gw.py:538` | Degree-31 spline signal interpolation and near-zero out-of-domain filling can contaminate boundaries. Orbit interpolation instead fills with nominal arm length. Neither is a physical prehistory policy. |

The exact indexing adapter should be:

| Paper MOSA / received link | Existing optical bench | Existing propagation arm |
|---|---:|---:|
| 12 | 1 | 3 |
| 13 | -1 | -2 |
| 23 | 2 | 1 |
| 21 | -2 | -3 |
| 31 | 3 | 2 |
| 32 | -3 | -1 |

For example, `obs[1]` is MOSA 12, while propagation arm 1 is link 23. `obs[-1]` is Python negative indexing into the seven-element bench list, not a separate dictionary key.

## 3. Bounded checks performed on this snapshot

The reproducible diagnostic and results are in `output/tdi_model_review/run_checks.py` and `checks.json`. Run from the repository root:

```sh
python3 output/tdi_model_review/run_checks.py
```

Environment: Python 3.14.0, NumPy 2.3.5, SciPy 1.16.3. The diagnostic uses deterministic signals and does not call the arm-lock controller.

| Status | Check | Evidence and interpretation |
|---|---|---|
| PASS, bounded | Construct Triangle with a 512 s duration | Import and short initialization work. This is not a full default-duration pipeline test. |
| PASS, bounded | Six distinct laser sinusoids of amplitude `1e-13`, 5 Hz sampling, constant equal arms of 10.123 s, GW/secondary noise off | On `180 < t < 480 s`, maximum absolute X1/X2 residuals are `3.34e-28` / `4.99e-28`. This supports equal-arm interior cancellation for smooth signals, not broadband or flexing-arm validation. |
| FAIL if all returned samples are treated as science data | Same run, without trimming | Full-array maxima are `2.09e-9` / `1.25e-8`, much larger than the injected lasers. Near-zero extension and repeated high-order interpolation require an explicit support policy; the precise division between missing history and spline amplification was not isolated. |
| WARN: approximation gap | Instantaneous distance versus iterated flat-spacetime moving-endpoint light time at `t=100 s` | For arms `3` and `-3`, current delays both equal `10.1350721 s`. Retarded values are `10.1355765 s` and `10.1345678 s`, differing by about +/-0.504 ms, or +/-151 km in light-distance units. This is a kinematic diagnostic, not a relativistic orbit benchmark or a measured ranging bias. |
| FAIL: internal response inconsistency | `ygwsr(3,100,...)` versus main `gw.ygw(3,100)` at an actual sample | With `f=0.007 Hz`, amplitude `1e-20`, beta 0.3, lambda 0.7, psi 0.2, results are `2.52e-21` and `8.34e-24`. Source inspection confirms both geometry and polarization differences. This does not identify the main path as independently validated. |
| NOT CHECKED | Full noise PSD normalization, unequal/flexing-arm cancellation, servo stability, clock correction, paper Figures 10-14 | These need the staged acceptance tests below. No claim of full paper reproduction is made. |

## 4. Conventions and equations to implement first

Use seconds for delays and clock deviations; metres for path displacements; Hz for laser/beatnote offsets and fluctuations; dimensionless fractional frequency for the legacy adapter. Phases, when used analytically, are in cycles. Use one-sided PSDs for positive frequencies and document lower-frequency regularization. For a Fourier transform with kernel `exp(-2*pi*i*f*t)`, a time derivative contributes `2*pi*i*f`.

Let `ij` mean reception on spacecraft i of light emitted on j. In receiver proper time:

\[
D_{ij}s(\tau)=s(\tau-d^o_{ij}(\tau)),\qquad
\mathcal D_{ij}s(\tau)=(1-\dot d^o_{ij}(\tau))D_{ij}s(\tau).
\]

The paper denotes the second operator by a dotted D. It is a Doppler-delay operator, not the derivative of the delayed function for arbitrary input. It follows from the chain rule when the input is already a frequency.

For one ISI carrier, before clock resampling and filtering:

\[
a^c_{ij}=\mathcal D_{ij}O_{ji}-\nu_0\dot d^o_{ij}-O_{ij},
\]
\[
s^\epsilon_{ij,c}=\mathcal D_{ij}\delta\nu_{ji}-\delta\nu_{ij}
 -(\nu_0+D_{ij}O_{ji})\dot H_{ij}+n_{ij}.
\]

Here `n_ij` is only shorthand for the explicitly located path/readout terms, not a recommendation to merge independent physical sources. These are Eqs. (82a), (83b) before clock processing. The current convention should reduce to `sci = D laser_remote - laser_local + y_gw + oms` after division by `nu0`, setting offset/clock/path terms appropriately and neglecting arm-rate factors. The mapping `y_gw = -H_dot` must be verified with a controlled injected phase/path perturbation rather than inferred from variable names.

For clock time `hat_tau_i = tau_i + delta_i(tau_i)`, solve for the inverse map `u_i(hat_tau_i)`. The exact frequency transformation is

\[
\nu^{\hat\tau_i}(\hat\tau_i)=
\frac{\nu^{\tau_i}(u_i(\hat\tau_i))}
 {1+\dot q_i(u_i(\hat\tau_i))}.
\]

Use the split implementation in Eqs. (75)-(77), retaining the term proportional to `-a_ij * q_dot_epsilon`. A uniform beatnote B and constant clock fractional offset y must give `B/(1+y)`. This is a simple independent acceptance test.

For ranging, use the receiver-proper-time equation

\[
R_{ij}=d^o_{ij}+\delta_i-D_{ij}\delta_j+N^R_{ij},
\]

then timestamp and filter it. Unlike a beatnote frequency, this time-difference observable is resampled as a value, without the frequency Jacobian. Convert any ranging-noise amplitude specified in metres to seconds by dividing by c.

For path and acceleration noise conversions:

\[
S_{\delta\nu_x}(f)=(\nu_0/c)^2(2\pi f)^2S_x(f),\qquad
S_{y_a}(f)=\frac{S_a(f)}{c^2(2\pi f)^2}.
\]

The latter is for single-pass test-mass velocity divided by c; apply the reflection factor once at the TMI measurement. Do not add a new factor of two to an already double-pass noise budget.

## 5. Step-by-step implementation sequence

Each step is a separately reviewable change. New modules remain alongside the current `TJ_*.py` files; add them to the explicit module list in `pyproject.toml`.

### Step 1: Freeze a useful legacy baseline and fix support handling

**Files:** `TJ_ob.py`, `TJ_gw.py`, `TJ_synthesis.py`, `TJ_tdi.py`, new `tests/test_*.py`.

- Record units, bench/link maps, sample epochs, interpolation policy, and current channel signs.
- Add deterministic no-noise, laser-only, and GW-only fixtures. Exercise all six links and X/Y/Z cyclic permutations.
- Reconcile the scalar and vectorized GW implementations using one common response kernel and an explicit polarization model.
- Generate padded prehistory/posthistory and return a valid-data mask. Compute padding from the complete propagation/locking/TDI dependency path and interpolation support, later including filter transients and clock offsets. A fixed arbitrary crop is not a general solution.
- Benchmark finite-support fractional-delay interpolation against the present degree-31 spline, including amplitude/phase error and cancellation residual. Do not choose an order solely because it is high.

**Gate:** recover the smooth interior baseline, reproduce identical scalar/vector GW outputs, and ensure invalid support cannot silently enter spectra or inference.

### Step 2: Introduce explicit signals and link identifiers

**Files:** new `TJ_signal.py`; extend `TJ_constant.py`, `TJ_ob.py`, `TJ_Triangle.py`.

- Introduce small data containers for a sampled signal (epoch, cadence, time frame, unit, valid support) and a frequency pair (offset, fluctuation).
- Keep `nu0` configurable. For paper comparisons, use the paper's optical reference; for Taiji, use an explicitly selected mission profile.
- Index internal links by `(receiver, sender)`, with adapters for signed arms and existing bench arrays.
- Preserve current fractional-frequency outputs through an explicit conversion adapter. Do not reconstruct tiny fluctuations by subtracting trends from summed MHz telemetry.
- Store epoch separately from relative sample times to protect long-duration time precision.

**Gate:** the new representation reduces to legacy outputs with extra effects off and preserves a tiny injected fluctuation alongside a large offset in its separate array.

### Step 3: Make propagation directed and time-frame aware

**Files:** extend `TJ_orbit.py`; new `TJ_delay.py` and, if needed, `TJ_time.py`.

- Define a provider for emission time, proper pseudorange, its rate, positions, and time transformations, with provenance and valid coverage.
- First implement static directed delays and a kinematic moving-endpoint solver. In the current light-second coordinates, solve `L = norm(x_receiver(t) - x_sender(t-L))`. Label this as a common-coordinate-time approximation.
- For paper-level relativistic fidelity, ingest validated proper-time products or implement separately validated TCB-to-TPS mappings and the required propagation corrections. A retarded Euclidean solution alone is not the complete paper model.
- Make both GW generation and laser propagation use the same emission events and conventions.
- Implement separate scalar/phase and frequency delay operations. For chained frequency delays, multiply Jacobians evaluated at each nested retarded time.
- Keep truth propagation separate from estimated ground-processing delays.
- Preserve existing list composition semantics through an adapter: `orbit.delay` traverses the stored arm list in reverse, and `delay_tdi` appends outer delays. Translate carefully rather than reversing lists globally.

**Gate:** static analytic light time; moving-endpoint null-path residual; direction asymmetry; noncommuting linear-in-time delays; numerical derivative of a delayed analytic phase agrees with Doppler-delayed frequency. Test changes in delay interpolation resolution.

### Step 4: Build the carrier optical measurement layer

**Files:** extend `TJ_ob.py`, `TJ_synthesis.py`, `TJ_noise.py`; optionally new `TJ_measurement.py`.

- Generate local, adjacent, and distant carrier frequency pairs.
- Assemble ISI, RFI, and TMI using Eqs. (19)-(24), (30)-(32), and (48)-(49), initially with perfect clocks and no locking.
- Add path/readout/backlink/test-mass disturbances at their physical entry points. Reuse realizations wherever the same physical path is sampled by more than one measurement.
- Keep physical source arrays distinct from combined channel noise. Avoid adding the old aggregate OMS noise on top of a new budget that already contains it.
- Expose raw measurement results without immediately creating eta.

**Gate:** single-source injection verifies signs and factors; no-noise frequency offsets match analytic beatnotes; the reduced model agrees with legacy fractional-frequency synthesis under matched approximations.

### Step 5: Add one clock model per spacecraft and frequency timestamping

**Files:** new `TJ_clock.py`; extend `TJ_Triangle.py`, measurement synthesis.

- Represent clock frequency offset/drift and stochastic fractional-frequency noise separately, together with initial timer offset.
- Integrate the clock model consistently to obtain time deviations; share one realization between both local benches and all clock-derived signals.
- Implement inverse clock mapping with convergence/residual checks. The paper's two iterations are a reference choice, not a guarantee for arbitrary user settings.
- Apply Eqs. (75)-(77) to offset/fluctuation beatnotes. Separate pre-telemetry clock readout from later filtering.

**Gate:** identity for ideal clocks; exact constant-offset/drift cases; correct `-B*q_dot` coupling; consistent timestamps and frequencies for an analytic phase; no false clock difference between co-located benches.

### Step 6: Implement frequency-plan phase locking as its own subsystem

**Files:** implement currently empty `TJ_freplan.py`; extend or separate `TJ_lock.py`; update `TJ_Triangle.py`.

- A frequency plan supplies signed reference beatnotes in each local clock time, initially constant and piecewise-linear schedules. Validate coverage and beatnote-band margins, including sidebands once Step 7 is available.
- Describe locking as a dependency graph: one primary laser, five ISI/RFI locks. Start with one verified topology and primary; add cyclic/reflected configurations only after reference checks.
- Solve offsets and fluctuations using Eqs. (97)-(100), including clock coupling and the subtraction of local optical path noise when translating the locked photodiode beam back to the source.
- Reuse locking readout-noise realizations in subsequent synthesis, so ideal locking actually constrains the measured beatnote.
- Keep arm stabilization independently selectable. The paper's baseline uses a cavity-stabilized primary; the existing single/dual/common controllers can be an optional alternative primary model after their own validation.
- Add explicit `none` choices for both arm stabilization and phase locking. Do not interpret `hasLaser=False` as permission to remove locked-laser responses driven by GW or other instrument disturbances.

**Gate:** each controlled beatnote follows its plan in local clock time before filtering, with only the expected numerical residual. Inject one GW/noise source and track its transfer through the lock graph; verify total and fluctuation channels separately.

### Step 7: Add upper sidebands and their noise correlations

**Files:** extend clock, signal, bench, measurement, and noise modules.

- Form upper-sideband frequency pairs with Eqs. (18), (84), and (86). The paper's profile assigns 2.4 GHz to 12/23/31 and 2.401 GHz to 13/32/21.
- Generate ISI and RFI sideband beatnotes, with shared carrier laser/path disturbances and appropriately separate readout/modulation noise. Do not add TMI sideband telemetry to the paper baseline.
- Add channel-specific readout and modulation PSDs, deterministic seed allocation per physical source, and explicit low-frequency cutoffs.
- Preserve legacy Taiji noise defaults as a separate profile. Add a named paper-2023 benchmark profile, recording any ambiguity or correction.

**Gate:** carrier/sideband differences reject shared laser fluctuations as predicted; remaining terms match clock/modulation/readout injections; same-spacecraft correlations are correct; frequency-plan margins include sideband offsets.

### Step 8: Produce measured pseudoranges and telemetry

**Files:** new `TJ_ranging.py`, `TJ_telemetry.py`; update `TJ_Triangle.py`.

- Generate measured pseudoranges with Eq. (103), using directed truth delays and clock deviations plus separately configurable bias/noise. Omit PRN ambiguity and GW ranging perturbations only as explicit paper-baseline assumptions.
- Resample measurements to each receiving clock, then apply FIR filtering and decimation. Use 16-to-4 Hz as the paper profile and configurable rates elsewhere.
- Appendix D specifies a Kaiser-window FIR, transition 1.1-2.9 Hz and 240 dB design attenuation above 2.9 Hz. Treat this as a design target requiring numerical verification. Its stopband is intentionally above the 2 Hz telemetry Nyquist frequency because the protected science band ends at 1 Hz.
- Track group delay, edge support, filter state, and the fact that filtering and time-varying delay generally do not commute.
- Export total beatnotes, optional split diagnostics, MPRs, per-spacecraft time axes, units, configuration, seed map, validity, and provenance. HDF5 is suitable but can remain an optional dependency; a simple array container is sufficient for the first milestone.

**Gate:** no-clock/no-ranging-noise MPR equals the truth delay; fixed clock offsets produce the correct signed bias; a deliberate ranging error affects only the estimate/processing path; alias injections above telemetry Nyquist remain below a declared science-band budget.

### Step 9: Separate ground processing and validate clock-corrected TDI

**Files:** new `TJ_preprocess.py`; adapt `TJ_synthesis.py`, `TJ_tdi.py`; optionally a PyTDI adapter.

- Define a processor that accepts telemetry plus explicit timing/delay estimates. Its production path must not read simulator truth noise arrays to subtract clock or laser noise.
- Start with an explicitly labelled truth-delay/perfect-clock diagnostic mode. Then introduce measured-delay estimates, synchronization, sideband-based clock calibration, and xi/eta formation under one consistent convention.
- Choose and document a processing route: resynchronize to a common time coordinate, or use a validated unsynchronized formulation. Do not mix formulas between the two routes.
- Derive or adopt the correction algorithm from the paper's Ref. [32], with source/version/convention checks. The instrument paper alone is insufficient to claim that this ground algorithm has been implemented.
- Retain the X/Y/Z channel structure but validate ordered Doppler-delay operators for frequency data. In the phase formulation use phase delays; do not attach Doppler factors indiscriminately to every legacy array.
- Compare with a pinned, compatible PyTDI/LISA Instrument reference on identical inputs before independent extensions. Agreement on stochastic distributions is appropriate unless realizations are shared.
- Pass valid support, units, and the final sample rate to plots and `utils/TJ_infer.py`; update analytic noise transfer functions when channels or conventions change.

**Gate:** static unequal-arm laser cancellation; flexing-arm X2 improvement relative to X1 within the expected approximation; laser/clock residuals below a declared secondary-noise budget; GW amplitude and phase recovery; deliberate clock/ranging perturbations give the expected degradation.

### Step 10: Reproduce the paper's hierarchy, then optimize

**Files:** new deterministic tests and benchmark scripts; extend the existing demo after APIs settle.

- Simple case: static arms, laser noise only, free lasers, ideal clocks, no decimation.
- Intermediate case: realistic directed delay products, phase locks/frequency plan, non-clock noises, telemetry filtering.
- Full case: clocks, sidebands, ranging, and validated ground correction.
- Compare the behavior of Figures 10-14: controlled/noncontrolled beatnotes, numerical floor with separated versus summed variables, MPR offsets/drifts, sideband clock diagnostics, and X2/Y2/Z2 GW recovery.
- Match the actual simulation duration, source amplitude, observation window, and PSD conventions. The paper's example rescales the binary amplitude to give a short run a multiyear-equivalent SNR; do not compare that SNR directly with an unscaled Taiji run.
- Profile once correctness is established. Adopt chunked array processing with delay-history buffers, persistent noise/filter states, and streamed output for long runs. Check chunked/full-array agreement with deterministic inputs.

**Gate:** reproducible configuration and outputs, documented approximation/error budget, converged numerical residuals, and bounded memory growth. A full C++ rewrite is not a prerequisite.

## 6. Source ambiguities that need explicit decisions

The rendered PDF confirms apparent typographical errors; do not transcribe its summary equations mechanically.

- **Eq. (84b), p. 15:** the subtracted local optical-path term carries the distant-path index `isi12 <- 21` again. The beam definitions in Eqs. (20b), (32b), and the distant-minus-local convention imply `isi12 <- 12` for that local term.
- **Eq. (86c), p. 15:** the left-hand side repeats the offset superscript although it is defining the total RFI sideband measurement. Follow the total = offset + fluctuation definition.
- **Eq. (94b), p. 16:** the coefficient multiplying clock timing fluctuation is printed as plan phase. Eqs. (91)-(92) and units require plan frequency. The later frequency-lock equations provide the appropriate implementation basis.
- **Eq. (104), p. 18:** the propagated timer for link 12 is printed with spacecraft index 3. Eqs. (101)-(103) require spacecraft 2. Implement the explicit receiver-minus-emitter equation.
- **Appendix B.3:** the stated acceleration budget includes a reflection factor while the optical measurement model explicitly contains `2*N_delta`. Define whether a noise parameter represents physical single-test-mass acceleration or an effective double-pass quantity, and resolve the normalization against reference implementation/benchmarks before calling the profile exact.
- **Appendix E and §IX:** duration statements are inconsistent. Use an explicit duration in seconds; the code's three-day value is 259200 s. Do not copy the prose's approximate `10^6 s` or treat the snippet as a complete reproduction of the full-noise result.

These are equation-consistency observations on the supplied paper, not an exhaustive erratum search.

## 7. Suggested first implementation milestone

Complete Steps 1-4 with ideal clocks and unlocked lasers: tested conventions, support masks, separated frequency variables, directed delay/Doppler-delay operators, and carrier ISI/TMI/RFI output that reduces to the legacy model. This gives a reviewable foundation and catches sign/unit/precision problems before clock and locking effects obscure them.

Then implement Steps 5-8 for instrument telemetry, and Steps 9-10 for a demonstrated end-to-end result. Frequency-plan optimization, new orbital dynamics, finite-bandwidth phase-lock servos, DWS/tilt-to-length, and mission-length performance are later extensions with separate validation needs.

Final verification matrix:

| Subject | Status | Scope |
|---|---|---|
| Paper-to-module mapping | PASS | Source inspection, equation tracing, and visual checks of optical layout and flagged equation typos. |
| Current equal-arm TDI | PASS, bounded | Smooth deterministic laser signals on the tested interior interval only. |
| Untrimmed output validity and duplicate GW implementations | FAIL in the tested cases | Demonstrated boundary amplification and disagreement between response paths; fixes proposed, not applied. |
| Current versus paper propagation | WARN | Demonstrated kinematic approximation gap; full relativistic correction not evaluated. |
| Full instrument/clock/noise/TDI reproduction | NOT CHECKED | Future implementation and validation gates are specified above. |

Residual uncertainty: this review does not validate the full analytical orbit model, noise generator spectra, all TDI polynomials, or arm-controller stability. The measured baseline checks establish only the narrow cases described above. Relativistic inputs, exact paper noise normalization, and the external ground-correction algorithm remain dependencies for a claim of paper-level agreement.
