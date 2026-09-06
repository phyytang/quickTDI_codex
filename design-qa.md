# quickTDI Dashboard Design QA

- Source visual truth: `docs/assets/quicktdi-dashboard-reference.png`
- Implementation: `src/TJ_dashboard.py`
- Intended viewport: 1440 × 1024
- State: default configuration, light theme, placeholder spectrum
- Implementation screenshot: unavailable — macOS returned `could not create image from display` when automated capture was attempted.

**Findings**

- [P1] Visual comparison is blocked
  - Location: full dashboard.
  - Evidence: the selected source mockup is available, and the Tkinter application launches without terminal errors, but the environment denied capture of the native application window.
  - Impact: typography, spacing, wrapping, panel proportions, and widget rendering cannot be compared from visible evidence.
  - Fix: capture the open 1440 × 1024 dashboard window and compare it with the source mockup before visual handoff.

**Required Fidelity Surfaces**

- Fonts and typography: blocked pending implementation screenshot.
- Spacing and layout rhythm: blocked pending implementation screenshot.
- Colors and visual tokens: code tokens follow the source palette, but visible rendering is blocked pending screenshot.
- Image quality and asset fidelity: the dashboard has no raster content beyond the saved source reference; plot rendering is blocked pending screenshot.
- Copy and content: statically inspected and aligned with quickTDI terminology; visible truncation remains unverified.

**Full-view Comparison Evidence**

Blocked. The implementation screenshot could not be captured, so no side-by-side evidence can be produced.

**Focused Region Comparison Evidence**

Blocked for the same reason. Configuration form density, pipeline tracker, chart legend, and status bar require focused visual inspection.

**Functional Evidence**

- `TJ_dashboard` compiles and imports successfully.
- The Tkinter application launches and remains active without terminal errors.
- A reduced orbit → GW → optical benches → locking → synthesis → X1 TDI run completed with 100 samples.
- Automated screen capture failed because display capture is unavailable in the current macOS execution context.

**Comparison History**

- Initial pass: blocked before visual comparison; no P0/P1/P2 visual fixes can be responsibly inferred without implementation evidence.

**Implementation Checklist**

1. Capture the default dashboard at 1440 × 1024.
2. Compare full-view composition with the source mockup.
3. Inspect configuration fields, pipeline tracker, plot typography, and bottom status bar at readable scale.
4. Fix any P0/P1/P2 drift and repeat capture.

**Follow-up Polish**

- Evaluate native widget differences across macOS, Windows, and Linux ttk themes after the primary macOS comparison passes.

final result: blocked
