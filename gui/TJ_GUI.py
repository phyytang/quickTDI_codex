"""
Taiji/LISA TDI Simulator - Graphical User Interface

A comprehensive GUI for running Time Delay Interferometry simulations
with interactive visualization and analysis tools.

Features:
- Orbit configuration (analytical and file-based)
- GW source parameters
- Noise configuration
- Multi-channel power spectra comparison
- Real-time interferometer data streams
- Time-windowed spectral analysis
- Interactive plotting and data export

Author: GUI wrapper for TY simulation modules
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from scipy import signal
import matplotlib
import sys
import platform

# Configure matplotlib backend for macOS compatibility
if platform.system() == 'Darwin':  # macOS
    try:
        # Try to use TkAgg, but don't force it if it causes issues
        import matplotlib
        matplotlib.use('TkAgg')
    except Exception as e:
        print(f"Warning: Could not set TkAgg backend: {e}")
        print("Using default backend instead")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from TJ_theme import (
    apply_theme, ChannelPicker, ModernFigure as Figure, show_empty_state, BG
)
import matplotlib.pyplot as plt
import threading
import json
from pathlib import Path

# Make the src/ directory (containing the TJ_*.py modules) importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Import TJ modules
try:
    from TJ_orbit import orbit
    from TJ_gw import gw
    from TJ_ob import ob
    from TJ_lock import lock
    from TJ_synthesis import synthesis
    from TJ_tdi import tdi, X1, X2, Y1, Y2, Z1, Z2, a1, a2, b1, b2, g1, g2, z1, z15, z2
    import TJ_constant as const
except ImportError as e:
    print(f"Error importing TJ modules: {e}")
    print("Make sure all TJ_*.py files are in the same directory as this GUI file.")
    raise


class LinkedCursor:
    """
    Synchronize cursor position across multiple matplotlib axes.

    Features:
    - Vertical cursor line on all axes
    - Shows same time position across all plots
    - Time value annotation
    """

    def __init__(self, axes_list):
        """
        Initialize linked cursor for multiple axes.

        Parameters
        ----------
        axes_list : list
            List of matplotlib axes to link
        """
        self.axes = axes_list
        self.cursor_lines = []
        self.enabled = False

        # Create vertical lines on each axis
        for ax in self.axes:
            line = ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.6, visible=False)
            self.cursor_lines.append(line)

    def enable(self):
        """Enable the linked cursor"""
        self.enabled = True
        for line in self.cursor_lines:
            line.set_visible(True)

    def disable(self):
        """Disable the linked cursor"""
        self.enabled = False
        for line in self.cursor_lines:
            line.set_visible(False)

    def update(self, x_position):
        """Update cursor position on all axes"""
        if not self.enabled:
            return

        for line in self.cursor_lines:
            line.set_xdata([x_position, x_position])

        # Redraw all axes
        for ax in self.axes:
            ax.figure.canvas.draw_idle()


class CollapsibleFrame(ttk.Frame):
    """
    A collapsible frame widget that can be expanded or collapsed.

    Features:
    - Click on header to toggle expand/collapse
    - Visual indicator (▼ expanded, ▶ collapsed)
    - Smooth user experience
    """

    def __init__(self, parent, text="", default_open=True, **kwargs):
        ttk.Frame.__init__(self, parent, padding=(0, 0, 0, 8), **kwargs)

        self.is_open = default_open

        # Header frame with toggle button
        self.header_frame = ttk.Frame(self)
        self.header_frame.pack(fill=tk.X, padx=2, pady=2)

        # Toggle button with arrow indicator
        self.toggle_text = tk.StringVar(value="▼ " + text if default_open else "▶ " + text)
        self.toggle_btn = ttk.Button(
            self.header_frame,
            textvariable=self.toggle_text,
            command=self.toggle,
            style="Section.TButton"
        )
        self.toggle_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Content frame (holds the actual content)
        self.content_frame = ttk.Frame(self, padding=(12, 8))
        if default_open:
            self.content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)

        self.text = text

    def toggle(self):
        """Toggle between expanded and collapsed states"""
        if self.is_open:
            # Collapse
            self.content_frame.pack_forget()
            self.toggle_text.set("▶ " + self.text)
            self.is_open = False
        else:
            # Expand
            self.content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)
            self.toggle_text.set("▼ " + self.text)
            self.is_open = True

    def get_content_frame(self):
        """Return the frame where content should be added"""
        return self.content_frame


class TaijiSimulatorGUI:
    """Main GUI application for Taiji/LISA TDI simulations"""

    def __init__(self, root):
        self.root = root
        self.root.title("Taiji/LISA TDI Simulator")
        self.root.geometry("1500x960")
        self.root.minsize(1200, 800)
        self.style = apply_theme(self.root)

        # Simulation state
        self.sim_running = False
        self.orbit_obj = None
        self.obs_list = None
        self.gw_obj = None
        self.results = {}
        self.orbit_files = {}

        # Animation state
        self.animation_running = False
        self.animation_id = None
        self.ts_animation_running = False
        self.ts_animation_id = None
        self.psd_animation_running = False
        self.psd_animation_id = None

        # TDI channel mapping
        self.tdi_channel_map = {
            'X1': X1, 'X2': X2, 'Y1': Y1, 'Y2': Y2, 'Z1': Z1, 'Z2': Z2,
            'a1': a1, 'a2': a2, 'b1': b1, 'b2': b2, 'g1': g1, 'g2': g2,
            'z1': z1, 'z15': z15, 'z2': z2
        }

        # Setup UI
        self.setup_ui()

    def setup_ui(self):
        """Setup main user interface"""
        # Create menu bar
        self.setup_menu()

        # One compact row reserves vertical space for the analysis workspace.
        header = ttk.Frame(self.root, padding=(16, 0))
        header.pack(fill=tk.X)
        ttk.Label(header, text="TDI Simulation Studio", style="Header.TLabel").pack(side=tk.LEFT)
        self.run_button = ttk.Button(header, text="▶  Run simulation",
                                     style="Header.Accent.TButton", command=self.run_simulation)
        self.run_button.pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(header, text="Export CSV", style="Header.TButton",
                   command=self.export_data).pack(side=tk.RIGHT)
        ttk.Button(header, text="Save configuration", style="Header.TButton",
                   command=self.save_config).pack(side=tk.RIGHT, padx=10)

        # Reserve space for status before the expanding workspace.
        self.setup_status_bar(self.root)
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=16, pady=(4, 8))
        left_panel = ttk.Frame(main_container, width=440)
        main_container.add(left_panel, weight=0)
        right_panel = ttk.Frame(main_container, style="Page.TFrame")
        main_container.add(right_panel, weight=1)
        self.setup_control_panel(left_panel)
        self.setup_visualization_panel(right_panel)
        self.root.after_idle(lambda: main_container.sashpos(0, 440))

    def setup_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Configuration", command=self.load_config)
        file_menu.add_command(label="Save Configuration", command=self.save_config)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results to CSV", command=self.export_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Refresh All Plots", command=self.update_all_displays)
        menubar.add_cascade(label="View", menu=view_menu)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def setup_control_panel(self, parent):
        """Create control panel with simulation parameters"""
        # Create scrollable frame
        heading = ttk.Frame(parent, padding=(16, 12))
        heading.pack(fill="x")
        ttk.Label(heading, text="SIMULATION SETUP", style="Eyebrow.TLabel").pack(anchor="w")
        ttk.Label(heading, text="Define the detector and signal", style="Muted.TLabel").pack(anchor="w")
        canvas = tk.Canvas(parent, width=430, background=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        window_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window_id, width=event.width))

        def scroll_controls(event):
            # Only consume wheel events over this sidebar; preserve plot zoom.
            widget = event.widget
            while widget is not None:
                if widget == scrollable_frame or widget == canvas:
                    if canvas.bbox("all")[3] > canvas.winfo_height():
                        delta = event.delta
                        steps = (-int(delta) if platform.system() == "Darwin"
                                 else -int(delta / 120))
                        if event.num in (4, 5):
                            steps = -1 if event.num == 4 else 1
                        canvas.yview_scroll(steps, "units")
                    return "break"
                widget = getattr(widget, "master", None)
        self.root.bind("<MouseWheel>", scroll_controls, add="+")
        self.root.bind("<Button-4>", scroll_controls, add="+")
        self.root.bind("<Button-5>", scroll_controls, add="+")
        canvas.configure(yscrollcommand=scrollbar.set)

        # A. Orbit Configuration
        orbit_collapsible = CollapsibleFrame(scrollable_frame, text="Orbit Configuration", default_open=True)
        orbit_collapsible.pack(fill="x", padx=5, pady=3)
        orbit_frame = orbit_collapsible.get_content_frame()

        ttk.Label(orbit_frame, text="Data Source:").grid(row=0, column=0, sticky="w")
        self.data_source_var = tk.StringVar(value="analytical")
        ttk.Radiobutton(orbit_frame, text="Analytical", variable=self.data_source_var,
                       value="analytical", command=self.toggle_orbit_inputs).grid(row=0, column=1)
        ttk.Radiobutton(orbit_frame, text="File", variable=self.data_source_var,
                       value="file", command=self.toggle_orbit_inputs).grid(row=0, column=2)

        # Time parameters (for analytical)
        self.time_params_frame = ttk.Frame(orbit_frame)
        self.time_params_frame.grid(row=1, column=0, columnspan=3, pady=5, sticky="ew")

        ttk.Label(self.time_params_frame, text="Start Time (s):").grid(row=0, column=0, sticky="w")
        self.t_start_var = tk.DoubleVar(value=0.0)
        ttk.Entry(self.time_params_frame, textvariable=self.t_start_var, width=15).grid(row=0, column=1, padx=5)

        ttk.Label(self.time_params_frame, text="End Time (s):").grid(row=1, column=0, sticky="w")
        self.t_end_var = tk.DoubleVar(value=20000.0)
        ttk.Entry(self.time_params_frame, textvariable=self.t_end_var, width=15).grid(row=1, column=1, padx=5)

        ttk.Label(self.time_params_frame, text="Time Step (s):").grid(row=2, column=0, sticky="w")
        self.t_step_var = tk.DoubleVar(value=1.0)
        ttk.Entry(self.time_params_frame, textvariable=self.t_step_var, width=15).grid(row=2, column=1, padx=5)

        ttk.Label(orbit_frame, text="Arm Length (light-s):").grid(row=2, column=0, sticky="w", pady=5)
        self.arm_len_var = tk.DoubleVar(value=10.0)
        ttk.Entry(orbit_frame, textvariable=self.arm_len_var, width=15).grid(row=2, column=1, padx=5)

        ttk.Label(orbit_frame, text="Orbit Type:").grid(row=3, column=0, sticky="w")
        self.orbit_type_var = tk.StringVar(value="heliocentric")
        ttk.Combobox(orbit_frame, textvariable=self.orbit_type_var,
                    values=["heliocentric", "geocentric"], width=13).grid(row=3, column=1, sticky="w", padx=5)

        # File selection (for file mode)
        self.file_params_frame = ttk.Frame(orbit_frame)
        ttk.Button(self.file_params_frame, text="Load Position File",
                  command=self.load_position_file).pack(pady=2, fill="x")
        ttk.Button(self.file_params_frame, text="Load Velocity File (opt)",
                  command=self.load_velocity_file).pack(pady=2, fill="x")
        ttk.Button(self.file_params_frame, text="Load Distance File (opt)",
                  command=self.load_distance_file).pack(pady=2, fill="x")
        self.file_status_label = ttk.Label(self.file_params_frame, text="No files loaded", foreground="gray")
        self.file_status_label.pack(pady=2)

        # B. GW Source Configuration
        gw_collapsible = CollapsibleFrame(scrollable_frame, text="GW Source Parameters", default_open=True)
        gw_collapsible.pack(fill="x", padx=5, pady=3)
        gw_frame = gw_collapsible.get_content_frame()

        ttk.Label(gw_frame, text="Frequency (Hz):").grid(row=0, column=0, sticky="w")
        self.fgw_var = tk.DoubleVar(value=0.007)
        ttk.Entry(gw_frame, textvariable=self.fgw_var, width=15).grid(row=0, column=1, padx=5)

        ttk.Label(gw_frame, text="Strain:").grid(row=1, column=0, sticky="w")
        self.strain_var = tk.DoubleVar(value=1e-24)
        ttk.Entry(gw_frame, textvariable=self.strain_var, width=15).grid(row=1, column=1, padx=5)

        ttk.Label(gw_frame, text="Ecliptic Latitude β (rad):").grid(row=2, column=0, sticky="w")
        self.beta_var = tk.DoubleVar(value=0.0)
        ttk.Entry(gw_frame, textvariable=self.beta_var, width=15).grid(row=2, column=1, padx=5)

        ttk.Label(gw_frame, text="Ecliptic Longitude λ (rad):").grid(row=3, column=0, sticky="w")
        self.lamda_var = tk.DoubleVar(value=0.0)
        ttk.Entry(gw_frame, textvariable=self.lamda_var, width=15).grid(row=3, column=1, padx=5)

        ttk.Label(gw_frame, text="Polarization ψ (rad):").grid(row=4, column=0, sticky="w")
        self.psi_var = tk.DoubleVar(value=0.0)
        ttk.Entry(gw_frame, textvariable=self.psi_var, width=15).grid(row=4, column=1, padx=5)

        # C. Noise Configuration
        noise_collapsible = CollapsibleFrame(scrollable_frame, text="Noise Settings", default_open=False)
        noise_collapsible.pack(fill="x", padx=5, pady=3)
        noise_frame = noise_collapsible.get_content_frame()

        self.laser_noise_var = tk.StringVar(value="On")
        ttk.Checkbutton(noise_frame, text="Laser Noise", variable=self.laser_noise_var,
                       onvalue="On", offvalue="Off").pack(anchor="w")

        self.acc_noise_var = tk.StringVar(value="Off")
        ttk.Checkbutton(noise_frame, text="Acceleration Noise", variable=self.acc_noise_var,
                       onvalue="On", offvalue="Off").pack(anchor="w")

        self.oms_noise_var = tk.StringVar(value="Off")
        ttk.Checkbutton(noise_frame, text="OMS Noise", variable=self.oms_noise_var,
                       onvalue="On", offvalue="Off").pack(anchor="w")

        ttk.Label(noise_frame, text="Sampling Freq (Hz):").pack(anchor="w", pady=(5,0))
        self.fsample_var = tk.DoubleVar(value=5.0)
        ttk.Entry(noise_frame, textvariable=self.fsample_var, width=15).pack(anchor="w")

        # D. Arm Locking Configuration
        lock_collapsible = CollapsibleFrame(scrollable_frame, text="Arm Locking (Laser Stabilization)", default_open=True)
        lock_collapsible.pack(fill="x", padx=5, pady=3)
        lock_frame = lock_collapsible.get_content_frame()

        self.enable_lock_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(lock_frame, text="Enable Arm Locking", variable=self.enable_lock_var,
                       command=self.toggle_lock_inputs).pack(anchor="w")

        # Lock parameters frame
        self.lock_params_frame = ttk.Frame(lock_frame)
        self.lock_params_frame.pack(fill="x", pady=5)

        ttk.Label(self.lock_params_frame, text="Lock Type:").pack(anchor="w")
        self.lock_type_var = tk.StringVar(value="dual")
        ttk.Radiobutton(self.lock_params_frame, text="Single Arm", variable=self.lock_type_var,
                       value="single", command=self.toggle_lock_params).pack(anchor="w")
        ttk.Radiobutton(self.lock_params_frame, text="Dual Arm (Recommended)", variable=self.lock_type_var,
                       value="dual", command=self.toggle_lock_params).pack(anchor="w")
        ttk.Radiobutton(self.lock_params_frame, text="Common Arm", variable=self.lock_type_var,
                       value="common", command=self.toggle_lock_params).pack(anchor="w")

        # Single arm parameters
        self.single_arm_frame = ttk.LabelFrame(self.lock_params_frame, text="Single Arm Parameters", padding=5)
        ttk.Label(self.single_arm_frame, text="Arm Delay (s):").grid(row=0, column=0, sticky="w", padx=5)
        self.arm_delay_var = tk.DoubleVar(value=20.0)
        ttk.Entry(self.single_arm_frame, textvariable=self.arm_delay_var, width=12).grid(row=0, column=1, padx=5)

        # Dual arm parameters
        self.dual_arm_frame = ttk.LabelFrame(self.lock_params_frame, text="Dual Arm Parameters", padding=5)
        ttk.Label(self.dual_arm_frame, text="Arm1 Delay (s):").grid(row=0, column=0, sticky="w", padx=5)
        self.arm1_delay_var = tk.DoubleVar(value=20.1)
        ttk.Entry(self.dual_arm_frame, textvariable=self.arm1_delay_var, width=12).grid(row=0, column=1, padx=5)

        ttk.Label(self.dual_arm_frame, text="Arm2 Delay (s):").grid(row=1, column=0, sticky="w", padx=5)
        self.arm2_delay_var = tk.DoubleVar(value=19.9)
        ttk.Entry(self.dual_arm_frame, textvariable=self.arm2_delay_var, width=12).grid(row=1, column=1, padx=5)

        # Common parameters
        common_params_frame = ttk.Frame(self.lock_params_frame)
        common_params_frame.pack(fill="x", pady=5)

        ttk.Label(common_params_frame, text="Gain Factor:").grid(row=0, column=0, sticky="w", padx=5)
        self.gfactor_var = tk.DoubleVar(value=10000.0)
        ttk.Entry(common_params_frame, textvariable=self.gfactor_var, width=12).grid(row=0, column=1, padx=5)

        ttk.Label(common_params_frame, text="Proportional Factor:").grid(row=1, column=0, sticky="w", padx=5)
        self.afactor_var = tk.DoubleVar(value=100.0)
        ttk.Entry(common_params_frame, textvariable=self.afactor_var, width=12).grid(row=1, column=1, padx=5)

        ttk.Label(common_params_frame, text="Corner Freq (Hz):").grid(row=2, column=0, sticky="w", padx=5)
        self.w0_freq_var = tk.DoubleVar(value=0.4)
        ttk.Entry(common_params_frame, textvariable=self.w0_freq_var, width=12).grid(row=2, column=1, padx=5)

        # E. TDI Channel Selection
        tdi_collapsible = CollapsibleFrame(scrollable_frame, text="TDI Channel Selection", default_open=True)
        tdi_collapsible.pack(fill="x", padx=5, pady=3)
        tdi_frame = tdi_collapsible.get_content_frame()

        channels = list(self.tdi_channel_map.keys())

        ttk.Label(tdi_frame, text="Channel 1:").pack(anchor="w")
        self.tdi_channel1_var = tk.StringVar(value="X1")
        ttk.Combobox(tdi_frame, textvariable=self.tdi_channel1_var,
                    values=channels, width=13, state='readonly').pack(anchor="w", pady=2)

        ttk.Label(tdi_frame, text="Channel 2:").pack(anchor="w", pady=(5,0))
        self.tdi_channel2_var = tk.StringVar(value="X2")
        ttk.Combobox(tdi_frame, textvariable=self.tdi_channel2_var,
                    values=channels, width=13, state='readonly').pack(anchor="w", pady=2)

        ttk.Label(tdi_frame, text="Delay Level:").pack(anchor="w")
        self.delay_level_var = tk.StringVar(value="1")
        ttk.Radiobutton(tdi_frame, text="Flexible ('1')", variable=self.delay_level_var,
                       value="1").pack(anchor="w")
        ttk.Radiobutton(tdi_frame, text="Constant ('0')", variable=self.delay_level_var,
                       value="0").pack(anchor="w")

        # E. Action Buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill="x", padx=5, pady=10)

        ttk.Button(button_frame, text="Clear Results",
                  command=self.clear_results).pack(fill="x", pady=2)

        # F. Master Timeline Control
        self.setup_master_timeline(scrollable_frame)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Initialize orbit input visibility
        self.toggle_orbit_inputs()

        # Initialize arm locking input visibility
        self.toggle_lock_inputs()

    def setup_master_timeline(self, parent):
        """Setup master timeline controller for unified navigation across all panels"""
        master_collapsible = CollapsibleFrame(parent, text="Master Timeline Control", default_open=True)
        master_collapsible.pack(fill="x", padx=5, pady=3)
        master_frame = master_collapsible.get_content_frame()

        # Main timeline slider
        ttk.Label(master_frame, text="Current Time (s):").pack(anchor="w", padx=5, pady=(5,0))

        slider_frame = ttk.Frame(master_frame)
        slider_frame.pack(fill=tk.X, padx=5, pady=2)

        self.master_time = tk.DoubleVar(value=0.0)
        self.master_slider = ttk.Scale(slider_frame, from_=0, to=10000,
                                       variable=self.master_time,
                                       orient=tk.HORIZONTAL,
                                       command=self.on_master_time_change)
        self.master_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.master_time_label = ttk.Label(slider_frame, text="0.0 s", width=8)
        self.master_time_label.pack(side=tk.LEFT, padx=2)

        # Window size and sync
        ttk.Label(master_frame, text="Window Size (s):").pack(anchor="w", padx=5, pady=(5,0))
        window_frame = ttk.Frame(master_frame)
        window_frame.pack(fill=tk.X, padx=5, pady=2)

        self.master_window = tk.DoubleVar(value=1000.0)
        ttk.Spinbox(window_frame, from_=100, to=5000, increment=100,
                   textvariable=self.master_window, width=10,
                   command=self.on_master_window_change).pack(side=tk.LEFT)

        self.sync_all_panels = tk.BooleanVar(value=True)
        ttk.Checkbutton(window_frame, text="Sync Panels",
                       variable=self.sync_all_panels).pack(side=tk.LEFT, padx=10)

        # Navigation buttons - compact layout
        ttk.Label(master_frame, text="Navigation:").pack(anchor="w", padx=5, pady=(5,0))

        nav_frame1 = ttk.Frame(master_frame)
        nav_frame1.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(nav_frame1, text="⏮ Start", width=8,
                  command=self.master_goto_start).pack(side=tk.LEFT, padx=1)
        ttk.Button(nav_frame1, text="◀◀ -1000s", width=10,
                  command=lambda: self.master_step_time(-1000)).pack(side=tk.LEFT, padx=1)
        ttk.Button(nav_frame1, text="◀ -100s", width=9,
                  command=lambda: self.master_step_time(-100)).pack(side=tk.LEFT, padx=1)

        nav_frame2 = ttk.Frame(master_frame)
        nav_frame2.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(nav_frame2, text="▶ Play", width=8,
                  command=self.master_start_animation).pack(side=tk.LEFT, padx=1)
        ttk.Button(nav_frame2, text="⏸ Pause", width=8,
                  command=self.master_stop_animation).pack(side=tk.LEFT, padx=1)
        ttk.Button(nav_frame2, text="⏭ End", width=8,
                  command=self.master_goto_end).pack(side=tk.LEFT, padx=1)

        nav_frame3 = ttk.Frame(master_frame)
        nav_frame3.pack(fill=tk.X, padx=5, pady=(2,5))
        ttk.Button(nav_frame3, text="▶ +100s", width=9,
                  command=lambda: self.master_step_time(100)).pack(side=tk.LEFT, padx=1)
        ttk.Button(nav_frame3, text="▶▶ +1000s", width=10,
                  command=lambda: self.master_step_time(1000)).pack(side=tk.LEFT, padx=1)

        # Master animation state
        self.master_animation_running = False
        self.master_animation_id = None

        # Linked cursor system
        self.linked_cursor_rt = None  # For realtime streams
        self.cursor_enabled = tk.BooleanVar(value=False)

    def setup_visualization_panel(self, parent):
        """Create visualization panel with multiple tabs"""
        # Create notebook for multiple visualization modes
        self.viz_notebook = ttk.Notebook(parent)
        self.viz_notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Power Spectra Comparison
        self.psd_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.psd_frame, text="Power Spectra")
        self.setup_psd_panel(self.psd_frame)

        # Tab 2: Real-time Interferometer Streams
        self.realtime_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.realtime_frame, text="Real-time Streams")
        self.setup_realtime_panel(self.realtime_frame)

        # Tab 3: Time Series
        self.timeseries_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.timeseries_frame, text="Time Series")
        self.setup_timeseries_panel(self.timeseries_frame)

    def setup_psd_panel(self, parent):
        """Setup power spectral density comparison panel"""
        # Control frame at top
        control_frame = ttk.LabelFrame(parent, text="Display options", padding=8)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Channel selection
        channel_frame = ChannelPicker(control_frame)
        channel_frame.pack(side=tk.TOP, fill=tk.X, pady=2)


        self.psd_channels = {
            'TDI': tk.BooleanVar(value=True),
            'Free Laser 1': tk.BooleanVar(value=False),
            'Free Laser 2': tk.BooleanVar(value=False),
            'Free Laser 3': tk.BooleanVar(value=False),
            'Locked Laser 1': tk.BooleanVar(value=False),
            'Locked Laser 2': tk.BooleanVar(value=False),
            'Locked Laser 3': tk.BooleanVar(value=False),
            'Acceleration Noise': tk.BooleanVar(value=False),
            'OMS Noise': tk.BooleanVar(value=False),
            'GW Signal': tk.BooleanVar(value=False),
        }

        for name, var in self.psd_channels.items():
            channel_frame.add_channel(text=name, variable=var,
                                      command=self.update_psd_plot)

        # Time window controls
        window_frame = ttk.LabelFrame(control_frame, text="Spectral Analysis Time Window", padding=5)
        window_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        window_frame.columnconfigure(1, weight=1)

        ttk.Label(window_frame, text="Window Start (s):").grid(row=0, column=0, sticky='w', padx=5)
        self.psd_window_start = tk.DoubleVar(value=1000.0)  # Skip initial transient (default 1000s)
        self.psd_start_scale = ttk.Scale(window_frame, from_=0, to=10000,
                                        variable=self.psd_window_start,
                                        orient=tk.HORIZONTAL, length=300,
                                        command=self.update_psd_labels)
        self.psd_start_scale.grid(row=0, column=1, padx=5, sticky="ew")
        self.psd_start_label = ttk.Label(window_frame, text="1000.0 s", width=10)
        self.psd_start_label.grid(row=0, column=2, padx=5)

        ttk.Label(window_frame, text="Window Length (s):").grid(row=1, column=0, sticky='w', padx=5)
        self.psd_window_length = tk.DoubleVar(value=1000.0)
        self.psd_length_scale = ttk.Scale(window_frame, from_=1000, to=10000,
                                         variable=self.psd_window_length,
                                         orient=tk.HORIZONTAL, length=300,
                                         command=self.update_psd_labels)
        self.psd_length_scale.grid(row=1, column=1, padx=5, sticky="ew")
        self.psd_length_label = ttk.Label(window_frame, text="1000.0 s", width=10)
        self.psd_length_label.grid(row=1, column=2, padx=5)

        ttk.Label(window_frame, text="FFT Window:").grid(row=2, column=0, sticky='w', padx=5)
        self.nperseg_var = tk.IntVar(value=16384)
        ttk.Combobox(window_frame, textvariable=self.nperseg_var,
                    values=[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
                    width=10, state='readonly').grid(row=2, column=1, sticky='w', padx=5)

        ttk.Button(window_frame, text="Update PSD",
                  command=self.update_window_psd).grid(row=2, column=2, padx=5)

        # A single compact row keeps navigation close to the spectrum.
        self.psd_nav_frame = ttk.Frame(control_frame)
        self.psd_nav_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        self.psd_auto_advance = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.psd_nav_frame, text="Auto-advance window",
                        style="Navigation.TCheckbutton",
                        variable=self.psd_auto_advance,
                        command=self.toggle_psd_auto_advance).pack(side=tk.LEFT, padx=(0, 8))

        actions = [
            ("◀◀ -1000s", lambda: self.step_psd_time(-1000)),
            ("◀ -100s", lambda: self.step_psd_time(-100)),
            ("▶ Play", self.start_psd_animation),
            ("⏸ Pause", self.stop_psd_animation),
            ("▶ +100s", lambda: self.step_psd_time(100)),
            ("▶▶ +1000s", lambda: self.step_psd_time(1000)),
        ]
        for label, command in actions:
            ttk.Button(self.psd_nav_frame, text=label, width=0,
                       style="Navigation.TButton", command=command).pack(side=tk.LEFT, padx=1)

        # Matplotlib figure for PSD
        self.psd_fig = Figure(figsize=(10, 6), dpi=100)
        self.psd_ax = self.psd_fig.add_subplot(111)
        show_empty_state(self.psd_ax, "Power spectral density")
        self.psd_ax.grid(True, alpha=0.3)

        self.psd_canvas = FigureCanvasTkAgg(self.psd_fig, parent)
        self.psd_canvas.draw()

        toolbar = NavigationToolbar2Tk(self.psd_canvas, parent)
        toolbar.update()

        self.psd_canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def setup_realtime_panel(self, parent):
        """Setup real-time data stream visualization panel"""
        # Controls
        control_frame = ttk.LabelFrame(parent, text="Display options", padding=8)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        mode_frame = ttk.Frame(control_frame)
        mode_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        ttk.Label(mode_frame, text="Display Mode:", style="Strong.TLabel").pack(side=tk.TOP, anchor="w", padx=5)

        self.realtime_mode = tk.StringVar(value='phasemeters')
        ttk.Radiobutton(mode_frame, text="Phasemeters (sci/tes/ref)",
                       variable=self.realtime_mode, value='phasemeters',
                       command=self.update_realtime_display).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="Eta Signals",
                       variable=self.realtime_mode, value='eta',
                       command=self.update_realtime_display).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="Laser Noise",
                       variable=self.realtime_mode, value='laser',
                       command=self.update_realtime_display).pack(side=tk.LEFT, padx=5)

        # Cursor toggle
        ttk.Checkbutton(mode_frame, text="Show Cursor",
                       variable=self.cursor_enabled,
                       command=self.toggle_cursor).pack(side=tk.LEFT, padx=15)

        # Arm pair selection
        arm_frame = ttk.Frame(control_frame)
        arm_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        ttk.Label(arm_frame, text="Show Arm Pair:", style="Strong.TLabel").pack(side=tk.LEFT, padx=5)
        self.arm_pair_selection = tk.StringVar(value='1')
        ttk.Radiobutton(arm_frame, text="OB1 & OB1p",
                       variable=self.arm_pair_selection, value='1',
                       command=self.update_realtime_display).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(arm_frame, text="OB2 & OB2p",
                       variable=self.arm_pair_selection, value='2',
                       command=self.update_realtime_display).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(arm_frame, text="OB3 & OB3p",
                       variable=self.arm_pair_selection, value='3',
                       command=self.update_realtime_display).pack(side=tk.LEFT, padx=5)

        # Time navigation
        nav_frame = ttk.LabelFrame(control_frame, text="Time Navigation", padding=5)
        nav_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        nav_frame.columnconfigure(1, weight=1)

        ttk.Label(nav_frame, text="Current Time (s):").grid(row=0, column=0, sticky='w', padx=5)
        self.realtime_position = tk.DoubleVar(value=0.0)
        self.realtime_slider = ttk.Scale(nav_frame, from_=0, to=10000,
                                        variable=self.realtime_position,
                                        orient=tk.HORIZONTAL, length=220,
                                        command=self.update_realtime_labels)
        self.realtime_slider.grid(row=0, column=1, padx=5, sticky="ew")
        self.realtime_time_label = ttk.Label(nav_frame, text="0.0 s", width=10)
        self.realtime_time_label.grid(row=0, column=2, padx=5)

        ttk.Label(nav_frame, text="Display Window (s):").grid(row=1, column=0, sticky='w', padx=5)
        self.realtime_window = tk.DoubleVar(value=100.0)
        ttk.Spinbox(nav_frame, from_=100, to=1000, increment=100,
                   textvariable=self.realtime_window, width=10).grid(row=1, column=1, sticky='w', padx=5)

        ttk.Button(nav_frame, text="Update Display",
                  command=self.update_realtime_display).grid(row=1, column=2, padx=5)

        # Animation controls
        anim_frame = ttk.Frame(nav_frame)
        anim_frame.grid(row=2, column=0, columnspan=3, pady=5)

        ttk.Button(anim_frame, text="◀◀ -1000s",
                  command=lambda: self.step_time(-1000)).pack(side=tk.LEFT, padx=2)
        ttk.Button(anim_frame, text="◀ -100s",
                  command=lambda: self.step_time(-100)).pack(side=tk.LEFT, padx=2)
        ttk.Button(anim_frame, text="▶ Play",
                  command=self.start_animation).pack(side=tk.LEFT, padx=2)
        ttk.Button(anim_frame, text="⏸ Pause",
                  command=self.stop_animation).pack(side=tk.LEFT, padx=2)
        ttk.Button(anim_frame, text="▶ +100s",
                  command=lambda: self.step_time(100)).pack(side=tk.LEFT, padx=2)
        ttk.Button(anim_frame, text="▶▶ +1000s",
                  command=lambda: self.step_time(1000)).pack(side=tk.LEFT, padx=2)

        # Create figure with 2 subplots (for selected arm pair)
        self.realtime_fig = Figure(figsize=(12, 6), dpi=100)

        self.realtime_axes = []
        for i in range(2):
            ax = self.realtime_fig.add_subplot(2, 1, i+1)
            ax.set_title(f"Optical Bench {i+1}")
            ax.grid(True, alpha=0.3)
            self.realtime_axes.append(ax)


        self.realtime_canvas = FigureCanvasTkAgg(self.realtime_fig, parent)
        self.realtime_canvas.draw()

        # Initialize linked cursor for realtime streams
        self.linked_cursor_rt = LinkedCursor(self.realtime_axes)

        # Connect mouse motion event
        self.realtime_canvas.mpl_connect('motion_notify_event', self.on_realtime_mouse_move)

        toolbar = NavigationToolbar2Tk(self.realtime_canvas, parent)
        toolbar.update()

        self.realtime_canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def setup_timeseries_panel(self, parent):
        """Setup real-time time series panel with navigation controls"""
        # Controls
        control_frame = ttk.LabelFrame(parent, text="Display options", padding=8)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Time navigation
        nav_frame = ttk.LabelFrame(control_frame, text="Time Navigation", padding=5)
        nav_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        nav_frame.columnconfigure(1, weight=1)

        ttk.Label(nav_frame, text="Current Time (s):").grid(row=0, column=0, sticky='w', padx=5)
        self.ts_position = tk.DoubleVar(value=0.0)
        self.ts_slider = ttk.Scale(nav_frame, from_=0, to=10000,
                                   variable=self.ts_position,
                                   orient=tk.HORIZONTAL, length=220,
                                   command=self.update_ts_labels)
        self.ts_slider.grid(row=0, column=1, padx=5, sticky="ew")
        self.ts_time_label = ttk.Label(nav_frame, text="0.0 s", width=10)
        self.ts_time_label.grid(row=0, column=2, padx=5)

        ttk.Label(nav_frame, text="Display Window (s):").grid(row=1, column=0, sticky='w', padx=5)
        self.ts_window = tk.DoubleVar(value=100.0)
        ttk.Spinbox(nav_frame, from_=100, to=1000, increment=100,
                   textvariable=self.ts_window, width=10).grid(row=1, column=1, sticky='w', padx=5)

        ttk.Button(nav_frame, text="Update Display",
                  command=self.update_timeseries_display).grid(row=1, column=2, padx=5)

        # Animation controls
        anim_frame = ttk.Frame(nav_frame)
        anim_frame.grid(row=2, column=0, columnspan=3, pady=5)

        ttk.Button(anim_frame, text="◀◀ -1000s",
                  command=lambda: self.step_ts_time(-1000)).pack(side=tk.LEFT, padx=2)
        ttk.Button(anim_frame, text="◀ -100s",
                  command=lambda: self.step_ts_time(-100)).pack(side=tk.LEFT, padx=2)
        ttk.Button(anim_frame, text="▶ Play",
                  command=self.start_ts_animation).pack(side=tk.LEFT, padx=2)
        ttk.Button(anim_frame, text="⏸ Pause",
                  command=self.stop_ts_animation).pack(side=tk.LEFT, padx=2)
        ttk.Button(anim_frame, text="▶ +100s",
                  command=lambda: self.step_ts_time(100)).pack(side=tk.LEFT, padx=2)
        ttk.Button(anim_frame, text="▶▶ +1000s",
                  command=lambda: self.step_ts_time(1000)).pack(side=tk.LEFT, padx=2)

        # Matplotlib figure
        self.ts_fig = Figure(figsize=(10, 6), dpi=100)
        self.ts_ax = self.ts_fig.add_subplot(111)
        show_empty_state(self.ts_ax, "TDI time series")
        self.ts_ax.grid(True, alpha=0.3)

        self.ts_canvas = FigureCanvasTkAgg(self.ts_fig, parent)
        self.ts_canvas.draw()

        toolbar = NavigationToolbar2Tk(self.ts_canvas, parent)
        toolbar.update()

        self.ts_canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def setup_status_bar(self, parent):
        """Create status bar with progress indicator"""
        status_frame = ttk.Frame(parent, padding=(24, 8))
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_var = tk.StringVar(value="Ready · Configure parameters and run a simulation")
        status_label = ttk.Label(status_frame, textvariable=self.status_var,
                                style="Muted.TLabel", anchor=tk.W)
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=2)

        self.progress_var = tk.IntVar(value=0)
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var,
                                           maximum=100, length=200, mode='determinate')
        self.progress_bar.pack(side=tk.RIGHT, padx=5, pady=2)

    # Helper methods
    def update_status(self, message):
        """Update status bar message"""
        self.status_var.set(message)
        self.root.update_idletasks()

    def toggle_orbit_inputs(self):
        """Toggle visibility of orbit input fields based on data source"""
        if self.data_source_var.get() == "analytical":
            self.time_params_frame.grid()
            self.file_params_frame.grid_remove()
        else:
            self.time_params_frame.grid_remove()
            self.file_params_frame.grid(row=1, column=0, columnspan=3, pady=5)

    def toggle_lock_inputs(self):
        """Toggle visibility of arm locking parameter inputs"""
        if self.enable_lock_var.get():
            self.lock_params_frame.pack(fill="x", pady=5)
            self.toggle_lock_params()
        else:
            self.lock_params_frame.pack_forget()

    def toggle_lock_params(self):
        """Toggle between single and dual/common arm locking parameters"""
        if self.lock_type_var.get() == "single":
            self.dual_arm_frame.pack_forget()
            self.single_arm_frame.pack(fill="x", pady=5)
        else:  # dual or common (both use two arm delays)
            self.single_arm_frame.pack_forget()
            self.dual_arm_frame.pack(fill="x", pady=5)

    def load_position_file(self):
        """Load orbit position file"""
        filename = filedialog.askopenfilename(
            title="Select Position File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.orbit_files['positions'] = filename
            self.update_file_status()

    def load_velocity_file(self):
        """Load orbit velocity file"""
        filename = filedialog.askopenfilename(
            title="Select Velocity File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.orbit_files['velocities'] = filename
            self.update_file_status()

    def load_distance_file(self):
        """Load orbit distance file"""
        filename = filedialog.askopenfilename(
            title="Select Distance File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.orbit_files['distances'] = filename
            self.update_file_status()

    def update_file_status(self):
        """Update file loading status label"""
        files_loaded = len(self.orbit_files)
        if files_loaded == 0:
            self.file_status_label.config(text="No files loaded", foreground="gray")
        else:
            file_list = ", ".join(self.orbit_files.keys())
            self.file_status_label.config(text=f"Loaded: {file_list}", foreground="green")

    def update_psd_labels(self, event=None):
        """Update PSD time window labels"""
        self.psd_start_label.config(text=f"{self.psd_window_start.get():.1f} s")
        self.psd_length_label.config(text=f"{self.psd_window_length.get():.1f} s")

    def update_realtime_labels(self, event=None):
        """Update real-time display labels"""
        self.realtime_time_label.config(text=f"{self.realtime_position.get():.1f} s")

    def update_ts_labels(self, event=None):
        """Update time series display labels"""
        self.ts_time_label.config(text=f"{self.ts_position.get():.1f} s")

    def toggle_cursor(self):
        """Toggle linked cursor on/off"""
        if self.cursor_enabled.get():
            if self.linked_cursor_rt:
                self.linked_cursor_rt.enable()
        else:
            if self.linked_cursor_rt:
                self.linked_cursor_rt.disable()
        # Redraw
        if self.linked_cursor_rt and len(self.linked_cursor_rt.axes) > 0:
            self.realtime_canvas.draw_idle()

    def on_realtime_mouse_move(self, event):
        """Handle mouse motion on realtime streams to update cursor"""
        if event.inaxes and self.cursor_enabled.get() and self.linked_cursor_rt:
            self.linked_cursor_rt.update(event.xdata)

    # Simulation methods
    def run_simulation(self):
        """Execute the full TDI simulation pipeline"""
        if self.sim_running:
            messagebox.showwarning("Warning", "Simulation already running!")
            return

        # Validate inputs
        if self.data_source_var.get() == "file" and not self.orbit_files:
            messagebox.showerror("Error", "Please load orbit data files first!")
            return

        # Run in separate thread
        thread = threading.Thread(target=self._run_simulation_thread, daemon=True)
        thread.start()

    def _run_simulation_thread(self):
        """Simulation thread worker"""
        try:
            self.sim_running = True

            # 1. Create orbit
            self.update_status("Creating orbit...")
            self.progress_var.set(10)

            if self.data_source_var.get() == "analytical":
                self.orbit_obj = orbit(
                    t_start=self.t_start_var.get(),
                    t_end=self.t_end_var.get(),
                    t_step=self.t_step_var.get(),
                    tri_arm=self.arm_len_var.get(),
                    orbit_type=self.orbit_type_var.get()
                )
            else:
                self.orbit_obj = orbit(
                    tri_arm=self.arm_len_var.get(),
                    data_source='file',
                    data_files=self.orbit_files
                )
                # Update time range from loaded data
                self.t_start_var.set(self.orbit_obj._t_start)
                self.t_end_var.set(self.orbit_obj._t_end)

            # 2. Create optical benches
            self.update_status("Creating optical benches...")
            self.progress_var.set(25)

            ob_params = {
                't_start': self.orbit_obj._t_start if self.data_source_var.get() == 'file' else self.t_start_var.get(),
                't_end': self.orbit_obj._t_end if self.data_source_var.get() == 'file' else self.t_end_var.get(),
                'fsample': self.fsample_var.get(),
                'hasLaser': self.laser_noise_var.get() == "On",
                'hasAcc': self.acc_noise_var.get() == "On",
                'hasOms': self.oms_noise_var.get() == "On"
            }

            self.obs_list = [None]
            for i in range(6):
                self.obs_list.append(ob(**ob_params))
                self.progress_var.set(25 + i * 5)
                self.update_status(f"Created optical bench {i+1}/6")

            # 3. Create GW signal
            self.update_status("Generating GW signal...")
            self.progress_var.set(60)

            self.gw_obj = gw(
                orbits=self.orbit_obj,
                t_start=ob_params['t_start'],
                t_end=ob_params['t_end'],
                fsample=self.fsample_var.get(),
                fgw=self.fgw_var.get(),
                strain=self.strain_var.get(),
                hasGW=True,
                beta=self.beta_var.get(),
                lamda=self.lamda_var.get(),
                psi=self.psi_var.get()
            )

            # 4. Apply Arm Locking (if enabled)
            if self.enable_lock_var.get():
                self.update_status("Applying arm locking...")
                self.progress_var.set(70)

                lock_params = {
                    'orbits': self.orbit_obj,
                    'obs': self.obs_list,
                    'gws': self.gw_obj,
                    'lock_type': self.lock_type_var.get(),
                    'gfactor': self.gfactor_var.get(),
                    'afactor': self.afactor_var.get(),
                    'w0': 2 * np.pi * self.w0_freq_var.get()
                }

                if self.lock_type_var.get() == 'single':
                    lock_params['arm_delay'] = self.arm_delay_var.get()
                    print(f"DEBUG: Arm locking (single) with arm_delay={self.arm_delay_var.get()}s")
                else:  # dual or common
                    lock_params['arm1_delay'] = self.arm1_delay_var.get()
                    lock_params['arm2_delay'] = self.arm2_delay_var.get()
                    print(f"DEBUG: Arm locking ({self.lock_type_var.get()}) with arm1_delay={self.arm1_delay_var.get()}s, arm2_delay={self.arm2_delay_var.get()}s")

                lk = lock(**lock_params)
                lk.run()
                print("DEBUG: Arm locking completed")
            else:
                print("DEBUG: Arm locking disabled")

            # 5. Synthesize
            self.update_status("Synthesizing signals...")
            self.progress_var.set(80)

            delay_level = self.delay_level_var.get()
            print(f"DEBUG: Using delay_level='{delay_level}' for synthesis")
            syn = synthesis(self.orbit_obj, self.obs_list, self.gw_obj,
                          delay_level=delay_level)

            # 6. Run TDI
            self.update_status("Computing TDI...")
            self.progress_var.set(90)

            tdi_obj = tdi(self.orbit_obj, self.obs_list)

            # Compute both TDI channels
            channel1_name = self.tdi_channel1_var.get()
            channel2_name = self.tdi_channel2_var.get()
            selected_channel1 = self.tdi_channel_map[channel1_name]
            selected_channel2 = self.tdi_channel_map[channel2_name]

            print(f"DEBUG: Computing TDI channels '{channel1_name}' and '{channel2_name}' with delay_level='{delay_level}'")
            tdi_output1 = tdi_obj.run(TDI_channel=selected_channel1, delay_level=delay_level)
            tdi_output2 = tdi_obj.run(TDI_channel=selected_channel2, delay_level=delay_level)

            # Store results
            self.results = {
                'tdi_output1': tdi_output1,
                'tdi_output2': tdi_output2,
                'channel1_name': channel1_name,
                'channel2_name': channel2_name,
                'time_array': self.obs_list[1]._tarray,
                'orbit': self.orbit_obj,
                'obs': self.obs_list,
                'gw': self.gw_obj
            }

            # Update slider ranges and set intelligent PSD window defaults
            max_time = self.results['time_array'][-1]
            self.root.after(0, lambda: self.psd_start_scale.config(to=max_time))
            self.root.after(0, lambda: self.realtime_slider.config(to=max_time))
            self.root.after(0, lambda: self.ts_slider.config(to=max_time))
            self.root.after(0, lambda: self.master_slider.config(to=max_time))

            # Skip initial and final transients (1000s each) for accurate PSD with flexible TDI
            # This matches the approach in test.ipynb: a_start = 1000*fphy
            transient_time = 1000.0  # seconds
            if max_time > 2 * transient_time:
                # Set PSD window to skip transients
                psd_start = transient_time
                psd_length = max_time - 2 * transient_time
                self.root.after(0, lambda: self.psd_window_start.set(psd_start))
                self.root.after(0, lambda: self.psd_window_length.set(psd_length))
                print(f"DEBUG: Set PSD window to t={psd_start:.0f}s to t={psd_start+psd_length:.0f}s (skipping {transient_time:.0f}s transients)")
            else:
                print(f"DEBUG: Simulation too short ({max_time:.0f}s) to skip transients, PSD may show artifacts with flexible TDI")

            self.update_status("Simulation complete!")
            self.progress_var.set(100)

            # Auto-update all displays
            self.root.after(0, self.update_all_displays)

            self.root.after(0, lambda: messagebox.showinfo("Success", "Initilization/Simulation completed successfully!"))

        except Exception as e:
            import traceback
            error_msg = f"Simulation failed:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            self.update_status("Error occurred")

        finally:
            self.sim_running = False

    def update_all_displays(self):
        """Update all visualization tabs"""
        if not self.results:
            return
        self.update_window_psd()
        self.update_realtime_display()
        self.update_timeseries_display()

    def update_window_psd(self):
        """Recompute and plot PSD for selected time window"""
        if 'tdi_output1' not in self.results:
            return

        try:
            # Get parameters
            t_start = self.psd_window_start.get()
            t_length = self.psd_window_length.get()
            t_array = self.results['time_array']
            fsample = self.fsample_var.get()

            # Find indices
            idx_start = np.searchsorted(t_array, t_start)
            idx_end = np.searchsorted(t_array, t_start + t_length)

            if idx_end - idx_start < 100:
                messagebox.showwarning("Warning", "Time window too small for PSD calculation")
                return

            # Get actual time values from data (accounting for sampling frequency)
            actual_t_start = t_array[idx_start]
            actual_t_end = t_array[min(idx_end - 1, len(t_array) - 1)]

            # Clear plot
            self.psd_ax.clear()

            nperseg = min(self.nperseg_var.get(), idx_end - idx_start)
            colors = plt.cm.tab10.colors
            color_idx = 0

            # Plot TDI channels
            if self.psd_channels['TDI'].get():
                # Plot TDI channel 1
                if 'tdi_output1' in self.results:
                    data = self.results['tdi_output1'][idx_start:idx_end]
                    freq, psd = signal.welch(data, fs=fsample, nperseg=nperseg)
                    self.psd_ax.loglog(freq, np.sqrt(psd),
                                      label=f"TDI {self.results['channel1_name']}",
                                      color=colors[color_idx % 10], linewidth=2)
                    color_idx += 1

                # Plot TDI channel 2
                if 'tdi_output2' in self.results:
                    data = self.results['tdi_output2'][idx_start:idx_end]
                    freq, psd = signal.welch(data, fs=fsample, nperseg=nperseg)
                    self.psd_ax.loglog(freq, np.sqrt(psd),
                                      label=f"TDI {self.results['channel2_name']}",
                                      color=colors[color_idx % 10], linewidth=2)
                    color_idx += 1

            # Plot free-running laser noises (before arm locking)
            free_laser_labels = ['Free Laser 1', 'Free Laser 2', 'Free Laser 3']
            free_laser_indices = [1, 2, 3]

            for label, idx in zip(free_laser_labels, free_laser_indices):
                if self.psd_channels[label].get():
                    data = self.obs_list[idx]._laser_noise[idx_start:idx_end]
                    freq, psd = signal.welch(data, fs=fsample, nperseg=nperseg)
                    self.psd_ax.loglog(freq, np.sqrt(psd), label=label,
                                      color=colors[color_idx % 10], alpha=0.7, linestyle=':')
                    color_idx += 1

            # Plot locked laser noises (after arm locking, if applied)
            locked_laser_labels = ['Locked Laser 1', 'Locked Laser 2', 'Locked Laser 3']
            locked_laser_indices = [1, 2, 3]

            for label, idx in zip(locked_laser_labels, locked_laser_indices):
                if self.psd_channels[label].get():
                    data = self.obs_list[idx]._laser[idx_start:idx_end]
                    freq, psd = signal.welch(data, fs=fsample, nperseg=nperseg)
                    self.psd_ax.loglog(freq, np.sqrt(psd), label=label,
                                      color=colors[color_idx % 10], alpha=0.7)
                    color_idx += 1

            # Show the raw OB1 noise realizations, using the same time window
            # and ASD estimator as the laser channels (not TDI-propagated noise).
            for label, attribute in (("Acceleration Noise", "_acc"), ("OMS Noise", "_oms")):
                if self.psd_channels[label].get():
                    data = getattr(self.obs_list[1], attribute)[idx_start:idx_end]
                    freq, psd = signal.welch(data, fs=fsample, nperseg=nperseg)
                    self.psd_ax.loglog(freq, np.sqrt(psd), label=f"{label} (OB1)",
                                      color=colors[color_idx % 10], linestyle="--")
                    color_idx += 1

            # Plot GW signal
            if self.psd_channels['GW Signal'].get():
                data = self.gw_obj.gw[1](t_array[idx_start:idx_end])
                freq, psd = signal.welch(data, fs=fsample, nperseg=nperseg)
                self.psd_ax.loglog(freq, np.sqrt(psd), label='GW Signal (Arm 1)',
                                  color='red', linewidth=2, linestyle='--')

            # Formatting
            self.psd_ax.set_xlabel('Frequency (Hz)', fontsize=12)
            self.psd_ax.set_ylabel('Amplitude Spectral Density [1/√Hz]', fontsize=12)
            self.psd_ax.set_title(f'Power Spectra (t = {actual_t_start:.1f} to {actual_t_end:.1f} s, Δt = {actual_t_end-actual_t_start:.1f} s)',
                                 fontsize=14)
            self.psd_ax.set_xlim([0.0001, 2.0])  # Set frequency range from 0.0001 Hz to 2 Hz
            self.psd_ax.grid(True, which='both', alpha=0.3)
            self.psd_ax.legend(loc='upper left', fontsize=11)

            self.psd_canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to update PSD:\n{str(e)}")

    def update_psd_plot(self):
        """Callback for channel checkbox changes"""
        self.update_window_psd()

    def update_realtime_display(self):
        """Update real-time data stream display"""
        if not self.obs_list:
            return

        try:
            t_center = self.realtime_position.get()
            window = self.realtime_window.get()
            t_array = self.results['time_array']

            # Find indices
            t_start = max(0, t_center - window/2)
            t_end = t_center + window/2
            idx_start = np.searchsorted(t_array, t_start)
            idx_end = np.searchsorted(t_array, t_end)

            if idx_end - idx_start < 10:
                return

            t_plot = t_array[idx_start:idx_end]

            # Get actual time values from data
            actual_t_start = t_plot[0] if len(t_plot) > 0 else t_start
            actual_t_end = t_plot[-1] if len(t_plot) > 0 else t_end

            # Clear axes
            for ax in self.realtime_axes:
                ax.clear()

            mode = self.realtime_mode.get()

            # Get selected arm pair
            arm_pair = self.arm_pair_selection.get()  # '1', '2', or '3'

            # Map arm pair to optical bench indices
            if arm_pair == '1':
                ob_indices = [1, -1]  # OB1 and OB1p
                arm_labels = ['OB1 (Arm 1)', 'OB1p (Arm -1)']
            elif arm_pair == '2':
                ob_indices = [2, -2]  # OB2 and OB2p
                arm_labels = ['OB2 (Arm 2)', 'OB2p (Arm -2)']
            else:  # '3'
                ob_indices = [3, -3]  # OB3 and OB3p
                arm_labels = ['OB3 (Arm 3)', 'OB3p (Arm -3)']

            for i in range(2):
                ob_idx = ob_indices[i]
                ob = self.obs_list[ob_idx]
                ax = self.realtime_axes[i]

                if mode == 'phasemeters':
                    ax.plot(t_plot, ob._sci[idx_start:idx_end], label='sci', alpha=0.7, linewidth=1.0)
                    ax.plot(t_plot, ob._tes[idx_start:idx_end], label='tes', alpha=0.7, linewidth=1.0)
                    ax.plot(t_plot, ob._ref[idx_start:idx_end], label='ref', alpha=0.7, linewidth=1.0)
                    ax.set_ylabel('Phase', fontsize=12)

                elif mode == 'eta':
                    ax.plot(t_plot, ob._eta[idx_start:idx_end], label='η', color='blue', linewidth=1.0)
                    ax.set_ylabel('η signal', fontsize=12)

                elif mode == 'laser':
                    ax.plot(t_plot, ob._laser[idx_start:idx_end], label='Laser', color='red', linewidth=1.0)
                    ax.set_ylabel('Laser Noise', fontsize=12)

                # Add time range to title
                title_with_time = f"{arm_labels[i]} (t = {actual_t_start:.1f} to {actual_t_end:.1f} s)"
                ax.set_title(title_with_time, fontsize=11)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper left', fontsize=11)
                ax.tick_params(labelsize=11)

                if i == 0:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel('Time (s)', fontsize=12)

            self.realtime_canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to update real-time display:\n{str(e)}")

    def update_timeseries_display(self):
        """Update time series plot with time window navigation"""
        if 'tdi_output1' not in self.results:
            return

        try:
            t_center = self.ts_position.get()
            window = self.ts_window.get()
            t_array = self.results['time_array']

            # Find indices for time window
            t_start = max(0, t_center - window/2)
            t_end = t_center + window/2
            idx_start = np.searchsorted(t_array, t_start)
            idx_end = np.searchsorted(t_array, t_end)

            if idx_end - idx_start < 10:
                return

            t_plot = t_array[idx_start:idx_end]

            # Get actual time values from data
            actual_t_start = t_plot[0] if len(t_plot) > 0 else t_start
            actual_t_end = t_plot[-1] if len(t_plot) > 0 else t_end

            # Clear and plot
            self.ts_ax.clear()

            # Plot both TDI channels
            tdi_data1 = self.results['tdi_output1'][idx_start:idx_end]
            tdi_data2 = self.results['tdi_output2'][idx_start:idx_end]

            self.ts_ax.plot(t_plot, tdi_data1, linewidth=0.8, alpha=0.8,
                           label=f"TDI {self.results['channel1_name']}")
            self.ts_ax.plot(t_plot, tdi_data2, linewidth=0.8, alpha=0.8,
                           label=f"TDI {self.results['channel2_name']}")

            self.ts_ax.set_xlabel('Time (s)', fontsize=12)
            self.ts_ax.set_ylabel('TDI Output', fontsize=12)
            self.ts_ax.set_title(f'TDI Channels - Time Series (t = {actual_t_start:.1f} to {actual_t_end:.1f} s)', fontsize=14)
            self.ts_ax.legend(loc='upper left')
            self.ts_ax.grid(True, alpha=0.3)

            self.ts_canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to update time series:\n{str(e)}")

    # Animation methods
    def start_animation(self):
        """Start animated scrolling"""
        if not self.results or self.animation_running:
            return

        self.animation_running = True

        def animate():
            if not self.animation_running:
                return

            current = self.realtime_position.get()
            max_time = self.results['time_array'][-1]

            new_time = current + 10
            if new_time > max_time:
                new_time = 0

            self.realtime_position.set(new_time)
            self.update_realtime_display()

            self.animation_id = self.root.after(100, animate)

        animate()

    def stop_animation(self):
        """Stop animation"""
        self.animation_running = False
        if self.animation_id:
            self.root.after_cancel(self.animation_id)
            self.animation_id = None

    def step_time(self, delta_t):
        """Step time forward or backward"""
        if not self.results:
            return

        current = self.realtime_position.get()
        max_time = self.results['time_array'][-1]
        new_time = np.clip(current + delta_t, 0, max_time)
        self.realtime_position.set(new_time)
        self.update_realtime_display()

    # Time series animation methods
    def start_ts_animation(self):
        """Start animated scrolling for time series"""
        if not self.results or self.ts_animation_running:
            return

        self.ts_animation_running = True

        def animate():
            if not self.ts_animation_running:
                return

            current = self.ts_position.get()
            max_time = self.results['time_array'][-1]

            new_time = current + 10
            if new_time > max_time:
                new_time = 0

            self.ts_position.set(new_time)
            self.update_timeseries_display()

            self.ts_animation_id = self.root.after(100, animate)

        animate()

    def stop_ts_animation(self):
        """Stop time series animation"""
        self.ts_animation_running = False
        if self.ts_animation_id:
            self.root.after_cancel(self.ts_animation_id)
            self.ts_animation_id = None

    def step_ts_time(self, delta_t):
        """Step time series time forward or backward"""
        if not self.results:
            return

        current = self.ts_position.get()
        max_time = self.results['time_array'][-1]
        new_time = np.clip(current + delta_t, 0, max_time)
        self.ts_position.set(new_time)
        self.update_timeseries_display()

    # PSD animation methods
    def toggle_psd_auto_advance(self):
        """Toggle PSD auto-advance mode"""
        if self.psd_auto_advance.get():
            # Start auto-advance if results exist
            if self.results:
                self.start_psd_animation()
        else:
            # Stop auto-advance
            self.stop_psd_animation()

    def start_psd_animation(self):
        """Start animated PSD window advancement"""
        if not self.results or self.psd_animation_running:
            return

        self.psd_animation_running = True

        def animate():
            if not self.psd_animation_running:
                return

            current = self.psd_window_start.get()
            window_length = self.psd_window_length.get()
            max_time = self.results['time_array'][-1]

            # Advance by window_length to show next segment
            new_start = current + window_length
            if new_start + window_length > max_time:
                # Wrap around to beginning (skip initial transient)
                new_start = 1000.0

            self.psd_window_start.set(new_start)
            self.update_window_psd()

            # Auto-advance every 2 seconds of wall-clock time
            self.psd_animation_id = self.root.after(2000, animate)

        animate()

    def stop_psd_animation(self):
        """Stop PSD animation"""
        self.psd_animation_running = False
        if self.psd_animation_id:
            self.root.after_cancel(self.psd_animation_id)
            self.psd_animation_id = None

    def step_psd_time(self, delta_t):
        """Step PSD window forward or backward"""
        if not self.results:
            return

        current = self.psd_window_start.get()
        window_length = self.psd_window_length.get()
        max_time = self.results['time_array'][-1]

        # Ensure we don't go past the end
        max_start = max_time - window_length
        new_start = np.clip(current + delta_t, 0, max_start)
        self.psd_window_start.set(new_start)
        self.update_window_psd()

    # Master timeline methods
    def on_master_time_change(self, event=None):
        """Handle master timeline slider change"""
        current_time = self.master_time.get()
        self.master_time_label.config(text=f"{current_time:.1f} s")

        if self.sync_all_panels.get() and self.results:
            # Sync all panels to master timeline
            self.sync_panels_to_master()

    def on_master_window_change(self):
        """Handle master window size change"""
        if self.sync_all_panels.get() and self.results:
            # Update all panel window sizes
            window_size = self.master_window.get()
            self.psd_window_length.set(window_size)
            self.realtime_window.set(min(window_size, 1000))  # Cap realtime at 1000s
            self.ts_window.set(min(window_size, 1000))
            self.sync_panels_to_master()

    def sync_panels_to_master(self):
        """Synchronize all panel positions to master timeline"""
        current_time = self.master_time.get()
        window_size = self.master_window.get()

        # Update PSD panel (window start)
        self.psd_window_start.set(current_time)
        self.update_window_psd()

        # Update realtime panel (center position)
        self.realtime_position.set(current_time + self.realtime_window.get() / 2)
        self.update_realtime_display()

        # Update time series panel (center position)
        self.ts_position.set(current_time + self.ts_window.get() / 2)
        self.update_timeseries_display()

    def master_step_time(self, delta_t):
        """Step master timeline forward or backward"""
        if not self.results:
            return

        current = self.master_time.get()
        max_time = self.results['time_array'][-1]
        new_time = np.clip(current + delta_t, 0, max_time)
        self.master_time.set(new_time)
        self.sync_panels_to_master()

    def master_goto_start(self):
        """Go to start of simulation"""
        if self.results:
            self.master_time.set(1000.0)  # Skip initial transient
            self.sync_panels_to_master()

    def master_goto_end(self):
        """Go to end of simulation"""
        if self.results:
            max_time = self.results['time_array'][-1]
            window_size = self.master_window.get()
            self.master_time.set(max(0, max_time - window_size - 1000))  # Skip final transient
            self.sync_panels_to_master()

    def master_start_animation(self):
        """Start master timeline animation"""
        if not self.results or self.master_animation_running:
            return

        self.master_animation_running = True

        def animate():
            if not self.master_animation_running:
                return

            current = self.master_time.get()
            max_time = self.results['time_array'][-1]
            window_size = self.master_window.get()

            new_time = current + 100  # Advance by 100s
            if new_time > max_time - window_size:
                new_time = 1000.0  # Wrap to start

            self.master_time.set(new_time)
            self.sync_panels_to_master()

            self.master_animation_id = self.root.after(200, animate)

        animate()

    def master_stop_animation(self):
        """Stop master timeline animation"""
        self.master_animation_running = False
        if self.master_animation_id:
            self.root.after_cancel(self.master_animation_id)
            self.master_animation_id = None

    # File I/O methods
    def save_config(self):
        """Save current configuration to JSON file"""
        filename = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            config = {
                'orbit': {
                    'data_source': self.data_source_var.get(),
                    't_start': self.t_start_var.get(),
                    't_end': self.t_end_var.get(),
                    't_step': self.t_step_var.get(),
                    'arm_length': self.arm_len_var.get(),
                    'orbit_type': self.orbit_type_var.get()
                },
                'gw': {
                    'frequency': self.fgw_var.get(),
                    'strain': self.strain_var.get(),
                    'beta': self.beta_var.get(),
                    'lambda': self.lamda_var.get(),
                    'psi': self.psi_var.get()
                },
                'noise': {
                    'laser': self.laser_noise_var.get(),
                    'acc': self.acc_noise_var.get(),
                    'oms': self.oms_noise_var.get(),
                    'fsample': self.fsample_var.get()
                },
                'arm_locking': {
                    'enabled': self.enable_lock_var.get(),
                    'lock_type': self.lock_type_var.get(),
                    'arm_delay': self.arm_delay_var.get(),
                    'arm1_delay': self.arm1_delay_var.get(),
                    'arm2_delay': self.arm2_delay_var.get(),
                    'gfactor': self.gfactor_var.get(),
                    'afactor': self.afactor_var.get(),
                    'w0_freq': self.w0_freq_var.get()
                },
                'tdi': {
                    'channel1': self.tdi_channel1_var.get(),
                    'channel2': self.tdi_channel2_var.get(),
                    'delay_level': self.delay_level_var.get()
                }
            }

            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)

            messagebox.showinfo("Success", f"Configuration saved to {filename}")

    def load_config(self):
        """Load configuration from JSON file"""
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'r') as f:
                    config = json.load(f)

                # Load orbit settings
                self.data_source_var.set(config['orbit']['data_source'])
                self.t_start_var.set(config['orbit']['t_start'])
                self.t_end_var.set(config['orbit']['t_end'])
                self.t_step_var.set(config['orbit']['t_step'])
                self.arm_len_var.set(config['orbit']['arm_length'])
                self.orbit_type_var.set(config['orbit']['orbit_type'])

                # Load GW settings
                self.fgw_var.set(config['gw']['frequency'])
                self.strain_var.set(config['gw']['strain'])
                self.beta_var.set(config['gw']['beta'])
                self.lamda_var.set(config['gw']['lambda'])
                self.psi_var.set(config['gw']['psi'])

                # Load noise settings
                self.laser_noise_var.set(config['noise']['laser'])
                self.acc_noise_var.set(config['noise']['acc'])
                self.oms_noise_var.set(config['noise']['oms'])
                self.fsample_var.set(config['noise']['fsample'])

                # Load arm locking settings (if present in config)
                if 'arm_locking' in config:
                    self.enable_lock_var.set(config['arm_locking']['enabled'])
                    self.lock_type_var.set(config['arm_locking']['lock_type'])
                    self.arm_delay_var.set(config['arm_locking']['arm_delay'])
                    self.arm1_delay_var.set(config['arm_locking']['arm1_delay'])
                    self.arm2_delay_var.set(config['arm_locking']['arm2_delay'])
                    self.gfactor_var.set(config['arm_locking']['gfactor'])
                    self.afactor_var.set(config['arm_locking']['afactor'])
                    self.w0_freq_var.set(config['arm_locking']['w0_freq'])

                # Load TDI settings (with backwards compatibility)
                if 'channel1' in config['tdi']:
                    self.tdi_channel1_var.set(config['tdi']['channel1'])
                    self.tdi_channel2_var.set(config['tdi']['channel2'])
                elif 'channel' in config['tdi']:  # Old format compatibility
                    self.tdi_channel1_var.set(config['tdi']['channel'])
                self.delay_level_var.set(config['tdi']['delay_level'])

                self.toggle_orbit_inputs()
                self.toggle_lock_inputs()
                messagebox.showinfo("Success", f"Configuration loaded from {filename}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration:\n{str(e)}")

    def export_data(self):
        """Export simulation results to CSV"""
        if not self.results:
            messagebox.showwarning("Warning", "No simulation results to export!")
            return

        filename = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filename:
            try:
                # Combine time and both TDI outputs
                data = np.column_stack((
                    self.results['time_array'],
                    self.results['tdi_output1'],
                    self.results['tdi_output2']
                ))

                header = f"time,tdi_{self.results['channel1_name']},tdi_{self.results['channel2_name']}"
                np.savetxt(filename, data, delimiter=',', header=header, comments='')

                messagebox.showinfo("Success", f"Results exported to {filename}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to export data:\n{str(e)}")

    def clear_results(self):
        """Clear all simulation results"""
        if messagebox.askyesno("Confirm", "Clear all simulation results?"):
            self.results = {}
            self.orbit_obj = None
            self.obs_list = None
            self.gw_obj = None

            # Clear plots
            self.psd_ax.clear()
            show_empty_state(self.psd_ax, "Power spectral density")
            self.psd_canvas.draw()

            for ax in self.realtime_axes:
                ax.clear()
            self.realtime_canvas.draw()

            self.ts_ax.clear()
            show_empty_state(self.ts_ax, "TDI time series")
            self.ts_canvas.draw()

            self.update_status("Results cleared")
            self.progress_var.set(0)

    def show_about(self):
        """Show about dialog"""
        about_text = """
Taiji/LISA TDI Simulator GUI
Version 1.0

A graphical interface for running Time Delay Interferometry
simulations for space-based gravitational wave detectors.

Features:
• Interactive parameter configuration
• Multi-channel power spectra comparison
• Real-time interferometer data visualization
• Time-windowed spectral analysis
• Configuration save/load
• Data export

Built with Python, Tkinter, and Matplotlib
        """
        messagebox.showinfo("About", about_text)


def main():
    """Main entry point"""
    root = tk.Tk()
    app = TaijiSimulatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
