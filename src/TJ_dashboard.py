"""Tkinter dashboard for configuring and running quickTDI simulations.

The dashboard intentionally depends only on the Python standard-library GUI,
Matplotlib, and quickTDI's existing numerical dependencies.  Long-running
simulation work is performed away from Tk's event loop and communicates back
through a thread-safe queue.
"""

from __future__ import annotations

import contextlib
import io
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from scipy import signal
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import TJ_tdi
import TJ_orbit
from TJ_Triangle import Triangle


BG = "#F5F7FA"
SURFACE = "#FFFFFF"
CONFIG_BG = "#EEF5FF"
SIDEBAR = "#EAF2FF"
SIDEBAR_ACTIVE = "#CFE0FF"
SIDEBAR_TEXT = "#173A68"
SIDEBAR_MUTED = "#5E7BA5"
TEXT = "#172033"
MUTED = "#657084"
BORDER = "#D9E0EA"
PRIMARY = "#145CE5"
SUCCESS = "#18A058"
WARNING = "#D97706"
PLOT_COLORS = {
    "Free laser": "#D946EF",
    "Arm lock": "#06A6C7",
    "X1": "#315CF5",
    "X2": "#10A88A",
    "Y1": "#E58B20",
    "Z1": "#7C55C7",
}


@dataclass
class RunConfig:
    """Validated values captured from the UI before a run."""

    t_start: float
    t_end: float
    arm_length: float
    orbit_type: str
    fsample_ob: float
    fsample_gw: float
    seed: Optional[int]
    sources: List[Dict[str, float]]
    has_laser: bool
    has_acc: bool
    has_oms: bool
    laser_amplitude: float
    lock_type: str
    delay_level: str
    channels: List[str]


class QueueWriter(io.TextIOBase):
    """Forward line-oriented output from numerical code to the Tk queue."""

    def __init__(self, events: queue.Queue):
        self.events = events
        self.buffer = ""

    def write(self, text: str) -> int:
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.strip():
                self.events.put(("log", line))
        return len(text)

    def flush(self) -> None:
        if self.buffer.strip():
            self.events.put(("log", self.buffer))
        self.buffer = ""


class ScrollablePanel(ttk.Frame):
    """A vertically scrollable ttk content frame."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, bg=CONFIG_BG, highlightthickness=0, width=350)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas, style="Config.TFrame", padding=(18, 12))
        self.window = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.content.bind("<Configure>", self._sync_scrollregion)
        self.canvas.bind("<Configure>", self._sync_width)

    def _sync_scrollregion(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _sync_width(self, event):
        self.canvas.itemconfigure(self.window, width=event.width)

    def scroll_to(self, fraction: float) -> None:
        self.canvas.yview_moveto(max(0.0, min(1.0, fraction)))


class DashboardApp(tk.Tk):
    """Main quickTDI desktop dashboard."""

    def __init__(self):
        super().__init__()
        self.title("quickTDI — Taiji/LISA Simulation Dashboard")
        self.geometry("1440x1024")
        self.minsize(1180, 760)
        self.configure(bg=BG)

        self.events: queue.Queue = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.triangle: Optional[Triangle] = None
        self.last_results: Dict[str, np.ndarray] = {}
        self.last_spectra: Dict[str, np.ndarray] = {}
        self.last_freqs: Optional[np.ndarray] = None
        self.last_time: Optional[np.ndarray] = None
        self.stage_labels: List[tk.Label] = []
        self.run_started = 0.0
        self.animation_job: Optional[str] = None

        self._create_variables()
        self._configure_styles()
        self._build_ui()
        self.duration.trace_add("write", lambda *_: self._update_metrics())
        self.fsample_ob.trace_add("write", lambda *_: self._update_metrics())
        self._insert_default_source()
        self._draw_placeholder_plots()
        self.after(100, self._drain_events)

    def _create_variables(self) -> None:
        self.t_start = tk.StringVar(value="0")
        self.duration = tk.StringVar(value="40000")
        self.arm_length = tk.StringVar(value="10")
        self.orbit_type = tk.StringVar(value="heliocentric")
        self.fsample_ob = tk.StringVar(value="5.0")
        self.fsample_gw = tk.StringVar(value="0.1")
        self.seed = tk.StringVar(value="1234")
        self.has_laser = tk.BooleanVar(value=True)
        self.has_acc = tk.BooleanVar(value=False)
        self.has_oms = tk.BooleanVar(value=False)
        self.laser_amplitude = tk.StringVar(value="1e-13")
        self.lock_type = tk.StringVar(value="dual")
        self.delay_level = tk.StringVar(value="1")
        self.plot_x_min = tk.StringVar(value="1e-4")
        self.plot_x_max = tk.StringVar(value="1")
        self.plot_y_min = tk.StringVar(value="1e-25")
        self.plot_y_max = tk.StringVar(value="1e-12")
        self.orbit_time = tk.StringVar(value="0")
        self.animation_step = tk.StringVar(value="2000")
        self.playback_text = tk.StringVar(value="Play")
        self.channels = {name: tk.BooleanVar(value=True) for name in ("X1", "X2", "Y1", "Z1")}
        self.status_text = tk.StringVar(value="Ready")
        self.progress_value = tk.DoubleVar(value=0.0)
        self.metric_samples = tk.StringVar(value="200,000")
        self.metric_sources = tk.StringVar(value="1 source")
        self.metric_outputs = tk.StringVar(value="4 channels")

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", font=("Helvetica Neue", 12), foreground=TEXT)
        style.configure("TFrame", background=BG)
        style.configure("Surface.TFrame", background=SURFACE)
        style.configure("Config.TFrame", background=CONFIG_BG)
        style.configure("Card.TFrame", background=SURFACE, relief="flat")
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure("Surface.TLabel", background=SURFACE, foreground=TEXT)
        style.configure("Muted.TLabel", background=SURFACE, foreground=MUTED, font=("Helvetica Neue", 10))
        style.configure("Config.TLabel", background=CONFIG_BG, foreground=TEXT)
        style.configure("ConfigMuted.TLabel", background=CONFIG_BG, foreground=MUTED, font=("Helvetica Neue", 10))
        style.configure("Config.TCheckbutton", background=CONFIG_BG, foreground=TEXT)
        style.map("Config.TCheckbutton", background=[("active", CONFIG_BG)])
        style.configure("Config.TLabelframe", background=CONFIG_BG, bordercolor="#BFD2EC", relief="solid")
        style.configure("Config.TLabelframe.Label", background=CONFIG_BG, foreground=TEXT, font=("Helvetica Neue", 13, "bold"))
        style.configure("Section.TLabel", background=SURFACE, foreground=TEXT, font=("Helvetica Neue", 14, "bold"))
        style.configure("Metric.TLabel", background=SURFACE, foreground=TEXT, font=("Helvetica Neue", 18, "bold"))
        style.configure("Primary.TButton", background=PRIMARY, foreground="white", padding=(20, 12), font=("Helvetica Neue", 12, "bold"), borderwidth=0)
        style.map("Primary.TButton", background=[("active", "#0F4CC8"), ("disabled", "#A8B8D6")])
        style.configure("Secondary.TButton", background=SURFACE, foreground=PRIMARY, padding=(12, 8), bordercolor=PRIMARY)
        style.configure("TEntry", fieldbackground="#FBFCFE", bordercolor=BORDER, padding=6)
        style.configure("TCombobox", fieldbackground="#FBFCFE", bordercolor=BORDER, padding=5)
        style.configure("Treeview", background=SURFACE, fieldbackground=SURFACE, rowheight=28, bordercolor=BORDER, font=("Menlo", 10))
        style.configure("Treeview.Heading", background="#EEF2F7", foreground=TEXT, font=("Helvetica Neue", 10, "bold"))
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(0, 9), font=("Helvetica Neue", 11), background="#EEF2F7", bordercolor="#AEB7C3", borderwidth=1, relief="solid")
        style.map("TNotebook.Tab", foreground=[("selected", PRIMARY)], background=[("selected", "#D7E6FF"), ("active", "#E4ECF8")], bordercolor=[("selected", PRIMARY)])
        style.configure("Horizontal.TProgressbar", background=PRIMARY, troughcolor="#E7ECF3", borderwidth=0)

    def _build_ui(self) -> None:
        shell = ttk.Frame(self)
        shell.pack(fill="both", expand=True)
        self._build_sidebar(shell)

        body = ttk.Frame(shell)
        body.pack(side="left", fill="both", expand=True)
        self._build_header(body)

        workspace = ttk.Panedwindow(body, orient="horizontal")
        workspace.pack(fill="both", expand=True)
        self.config_panel = ScrollablePanel(workspace)
        workspace.add(self.config_panel, weight=0)
        self._build_configuration(self.config_panel.content)

        main = ttk.Frame(workspace, style="TFrame", padding=(20, 12, 20, 8))
        workspace.add(main, weight=1)
        self._build_main_workspace(main)
        self._build_status_bar(body)

    def _build_sidebar(self, master) -> None:
        sidebar = tk.Frame(master, bg=SIDEBAR, width=190)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        brand = tk.Frame(sidebar, bg=SIDEBAR, height=104)
        brand.pack(fill="x")
        brand.pack_propagate(False)
        tk.Label(brand, text="quickTDI", bg=SIDEBAR, fg=SIDEBAR_TEXT, font=("Helvetica Neue", 20, "bold"), anchor="w").pack(fill="x", padx=20, pady=(24, 0))
        tk.Label(brand, text="TAIJI / LISA WORKBENCH", bg=SIDEBAR, fg=SIDEBAR_MUTED, font=("Helvetica Neue", 8, "bold"), anchor="w").pack(fill="x", padx=20, pady=(2, 0))
        nav = [
            ("Dashboard", lambda: self._navigate("spectrum", 0.0)),
            ("GW Sources", lambda: self._navigate("spectrum", 0.26)),
            ("Orbit", lambda: self._navigate("orbit", None)),
            ("Results", lambda: self._navigate("spectrum", None)),
            ("Run History", lambda: self._navigate("log", None)),
            ("Settings", lambda: self._navigate("spectrum", 0.0)),
        ]
        for index, (text, command) in enumerate(nav):
            color = SIDEBAR_ACTIVE if index == 0 else SIDEBAR
            button = tk.Button(sidebar, text=text, command=command, bg=color, fg=SIDEBAR_TEXT, activebackground=SIDEBAR_ACTIVE, activeforeground=SIDEBAR_TEXT, relief="flat", borderwidth=0, anchor="w", padx=20, pady=13, font=("Helvetica Neue", 11, "bold" if index == 0 else "normal"))
            button.pack(fill="x")
        tk.Label(sidebar, text="v0.1.0", bg=SIDEBAR, fg=SIDEBAR_MUTED, font=("Menlo", 9)).pack(side="bottom", pady=18)

    def _build_header(self, master) -> None:
        header = ttk.Frame(master, style="Surface.TFrame", padding=(22, 15))
        header.pack(fill="x")
        title_box = ttk.Frame(header, style="Surface.TFrame")
        title_box.pack(side="left")
        ttk.Label(title_box, text="Simulation Dashboard", style="Section.TLabel", font=("Helvetica Neue", 18, "bold")).pack(anchor="w")
        ttk.Label(title_box, text="Configure, run, and inspect the orbit-to-TDI pipeline", style="Muted.TLabel").pack(anchor="w", pady=(2, 0))
        self.run_button = ttk.Button(header, text="Run simulation", style="Primary.TButton", command=self.start_run)
        self.run_button.pack(side="right")
        self.cancel_button = ttk.Button(header, text="Cancel", style="Secondary.TButton", command=self.cancel_run, state="disabled")
        self.cancel_button.pack(side="right", padx=(0, 10))

    def _section(self, parent, title: str) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=title, padding=(10, 8), style="Config.TLabelframe")
        frame.pack(fill="x", pady=(0, 12))
        frame.columnconfigure(1, weight=1)
        return frame

    def _form_row(self, parent, row: int, label: str, variable: tk.Variable, unit: str = "", values: Optional[Iterable[str]] = None) -> None:
        ttk.Label(parent, text=label, style="Config.TLabel").grid(row=row, column=0, sticky="w", pady=5)
        if values is None:
            widget = ttk.Entry(parent, textvariable=variable)
        else:
            widget = ttk.Combobox(parent, textvariable=variable, values=list(values), state="readonly")
        widget.grid(row=row, column=1, sticky="ew", padx=(12, 4), pady=5)
        if unit:
            ttk.Label(parent, text=unit, style="ConfigMuted.TLabel").grid(row=row, column=2, sticky="w")

    def _build_configuration(self, parent) -> None:
        sim = self._section(parent, "Simulation")
        self._form_row(sim, 0, "Start time", self.t_start, "s")
        self._form_row(sim, 1, "Duration", self.duration, "s")
        self._form_row(sim, 2, "Arm length", self.arm_length, "light-s")
        self._form_row(sim, 3, "Orbit", self.orbit_type, values=("heliocentric", "geocentric"))
        self._form_row(sim, 4, "OB sampling", self.fsample_ob, "Hz")
        self._form_row(sim, 5, "GW sampling", self.fsample_gw, "Hz")
        self._form_row(sim, 6, "Random seed", self.seed)

        sources = self._section(parent, "GW Sources")
        columns = ("f", "h0", "beta", "lambda", "psi")
        self.source_tree = ttk.Treeview(sources, columns=columns, show="headings", height=4, selectmode="browse")
        headings = {"f": "f [Hz]", "h0": "h0", "beta": "β", "lambda": "λ", "psi": "ψ"}
        for col in columns:
            self.source_tree.heading(col, text=headings[col])
            self.source_tree.column(col, width=58, anchor="center", stretch=True)
        self.source_tree.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(2, 8))
        ttk.Button(sources, text="Add", command=self.add_source_dialog).grid(row=1, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(sources, text="Edit", command=self.edit_source_dialog).grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Button(sources, text="Remove", command=self.remove_source).grid(row=1, column=2, sticky="ew", padx=(4, 0))
        for column in range(3):
            sources.columnconfigure(column, weight=1)

        noise = self._section(parent, "Instrument Noise")
        ttk.Checkbutton(noise, text="Laser noise", variable=self.has_laser, style="Config.TCheckbutton").grid(row=0, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Checkbutton(noise, text="Acceleration noise", variable=self.has_acc, style="Config.TCheckbutton").grid(row=1, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Checkbutton(noise, text="Optical metrology noise", variable=self.has_oms, style="Config.TCheckbutton").grid(row=2, column=0, columnspan=2, sticky="w", pady=4)
        self._form_row(noise, 3, "Laser amplitude", self.laser_amplitude)

        locking = self._section(parent, "Laser Locking")
        self._form_row(locking, 0, "Mode", self.lock_type, values=("single", "dual", "common"))
        ttk.Label(locking, text="Advanced parameters use quickTDI defaults.", style="ConfigMuted.TLabel", wraplength=280).grid(row=1, column=0, columnspan=3, sticky="w", pady=(5, 0))

        tdi = self._section(parent, "TDI")
        self._form_row(tdi, 0, "Arm delay", self.delay_level, values=("1", "0"))
        ttk.Label(tdi, text="1 = flexible, 0 = constant", style="ConfigMuted.TLabel").grid(row=1, column=0, columnspan=3, sticky="w")
        channel_frame = ttk.Frame(tdi, style="Config.TFrame")
        channel_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(7, 3))
        for idx, (name, variable) in enumerate(self.channels.items()):
            ttk.Checkbutton(channel_frame, text=name, variable=variable, command=self._update_metrics, style="Config.TCheckbutton").grid(row=0, column=idx, sticky="w", padx=(0, 8))

        ttk.Button(parent, text="Reset to defaults", style="Secondary.TButton", command=self.reset_defaults).pack(fill="x", pady=(0, 10))

    def _build_main_workspace(self, parent) -> None:
        self._build_pipeline(parent)
        metrics = ttk.Frame(parent)
        metrics.pack(fill="x", pady=(12, 10))
        for idx, (title, variable) in enumerate((("Optical-bench samples", self.metric_samples), ("GW model", self.metric_sources), ("TDI outputs", self.metric_outputs))):
            card = ttk.Frame(metrics, style="Card.TFrame", padding=(18, 11))
            card.grid(row=0, column=idx, sticky="ew", padx=(0 if idx == 0 else 6, 0 if idx == 2 else 6))
            ttk.Label(card, text=title, style="Muted.TLabel").pack(anchor="w")
            ttk.Label(card, textvariable=variable, style="Metric.TLabel").pack(anchor="w", pady=(2, 0))
            metrics.columnconfigure(idx, weight=1)

        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True)
        spectrum_tab = ttk.Frame(self.notebook, style="Surface.TFrame")
        time_tab = ttk.Frame(self.notebook, style="Surface.TFrame")
        orbit_tab = ttk.Frame(self.notebook, style="Surface.TFrame")
        log_tab = ttk.Frame(self.notebook, style="Surface.TFrame")
        self.notebook.add(spectrum_tab, text="Spectrum")
        self.notebook.add(time_tab, text="Time series")
        self.notebook.add(orbit_tab, text="Orbit geometry")
        self.notebook.add(log_tab, text="Run log")
        self._equalize_notebook_tabs()

        plot_controls = ttk.Frame(spectrum_tab, style="Surface.TFrame", padding=(10, 8, 10, 0))
        plot_controls.pack(fill="x")
        ttk.Label(plot_controls, text="Frequency", style="Surface.TLabel", font=("Helvetica Neue", 10, "bold")).pack(side="left")
        ttk.Label(plot_controls, text="min", style="Muted.TLabel").pack(side="left", padx=(10, 3))
        ttk.Entry(plot_controls, textvariable=self.plot_x_min, width=9).pack(side="left")
        ttk.Label(plot_controls, text="max", style="Muted.TLabel").pack(side="left", padx=(8, 3))
        ttk.Entry(plot_controls, textvariable=self.plot_x_max, width=9).pack(side="left")
        ttk.Label(plot_controls, text="ASD", style="Surface.TLabel", font=("Helvetica Neue", 10, "bold")).pack(side="left", padx=(18, 0))
        ttk.Label(plot_controls, text="min", style="Muted.TLabel").pack(side="left", padx=(10, 3))
        ttk.Entry(plot_controls, textvariable=self.plot_y_min, width=9).pack(side="left")
        ttk.Label(plot_controls, text="max", style="Muted.TLabel").pack(side="left", padx=(8, 3))
        ttk.Entry(plot_controls, textvariable=self.plot_y_max, width=9).pack(side="left")
        ttk.Button(plot_controls, text="Update view", style="Secondary.TButton", command=self.update_plot_limits).pack(side="right")

        self.spectrum_figure = Figure(figsize=(8, 5), dpi=100, facecolor=SURFACE)
        self.spectrum_ax = self.spectrum_figure.add_subplot(111)
        self.spectrum_canvas = FigureCanvasTkAgg(self.spectrum_figure, spectrum_tab)
        self.spectrum_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(8, 0))
        toolbar = NavigationToolbar2Tk(self.spectrum_canvas, spectrum_tab, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="left", fill="x", padx=8, pady=4)
        ttk.Button(spectrum_tab, text="Export plot", command=self.export_plot).pack(side="right", padx=8, pady=4)
        ttk.Button(spectrum_tab, text="Export data", command=self.export_data).pack(side="right", pady=4)

        time_controls = ttk.Frame(time_tab, style="Surface.TFrame", padding=(12, 10))
        time_controls.pack(fill="x")
        ttk.Label(time_controls, text="Window / step", style="Surface.TLabel", font=("Helvetica Neue", 11, "bold")).pack(side="left")
        ttk.Entry(time_controls, textvariable=self.animation_step, width=12).pack(side="left", padx=(10, 4))
        ttk.Label(time_controls, text="s", style="Muted.TLabel").pack(side="left")
        ttk.Button(time_controls, text="Previous", command=lambda: self.step_visualizations(-1)).pack(side="left", padx=(14, 5))
        ttk.Button(time_controls, text="Next", command=lambda: self.step_visualizations(1)).pack(side="left", padx=5)
        ttk.Button(time_controls, textvariable=self.playback_text, style="Secondary.TButton", command=self.toggle_animation).pack(side="left", padx=(5, 0))
        ttk.Label(time_controls, text="Time series and orbit advance together.", style="Muted.TLabel").pack(side="right")

        self.time_figure = Figure(figsize=(8, 4), dpi=100, facecolor=SURFACE)
        self.time_ax = self.time_figure.add_subplot(111)
        self.time_canvas = FigureCanvasTkAgg(self.time_figure, time_tab)
        self.time_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

        orbit_controls = ttk.Frame(orbit_tab, style="Surface.TFrame", padding=(12, 10))
        orbit_controls.pack(fill="x")
        ttk.Label(orbit_controls, text="Constellation time", style="Surface.TLabel", font=("Helvetica Neue", 11, "bold")).pack(side="left")
        ttk.Entry(orbit_controls, textvariable=self.orbit_time, width=14).pack(side="left", padx=(10, 4))
        ttk.Label(orbit_controls, text="s", style="Muted.TLabel").pack(side="left")
        ttk.Button(orbit_controls, text="Update", style="Secondary.TButton", command=self.update_visualizations).pack(side="left", padx=(12, 0))
        ttk.Label(orbit_controls, text="Step", style="Muted.TLabel").pack(side="left", padx=(18, 3))
        ttk.Entry(orbit_controls, textvariable=self.animation_step, width=10).pack(side="left")
        ttk.Label(orbit_controls, text="s", style="Muted.TLabel").pack(side="left", padx=(3, 0))
        ttk.Button(orbit_controls, text="Previous", command=lambda: self.step_visualizations(-1)).pack(side="left", padx=(14, 4))
        ttk.Button(orbit_controls, text="Next", command=lambda: self.step_visualizations(1)).pack(side="left", padx=4)
        ttk.Button(orbit_controls, textvariable=self.playback_text, style="Secondary.TButton", command=self.toggle_animation).pack(side="left", padx=(4, 0))
        ttk.Label(orbit_controls, text="Barycenter-relative coordinates", style="Muted.TLabel").pack(side="right")
        self.orbit_figure = Figure(figsize=(8, 5), dpi=100, facecolor=SURFACE)
        self.orbit_ax = self.orbit_figure.add_subplot(111, projection="3d")
        self.orbit_canvas = FigureCanvasTkAgg(self.orbit_figure, orbit_tab)
        self.orbit_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self.log_text = tk.Text(log_tab, bg="#F8FAFC", fg=TEXT, insertbackground=TEXT, relief="flat", font=("Menlo", 10), padx=14, pady=12, wrap="word", state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=8, pady=8)
        self.log_text.tag_configure("time", foreground=MUTED)
        self.log_text.tag_configure("success", foreground=SUCCESS)
        self.log_text.tag_configure("error", foreground="#C53A3A")

    def _equalize_notebook_tabs(self) -> None:
        """Give every notebook tab the same measured pixel width."""

        labels = [self.notebook.tab(tab_id, "text") for tab_id in self.notebook.tabs()]
        font = tkfont.Font(font=("Helvetica Neue", 11))
        target_width = max(150, max(font.measure(label) for label in labels) + 36)
        for tab_id, label in zip(self.notebook.tabs(), labels):
            horizontal = max(12, (target_width - font.measure(label)) // 2)
            self.notebook.tab(tab_id, padding=(horizontal, 9, horizontal, 9))

    def _build_pipeline(self, parent) -> None:
        frame = ttk.Frame(parent, style="Surface.TFrame", padding=(16, 12))
        frame.pack(fill="x")
        stages = ("Orbit", "GW", "Benches", "Locking", "Synthesis", "TDI")
        for idx, stage in enumerate(stages):
            box = ttk.Frame(frame, style="Surface.TFrame")
            box.grid(row=0, column=idx, sticky="ew")
            marker = tk.Label(box, text=str(idx + 1), bg="#E9EEF6", fg=MUTED, width=3, height=1, font=("Helvetica Neue", 10, "bold"))
            marker.pack()
            label = tk.Label(box, text=stage, bg=SURFACE, fg=MUTED, font=("Helvetica Neue", 10))
            label.pack(pady=(4, 0))
            self.stage_labels.append(marker)
            frame.columnconfigure(idx, weight=1)

    def _build_status_bar(self, parent) -> None:
        bar = ttk.Frame(parent, style="Surface.TFrame", padding=(16, 9))
        bar.pack(fill="x")
        self.status_dot = tk.Label(bar, text="●", bg=SURFACE, fg=SUCCESS, font=("Helvetica Neue", 11))
        self.status_dot.pack(side="left")
        ttk.Label(bar, textvariable=self.status_text, style="Surface.TLabel").pack(side="left", padx=(6, 14))
        self.progress = ttk.Progressbar(bar, variable=self.progress_value, maximum=100, length=260)
        self.progress.pack(side="left", fill="x", expand=True)
        ttk.Label(bar, text="Python / NumPy / SciPy", style="Muted.TLabel").pack(side="right", padx=(16, 0))

    def _navigate(self, tab: str, scroll: Optional[float]) -> None:
        tab_index = {"spectrum": 0, "time": 1, "orbit": 2, "log": 3}[tab]
        self.notebook.select(tab_index)
        if tab == "orbit":
            self.update_orbit_plot(show_errors=False)
        if scroll is not None:
            self.config_panel.scroll_to(scroll)

    def _insert_default_source(self) -> None:
        self.source_tree.insert("", "end", values=("0.007", "1e-23", "0", "0", "0"))
        self._update_metrics()

    def _source_dialog(self, title: str, item: Optional[str] = None) -> None:
        dialog = tk.Toplevel(self)
        dialog.title(title)
        dialog.transient(self)
        dialog.grab_set()
        dialog.configure(bg=SURFACE)
        fields = (("Frequency [Hz]", "0.007"), ("Strain h0", "1e-23"), ("β [rad]", "0"), ("λ [rad]", "0"), ("ψ [rad]", "0"))
        if item:
            current = self.source_tree.item(item, "values")
            fields = tuple((fields[i][0], str(current[i])) for i in range(5))
        variables = []
        body = ttk.Frame(dialog, style="Surface.TFrame", padding=18)
        body.pack(fill="both", expand=True)
        for row, (label, value) in enumerate(fields):
            ttk.Label(body, text=label, style="Surface.TLabel").grid(row=row, column=0, sticky="w", pady=5)
            variable = tk.StringVar(value=value)
            ttk.Entry(body, textvariable=variable, width=22).grid(row=row, column=1, sticky="ew", padx=(12, 0), pady=5)
            variables.append(variable)

        def save():
            try:
                values = tuple(float(var.get()) for var in variables)
                if values[0] <= 0 or values[1] <= 0:
                    raise ValueError("Frequency and strain must be positive.")
            except ValueError as exc:
                messagebox.showerror("Invalid source", str(exc), parent=dialog)
                return
            display = tuple(f"{value:.5g}" for value in values)
            if item:
                self.source_tree.item(item, values=display)
            else:
                self.source_tree.insert("", "end", values=display)
            self._update_metrics()
            dialog.destroy()

        buttons = ttk.Frame(body, style="Surface.TFrame")
        buttons.grid(row=len(fields), column=0, columnspan=2, sticky="e", pady=(14, 0))
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="Save source", style="Primary.TButton", command=save).pack(side="left")
        body.columnconfigure(1, weight=1)
        dialog.wait_visibility()
        dialog.focus_force()

    def add_source_dialog(self) -> None:
        self._source_dialog("Add gravitational-wave source")

    def edit_source_dialog(self) -> None:
        selection = self.source_tree.selection()
        if not selection:
            messagebox.showinfo("Select a source", "Select a source row to edit.", parent=self)
            return
        self._source_dialog("Edit gravitational-wave source", selection[0])

    def remove_source(self) -> None:
        selection = self.source_tree.selection()
        if not selection:
            return
        self.source_tree.delete(selection[0])
        self._update_metrics()

    def _update_metrics(self) -> None:
        try:
            samples = int(float(self.duration.get()) * float(self.fsample_ob.get()))
            self.metric_samples.set(f"{samples:,}")
        except ValueError:
            self.metric_samples.set("—")
        count = len(self.source_tree.get_children()) if hasattr(self, "source_tree") else 0
        self.metric_sources.set(f"{count} source" + ("" if count == 1 else "s"))
        selected = sum(variable.get() for variable in self.channels.values())
        self.metric_outputs.set(f"{selected} channel" + ("" if selected == 1 else "s"))

    def reset_defaults(self) -> None:
        self.stop_animation()
        for variable, value in ((self.t_start, "0"), (self.duration, "40000"), (self.arm_length, "10"), (self.orbit_type, "heliocentric"), (self.fsample_ob, "5.0"), (self.fsample_gw, "0.1"), (self.seed, "1234"), (self.laser_amplitude, "1e-13"), (self.lock_type, "dual"), (self.delay_level, "1"), (self.plot_x_min, "1e-4"), (self.plot_x_max, "1"), (self.plot_y_min, "1e-25"), (self.plot_y_max, "1e-12"), (self.orbit_time, "0"), (self.animation_step, "2000")):
            variable.set(value)
        self.has_laser.set(True)
        self.has_acc.set(False)
        self.has_oms.set(False)
        for variable in self.channels.values():
            variable.set(True)
        for item in self.source_tree.get_children():
            self.source_tree.delete(item)
        self._insert_default_source()
        self._draw_placeholder_plots()
        self.status_text.set("Defaults restored")

    def _read_config(self) -> RunConfig:
        try:
            t_start = float(self.t_start.get())
            duration = float(self.duration.get())
            arm = float(self.arm_length.get())
            fs_ob = float(self.fsample_ob.get())
            fs_gw = float(self.fsample_gw.get())
            amplitude = float(self.laser_amplitude.get())
            seed_text = self.seed.get().strip()
            seed = int(seed_text) if seed_text else None
        except ValueError as exc:
            raise ValueError("Simulation values must be valid numbers.") from exc
        if duration <= 0 or arm <= 0 or fs_ob <= 0 or fs_gw <= 0 or amplitude <= 0:
            raise ValueError("Duration, arm length, sampling rates, and noise amplitude must be positive.")
        if duration * fs_ob < 64 or duration * fs_gw < 4:
            raise ValueError("The selected duration and sampling rates produce too few samples.")
        sources = []
        for item in self.source_tree.get_children():
            fgw, strain_value, beta, lamda, psi = (float(value) for value in self.source_tree.item(item, "values"))
            sources.append({"fgw": fgw, "strain": strain_value, "beta": beta, "lamda": lamda, "psi": psi})
        if not sources:
            raise ValueError("Add at least one gravitational-wave source.")
        channels = [name for name, variable in self.channels.items() if variable.get()]
        if not channels:
            raise ValueError("Select at least one TDI channel.")
        return RunConfig(t_start=t_start, t_end=t_start + duration, arm_length=arm, orbit_type=self.orbit_type.get(), fsample_ob=fs_ob, fsample_gw=fs_gw, seed=seed, sources=sources, has_laser=self.has_laser.get(), has_acc=self.has_acc.get(), has_oms=self.has_oms.get(), laser_amplitude=amplitude, lock_type=self.lock_type.get(), delay_level=self.delay_level.get(), channels=channels)

    def start_run(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            config = self._read_config()
        except ValueError as exc:
            messagebox.showerror("Check the configuration", str(exc), parent=self)
            return
        self.stop_event.clear()
        self.run_started = time.monotonic()
        self.run_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.progress_value.set(0)
        self.status_text.set("Starting simulation…")
        self.status_dot.configure(fg=PRIMARY)
        self._reset_stages()
        self._append_log("Simulation configuration validated.")
        self.worker = threading.Thread(target=self._run_pipeline, args=(config,), daemon=True)
        self.worker.start()

    def cancel_run(self) -> None:
        self.stop_event.set()
        self.cancel_button.configure(state="disabled")
        self.status_text.set("Cancelling after the current stage…")
        self._append_log("Cancellation requested; the current numerical stage will finish first.")

    def _check_cancelled(self) -> None:
        if self.stop_event.is_set():
            raise InterruptedError("Simulation cancelled.")

    def _run_pipeline(self, config: RunConfig) -> None:
        writer = QueueWriter(self.events)
        try:
            if config.seed is not None:
                np.random.seed(config.seed)
            channel_map = {"X1": TJ_tdi.X1, "X2": TJ_tdi.X2, "Y1": TJ_tdi.Y1, "Z1": TJ_tdi.Z1}
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                self.events.put(("stage", 0, "Building orbit"))
                triangle = Triangle(t_start=config.t_start, t_end=config.t_end, tri_arm=config.arm_length, orbit_type=config.orbit_type, fsample_ob=config.fsample_ob, fsample_gw=config.fsample_gw)
                self._check_cancelled()
                triangle.set_gw_sources(config.sources)
                self.events.put(("stage", 1, "Generating GW response"))
                triangle.create_gw_object(hasGW=True)
                self._check_cancelled()
                self.events.put(("stage", 2, "Creating optical benches"))
                triangle.setup_optical_benches(hasLaser=config.has_laser, hasAcc=config.has_acc, hasOms=config.has_oms, LNamp=config.laser_amplitude)
                self._check_cancelled()
                self.events.put(("stage", 3, "Applying laser lock"))
                triangle.apply_laser_lock(lock_type=config.lock_type)
                self._check_cancelled()
                self.events.put(("stage", 4, "Synthesizing signals"))
                triangle.synthesize_signals(delay_level=config.delay_level)
                self._check_cancelled()
                self.events.put(("stage", 5, "Computing TDI channels"))
                results = {}
                for name in config.channels:
                    self._check_cancelled()
                    results[name] = triangle.run_tdi(channel_map[name], delay_level=config.delay_level, store_name=name)
                self._check_cancelled()
            writer.flush()
            freqs, spectra = self._compute_spectra(triangle, results)
            self.events.put(("complete", triangle, results, freqs, spectra))
        except InterruptedError as exc:
            writer.flush()
            self.events.put(("cancelled", str(exc)))
        except Exception as exc:  # UI boundary: surface numerical failures to the user.
            writer.flush()
            self.events.put(("error", f"{type(exc).__name__}: {exc}"))

    @staticmethod
    def _compute_spectra(triangle: Triangle, results: Dict[str, np.ndarray]):
        free_laser, locked_laser, _, fsample = triangle.get_laser_data()
        trim = min(int(100 * fsample), max(0, len(free_laser) // 10))
        data_slice = slice(trim, -trim if trim else None)
        available = len(free_laser[data_slice])
        nperseg = max(32, min(available, max(256, available // 2)))
        spectra = {}
        freqs, psd = signal.welch(free_laser[data_slice], fsample, nperseg=nperseg)
        spectra["Free laser"] = np.sqrt(np.maximum(psd, np.finfo(float).tiny))
        _, psd = signal.welch(locked_laser[data_slice], fsample, nperseg=nperseg)
        spectra["Arm lock"] = np.sqrt(np.maximum(psd, np.finfo(float).tiny))
        for name, values in results.items():
            _, psd = signal.welch(values[data_slice], fsample, nperseg=nperseg)
            spectra[name] = np.sqrt(np.maximum(psd, np.finfo(float).tiny))
        return freqs, spectra

    def _drain_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "log":
                    self._append_log(event[1])
                elif kind == "stage":
                    index, text = event[1], event[2]
                    self._activate_stage(index)
                    self.progress_value.set(index / 6 * 100)
                    self.status_text.set(text)
                elif kind == "complete":
                    self._finish_run(*event[1:])
                elif kind == "cancelled":
                    self._finish_cancelled(event[1])
                elif kind == "error":
                    self._finish_error(event[1])
        except queue.Empty:
            pass
        self.after(100, self._drain_events)

    def _reset_stages(self) -> None:
        for label in self.stage_labels:
            label.configure(bg="#E9EEF6", fg=MUTED)

    def _activate_stage(self, index: int) -> None:
        for idx, label in enumerate(self.stage_labels):
            if idx < index:
                label.configure(bg="#DDF4E7", fg=SUCCESS, text="✓")
            elif idx == index:
                label.configure(bg=PRIMARY, fg="white", text=str(idx + 1))
            else:
                label.configure(bg="#E9EEF6", fg=MUTED, text=str(idx + 1))

    def _finish_run(self, triangle: Triangle, results: Dict[str, np.ndarray], freqs: np.ndarray, spectra: Dict[str, np.ndarray]) -> None:
        self.triangle = triangle
        self.last_results = results
        self.last_freqs = freqs
        self.last_spectra = spectra
        self.last_time = triangle.obs[1]._tarray if triangle.obs[1] is not None else None
        for label in self.stage_labels:
            label.configure(bg="#DDF4E7", fg=SUCCESS, text="✓")
        self.progress_value.set(100)
        elapsed = time.monotonic() - self.run_started
        self.status_text.set(f"Completed in {elapsed:.1f} s")
        self.status_dot.configure(fg=SUCCESS)
        self.run_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        self._append_log(f"Simulation completed successfully in {elapsed:.1f} seconds.", "success")
        self._draw_results()
        self.update_orbit_plot(show_errors=False)
        self.notebook.select(0)

    def _finish_cancelled(self, text: str) -> None:
        self.status_text.set(text)
        self.status_dot.configure(fg=WARNING)
        self.run_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        self._append_log(text)

    def _finish_error(self, text: str) -> None:
        self.status_text.set("Simulation failed")
        self.status_dot.configure(fg="#C53A3A")
        self.run_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        self._append_log(text, "error")
        messagebox.showerror("Simulation failed", text, parent=self)

    def _append_log(self, text: str, tag: Optional[str] = None) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{timestamp}] ", "time")
        self.log_text.insert("end", text.rstrip() + "\n", tag or "")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _style_axis(self, axis, title: str, xlabel: str, ylabel: str) -> None:
        axis.set_facecolor(SURFACE)
        axis.set_title(title, color=TEXT, fontsize=13, pad=12)
        axis.set_xlabel(xlabel, color=TEXT)
        axis.set_ylabel(ylabel, color=TEXT)
        axis.grid(True, which="both", color="#D5DCE7", linestyle="-", linewidth=0.5, alpha=0.8)
        axis.tick_params(colors=MUTED, labelsize=9)
        for spine in axis.spines.values():
            spine.set_color(BORDER)

    def _plot_limits(self):
        """Return validated positive spectrum limits from the UI."""

        try:
            x_min = float(self.plot_x_min.get())
            x_max = float(self.plot_x_max.get())
            y_min = float(self.plot_y_min.get())
            y_max = float(self.plot_y_max.get())
        except ValueError as exc:
            raise ValueError("Plot limits must be valid numbers.") from exc
        if min(x_min, x_max, y_min, y_max) <= 0:
            raise ValueError("Logarithmic plot limits must be positive.")
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("Each minimum plot limit must be smaller than its maximum.")
        return x_min, x_max, y_min, y_max

    def _apply_plot_limits(self) -> None:
        x_min, x_max, y_min, y_max = self._plot_limits()
        self.spectrum_ax.set_xlim(x_min, x_max)
        self.spectrum_ax.set_ylim(y_min, y_max)

    def update_plot_limits(self) -> None:
        """Apply user-entered axis ranges to the spectrum plot."""

        try:
            self._apply_plot_limits()
        except ValueError as exc:
            messagebox.showerror("Invalid plot range", str(exc), parent=self)
            return
        self.spectrum_canvas.draw_idle()
        self.status_text.set("Spectrum range updated")

    def _animation_bounds(self):
        if self.last_time is not None and len(self.last_time):
            return float(self.last_time[0]), float(self.last_time[-1])
        start = float(self.t_start.get())
        return start, start + float(self.duration.get())

    def _animation_step_value(self) -> float:
        try:
            step = float(self.animation_step.get())
        except ValueError as exc:
            raise ValueError("Visualization step must be a valid number.") from exc
        if step <= 0:
            raise ValueError("Visualization step must be positive.")
        return step

    def update_visualizations(self, show_errors: bool = True) -> None:
        """Update the synchronized time-series window and orbit geometry."""

        try:
            float(self.orbit_time.get())
            self._animation_step_value()
        except ValueError as exc:
            if show_errors:
                messagebox.showerror("Invalid visualization time", str(exc), parent=self)
            return
        self.update_time_series_window(show_errors=show_errors)
        self.update_orbit_plot(show_errors=show_errors)

    def step_visualizations(self, direction: int) -> None:
        """Advance or rewind both visualizations by the configured data step."""

        try:
            step = self._animation_step_value()
            start, end = self._animation_bounds()
            current = float(self.orbit_time.get())
        except ValueError as exc:
            messagebox.showerror("Invalid visualization step", str(exc), parent=self)
            self.stop_animation()
            return
        if direction >= 0:
            next_time = current + step
            if current < start or next_time >= end:
                next_time = start
        else:
            next_time = current - step
            if current > end or next_time < start:
                next_time = max(start, end - step)
        self.orbit_time.set(f"{next_time:g}")
        self.update_visualizations(show_errors=False)

    def toggle_animation(self) -> None:
        """Start or pause synchronized playback."""

        if self.animation_job is not None:
            self.stop_animation()
            return
        try:
            self._animation_step_value()
            self._animation_bounds()
        except ValueError as exc:
            messagebox.showerror("Invalid visualization step", str(exc), parent=self)
            return
        self.playback_text.set("Pause")
        self.step_visualizations(1)
        self.animation_job = self.after(900, self._animation_tick)

    def _animation_tick(self) -> None:
        self.animation_job = None
        if self.playback_text.get() != "Pause":
            return
        self.step_visualizations(1)
        if self.playback_text.get() == "Pause":
            self.animation_job = self.after(900, self._animation_tick)

    def stop_animation(self) -> None:
        if self.animation_job is not None:
            self.after_cancel(self.animation_job)
            self.animation_job = None
        self.playback_text.set("Play")

    def update_time_series_window(self, show_errors: bool = True) -> None:
        """Draw one moving data window using the shared orbit/playback time."""

        try:
            window_start = float(self.orbit_time.get())
            step = self._animation_step_value()
            bound_start, bound_end = self._animation_bounds()
        except ValueError as exc:
            if show_errors:
                messagebox.showerror("Invalid time-series view", str(exc), parent=self)
            return
        if window_start < bound_start or window_start >= bound_end:
            window_start = bound_start
            self.orbit_time.set(f"{window_start:g}")
        window_end = min(window_start + step, bound_end)
        self.time_ax.clear()
        if self.last_time is not None and self.last_results:
            first = int(np.searchsorted(self.last_time, window_start, side="left"))
            last = int(np.searchsorted(self.last_time, window_end, side="right"))
            count = max(0, last - first)
            stride = max(1, count // 5000)
            if count:
                selection = slice(first, last, stride)
                for name, values in self.last_results.items():
                    self.time_ax.plot(self.last_time[selection], values[selection], label=name, color=PLOT_COLORS.get(name), linewidth=0.9)
                self.time_ax.legend(frameon=False, fontsize=9)
        else:
            t = np.linspace(window_start, window_end, 800)
            frequency = 0.007
            children = self.source_tree.get_children() if hasattr(self, "source_tree") else ()
            if children:
                frequency = float(self.source_tree.item(children[0], "values")[0])
            self.time_ax.plot(t, 1e-23 * np.sin(2 * np.pi * frequency * t), color=PLOT_COLORS["X1"], linewidth=1.1, label="X1 preview")
            self.time_ax.legend(frameon=False, fontsize=9)
        self._style_axis(self.time_ax, f"TDI time series: {window_start:g}–{window_end:g} s", "Time [s]", "Response")
        self.time_ax.set_xlim(window_start, window_end)
        self.time_figure.tight_layout()
        self.time_canvas.draw_idle()

    def update_orbit_plot(self, show_errors: bool = True) -> None:
        """Draw the centered three-spacecraft constellation at a chosen time."""

        try:
            orbit_time = float(self.orbit_time.get())
            arm_length = float(self.arm_length.get())
            if arm_length <= 0:
                raise ValueError("Arm length must be positive.")
            if self.triangle is not None:
                positions = np.array([self.triangle.orbit.position(sc, orbit_time) for sc in (1, 2, 3)])
                orbit_label = self.triangle.orbit_type
            else:
                orbit_label = self.orbit_type.get()
                positions = np.array([
                    TJ_orbit.sc_pos_analytic(sc, orbit_time, arm_length, orbit_label)
                    for sc in (1, 2, 3)
                ])
        except (ValueError, TypeError) as exc:
            if show_errors:
                messagebox.showerror("Invalid orbit view", str(exc), parent=self)
            return

        centered = positions - positions.mean(axis=0)
        closed = np.vstack((centered, centered[0]))
        self.orbit_ax.clear()
        self.orbit_ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color=PRIMARY, linewidth=2.2)
        colors = ("#145CE5", "#10A88A", "#E58B20")
        for index, (point, color) in enumerate(zip(centered, colors), start=1):
            self.orbit_ax.scatter(*point, color=color, s=85, depthshade=False)
            self.orbit_ax.text(*point, f"  SC{index}", color=TEXT, fontsize=10, fontweight="bold")
        span = max(float(np.ptp(centered[:, axis])) for axis in range(3))
        half = max(span * 0.62, arm_length * 0.6)
        for setter in (self.orbit_ax.set_xlim, self.orbit_ax.set_ylim, self.orbit_ax.set_zlim):
            setter(-half, half)
        self.orbit_ax.set_box_aspect((1, 1, 0.72))
        self.orbit_ax.set_title(f"{orbit_label.title()} constellation at t = {orbit_time:g} s", color=TEXT, pad=14)
        self.orbit_ax.set_xlabel("x − barycenter [light-s]", color=TEXT)
        self.orbit_ax.set_ylabel("y − barycenter [light-s]", color=TEXT)
        self.orbit_ax.set_zlabel("z − barycenter [light-s]", color=TEXT)
        self.orbit_ax.grid(True, color="#D5DCE7", linewidth=0.5)
        self.orbit_figure.tight_layout()
        self.orbit_canvas.draw_idle()
        self.status_text.set("Orbit geometry updated")

    def _draw_placeholder_plots(self) -> None:
        f = np.logspace(-4, 1, 600)
        valley = 2e-25 * ((f / 0.02) ** -1.45 + 0.55 * (f / 0.02) ** 1.2)
        curves = {
            "Free laser": valley * 2.5e4,
            "Arm lock": valley * 1.2e3,
            "X1": valley * 1.25,
            "X2": valley,
        }
        self.spectrum_ax.clear()
        for name, values in curves.items():
            self.spectrum_ax.loglog(f, values, label=name, color=PLOT_COLORS[name], linewidth=1.8, linestyle="-")
        self._style_axis(self.spectrum_ax, "TDI Channel ASD — preview", "Frequency [Hz]", "ASD [1/√Hz]")
        self._apply_plot_limits()
        self.spectrum_ax.legend(frameon=True, facecolor=SURFACE, edgecolor=BORDER, fontsize=9)
        self.spectrum_figure.tight_layout()
        self.spectrum_canvas.draw_idle()
        self.update_time_series_window(show_errors=False)
        self.update_orbit_plot(show_errors=False)

    def _draw_results(self) -> None:
        if self.last_freqs is None:
            return
        self.spectrum_ax.clear()
        positive = self.last_freqs > 0
        for name, values in self.last_spectra.items():
            self.spectrum_ax.loglog(self.last_freqs[positive], values[positive], label=name, color=PLOT_COLORS.get(name), linewidth=1.5, linestyle="-")
        self._style_axis(self.spectrum_ax, "TDI Channel ASD", "Frequency [Hz]", "ASD [1/√Hz]")
        self._apply_plot_limits()
        self.spectrum_ax.legend(frameon=True, facecolor=SURFACE, edgecolor=BORDER, fontsize=9)
        self.spectrum_figure.tight_layout()
        self.spectrum_canvas.draw_idle()
        self.update_time_series_window(show_errors=False)

    def export_plot(self) -> None:
        path = filedialog.asksaveasfilename(parent=self, title="Export spectrum plot", defaultextension=".png", filetypes=(("PNG image", "*.png"), ("PDF", "*.pdf")))
        if path:
            self.spectrum_figure.savefig(path, dpi=180, bbox_inches="tight")
            self._append_log(f"Plot exported to {path}", "success")

    def export_data(self) -> None:
        if self.last_freqs is None or not self.last_results:
            messagebox.showinfo("No simulation data", "Run a simulation before exporting numerical results.", parent=self)
            return
        path = filedialog.asksaveasfilename(parent=self, title="Export simulation data", defaultextension=".npz", filetypes=(("NumPy archive", "*.npz"),))
        if path:
            payload = {"frequency": self.last_freqs, **{f"asd_{key}": value for key, value in self.last_spectra.items()}, **{f"tdi_{key}": value for key, value in self.last_results.items()}}
            if self.last_time is not None:
                payload["time"] = self.last_time
            np.savez_compressed(path, **payload)
            self._append_log(f"Data exported to {path}", "success")


def main() -> None:
    """Launch the quickTDI desktop dashboard."""

    app = DashboardApp()
    app.mainloop()


if __name__ == "__main__":
    main()
