"""Shared visual styling for the desktop simulator."""

import tkinter as tk
from tkinter import ttk, font as tkfont

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from cycler import cycler

BG = "#f3f6fb"
SURFACE = "#ffffff"
TEXT = "#17263c"
MUTED = "#475569"
BORDER = "#dce4ef"
ACCENT = "#2563eb"


def apply_theme(root):
    """Apply a consistent, platform-independent ttk theme."""
    style = ttk.Style(root)
    style.theme_use("clam")
    # Preserve Tk's platform font and display scaling for native text rendering.
    family = tkfont.nametofont("TkDefaultFont", root=root).actual("family")
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
        tkfont.nametofont(name, root=root).configure(family=family, size=14)
    font = (family, 14)
    root.configure(background=BG)
    root.option_add("*TCombobox*Listbox.font", font)
    root.option_add("*TCombobox*Listbox.background", SURFACE)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
    style.configure(".", font=font, background=SURFACE, foreground=TEXT,
                    bordercolor=BORDER, lightcolor=SURFACE, darkcolor=BORDER,
                    focuscolor=ACCENT)
    # Explicit fonts prevent theme-specific widget defaults overriding the size.
    for widget_style in ("TLabel", "TButton", "TEntry", "TCombobox", "TSpinbox",
                         "TCheckbutton", "TRadiobutton", "TNotebook.Tab"):
        style.configure(widget_style, font=font)
    style.configure("Strong.TLabel", font=(family, 14, "bold"))
    style.configure("TFrame", background=SURFACE)
    style.configure("Page.TFrame", background=BG)
    style.configure("TLabel", padding=(0, 2))
    style.configure("Muted.TLabel", foreground=MUTED)
    style.configure("Title.TLabel", font=(family, 24, "bold"))
    style.configure("Eyebrow.TLabel", font=(family, 13, "bold"), foreground=ACCENT)
    style.configure("TButton", padding=(10, 7), borderwidth=1, relief="flat")
    style.map("TButton", background=[("pressed", "#dbeafe"), ("active", "#eff6ff")],
              foreground=[("disabled", "#94a3b8")])
    style.configure("Accent.TButton", background=ACCENT, foreground="white",
                    font=(family, 14, "bold"), bordercolor=ACCENT)
    style.map("Accent.TButton", background=[("pressed", "#1e40af"), ("active", "#1d4ed8")],
              foreground=[("disabled", "#cbd5e1"), ("!disabled", "white")])
    style.configure("Header.TLabel", font=(family, 14, "bold"), padding=0)
    style.configure("Header.TButton", padding=(8, 1))
    style.configure("Navigation.TButton", padding=(4, 1))
    style.configure("Navigation.TCheckbutton", padding=(0, 1))
    style.configure("Header.Accent.TButton", padding=(8, 1))
    style.configure("Section.TButton", anchor="w", padding=(12, 10),
                    font=(family, 14, "bold"), background="#edf3fc",
                    borderwidth=0)
    style.configure("TEntry", padding=(7, 5), fieldbackground=SURFACE)
    style.configure("TCombobox", padding=(6, 5), fieldbackground=SURFACE,
                    arrowsize=12)
    style.map("TCombobox", fieldbackground=[("readonly", SURFACE)],
              foreground=[("readonly", TEXT)])
    style.configure("TSpinbox", padding=(6, 5), fieldbackground=SURFACE)
    style.configure("TCheckbutton", padding=(3, 4))
    style.configure("TRadiobutton", padding=(3, 4))
    style.map("TCheckbutton", background=[("active", "#eff6ff")])
    style.map("TRadiobutton", background=[("active", "#eff6ff")])
    style.configure("TLabelframe", borderwidth=1, relief="solid")
    style.configure("TLabelframe.Label", font=(family, 13, "bold"), foreground=MUTED)
    style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(0, 0, 0, 8))
    style.configure("TNotebook.Tab", padding=(18, 10), background=BG, foreground=MUTED)
    style.map("TNotebook.Tab", background=[("selected", SURFACE)],
              foreground=[("selected", ACCENT)])
    style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=BG,
                    borderwidth=0, thickness=6)
    style.configure("Horizontal.TScale", background=SURFACE, troughcolor=BORDER)
    style.configure("Vertical.TScrollbar", arrowsize=0, width=9, background=BORDER,
                    troughcolor=BG, borderwidth=0)
    style.configure("TPanedwindow", background=BG, sashwidth=10)
    return style


class ChannelPicker(ttk.Frame):
    """Wrap channel controls to the available width without clipping."""

    def __init__(self, parent):
        super().__init__(parent)
        self.items = []
        self.columns = None
        self.bind("<Configure>", self._reflow)

    def add_channel(self, **kwargs):
        self.items.append(ttk.Checkbutton(self, **kwargs))
        self.columns = None
        self._reflow()

    def _reflow(self, event=None):
        width = event.width if event else self.winfo_width()
        cell_width = max([item.winfo_reqwidth() + 12 for item in self.items] or [150])
        columns = max(1, min(len(self.items), width // cell_width))
        if columns == self.columns:
            return
        self.columns = columns
        for index, item in enumerate(self.items):
            item.grid(row=index // columns, column=index % columns,
                      sticky="w", padx=(0, 12), pady=1)


class ModernAxes(Axes):
    """Keep plot styling consistent after data updates clear the axes."""

    def clear(self):
        super().clear()
        self.set_facecolor(SURFACE)
        self.set_prop_cycle(cycler(color=[ACCENT, "#0d9488", "#f59e0b", "#8b5cf6",
                                         "#e11d48", "#0891b2", "#475569"]))
        for name, spine in self.spines.items():
            spine.set_color(BORDER)
            spine.set_visible(name in ("left", "bottom"))
        self.tick_params(colors=MUTED, labelsize=11, length=3)
        self.xaxis.label.set_color(MUTED)
        self.yaxis.label.set_color(MUTED)
        self.title.set_color(TEXT)
        self.title.set_fontsize(14)
        self.grid(True, color=BORDER, linewidth=0.7, alpha=0.65)
        self.set_axisbelow(True)


class ModernFigure(Figure):
    """Figure with local styling, leaving global Matplotlib defaults intact."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("facecolor", SURFACE)
        kwargs.setdefault("layout", "constrained")
        super().__init__(*args, **kwargs)

    def add_subplot(self, *args, **kwargs):
        kwargs.setdefault("axes_class", ModernAxes)
        return super().add_subplot(*args, **kwargs)


def show_empty_state(ax, title):
    """Present useful instructions before simulation results are available."""
    ax.set_title(title, pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.5, 0.65, "Ready to explore", transform=ax.transAxes,
            ha="center", va="center", fontsize=18, color=TEXT, weight="medium")
    ax.text(0.5, 0.35, "Configure your simulation, then select Run simulation.\n"
            "Results will appear here when the run completes.", transform=ax.transAxes,
            ha="center", va="center", fontsize=13, color=MUTED, linespacing=1.8)
