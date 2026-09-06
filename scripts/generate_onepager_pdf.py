#!/usr/bin/env python3
"""Generate a one-page PDF summary for quickTDI."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

MPL_CACHE_DIR = Path(__file__).resolve().parents[1] / "tmp" / "pdfs" / "mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from TJ_orbit import orbit


ROOT = Path(__file__).resolve().parents[1]
TMP_DIR = ROOT / "tmp" / "pdfs"
ASSET_DIR = TMP_DIR / "assets"
OUT_DIR = ROOT / "output" / "pdf"
OUT_FILE = OUT_DIR / "quickTDI_onepage_summary.pdf"


def _setup_dirs() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(TMP_DIR / "mplconfig")


def _plot_constellation(path: Path) -> None:
    """Plot instantaneous triangular constellation geometry in barycentric frame."""
    orb = orbit(t_start=0.0, t_end=10000.0, tri_arm=10.0, orbit_type="heliocentric")
    t0 = 0.0
    points = np.array([orb.position(i, t0) for i in (1, 2, 3)])
    center = points.mean(axis=0)
    rel = points - center

    fig, ax = plt.subplots(figsize=(4.6, 3.1), dpi=200)
    ax.set_facecolor("#f8fafc")
    color = "#0f766e"

    for i, j in ((0, 1), (1, 2), (2, 0)):
        ax.plot([rel[i, 0], rel[j, 0]], [rel[i, 1], rel[j, 1]], color=color, lw=2.0)

    ax.scatter(rel[:, 0], rel[:, 1], s=60, color="#0ea5a4", edgecolor="white", linewidth=1.4, zorder=3)
    for k, (x, y, _) in enumerate(rel, start=1):
        ax.text(x, y, f"SC{k}", fontsize=9, ha="left", va="bottom", color="#111827")

    ax.set_title("Constellation Geometry (t = 0 s, barycentric frame)", fontsize=9, color="#111827")
    ax.set_xlabel("x [light-seconds]", fontsize=8)
    ax.set_ylabel("y [light-seconds]", fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_arm_breathing(path: Path) -> None:
    """Plot breathing of three representative arms in flexible-orbit mode."""
    t_end = 14 * 24 * 3600.0
    orb = orbit(t_start=0.0, t_end=t_end, t_step=3600.0, tri_arm=10.0, orbit_type="heliocentric")
    t = orb._tarray / 86400.0
    d1 = np.array([orb.dij(1, ti * 86400.0) for ti in t])
    d2 = np.array([orb.dij(2, ti * 86400.0) for ti in t])
    d3 = np.array([orb.dij(3, ti * 86400.0) for ti in t])

    fig, ax = plt.subplots(figsize=(4.6, 3.1), dpi=200)
    ax.set_facecolor("#f8fafc")
    ax.plot(t, d1, label="Arm 1", color="#1d4ed8", lw=1.8)
    ax.plot(t, d2, label="Arm 2", color="#0ea5a4", lw=1.8)
    ax.plot(t, d3, label="Arm 3", color="#f97316", lw=1.8)
    ax.set_title("Arm-Length Breathing over 14 Days", fontsize=9, color="#111827")
    ax.set_xlabel("Time [days]", fontsize=8)
    ax.set_ylabel("Distance [light-seconds]", fontsize=8)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _draw_header(c: canvas.Canvas, width: float, height: float) -> float:
    """Draw top banner and return current y position."""
    top_h = 88
    c.setFillColor(colors.HexColor("#0b132b"))
    c.rect(0, height - top_h, width, top_h, stroke=0, fill=1)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(34, height - 44, "quickTDI")

    c.setFont("Helvetica", 11)
    c.drawString(34, height - 63, "One-page summary for Taiji/LISA Time Delay Interferometry simulation")

    c.setFillColor(colors.HexColor("#38bdf8"))
    c.roundRect(width - 230, height - 71, 196, 28, 8, stroke=0, fill=1)
    c.setFillColor(colors.HexColor("#0b132b"))
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(width - 132, height - 53, "Expert research workflow")
    return height - top_h - 16


def _draw_section_title(c: canvas.Canvas, x: float, y: float, text: str) -> float:
    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, text)
    c.setStrokeColor(colors.HexColor("#0ea5a4"))
    c.setLineWidth(1.5)
    c.line(x, y - 3, x + 120, y - 3)
    return y - 18


def _draw_bullets(c: canvas.Canvas, x: float, y: float, lines: list[str], leading: float = 14) -> float:
    c.setFillColor(colors.HexColor("#111827"))
    c.setFont("Helvetica", 9.5)
    for line in lines:
        c.drawString(x, y, f"- {line}")
        y -= leading
    return y


def _draw_pipeline_card(c: canvas.Canvas, x: float, y: float, w: float, h: float) -> None:
    c.setFillColor(colors.HexColor("#f1f5f9"))
    c.roundRect(x, y - h, w, h, 10, stroke=0, fill=1)

    stages = ["Orbit", "GW", "Optical Benches", "Locking", "Synthesis", "TDI"]
    sx = x + 12
    sy = y - 24
    box_h = 18
    box_w = (w - 24 - 5 * 10) / 6
    for idx, stage in enumerate(stages):
        bx = sx + idx * (box_w + 10)
        c.setFillColor(colors.HexColor("#ffffff"))
        c.roundRect(bx, sy, box_w, box_h, 4, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#0f172a"))
        c.setFont("Helvetica-Bold", 8)
        c.drawCentredString(bx + box_w / 2, sy + 6, stage)
        if idx < len(stages) - 1:
            c.setStrokeColor(colors.HexColor("#0ea5a4"))
            c.setLineWidth(1.3)
            x1 = bx + box_w + 2
            x2 = x1 + 6
            ymid = sy + box_h / 2
            c.line(x1, ymid, x2, ymid)
            c.line(x2 - 2, ymid + 1.5, x2, ymid)
            c.line(x2 - 2, ymid - 1.5, x2, ymid)

    c.setFillColor(colors.HexColor("#334155"))
    c.setFont("Helvetica", 8.2)
    c.drawString(x + 12, y - h + 9, "Triangle.run_full_pipeline() orchestrates this end-to-end chain.")


def _make_pdf(constellation_png: Path, breathing_png: Path) -> None:
    width, height = A4
    c = canvas.Canvas(str(OUT_FILE), pagesize=A4)
    y = _draw_header(c, width, height)

    left_x = 34
    left_w = 332
    right_x = 382
    right_w = width - right_x - 24

    y = _draw_section_title(c, left_x, y, "Physical Background")
    y = _draw_bullets(
        c,
        left_x,
        y,
        [
            "Space-based laser interferometers (Taiji/LISA class) target the mHz GW band.",
            "A passing GW perturbs optical path lengths and imprints phase fluctuations.",
            "Arm lengths are ~10 light-seconds and vary with orbital breathing.",
            "Laser frequency noise dominates raw measurements and must be suppressed.",
        ],
    )

    y -= 4
    y = _draw_section_title(c, left_x, y, "Principles")
    y = _draw_bullets(
        c,
        left_x,
        y,
        [
            "Measurement model: y_ij(t)=C_i(t-L_ij)-C_j(t)+h_ij(t)+n_ij(t).",
            "Arm-locking (single/dual/common) stabilizes the master laser before TDI.",
            "Signal synthesis builds sci/tes/ref and eta streams with physical delays.",
            "TDI combines delayed eta terms to cancel C_i while preserving h_ij.",
        ],
    )

    card_top = y - 6
    _draw_section_title(c, left_x, card_top, "Pipeline")
    _draw_pipeline_card(c, left_x, card_top - 4, left_w, 56)

    io_top = card_top - 74
    _draw_section_title(c, left_x, io_top, "Input / Output")
    c.setFillColor(colors.HexColor("#f8fafc"))
    c.roundRect(left_x, io_top - 128, left_w, 114, 10, stroke=0, fill=1)

    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(left_x + 10, io_top - 30, "Core Inputs")
    c.drawString(left_x + 168, io_top - 30, "Core Outputs")

    c.setFont("Helvetica", 8.8)
    inputs = [
        "t_start, t_end, tri_arm, orbit_type",
        "GW sources: (fgw, strain, beta, lamda, psi)",
        "Noise toggles: hasLaser / hasAcc / hasOms",
        "Lock settings: lock_type, arm delays, gains",
        "Delay mode: delay_level in {'0','1'}",
    ]
    outputs = [
        "Phasemeter streams: sci, tes, ref",
        "Eta channels for TDI synthesis",
        "TDI observables: X1/X2/Y1/Z1 (+others)",
        "Stored results in Triangle.tdi_results",
        "Optional laser diagnostics via get_laser_data()",
    ]

    yy = io_top - 45
    for line in inputs:
        c.drawString(left_x + 10, yy, f"- {line}")
        yy -= 14
    yy = io_top - 45
    for line in outputs:
        c.drawString(left_x + 168, yy, f"- {line}")
        yy -= 14

    c.setFillColor(colors.HexColor("#475569"))
    c.setFont("Helvetica", 7.8)
    c.drawString(left_x + 10, io_top - 120, "Package entry point: from TJ_Triangle import Triangle")

    c.setFillColor(colors.HexColor("#f8fafc"))
    c.roundRect(right_x, y - 2, right_w, 180, 10, stroke=0, fill=1)
    c.drawImage(ImageReader(str(constellation_png)), right_x + 8, y + 82, width=right_w - 16, height=84, preserveAspectRatio=True, anchor="c")
    c.drawImage(ImageReader(str(breathing_png)), right_x + 8, y - 6, width=right_w - 16, height=84, preserveAspectRatio=True, anchor="c")

    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 10)
    c.drawString(right_x + 8, y + 168, "Pictures")
    c.setFillColor(colors.HexColor("#334155"))
    c.setFont("Helvetica", 8.2)
    c.drawString(right_x + 8, y + 70, "Top: triangular geometry in local frame")
    c.drawString(right_x + 8, y - 18, "Bottom: representative arm-length breathing")

    footer_y = 22
    c.setStrokeColor(colors.HexColor("#cbd5e1"))
    c.setLineWidth(1)
    c.line(28, footer_y + 12, width - 28, footer_y + 12)
    c.setFillColor(colors.HexColor("#64748b"))
    c.setFont("Helvetica", 7.8)
    c.drawString(30, footer_y, "Generated from quickTDI source modules: TJ_orbit, TJ_gw, TJ_ob, TJ_lock, TJ_synthesis, TJ_tdi")
    c.drawRightString(width - 30, footer_y, "A4 one-page summary")

    c.showPage()
    c.save()


def main() -> None:
    _setup_dirs()
    constellation_png = ASSET_DIR / "constellation.png"
    breathing_png = ASSET_DIR / "arm_breathing.png"
    _plot_constellation(constellation_png)
    _plot_arm_breathing(breathing_png)
    _make_pdf(constellation_png, breathing_png)
    print(f"Generated: {OUT_FILE}")


if __name__ == "__main__":
    main()
