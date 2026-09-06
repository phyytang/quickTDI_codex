from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output" / "pdf"
OUTPUT_PATH = OUTPUT_DIR / "quickTDI_app_summary.pdf"


def build_styles():
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=22,
            textColor=colors.HexColor("#15324b"),
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=11,
            textColor=colors.HexColor("#476274"),
            spaceAfter=8,
        ),
        "section": ParagraphStyle(
            "Section",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=13,
            textColor=colors.white,
            backColor=colors.HexColor("#234d6f"),
            borderPadding=(3, 6, 3, 6),
            spaceBefore=4,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.3,
            leading=10.0,
            textColor=colors.HexColor("#172b38"),
            spaceAfter=3,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.7,
            leading=9.2,
            textColor=colors.HexColor("#172b38"),
            spaceAfter=0,
        ),
    }


def bullets(items, style):
    marker_style = ParagraphStyle(
        "Marker",
        parent=style,
        alignment=1,
        spaceAfter=0,
    )
    rows = [[Paragraph("-", marker_style), Paragraph(item, style)] for item in items]
    return Table(
        rows,
        colWidths=[5 * mm, 162 * mm],
        style=TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ]
        ),
    )


def build_story():
    s = build_styles()
    story = [
        Paragraph("quickTDI App Summary", s["title"]),
        Paragraph(
            "Evidence source: repository files including <b>pyproject.toml</b>, "
            "<b>src/TJ_Triangle.py</b>, <b>docs/SOFTWARE_DOCUMENTATION.md</b>, and <b>CLAUDE.md</b>.",
            s["subtitle"],
        ),
    ]

    story.extend(
        [
            Paragraph("What It Is", s["section"]),
            Paragraph(
                "quickTDI is a Python simulation framework for Time Delay Interferometry (TDI) in "
                "space-based gravitational-wave detectors such as Taiji and LISA. The repo presents "
                "it as an end-to-end chain from orbit modeling through TDI output.",
                s["body"],
            ),
            Paragraph("Who It’s For", s["section"]),
            Paragraph(
                "Primary persona: researchers running Taiji/LISA physical simulations and sensitivity "
                "analysis. Secondary audiences mentioned in the repo are developers extending modules "
                "and engineering teams integrating the simulator.",
                s["body"],
            ),
            Paragraph("What It Does", s["section"]),
            bullets(
                [
                    "Provides a high-level <b>Triangle</b> wrapper that orchestrates the full TDI simulation pipeline.",
                    "Builds spacecraft orbit and arm-delay models, including heliocentric or geocentric setups.",
                    "Generates gravitational-wave responses for single sources, multiple sources, or frequency sweeps.",
                    "Creates six optical bench objects with configurable laser, acceleration, and OMS noise inputs.",
                    "Applies laser locking in single-arm, dual-arm, or common-arm modes.",
                    "Synthesizes phasemeter-style signals and evaluates predefined TDI channels such as X, Y, and Z variants.",
                    "Optionally supports Bayesian inference through <b>dynesty</b> utilities and a CLI script.",
                ],
                s["body"],
            ),
        ]
    )

    architecture_rows = [
        [
            Paragraph("<b>Components</b>", s["small"]),
            Paragraph(
                "<b>TJ_orbit</b> -> <b>TJ_gw</b> -> <b>TJ_ob</b> -> <b>TJ_lock</b> -> "
                "<b>TJ_synthesis</b> -> <b>TJ_tdi</b>, coordinated by <b>TJ_Triangle.Triangle</b>.",
                s["small"],
            ),
        ],
        [
            Paragraph("<b>Data Flow</b>", s["small"]),
            Paragraph(
                "Simulation parameters create orbit and delay models; those feed GW responses; responses and noise feed six optical benches; "
                "locking updates bench states; synthesis produces sci/tes/ref and eta signals; TDI combines them into output channel series.",
                s["small"],
            ),
        ],
        [
            Paragraph("<b>Interfaces</b>", s["small"]),
            Paragraph(
                "Python module entry points in <b>src/</b>, manual workflow in <b>test.ipynb</b>, and optional inference CLI at "
                "<b>scripts/run_dynesty.py</b>.",
                s["small"],
            ),
        ],
        [
            Paragraph("<b>Services</b>", s["small"]),
            Paragraph("Not found in repo.", s["small"]),
        ],
    ]

    story.extend(
        [
            Paragraph("How It Works", s["section"]),
            Table(
                architecture_rows,
                colWidths=[31 * mm, 136 * mm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8eff4")),
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#91a7b7")),
                        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c3d0d9")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 5),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                ),
            ),
            Spacer(1, 5),
            Paragraph("How To Run", s["section"]),
            bullets(
                [
                    "Install runtime dependencies: <b>python -m pip install numpy scipy</b>.",
                    "Install the repo for local imports: <b>python -m pip install -e .</b>.",
                    "Optional inference extras: <b>python -m pip install -e .[inference]</b>.",
                    "Run a minimal smoke check: <b>PYTHONPATH=src python -c \"from TJ_Triangle import Triangle; Triangle()\"</b>.",
                    "Use <b>test.ipynb</b> for the interactive validation workflow.",
                ],
                s["body"],
            ),
        ]
    )
    return story


def add_page_frame(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(colors.HexColor("#f6f9fb"))
    canvas.rect(0, 0, width, height, stroke=0, fill=1)
    canvas.setStrokeColor(colors.HexColor("#d7e1e8"))
    canvas.setLineWidth(0.8)
    canvas.rect(12 * mm, 12 * mm, width - 24 * mm, height - 24 * mm, stroke=1, fill=0)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#526a79"))
    canvas.drawRightString(width - 16 * mm, 14 * mm, "Generated from repo evidence")
    canvas.restoreState()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=18 * mm,
        title="quickTDI App Summary",
        author="Codex",
    )
    doc.build(build_story(), onFirstPage=add_page_frame)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
