# Repository Guidelines

## Project Structure & Module Organization
- Core simulation modules live in `src/` (`src/TJ_*.py`).
- `src/TJ_Triangle.py` provides the high-level pipeline wrapper (orbit → GW → optical benches → locking → synthesis → TDI).
- `test.ipynb` is the primary usage demo and manual validation notebook.
- There are no subpackages; keep new modules alongside existing `TJ_*.py` files in `src/` and update imports accordingly.

## Build, Test, and Development Commands
- `python -m pip install numpy scipy` — install runtime dependencies (no requirements file in the repo).
- `python -m pip install -e .` — install editable package for local imports.
- `python -m pip install -e .[inference]` — include dynesty for Bayesian inference.
- `PYTHONPATH=src python -c "from TJ_Triangle import Triangle; Triangle()"` — quick smoke check that imports and defaults work.
- `jupyter notebook test.ipynb` — run the interactive workflow used for validation.

## Coding Style & Naming Conventions
- Python with 4-space indentation and PEP 8-style spacing.
- Module names follow the `TJ_*` prefix; class names use `PascalCase` (e.g., `Triangle`), functions use `snake_case`, constants use `UPPER_SNAKE_CASE` (see `TJ_constant.py`).
- Docstrings use NumPy-style sections (`Parameters`, `Returns`, `Examples`); keep this pattern in English.

## Testing Guidelines
- No automated test suite currently.
- Use `test.ipynb` for manual checks and to compare plots/arrays.
- If adding tests, prefer `pytest` with files named `test_*.py` in a new `tests/` folder and small, deterministic fixtures.

## Commit & Pull Request Guidelines
- No git history is available in this folder; follow your team’s standard. Suggested: short, imperative commit messages (e.g., “Add TDI delay helper”).
- PRs should include: a concise summary, rationale for numerical changes, steps to reproduce, and plots/screenshots when results are visual.
