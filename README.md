# Maximum-Probability-Momentum-Trading-System
This repository contains project that calculates momentum score for a given stock based on predefined set of rules.

# MPTS — Maximum-Probability-Momentum-Trading-System

**Project:** MPTS (Maximum Probability Trading System)  
**Scope:** A Streamlit-based prototype to scan NIFTY segments, provide momentum score stocks using multi-factor signals (technical + fundamental + volatility), run simple backtests, and allow single-ticker analysis. Designed for Indian equity markets (NSE).

---

## Project description

MPTS is a prototype trading-tool that combines rule-based technical indicators, fundamental sanity checks, and a composite scoring engine to identify high-probability momentum candidates in NIFTY segments. It supports:
- Constituents fetching (NSE API / CSV / fallback sources),
- Uploading local snapshots,
- Single-ticker search (chart, score, backtest),
- Segment scanning for Top N candidates,
- Simple serial backtesting with stop/target logic.

This tool is intended as a research/prototyping platform — not production trading software.

---

## File list

- `mpts_app.py` — main Streamlit application (single-file).
- `README.md` — project documentation (this file).
- `snapshots/` — directory where constituency CSV snapshots will be saved (created automatically).

---

## How to run (local dev)

1. **Create and activate a Python env** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # macOS/Linux
   .venv\Scripts\activate         # Windows

2. **Install required packages**
    pip install streamlit pandas numpy yfinance plotly requests beautifulsoup4 lxml ta

3. **Run the app**
    streamlit run mpts_app.py

4. **browser will open http://localhost:8501**