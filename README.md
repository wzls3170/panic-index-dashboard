# Market Sentiment & Panic Index Dashboard

> **Exploring Market Asymmetry**: Can a Composite Panic Index (VIX + CNN Fear & Greed) identify tactical entry and exit opportunities across major indices and the Semiconductor sector (2021–2026)?

**Live Demo**: [https://panic-index-dashboard-gvjixdtgeazuqzzm3kqqv4.streamlit.app](#)

---

## Overview

An interactive financial dashboard that combines VIX and CNN Fear & Greed Index into a single **Composite Panic Index** to identify buy/sell opportunities in tech and semiconductor ETFs. The dashboard covers a full market cycle from May 2021 to April 2026, including the 2022 bear market, 2023–2024 AI bull run, and 2025 tariff shock.

**Key Finding**: The Panic Index is asymmetric — buy signals (extreme panic) are statistically validated with 64–74% win rates across 1-week to 3-month horizons, while sell signals (extreme greed) are less reliable in sustained bull markets.

---

## Features

- **💹 Price Signal Map**: Buy/sell signals overlaid on price chart with MA50 & MA200
- **🔍 Sentiment vs. MA50 Deviation**: Scatter plot revealing structural relationship between sentiment and price level
- **⚖️ Interactive Backtest**: Position sizing strategy vs Buy & Hold with real-time parameter adjustment
- **🎯 Forward Return Analysis**: Statistical validation of signal effectiveness across 1-week, 1-month, 3-month horizons
- **🎛️ Interactive Controls**: Date range filter, index selector, and position sizing sliders — all charts update simultaneously

---

## Composite Panic Index

```
vix_norm    = (VIX - VIX_min) / (VIX_max - VIX_min) × 100
fg_fear     = 100 - Fear_Greed_Score
panic_index = vix_norm × 0.5 + fg_fear × 0.5
```

| Signal | Threshold | Basis |
|--------|-----------|-------|
| Buy Zone | Panic Index > 64 | 95th percentile of 5-year history |
| Sell Zone | Panic Index < 15 | 5th percentile of 5-year history |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Visualization | Plotly |
| Data | yfinance, CNN Fear & Greed API |
| Language | Python 3.12 |
| Deployment | Streamlit Community Cloud |

---

## Project Structure

```
panic-index-dashboard/
├── src/
│   └── app.py              # Main Streamlit application
├── data/
│   └── market_data_5.csv   # 5-year historical data (2021–2026)
├── tests/
│   └── test_strategy.py    # Unit tests for backtest logic
├── requirements.txt
└── README.md
```

---

## Local Setup

```bash
git clone https://github.com/wzls3170/panic-index-dashboard.git
cd panic-index-dashboard
pip install -r requirements.txt
streamlit run src/app.py
```

---

## Data Sources

- **Fear & Greed Index (2021–2022)**: [MacroMicro](https://sc.macromicro.me/charts/50108/cnn-fear-and-greed)
- **Fear & Greed Index (2023–2026)**: CNN internal endpoint (`production.dataviz.cnn.io`)
- **Price data**: Yahoo Finance via `yfinance` — QQQ, SMH, SPX, DJI, VIX
- **Missing values**: Market holidays handled via linear interpolation

---

## Key Results (Full Period: May 2021 – Apr 2026)

## Key Results

**Price Signal Map:**
- Buy signals (Panic Index > 64) cluster around major market bottoms: 2022 bear market, 2025 tariff shock.
- Sell signals (Panic Index < 15) appear during sustained greed periods in 2023–2024 bull market.

**Forward Return Analysis (full 5-year period):**
- Extreme panic signals: win rates **64–74%** across 1-week to 3-month horizons ✅
- Extreme greed signals: prices continued rising after signals ❌
- **The Panic Index is asymmetric — buy signals validated, sell signals are not.**

**Backtest vs Buy & Hold:**
- Strategy reduces maximum drawdown by ~56% (-15% vs -35%)
- Buy & Hold outperforms on absolute return in this bull market period
- Strategy is more suitable for **risk-averse investors** who prioritize capital preservation

**Correlation Analysis:**
- Day-ahead predictive correlation near zero — consistent with Efficient Market Hypothesis
- Medium-term regime identification (1-week to 3-month) shows significant signal effectiveness
---

