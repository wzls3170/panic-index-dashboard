import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

st.set_page_config(layout="wide", page_title="Market Sentiment Dashboard")

# ── Constants ─────────────────────────────────────────────────────────────────
BUY_THRESHOLD  = 64
SELL_THRESHOLD = 15
RISK_FREE_RATE = 0.04 / 252


# ── Data Loading & Processing ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    base_path = os.path.dirname(os.path.dirname(__file__))
    csv_path  = os.path.join(base_path, "data", "market_data_5.csv")
    df = pd.read_csv(csv_path)

    df['date_dt']   = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df['date_only'] = df['date_dt'].dt.date
    df = df.sort_values('date_dt')
    df = df.drop_duplicates(subset=['date_only'], keep='last')
    df = df.dropna(subset=['date_dt'])

    price_cols = ['qqq_close', 'smh_close', 'spx_close', 'dji_close', 'vix']
    df[price_cols] = df[price_cols].interpolate(method='linear')

    def score_to_rating(score):
        if pd.isna(score): return None
        if score < 25:     return "extreme fear"
        elif score < 45:   return "fear"
        elif score < 55:   return "neutral"
        elif score < 75:   return "greed"
        else:              return "extreme greed"

    df['fear_greed_rating'] = df.apply(
        lambda row: score_to_rating(row['fear_greed_index'])
        if pd.isna(row.get('fear_greed_rating')) else row['fear_greed_rating'],
        axis=1
    )

    for col in ['qqq', 'smh', 'spx', 'dji']:
        df[f'{col}_return'] = df[f'{col}_close'].pct_change(fill_method=None) * 100


    df['vix_norm']    = (df['vix'] - df['vix'].min()) / (df['vix'].max() - df['vix'].min()) * 100
    df['fg_fear']     = 100 - df['fear_greed_index']
    df['panic_index'] = df['vix_norm'] * 0.5 + df['fg_fear'] * 0.5

    return df


def run_backtest(dff, target_col, pos_initial=50, add_amount=20,
                 reduce_amount=20, pos_max=100, pos_min=0):
    bt = dff[['date_dt', target_col, 'panic_index']].dropna().copy().reset_index(drop=True)

    position = pos_initial / 100  # 转成小数
    positions = []

    for _, row in bt.iterrows():
        if row['panic_index'] > BUY_THRESHOLD:
            position = min(position + add_amount/100, pos_max/100)
        elif row['panic_index'] < SELL_THRESHOLD:
            position = max(position - reduce_amount/100, pos_min/100)
        positions.append(position)

    bt['position']            = positions
    bt['daily_return']        = bt[target_col].pct_change(fill_method=None)
    bt['strategy_return'] = bt['daily_return'] * bt['position'].shift(1).fillna(0)
    bt['cumulative_strategy'] = (1 + bt['strategy_return'].fillna(0)).cumprod() * 100
    bt['cumulative_buyhold']  = (1 + bt['daily_return'].fillna(0)).cumprod() * 100
    return bt


def max_drawdown(series):
    peak = series.cummax()
    return ((series - peak) / peak).min() * 100


def sharpe_ratio(returns):
    excess = returns.fillna(0) - RISK_FREE_RATE
    return excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0.0


# ── Load ──────────────────────────────────────────────────────────────────────
df = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.title(" Market Sentiment & Panic Index Dashboard")
st.markdown(
    "Exploring Market Asymmetry: Can a Composite Panic Index (VIX + CNN Fear & Greed) Identify Tactical Entry and Exit Opportunities Across Major Indices and the Semiconductor Sector (2021–2026)?"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("🎛️ Controls")

target_map = {
    "Nasdaq 100 (QQQ)":     "qqq_close",
    "Semiconductors (SMH)": "smh_close",
    "S&P 500 (SPX)":        "spx_close",
    "Dow Jones (DJI)":      "dji_close",
}
selected_label = st.sidebar.selectbox("Select Index", list(target_map.keys()))
target_col     = target_map[selected_label]

min_date   = df['date_dt'].min().date()
max_date   = df['date_dt'].max().date()
date_range = st.sidebar.date_input(
    "Date Range", value=(min_date, max_date),
    min_value=min_date, max_value=max_date
)

if len(date_range) == 2:
    dff = df[
        (df['date_dt'].dt.date >= date_range[0]) &
        (df['date_dt'].dt.date <= date_range[1])
    ].copy()
else:
    dff = df.copy()
# Recalculate MA based on selected index
dff['ma50']  = dff[target_col].rolling(window=50).mean()
dff['ma200'] = dff[target_col].rolling(window=200).mean()
dff['price_dev_pct'] = (dff[target_col] - dff['ma50']) / dff['ma50'] * 100

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Position Sizing")


# Sidebar 滑块
pos_initial  = st.sidebar.slider("Initial position (%)",    0, 100, 50, 10)
reduce_amount = st.sidebar.slider("Reduce on sell signal (%)", 0, 50, 10, 5)  # 默认10%
add_amount    = st.sidebar.slider("Add on buy signal (%)",    0, 50, 20, 5)  # 默认20%
pos_max      = st.sidebar.slider("Max position (%)",        50, 100, 100, 10)
pos_min      = st.sidebar.slider("Min position (%)",        0, 50, 0, 10)


bt = run_backtest(dff, target_col, pos_initial, add_amount, reduce_amount, pos_max, pos_min)

strat_return = bt['cumulative_strategy'].iloc[-1] - 100
bnh_return   = bt['cumulative_buyhold'].iloc[-1] - 100
strat_sharpe = sharpe_ratio(bt['strategy_return'])
bnh_sharpe   = sharpe_ratio(bt['daily_return'])
strat_dd     = max_drawdown(bt['cumulative_strategy'])
bnh_dd       = max_drawdown(bt['cumulative_buyhold'])

strat_wins = sum([
    strat_return > bnh_return,
    strat_sharpe > bnh_sharpe,
    strat_dd     > bnh_dd,
])

if strat_wins == 3:
    verdict = "The strategy outperforms Buy & Hold on all three metrics: return, risk-adjusted return, and drawdown protection."
elif strat_wins == 2:
    verdict = "The strategy outperforms Buy & Hold on two of three metrics."
elif strat_dd > bnh_dd:
    verdict = (
        f"Buy & Hold outperforms on absolute return and Sharpe ratio in this period. "
        f"However, the strategy reduces maximum drawdown significantly "
        f"({strat_dd:.1f}% vs {bnh_dd:.1f}%), "
        "making it more suitable for risk-averse investors."
    )
else:
    verdict = "Buy & Hold outperforms the strategy on all metrics in this period."
st.sidebar.subheader("🔗 Correlation Analysis")
if not dff.empty:
    next_day_return = dff[target_col].pct_change().shift(-1) * 100

    corr_panic_price = dff['panic_index'].corr(next_day_return)
    corr_fg_price    = dff['fear_greed_index'].corr(next_day_return)
    corr_vix_price   = dff['vix'].corr(next_day_return)


    st.sidebar.metric("Panic Index → Next Day Return",  f"{corr_panic_price:.3f}")
    st.sidebar.metric("Fear & Greed → Next Day Return", f"{corr_fg_price:.3f}")
    st.sidebar.metric("VIX → Next Day Return",          f"{corr_vix_price:.3f}")


# ── Chart 1+2: Price + Panic Index ───────────────────────────────────────────
st.subheader("📈 Price Signal Map  &   Composite Panic Index")

buy_dates  = dff[dff['panic_index'] > BUY_THRESHOLD]['date_dt']
sell_dates = dff[dff['panic_index'] < SELL_THRESHOLD]['date_dt']

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    subplot_titles=(
        f"Price & MA50/MA200 — {selected_label} ",
        "Composite Panic Index  (VIX 50% + Inverted Fear & Greed 50%)"
    ),
    row_heights=[0.6, 0.4]
)

# 把所有形状一次性传入
shapes = []

for d in buy_dates:
    shapes.append(dict(
        type="line",
        x0=d, x1=d, y0=0, y1=1,
        xref="x", yref="y domain",
        line=dict(color="rgba(239,85,59,0.3)", width=1.5),
    ))

for d in sell_dates:
    shapes.append(dict(
        type="line",
        x0=d, x1=d, y0=0, y1=1,
        xref="x", yref="y domain",
        line=dict(color="rgba(0,204,150,0.3)", width=1.5),
    ))

fig.update_layout(shapes=shapes)

fig.add_trace(go.Scatter(x=dff['date_dt'], y=dff[target_col], name="Price",
                         line=dict(color='black', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=dff['date_dt'], y=dff['ma50'], name="MA50",
                         line=dict(color='orange', dash='dot')), row=1, col=1)
fig.add_trace(go.Scatter(x=dff['date_dt'], y=dff['ma200'], name="MA200",
                         line=dict(color='blue', dash='dot', width=1.5)), row=1, col=1)


# main panic index area
fig.add_trace(go.Scatter(
    x=dff['date_dt'], y=dff['panic_index'],
    name="Panic Index",
    line=dict(color='#aaaaaa', width=1),
    fill='tozeroy', fillcolor='rgba(170,170,170,0.1)'
), row=2, col=1)

# buy signal
buy_df = dff[dff['panic_index'] > BUY_THRESHOLD]
fig.add_trace(go.Scatter(
    x=buy_df['date_dt'], y=buy_df['panic_index'],
    mode='markers', name="Buy Signal",
    marker=dict(color='#EF553B', size=8, symbol='circle')
), row=2, col=1)

# Sell Signal
sell_df = dff[dff['panic_index'] < SELL_THRESHOLD]
fig.add_trace(go.Scatter(
    x=sell_df['date_dt'], y=sell_df['panic_index'],
    mode='markers', name="Sell Signal",
    marker=dict(color='#00CC96', size=8, symbol='circle')
), row=2, col=1)
fig.add_hline(y=BUY_THRESHOLD,  line_color="red",   line_dash="dash",
              annotation_text="Buy Zone (95th pct)",  row=2, col=1)
fig.add_hline(y=SELL_THRESHOLD, line_color="green", line_dash="dash",
              annotation_text="Sell Zone (5th pct)", row=2, col=1)

fig.update_layout(
    height=750, hovermode="x unified", template="plotly_white",
    legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
    margin=dict(t=80, b=40, l=60, r=40)
)
fig.update_xaxes(tickformat="%b %Y", nticks=15, tickangle=45)
fig.update_xaxes(hoverformat="%b %d, %Y")
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="Panic Index (0-100)", row=2, col=1)

st.plotly_chart(fig, use_container_width=True, key="main_chart")

st.divider()

# ── Chart 3: Scatter Plot ──────────────────────────────────────────────────────────────


st.subheader("🔍 Sentiment vs. MA50 Deviation")
st.caption("Each dot = one trading day · Color & size = VIX · "
        "Bottom-left = panic bottom (best historical entry points)")

vix_min = float(dff['vix'].min())
vix_max = float(dff['vix'].max())

fig_sc = go.Figure(go.Scatter(
    x=dff['fear_greed_index'],
    y=dff['price_dev_pct'],
    mode='markers',
    marker=dict(
    color=dff['vix'],
        line=dict(width=0.5, color='DarkSlateGrey'),
        colorscale='RdYlGn_r',
        cmin=vix_min, cmax=vix_max,
        size=dff['vix'].apply(
            lambda x: 6 + (x - vix_min) / max(vix_max - vix_min, 1) * 14
        ),
        showscale=True,
        colorbar=dict(title="VIX")
    ),
    text=dff['date_dt'].dt.strftime('%b %d, %Y'),
    customdata=dff['vix'],
    hovertemplate=(
        "Date: %{text}<br>"
        "Fear & Greed: %{x:.1f}<br>"
        "Price Dev from MA50: %{y:.2f}%<br>"
        "VIX: %{customdata:.2f}"
        "<extra></extra>"
    )
))
fig_sc.update_layout(
    xaxis_title="Fear & Greed Index  (0 = Extreme Fear  →  100 = Extreme Greed)",
    yaxis_title=f"{selected_label} Deviation from MA50 (%)",
    template="plotly_white", height=430,
    margin=dict(t=20, b=60, l=60, r=40),
    shapes=[dict(type='line', y0=0, y1=0, x0=0, x1=100,
                line=dict(color='gray', dash='dash'))]
)
st.plotly_chart(fig_sc, use_container_width=True, key="scatter_chart")
st.divider()


# ── Chart 4: Backtest ─────────────────────────────────────────────────────────

st.subheader("⚖️ Strategy vs Buy & Hold — Backtest")
st.caption(
    f"Position sizing: {int(pos_initial)}% initial · "
    f"+{int(add_amount)}% on buy signal (max {int(pos_max)}%) · "
    f"-{int(reduce_amount)}% on sell signal (min {int(pos_min)}%)"
)

fig_bt = go.Figure()
fig_bt.add_trace(go.Scatter(x=bt['date_dt'], y=bt['cumulative_strategy'],
                            name="Panic Index Strategy",
                            line=dict(color='#EF553B', width=2)))
fig_bt.add_trace(go.Scatter(x=bt['date_dt'], y=bt['cumulative_buyhold'],
                            name="Buy & Hold",
                            line=dict(color='black', width=2, dash='dot')))
fig_bt.update_layout(
    height=380, hovermode="x unified", template="plotly_white",
    legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
    margin=dict(t=40, b=40, l=60, r=40),
    yaxis_title="Cumulative Return (base = 100)"
)
fig_bt.update_xaxes(tickformat="%b %Y", nticks=15, tickangle=45,
                    hoverformat="%b %d, %Y")
st.plotly_chart(fig_bt, use_container_width=True, key="backtest_chart")


st.divider()
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Strategy Return",   f"{bt['cumulative_strategy'].iloc[-1]-100:.1f}%")
c2.metric("Buy & Hold Return", f"{bt['cumulative_buyhold'].iloc[-1]-100:.1f}%")
c3.metric("Strategy Max DD",   f"{max_drawdown(bt['cumulative_strategy']):.1f}%")
c4.metric("B&H Max DD",        f"{max_drawdown(bt['cumulative_buyhold']):.1f}%")
c5.metric("Strategy Sharpe",   f"{sharpe_ratio(bt['strategy_return']):.3f}")
c6.metric("B&H Sharpe",        f"{sharpe_ratio(bt['daily_return']):.3f}")

st.divider()

st.subheader("📊 Forward Return Analysis — Buy the Panic, Sell the Greed?")
def calc_forward_returns(dff, signal_mask, target_col, label):
    signal_dates = dff[signal_mask].copy()
    rows = []
    for idx, row in signal_dates.iterrows():
        signal_price = row[target_col]
        for horizon, h_label in [(5, '1 Week'), (21, '1 Month'), (63, '3 Months')]:
            future = dff[dff['date_dt'] > row['date_dt']].head(horizon)
            if len(future) == horizon:
                fwd_return = (future[target_col].iloc[-1] - signal_price) / signal_price * 100
                rows.append({
                    'Signal': label,
                    'Horizon': h_label,
                    'Forward Return (%)': round(fwd_return, 2)
                })
    return pd.DataFrame(rows)

if not dff.empty:
    df_panic = calc_forward_returns(
        dff, dff['panic_index'] > BUY_THRESHOLD, target_col, '🔴 Extreme Panic (Buy)'
    )
    df_greed = calc_forward_returns(
        dff, dff['panic_index'] < SELL_THRESHOLD, target_col, '🟢 Extreme Greed (Sell)'
    )
    df_fwd = pd.concat([df_panic, df_greed])

    if not df_fwd.empty:
        summary = df_fwd.groupby(['Signal', 'Horizon'])['Forward Return (%)'].agg(
            Mean=('mean'),
            Median=('median'),
            Win_Rate=lambda x: (x > 0).mean() * 100,
            Count='count'
        ).round(2)
        summary.columns = ['Mean Return (%)', 'Median Return (%)', 'Win Rate (%)', 'Signals']
        summary = summary.reindex([
            ('🔴 Extreme Panic (Buy)', '1 Week'),
            ('🔴 Extreme Panic (Buy)', '1 Month'),
            ('🔴 Extreme Panic (Buy)', '3 Months'),
            ('🟢 Extreme Greed (Sell)', '1 Week'),
            ('🟢 Extreme Greed (Sell)', '1 Month'),
            ('🟢 Extreme Greed (Sell)', '3 Months'),
        ])
        st.dataframe(summary, use_container_width=True)
        st.caption(
            "Panic signals: Win Rate > 50% and Mean Return > 0 validates 'buy the panic'. "
            "Greed signals: Win Rate < 50% and Mean Return < 0 validates 'sell the greed'."
        )

# ── Write-up ──────────────────────────────────────────────────────────────────
with st.expander("📝 Project Write-up", expanded=False):
    st.markdown(f"""



## Key Findings

### Price Signal Map & Composite Panic Index:
- Buy signals (Panic Index > {BUY_THRESHOLD}) cluster around major market
  bottoms: 2022 bear market, 2025 tariff shock.
- Sell signals (Panic Index < {SELL_THRESHOLD}) appear during sustained
  greed periods in 2023–2024 bull market.

### Scatter Plot (Sentiment vs. MA50 Deviation):
- Strong positive correlation between Fear & Greed and price deviation from MA50.
- Extreme panic days (bottom-left, red/large dots) cluster well below MA50,
  confirming historically oversold conditions.
- Day-ahead correlation is near zero, consistent with the
  [Efficient Market Hypothesis](https://en.wikipedia.org/wiki/Efficient-market_hypothesis) — sentiment has no short-term predictive power,
  but identifies medium-term regime shifts.
- **Conclusion**: The Panic Index is a medium-term regime indicator, not a
  short-term timing tool. Its value lies in identifying multi-week buying
  opportunities during market stress.



### Backtest ({date_range[0]} – {date_range[1]}):

| Metric | Strategy | Buy & Hold |
|--------|----------|------------|
| Total Return | {bt['cumulative_strategy'].iloc[-1]-100:.1f}% | {bt['cumulative_buyhold'].iloc[-1]-100:.1f}% |
| Max Drawdown | {max_drawdown(bt['cumulative_strategy']):.1f}% | {max_drawdown(bt['cumulative_buyhold']):.1f}% |
| Sharpe Ratio | {sharpe_ratio(bt['strategy_return']):.3f} | {sharpe_ratio(bt['daily_return']):.3f} |

{verdict}

### Forward Return Analysis:
- Extreme panic signals generate positive returns over 1-week to 3-month
  horizons with win rates of 64–74%, confirming the ["buy the panic hypothesis"](https://www.investopedia.com/terms/p/panicbuying.asp) .
- Extreme greed signals do NOT reliably predict subsequent declines — prices
  continued rising after greed signals during this bull market, indicating the
  sell signal is less reliable in sustained uptrends.
- **The Panic Index is asymmetric: buy signals are validated, sell signals are not.**

---

## Design Decisions
- **Four-chart layout**: Each chart serves a distinct analytical purpose:
  - **Chart 1 (Price Signal Map & Composite Panic Index)**: Shows *where* buy/sell signals appear on the
    price chart, allowing users to visually validate signal timing against
    historical price movements.
  - **Chart 2 (Sentiment vs. MA50 Deviation)**: Reveals the *structural relationship*
    between sentiment and price level, removing long-term trend bias via MA50 deviation.
  - **Chart 3 (Backtest)**: Quantifies *whether acting on signals generates returns*,
    comparing the strategy against passive Buy & Hold.
  - **Chart 4 (Forward Return Analysis)**: Statistically validates signal effectiveness
    across multiple time horizons, providing evidence beyond visual pattern recognition.

- **Composite Panic Index**: Normalized VIX (50%) + Inverted Fear & Greed (50%),
  scaled to 0–100. Buy threshold (>{BUY_THRESHOLD}) = 95th percentile;
  Sell threshold (<{SELL_THRESHOLD}) = 5th percentile of full 5-year history.
  Thresholds are fixed — consistent with real-world quant strategy development
  where thresholds are calibrated on historical data and held constant during deployment.

- **Removed standalone VIX and F&G charts**: Both indicators are already encoded
  in the Panic Index. Keeping them separately would be redundant.

- **Sharpe Ratio & Maximum Drawdown**: Used alongside total return to evaluate
  strategy quality. Sharpe measures risk-adjusted return (excess return per unit
  of volatility); Max Drawdown measures the largest peak-to-trough decline —
  critical for risk-averse investors.

- **Scatter plot (MA50 deviation)**: Uses price deviation from MA50 instead of
  absolute price to remove long-term trend bias, revealing the true relationship
  between sentiment and relative price level. MA50 chosen as the medium-term
  benchmark widely used by institutional investors.

- **Forward Return Analysis**: Validates signal effectiveness statistically
  across 1-week, 1-month, and 3-month horizons, providing evidence beyond
  visual pattern recognition.

- **Shared x-axis (Chart 1 + 2)**: Synchronized hover and zoom lets users trace
  any Panic Index spike directly to its corresponding price movement.

- **Date range filter**: All charts and backtest update simultaneously, enabling
  period-specific analysis (e.g., isolating the 2022 bear market or 2025 tariff shock).

- **Interactive position sizing**: Users can set initial position, add/reduce
  amounts per signal, and position limits — backtest updates in real time.

---

## Data Sources
- **Fear & Greed Index (2021–2026)**:
  [MacroMicro](https://sc.macromicro.me/charts/50108/cnn-fear-and-greed),
  [CNN](https://www.cnn.com/markets/fear-and-greed)
- **Price data**: Yahoo Finance via `yfinance` — QQQ, SMH, SPX, DJI, VIX
  daily adjusted closing prices.
- **Missing values**: Market holidays (e.g., Christmas, Thanksgiving, Good Friday,
  Easter) have no trading data and are handled via linear interpolation for
  chart continuity.

---

## References
- Shneiderman, B. (1994). Dynamic queries for visual information seeking.
  *IEEE Software*, 11(6), 70–77.
- Chart design inspired by standard financial dashboard conventions
  (TradingView, Bloomberg Terminal layout).
- Efficient Market Hypothesis. https://en.wikipedia.org/wiki/Efficient-market_hypothesis
- Buy the Panic. https://www.investopedia.com/terms/p/panicbuying.asp
---

## Development Commentary
- **Total time**: ~4 person-days
- **Most time-consuming**: Data pipeline — fetching, cleaning, and aligning
  Fear & Greed history from two different sources (MacroMicro CSV + CNN API)
  with yfinance price data across a 5-year period. Chart interaction logic
  (synchronized date filtering, buy/sell signal rendering, real-time backtest
  updates) also required significant iteration.
    """)