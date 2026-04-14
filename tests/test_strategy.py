# tests/test_strategy.py
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.app import run_backtest, max_drawdown, sharpe_ratio, BUY_THRESHOLD, SELL_THRESHOLD

def make_dummy_data():
    """Create minimal test DataFrame."""
    dates = pd.date_range('2022-01-01', periods=100, freq='B')
    np.random.seed(42)
    prices = 100 * (1 + np.random.randn(100) * 0.01).cumprod()
    panic  = np.random.uniform(0, 100, 100)
    return pd.DataFrame({
        'date_dt':     dates,
        'qqq_close':   prices,
        'panic_index': panic,
    })

def test_backtest_returns_correct_columns():
    dff = make_dummy_data()
    bt = run_backtest(dff, 'qqq_close')
    assert 'cumulative_strategy' in bt.columns
    assert 'cumulative_buyhold' in bt.columns
    assert 'position' in bt.columns

def test_position_never_exceeds_max():
    dff = make_dummy_data()
    bt = run_backtest(dff, 'qqq_close', pos_max=80)
    assert bt['position'].max() <= 0.80 + 1e-9

def test_position_never_below_min():
    dff = make_dummy_data()
    bt = run_backtest(dff, 'qqq_close', pos_min=20)
    assert bt['position'].min() >= 0.20 - 1e-9

def test_max_drawdown_negative():
    series = pd.Series([100, 110, 90, 95, 105])
    dd = max_drawdown(series)
    assert dd < 0

def test_sharpe_ratio_positive_for_uptrend():
    returns = pd.Series([0.01] * 252)  # 1% daily gain
    sr = sharpe_ratio(returns)
    assert sr > 0

if __name__ == "__main__":
    test_backtest_returns_correct_columns()
    test_position_never_exceeds_max()
    test_position_never_below_min()
    test_max_drawdown_negative()
    test_sharpe_ratio_positive_for_uptrend()
    print("All tests passed ✅")