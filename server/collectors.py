import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.tools.api import (
    get_company_facts,
    get_financial_metrics,
    get_price_data,
    get_market_cap,
    get_company_news,
)
from src.backtesting.metrics import PerformanceMetricsCalculator


def _safe_pct(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _compute_cagr(series: List[float], years: float) -> Optional[float]:
    if not series or len(series) < 2 or years <= 0:
        return None
    start, end = float(series[0]), float(series[-1])
    if start <= 0:
        return None
    try:
        return (end / start) ** (1.0 / years) - 1.0
    except Exception:
        return None


def collect_quick_facts(symbol: str, as_of: str) -> Dict[str, Any]:
    facts = get_company_facts(symbol)
    mcap = get_market_cap(symbol, as_of)
    return {
        "ticker": symbol,
        "name": getattr(facts, "name", None) if facts else None,
        "sector": getattr(facts, "sector", None) if facts else None,
        "industry": getattr(facts, "industry", None) if facts else None,
        "exchange": getattr(facts, "exchange", None) if facts else None,
        "location": getattr(facts, "location", None) if facts else None,
        "market_cap": mcap,
    }


def collect_valuation_metrics(symbol: str, as_of: str) -> Dict[str, Any]:
    metrics = get_financial_metrics(symbol, as_of, period="ttm", limit=1)
    if not metrics:
        return {}
    m = metrics[0]
    return {
        "pe": m.price_to_earnings_ratio,
        "ev_ebitda": m.enterprise_value_to_ebitda_ratio,
        "ps": m.price_to_sales_ratio,
        "p_fcf": None if m.free_cash_flow_yield is None else (1.0 / m.free_cash_flow_yield if m.free_cash_flow_yield != 0 else None),
        "ev": m.enterprise_value,
    }


def collect_growth_profit_quality(symbol: str, as_of: str) -> Dict[str, Any]:
    # Get last 9 quarterly or 3 yearly datapoints if available by using limit
    metrics = get_financial_metrics(symbol, as_of, period="ttm", limit=6)
    if not metrics:
        return {}
    # Use available fields for growth approximations
    rev_g = metrics[0].revenue_growth
    eps_g = metrics[0].earnings_per_share_growth
    fcf_g = metrics[0].free_cash_flow_growth
    return {
        "growth": {
            "revenue_cagr_3y": _safe_pct(rev_g),
            "eps_cagr_3y": _safe_pct(eps_g),
            "fcf_cagr_3y": _safe_pct(fcf_g),
        },
        "profit_quality": {
            "fcf_margin": metrics[0].free_cash_flow_per_share,  # placeholder if margin not available
            "cfo_to_net_income": metrics[0].operating_cash_flow_ratio,
        },
    }


def collect_volatility_and_beta(symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
    # Prices for the symbol
    px = get_price_data(symbol, start_date, end_date)
    if px.empty:
        return {}
    # Prices for benchmark (SPY) to compute beta
    bench = get_price_data("SPY", start_date, end_date)

    # Prepare series
    sym_ret = px["close"].pct_change().dropna()
    vol_calc = PerformanceMetricsCalculator()
    # Build DataFrame for metrics calculator
    portfolio_series = pd.DataFrame({
        "Date": sym_ret.index,
        "Portfolio Value": (1.0 + sym_ret).cumprod(),
    })
    perf = vol_calc.compute_metrics(portfolio_series.to_dict("records"))

    beta = None
    if not bench.empty:
        b_ret = bench["close"].pct_change().reindex(sym_ret.index).dropna()
        # Align again after dropna
        aligned = pd.concat([sym_ret, b_ret], axis=1).dropna()
        if not aligned.empty and aligned.shape[0] > 2:
            cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
            var = aligned.iloc[:, 1].var()
            if var and var != 0:
                beta = float(cov / var)

    return {
        "volatility": {
            "sharpe_1y": perf.get("sharpe_ratio"),
            "sortino_1y": perf.get("sortino_ratio"),
            "max_drawdown_1y_pct": perf.get("max_drawdown"),
        },
        "beta": beta,
    }


def collect_enriched_stock(symbol: str, months_back: int = 12) -> Dict[str, Any]:
    end_date = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.utcnow() - datetime.timedelta(days=30 * months_back)).strftime("%Y-%m-%d")

    facts = collect_quick_facts(symbol, end_date)
    valuation = collect_valuation_metrics(symbol, end_date)
    growth_profit = collect_growth_profit_quality(symbol, end_date)
    vol_beta = collect_volatility_and_beta(symbol, start_date, end_date)
    news_items = collect_latest_news(symbol, days_back=30, max_items=20)

    enriched: Dict[str, Any] = {
        "quick_facts": facts,
        "valuation": valuation,
        **growth_profit,
        **vol_beta,
        "news": news_items,
    }
    return enriched


def collect_latest_news(symbol: str, days_back: int = 30, max_items: int = 20) -> List[Dict[str, Any]]:
    end_date = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.utcnow() - datetime.timedelta(days=days_back)).strftime("%Y-%m-%d")
    try:
        items = get_company_news(symbol, end_date=end_date, start_date=start_date, limit=max_items)
    except Exception:
        return []

    results: List[Dict[str, Any]] = []
    for it in items[:max_items]:
        results.append(
            {
                "title": getattr(it, "title", None),
                "source": getattr(it, "source", None),
                "author": getattr(it, "author", None),
                "date": getattr(it, "date", None),
                "url": getattr(it, "url", None),
                "sentiment": getattr(it, "sentiment", None),
            }
        )
    return results


