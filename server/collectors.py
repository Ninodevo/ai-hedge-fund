import datetime
from typing import Any, Dict, List, Optional
from dateutil.relativedelta import relativedelta

import pandas as pd

from src.tools.api import (
    get_company_facts,
    get_financial_metrics,
    get_price_data,
    get_market_cap,
    get_company_news,
)
from src.backtesting.metrics import PerformanceMetricsCalculator
from server.data_provenance import DataProvenance, _get_exact_sec_filing_url


def _estimate_next_report_date(report_period: str, fiscal_period: Optional[str] = None) -> Optional[str]:
    """
    Estimate the next expected report date based on the last report period.
    
    SEC filing deadlines:
    - Quarterly (10-Q): ~45 days after quarter end (40 for accelerated filers)
    - Annual (10-K): ~60-90 days after fiscal year end
    
    Args:
        report_period: Last fiscal period end date (YYYY-MM-DD)
        fiscal_period: Fiscal period identifier (e.g., "2024-Q1", "2023-Annual")
    
    Returns:
        Estimated next report date (YYYY-MM-DD) or None if cannot determine
    """
    try:
        last_report_date = datetime.datetime.strptime(report_period, "%Y-%m-%d")
        
        # Determine if this was quarterly or annual
        is_annual = fiscal_period and ("annual" in fiscal_period.lower() or "year" in fiscal_period.lower())
        
        if is_annual:
            # Annual report: typically 60-90 days after fiscal year end
            # Use 75 days as average
            next_report = last_report_date + relativedelta(days=75)
        else:
            # Quarterly report: typically 40-45 days after quarter end
            # Use 45 days as average, then add 3 months for next quarter
            next_report = last_report_date + relativedelta(days=45)
            # Add 3 months to get to next quarter's report date
            next_report = next_report + relativedelta(months=3)
        
        return next_report.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


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


def collect_quick_facts(symbol: str, as_of: str, provenance: Optional[DataProvenance] = None) -> Dict[str, Any]:
    facts = get_company_facts(symbol)
    mcap = get_market_cap(symbol, as_of)
    
    # Track provenance
    if provenance:
        if mcap is not None:
            provenance.add_market_cap(mcap, as_of)
        if facts:
            for field in ["name", "sector", "industry", "exchange", "location"]:
                value = getattr(facts, field, None)
                if value is not None:
                    provenance.add_metric(
                        name=field,
                        value=value,
                        source="company_facts_api",
                        date=as_of,
                        period_type="latest",
                        is_latest=True,
                    )
            # Add SEC filings URL if available
            sec_filings_url = getattr(facts, "sec_filings_url", None)
            if sec_filings_url:
                provenance.add_metric(
                    name="sec_filings_url",
                    value=sec_filings_url,
                    source="company_facts_api",
                    date=as_of,
                    period_type="latest",
                    is_latest=True,
                )
    
    return {
        "ticker": symbol,
        "name": getattr(facts, "name", None) if facts else None,
        "sector": getattr(facts, "sector", None) if facts else None,
        "industry": getattr(facts, "industry", None) if facts else None,
        "exchange": getattr(facts, "exchange", None) if facts else None,
        "location": getattr(facts, "location", None) if facts else None,
        "market_cap": mcap,
    }


def collect_valuation_metrics(symbol: str, as_of: str, period_type: str = "ttm", provenance: Optional[DataProvenance] = None, cik: Optional[str] = None) -> Dict[str, Any]:
    """
    Collect valuation metrics.
    
    Args:
        symbol: Stock ticker symbol
        as_of: Date string (YYYY-MM-DD)
        period_type: "ttm" or "quarterly" - which period type to use
        provenance: Optional DataProvenance tracker
    """
    # Determine behavior based on period_type
    if period_type == "quarterly":
        # Use quarterly directly, no fallback needed
        metrics = get_financial_metrics(
            symbol, 
            as_of, 
            period="quarterly", 
            limit=1,
            check_freshness=False,  # Quarterly is always fresh by definition
            fallback_to_quarterly=False,
            prefer_fresh=False,
        )
    else:
        # Default TTM behavior with smart fallback
        metrics = get_financial_metrics(
            symbol, 
            as_of, 
            period="ttm", 
            limit=1,
            check_freshness=True,
            fallback_to_quarterly=True,  # Enable fallback for current reports
            prefer_fresh=True,  # Prefer quarterly if it's fresher
        )
    if not metrics:
        return {}
    m = metrics[0]
    
    # Track provenance
    if provenance:
        provenance.add_from_financial_metrics(
            metrics,
            field_mapping={
                "pe": "price_to_earnings_ratio",
                "ev_ebitda": "enterprise_value_to_ebitda_ratio",
                "ps": "price_to_sales_ratio",
                "ev": "enterprise_value",
            },
            as_of_date=as_of,
            period_type=period_type,
            cik=cik,
        )
        # Track free cash flow yield separately (calculated)
        if m.free_cash_flow_yield is not None:
            filing_url = _get_exact_sec_filing_url(cik, m.report_period, getattr(m, "fiscal_period", None)) if cik else None
            provenance.add_metric(
                name="p_fcf",
                value=1.0 / m.free_cash_flow_yield if m.free_cash_flow_yield != 0 else None,
                source="financial_metrics_api",
                date=m.report_period,
                period_type=period_type,
                is_latest=True,
                report_period=m.report_period,
                fiscal_period=getattr(m, "fiscal_period", None),
                filing_url=filing_url,
            )
    
    result = {
        "pe": m.price_to_earnings_ratio,
        "ev_ebitda": m.enterprise_value_to_ebitda_ratio,
        "ps": m.price_to_sales_ratio,
        "p_fcf": None if m.free_cash_flow_yield is None else (1.0 / m.free_cash_flow_yield if m.free_cash_flow_yield != 0 else None),
        "ev": m.enterprise_value,
        "date_reported": m.report_period,  # Fiscal period end date from SEC filings (10-Q/10-K)
        "fiscal_period": getattr(m, "fiscal_period", None),  # e.g., "2024-Q1"
        "next_report_estimated": _estimate_next_report_date(m.report_period, getattr(m, "fiscal_period", None)),
    }
    
    return result


def collect_growth_profit_quality(symbol: str, as_of: str, period_type: str = "ttm", provenance: Optional[DataProvenance] = None, cik: Optional[str] = None) -> Dict[str, Any]:
    """
    Collect growth and profit quality metrics.
    
    Args:
        symbol: Stock ticker symbol
        as_of: Date string (YYYY-MM-DD)
        period_type: "ttm" or "quarterly" - which period type to use
        provenance: Optional DataProvenance tracker
    """
    # Determine behavior based on period_type
    if period_type == "quarterly":
        # Use quarterly directly, no fallback needed
        metrics = get_financial_metrics(
            symbol, 
            as_of, 
            period="quarterly", 
            limit=6,
            check_freshness=False,  # Quarterly is always fresh by definition
            fallback_to_quarterly=False,
            prefer_fresh=False,
        )
    else:
        # Default TTM behavior with smart fallback
        metrics = get_financial_metrics(
            symbol, 
            as_of, 
            period="ttm", 
            limit=6,
            check_freshness=True,
            fallback_to_quarterly=True,  # Enable fallback for current reports
            prefer_fresh=True,  # Prefer quarterly if it's fresher
        )
    if not metrics:
        return {}
    # Use available fields for growth approximations
    rev_g = metrics[0].revenue_growth
    eps_g = metrics[0].earnings_per_share_growth
    fcf_g = metrics[0].free_cash_flow_growth
    
    # Track provenance
    if provenance:
        # Get filing URL for growth/profit metrics
        report_period = metrics[0].report_period
        fiscal_period = getattr(metrics[0], "fiscal_period", None)
        filing_url = _get_exact_sec_filing_url(cik, report_period, fiscal_period) if cik else None
        
        # Track the underlying metrics
        provenance.add_from_financial_metrics(
            metrics,
            field_mapping={
                "revenue_growth": "revenue_growth",
                "eps_growth": "earnings_per_share_growth",
                "fcf_growth": "free_cash_flow_growth",
                "fcf_per_share": "free_cash_flow_per_share",
                "operating_cash_flow_ratio": "operating_cash_flow_ratio",
            },
            as_of_date=as_of,
            period_type=period_type,
            cik=cik,
        )
        
        # Also track the response parameter names with filing URLs
        if rev_g is not None:
            provenance.add_metric(
                name="revenue_cagr_3y",
                value=_safe_pct(rev_g),
                source="financial_metrics_api",
                date=report_period,
                period_type=period_type,
                is_latest=True,
                report_period=report_period,
                fiscal_period=fiscal_period,
                filing_url=filing_url,
            )
        if eps_g is not None:
            provenance.add_metric(
                name="eps_cagr_3y",
                value=_safe_pct(eps_g),
                source="financial_metrics_api",
                date=report_period,
                period_type=period_type,
                is_latest=True,
                report_period=report_period,
                fiscal_period=fiscal_period,
                filing_url=filing_url,
            )
        if fcf_g is not None:
            provenance.add_metric(
                name="fcf_cagr_3y",
                value=_safe_pct(fcf_g),
                source="financial_metrics_api",
                date=report_period,
                period_type=period_type,
                is_latest=True,
                report_period=report_period,
                fiscal_period=fiscal_period,
                filing_url=filing_url,
            )
        if metrics[0].free_cash_flow_per_share is not None:
            provenance.add_metric(
                name="fcf_margin",
                value=metrics[0].free_cash_flow_per_share,
                source="financial_metrics_api",
                date=report_period,
                period_type=period_type,
                is_latest=True,
                report_period=report_period,
                fiscal_period=fiscal_period,
                filing_url=filing_url,
            )
        if metrics[0].operating_cash_flow_ratio is not None:
            provenance.add_metric(
                name="cfo_to_net_income",
                value=metrics[0].operating_cash_flow_ratio,
                source="financial_metrics_api",
                date=report_period,
                period_type=period_type,
                is_latest=True,
                report_period=report_period,
                fiscal_period=fiscal_period,
                filing_url=filing_url,
            )
    
    result = {
        "growth": {
            "revenue_cagr_3y": _safe_pct(rev_g),
            "eps_cagr_3y": _safe_pct(eps_g),
            "fcf_cagr_3y": _safe_pct(fcf_g),
        },
        "profit_quality": {
            "fcf_margin": metrics[0].free_cash_flow_per_share,  # placeholder if margin not available
            "cfo_to_net_income": metrics[0].operating_cash_flow_ratio,
        },
        "date_reported": metrics[0].report_period,  # Fiscal period end date from SEC filings (10-Q/10-K)
        "fiscal_period": getattr(metrics[0], "fiscal_period", None),  # e.g., "2024-Q1"
    }
    
    return result


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


def collect_enriched_stock(symbol: str, months_back: int = 12, period_type: str = "ttm") -> Dict[str, Any]:
    """
    Collect enriched stock data including valuation, growth, and quality metrics.
    
    Args:
        symbol: Stock ticker symbol
        months_back: Number of months of historical data to include
        period_type: "ttm" or "quarterly" - which period type to use for financial metrics
    """
    end_date = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.utcnow() - datetime.timedelta(days=30 * months_back)).strftime("%Y-%m-%d")

    # Initialize provenance tracker
    provenance = DataProvenance()

    facts = collect_quick_facts(symbol, end_date, provenance=provenance)
    # Get CIK from company facts for constructing filing URLs
    company_facts_obj = get_company_facts(symbol)
    cik = getattr(company_facts_obj, "cik", None) if company_facts_obj else None
    
    valuation = collect_valuation_metrics(symbol, end_date, period_type=period_type, provenance=provenance, cik=cik)
    growth_profit = collect_growth_profit_quality(symbol, end_date, period_type=period_type, provenance=provenance, cik=cik)
    vol_beta = collect_volatility_and_beta(symbol, start_date, end_date)
    news_items = collect_latest_news(symbol, days_back=30, max_items=20)

    # Extract next_report_estimated from valuation if available
    next_report_estimated = valuation.get("next_report_estimated") if valuation else None

    enriched: Dict[str, Any] = {
        "quick_facts": facts,
        "valuation": valuation,
        **growth_profit,
        **vol_beta,
        "news": news_items,
        "next_report_estimated": next_report_estimated,  # Estimated next report/filing date
        "data_provenance": {
            "parameters": provenance.get_provenance(),
            "summary": provenance.get_summary(),
        },
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


