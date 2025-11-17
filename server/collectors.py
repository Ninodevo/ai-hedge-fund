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
# from server.news_analyzer import analyze_news  # Commented out - not using structured news analyzer
from langchain_core.prompts import ChatPromptTemplate
from src.llm.models import get_model, get_model_info, ModelProvider
import os


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


def _parse_analysis_by_category(analysis_text: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse the analysis text and group results by category.
    
    Args:
        analysis_text: The raw analysis text from LLM
        
    Returns:
        Dictionary with category keys and dicts containing category_name and items list
    """
    if not analysis_text:
        return {}
    
    # Category mapping from numbers/names to standardized keys
    # Patterns match categories with or without number prefixes, and handle parenthetical text
    category_patterns = [
        (r"^(?:\d+\.?\s*)?relevant\s+news(?:\s+and\s+their\s+implications)?", "relevant_news"),
        (r"^(?:\d+\.?\s*)?financial\s+moves", "financial_moves"),
        (r"^(?:\d+\.?\s*)?upcoming\s+events", "upcoming_events"),
        (r"^(?:\d+\.?\s*)?short[- ]term\s+outlook", "short_term_outlook"),
        (r"^(?:\d+\.?\s*)?long[- ]term\s+outlook", "long_term_outlook"),
    ]
    
    # Category display names
    category_names = {
        "relevant_news": "Relevant news and their implications",
        "financial_moves": "Financial moves",
        "upcoming_events": "Upcoming events",
        "short_term_outlook": "Short-term outlook",
        "long_term_outlook": "Long-term outlook",
    }
    
    import re
    
    result = {}
    lines = analysis_text.split('\n')
    current_category = None
    current_bullets = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if this line starts a new category
        category_found = False
        line_lower = line.lower()
        # Remove parenthetical text for matching (e.g., "Financial moves (acquisitions...)" -> "Financial moves")
        line_for_matching = re.sub(r'\s*\([^)]*\).*$', '', line_lower).strip()
        
        for pattern, cat_key in category_patterns:
            if re.match(pattern, line_for_matching):
                # Save previous category if exists
                if current_category and current_bullets:
                    result[current_category] = {
                        "category_name": category_names.get(current_category, current_category),
                        "items": current_bullets
                    }
                # Start new category
                current_category = cat_key
                current_bullets = []
                category_found = True
                break
        
        # If it's a bullet point (starts with • or -)
        if not category_found and (line.startswith('•') or line.startswith('-')):
            if current_category:
                # Remove bullet marker and clean up
                bullet_text = line.lstrip('•-').strip()
                if bullet_text:
                    current_bullets.append(bullet_text)
    
    # Save last category
    if current_category and current_bullets:
        result[current_category] = {
            "category_name": category_names.get(current_category, current_category),
            "items": current_bullets
        }
    
    return result


def collect_enriched_stock(symbol: str, months_back: int = 12, period_type: str = "ttm", api_keys: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Collect enriched stock data including valuation, growth, and quality metrics.
    
    Args:
        symbol: Stock ticker symbol
        months_back: Number of months of historical data to include
        period_type: "ttm" or "quarterly" - which period type to use for financial metrics
        api_keys: Optional dictionary of API keys for LLM calls
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
    
    # Use _analyze_news_with_llm directly instead of collect_latest_news
    news_analysis_result = None
    news_analysis_grouped = {}
    try:
        print(f"Analyzing news for {symbol} using LLM with web search...")
        news_analysis_result = _analyze_news_with_llm(
            symbol,
            news_items=[],  # Not used, kept for compatibility
            api_keys=api_keys
        )
        
        # Parse and group analysis by category
        if news_analysis_result and isinstance(news_analysis_result, dict):
            analysis_text = news_analysis_result.get("analysis")
            if analysis_text:
                news_analysis_grouped = _parse_analysis_by_category(analysis_text)
                print(f"News analysis completed for {symbol}, found {len(news_analysis_grouped)} categories")
    except Exception as e:
        import traceback
        print(f"Error analyzing news for {symbol}: {e}")
        traceback.print_exc()

    # Extract next_report_estimated from valuation if available
    next_report_estimated = valuation.get("next_report_estimated") if valuation else None

    enriched: Dict[str, Any] = {
        "quick_facts": facts,
        "valuation": valuation,
        **growth_profit,
        **vol_beta,
        "news_analysis": {
            "raw": news_analysis_result.get("analysis") if news_analysis_result and isinstance(news_analysis_result, dict) else None,
            "grouped_by_category": news_analysis_grouped,
            "token_usage": news_analysis_result.get("token_usage") if news_analysis_result and isinstance(news_analysis_result, dict) else None,
            "cost_usd": news_analysis_result.get("cost_usd") if news_analysis_result and isinstance(news_analysis_result, dict) else None,
        },
        "next_report_estimated": next_report_estimated,  # Estimated next report/filing date
        "data_provenance": {
            "parameters": provenance.get_provenance(),
            "summary": provenance.get_summary(),
        },
    }
    return enriched


def collect_latest_news(
    symbol: str, 
    days_back: int = 15, 
    max_items: int = 20,
    analyze: bool = True,
    api_keys: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Collect latest news for a symbol and optionally analyze it.
    
    Args:
        symbol: Stock ticker symbol
        days_back: Number of days to look back for news
        max_items: Maximum number of news items to return
        analyze: Whether to analyze the news using LLM
        api_keys: Optional dictionary of API keys for LLM calls
        
    Returns:
        Dictionary with 'items' (list of news items) and 'analysis' (optional NewsAnalysis)
    """
    end_date = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.utcnow() - datetime.timedelta(days=days_back)).strftime("%Y-%m-%d")
    try:
        items = get_company_news(symbol, end_date=end_date, start_date=start_date, limit=max_items)
    except Exception:
        return {"items": [], "analysis": None}

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
    
    # Analyze news if requested and we have items
    analysis = None
    if analyze and results:
        try:
            print(f"Analyzing news for {symbol}: {len(results)} articles")
            analysis_result = _analyze_news_with_llm(
                symbol,
                results,
                api_keys=api_keys
            )
            # Extract analysis text from result (backward compatibility)
            if isinstance(analysis_result, dict):
                analysis = analysis_result.get("analysis")
            else:
                analysis = analysis_result
            print(f"News analysis completed for {symbol}")
        except Exception as e:
            import traceback
            print(f"Error analyzing news for {symbol}: {e}")
            traceback.print_exc()
            analysis = None
    
    if not results:
        print(f"No news items to analyze for {symbol}")
    
    return {
        "items": results,
        "analysis": analysis
    }


def _analyze_news_with_llm(
    symbol: str,
    news_items: List[Dict[str, Any]],
    api_keys: Optional[Dict[str, str]] = None,
    model_name: str = "gpt-5-mini",
    model_provider: str = "OPENAI",
    reasoning_depth: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Analyze stock using LLM with web search capabilities.
    This replaces the structured news analyzer with a direct LLM query that searches the web.
    
    Args:
        symbol: Stock ticker symbol
        news_items: List of news items (not used, kept for compatibility)
        api_keys: Optional dictionary of API keys
        model_name: LLM model name to use
        model_provider: LLM provider name
        reasoning_depth: Optional reasoning depth for OpenAI models ("low", "medium", or "high")
        
    Returns:
        Dictionary with 'analysis' (text), 'token_usage' (dict), and 'cost' (float), 
        or None if analysis fails
    """
    # Get model and API keys - use environment variables as fallback
    if api_keys is None:
        api_keys = {}
    
    # Add environment variables as fallback for common API keys
    if "OPENAI_API_KEY" not in api_keys:
        api_keys["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
    
    try:
        # Convert model_provider string to enum
        if isinstance(model_provider, str):
            try:
                model_provider_enum = ModelProvider[model_provider]
            except KeyError:
                model_provider_enum = ModelProvider(model_provider)
        else:
            model_provider_enum = model_provider
        
        # Use DuckDuckGo for web search (free, no API key required)
        use_web_search = True  # Always use web search for better results
        
        # Get model with reasoning_depth if needed
        if reasoning_depth and model_provider_enum == ModelProvider.OPENAI:
            from langchain_openai import ChatOpenAI
            api_key = (api_keys or {}).get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_API_BASE")
            
            # Configure reasoning parameter based on model
            if model_name.startswith("gpt-5") and reasoning_depth.lower() in ["low", "medium", "high", "minimal"]:
                # GPT-5 uses reasoning_effort - pass it explicitly, not in model_kwargs
                llm = ChatOpenAI(
                    model=model_name,
                    api_key=api_key,
                    base_url=base_url,
                    reasoning_effort=reasoning_depth.lower()  # Pass explicitly
                )
                print(f"Configured reasoning_effort={reasoning_depth.lower()} for GPT-5")
            else:
                # Use standard model creation
                llm = get_model(model_name, model_provider_enum, api_keys)
        else:
            llm = get_model(model_name, model_provider_enum, api_keys)
        
        # Perform web search using Tavily
        web_search_results = []
        if use_web_search:
            try:
                # Use Tavily Search API - optimized for AI/LLM use cases
                from tavily import TavilyClient
                
                # Get Tavily API key from environment or api_keys
                tavily_api_key = api_keys.get("TAVILY_API_KEY") if api_keys else None
                if not tavily_api_key:
                    tavily_api_key = os.environ.get("TAVILY_API_KEY")
                
                if not tavily_api_key:
                    print("Warning: TAVILY_API_KEY not found. Skipping web search.")
                    web_search_results = []
                else:
                    # Search queries for comprehensive stock information
                    # Note: Only searching for recent news (last month)
                    search_queries = [
                        f"{symbol} stock news",
                        f"{symbol} financial moves earnings acquisitions",
                        f"{symbol} upcoming events earnings product launch conference",
                        f"{symbol} regulatory decisions FDA approval",
                        f"{symbol} analyst upgrades downgrades",
                        f"{symbol} partnerships deals",
                        f"{symbol} stock outlook forecast"
                    ]
                    
                    # Calculate date threshold (1 month ago)
                    from datetime import datetime, timedelta
                    from urllib.parse import urlparse
                    one_month_ago = datetime.now() - timedelta(days=30)
                    
                    print(f"Performing web search for {symbol} using Tavily...")
                    tavily_client = TavilyClient(api_key=tavily_api_key)
                    
                    for query in search_queries:
                        try:
                            print(f"Searching: {query}")
                            # Tavily search with max_results and search_depth
                            response = tavily_client.search(
                                query=query,
                                max_results=5,
                                search_depth="advanced",  # "basic" or "advanced"
                                include_domains=None,  # Can specify domains to include
                                exclude_domains=[]  # Exclude Yahoo
                            )
                            
                            results = response.get("results", [])
                            print(f"Got {len(results)} results for '{query}'")
                            
                            for result in results:
                                url = result.get("url", "")
                                title = result.get("title", "")
                                content = result.get("content", "")  # Tavily provides content directly
                                published_date = result.get("published_date")
                                
                                # Check if article is within last month
                                should_include = True
                                if published_date:
                                    try:
                                        from dateutil import parser as date_parser
                                        # Map common timezone abbreviations to avoid warnings
                                        tzinfos = {
                                            'EST': -5*3600,  # Eastern Standard Time UTC-5
                                            'EDT': -4*3600,  # Eastern Daylight Time UTC-4
                                            'PST': -8*3600,  # Pacific Standard Time UTC-8
                                            'PDT': -7*3600,  # Pacific Daylight Time UTC-7
                                            'CST': -6*3600,  # Central Standard Time UTC-6
                                            'CDT': -5*3600,  # Central Daylight Time UTC-5
                                            'MST': -7*3600,  # Mountain Standard Time UTC-7
                                            'MDT': -6*3600,  # Mountain Daylight Time UTC-6
                                        }
                                        
                                        article_date = date_parser.parse(published_date, tzinfos=tzinfos)
                                        
                                        # Make sure one_month_ago is timezone-aware for comparison
                                        if article_date.tzinfo is not None and one_month_ago.tzinfo is None:
                                            from datetime import timezone
                                            one_month_ago_aware = one_month_ago.replace(tzinfo=timezone.utc)
                                        elif article_date.tzinfo is None and one_month_ago.tzinfo is not None:
                                            from datetime import timezone
                                            article_date = article_date.replace(tzinfo=timezone.utc)
                                            one_month_ago_aware = one_month_ago
                                        else:
                                            one_month_ago_aware = one_month_ago
                                        
                                        # Check if article is within last month
                                        if article_date < one_month_ago_aware:
                                            should_include = False
                                            print(f"Skipping article from {article_date.strftime('%Y-%m-%d')} (older than 1 month)")
                                    except Exception as date_error:
                                        # If date parsing fails, include the article to be safe
                                        pass
                                
                                # Skip this result if article is too old
                                if not should_include:
                                    continue
                                
                                # Extract domain for source prioritization
                                domain = urlparse(url).netloc.lower() if url else ""
                                
                                # Prioritize non-Yahoo sources
                                priority_sources = ['reuters.com', 'bloomberg.com', 'cnbc.com', 'wsj.com', 
                                                  'ft.com', 'seekingalpha.com', 'marketwatch.com', 'techcrunch.com',
                                                  'sec.gov', 'prnewswire.com', 'businesswire.com']
                                is_priority_source = any(ps in domain for ps in priority_sources)
                                is_yahoo = 'yahoo.com' in domain or 'finance.yahoo.com' in domain
                                
                                # Use Tavily's content directly (already extracted), limit to 5000 chars
                                full_text = content[:5000] + "..." if len(content) > 5000 else content
                                
                                web_search_results.append({
                                    "title": title,
                                    "snippet": content[:500] if content else "",  # Use first 500 chars as snippet
                                    "url": url,
                                    "full_text": full_text,  # Tavily provides content directly
                                    "domain": domain,
                                    "is_priority": is_priority_source,
                                    "is_yahoo": is_yahoo
                                })
                        except Exception as e:
                            print(f"Error searching for '{query}': {e}")
                            import traceback
                            traceback.print_exc()
                            continue
                    
                    # Sort results: priority sources first, then non-Yahoo, then Yahoo
                    web_search_results.sort(key=lambda x: (
                        not x.get('is_priority', False),  # Priority sources first
                        x.get('is_yahoo', False),  # Yahoo last
                    ))
                    
                    print(f"Found {len(web_search_results)} web search results")
                    yahoo_count = sum(1 for r in web_search_results if r.get('is_yahoo', False))
                    priority_count = sum(1 for r in web_search_results if r.get('is_priority', False))
                    print(f"  - Yahoo Finance: {yahoo_count}, Priority sources: {priority_count}")
            except ImportError:
                print("Warning: tavily-python package not installed. Install with: pip install tavily-python")
                print("Get your API key from: https://tavily.com")
                web_search_results = []
            except Exception as e:
                print(f"Error performing web search with Tavily: {e}")
                import traceback
                traceback.print_exc()
                web_search_results = []
        
        llm_with_tools = llm  # No need for tool binding with Tavily
        
        # Create prompt template with web search results
        web_search_context = ""
        if web_search_results:
            web_search_context = "\n\nWEB SEARCH RESULTS:\n"
            for i, result in enumerate(web_search_results[:35], 1):  # Limit to 35 results
                title = result.get('title', 'No title')
                snippet = result.get('snippet', 'No snippet')
                url = result.get('url', 'No URL')
                full_text = result.get('full_text')
                
                web_search_context += f"\n[{i}] {title}\n"
                web_search_context += f"   URL: {url}\n"
                
                # Use full article text if available, otherwise use snippet
                if full_text:
                    web_search_context += f"   Content: {full_text}\n"
                else:
                    web_search_context += f"   Snippet: {snippet}\n"
        
        template = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a financial analyst providing comprehensive stock analysis. 
                Your task is to analyze web search results and extract factual, valuable information with sufficient context.
                
                CRITICAL RULES:
                1. Only include factual information found in the search results provided
                2. Never include meta-commentary like "check the feed", "monitor for updates", "no information found", or asking for more sources
                3. Provide context and details - include numbers, dates, and specific facts
                4. MANDATORY SOURCE DIVERSIFICATION: 
                   - Prioritize Yahoo Finance, Reuters, Bloomberg, CNBC, WSJ, Financial Times, Seeking Alpha, MarketWatch, TechCrunch, company press releases, regulatory filings
                   - Never cite the same URL twice in the same category
                5. NO DUPLICATION ACROSS CATEGORIES: 
                   - Each piece of information should appear in ONLY ONE category, even if it could fit multiple categories
                   - If information appears in "Relevant news", do NOT repeat it in "Financial moves" or other categories
                   - Choose the most appropriate category for each piece of information and use it there only
                   - Example: A buyback authorization should appear in "Financial moves" OR "Relevant news", but NOT both
                6. Write clear, informative sentences with relevant context
                7. If no valuable information exists for a category, omit that category entirely - do NOT mention lack of sources
                8. Each bullet should be self-contained with enough detail to be meaningful
                9. Include specific details like amounts, percentages, dates, company names, etc.
                10. Work with the search results provided - do NOT ask for more sources or mention limitations
                
                Format: • [Detailed factual statement with context] - Source: [URL]"""
            ),
            (
                "human",
                """Analyze the stock {symbol} using the web search results below. Extract factual, valuable information with sufficient context and detail.

Categories (max 5 bullets each, only include if valuable information exists):
1. Relevant news and their implications
2. Financial moves (acquisitions, earnings, dividends, buybacks, guidance, partnerships)
3. Upcoming events (earnings dates, product launches, regulatory decisions, conferences, analyst days, investor events)
4. Short-term outlook (next 1-3 months)
5. Long-term outlook (6-12 months)

REQUIREMENTS:
- Write informative sentences with context - include specific details, numbers, dates, and facts
- Each bullet must be factual and self-contained with enough detail
- MANDATORY: Use DIFFERENT URLs for each bullet - even if from same domain, use different pages/articles
- MANDATORY: Prioritize Yahoo Finance, Reuters, Bloomberg, CNBC, WSJ, Financial Times, Seeking Alpha, MarketWatch, TechCrunch, company websites, SEC filings
- CRITICAL: NO DUPLICATION - Each piece of information must appear in ONLY ONE category. If a buyback appears in "Financial moves", do NOT also include it in "Relevant news". Choose the most appropriate category for each fact.
- For upcoming events: include earnings dates, product launches, regulatory decisions, conferences, analyst days, investor meetings, and other relevant events
- Omit categories with no valuable information - do NOT mention lack of sources or ask for more
- Never mention "check feeds", "monitor for updates", "no information found", or ask for additional sources
- Format: • [Detailed statement with context] - Source: [URL]
- Maximum 5 bullets per category
- Work with the provided search results - extract and present the information available

Web search results:{web_search_context}"""
            )
        ])
        
        prompt = template.invoke({
            "symbol": symbol,
            "web_search_context": web_search_context
        })
        
        # Call LLM with web search results as context
        print(f"Invoking LLM for {symbol} with model {model_name}...")
        print(f"Using {len(web_search_results)} web search results as context")
        
        # Call LLM - web search results are already included in the prompt
        result = llm_with_tools.invoke(prompt)
        
        # Extract text content from response
        if hasattr(result, 'content'):
            response_text = result.content
            print(f"Got content: {len(response_text) if response_text else 0} chars")
        elif isinstance(result, str):
            response_text = result
            print(f"Got string response: {len(response_text)} chars")
        else:
            response_text = str(result)
            print(f"Converted to string: {len(response_text)} chars")
        
        if not response_text or (isinstance(response_text, str) and len(response_text.strip()) == 0):
            print(f"Warning: Empty response from LLM for {symbol}")
            return None
        
        # Extract token usage and calculate cost
        token_usage = {}
        cost_usd = 0.0
        
        # Try to get token usage from response metadata
        if hasattr(result, 'response_metadata'):
            metadata = result.response_metadata
            if 'token_usage' in metadata:
                token_usage = metadata['token_usage']
            elif 'usage' in metadata:
                token_usage = metadata['usage']
        
        # Also check usage_metadata attribute
        if not token_usage and hasattr(result, 'usage_metadata'):
            usage_meta = result.usage_metadata
            token_usage = {
                'input_tokens': getattr(usage_meta, 'input_tokens', 0),
                'output_tokens': getattr(usage_meta, 'output_tokens', 0),
                'total_tokens': getattr(usage_meta, 'total_tokens', 0)
            }
        
        # Calculate cost based on model pricing
        if token_usage and model_provider_enum == ModelProvider.OPENAI:
            input_tokens = token_usage.get('prompt_tokens') or token_usage.get('input_tokens', 0)
            output_tokens = token_usage.get('completion_tokens') or token_usage.get('output_tokens', 0)
            
            # OpenAI pricing per million tokens (as of 2025)
            # GPT-5: $1.25 input, $10.00 output
            # GPT-5-mini: $0.25 input, $2.00 output
            # GPT-5-nano: lower pricing
            # GPT-4o: $2.50 input, $10.00 output
            # GPT-4o-mini: $0.15 input, $0.60 output
            
            pricing = {
                "gpt-5": {"input": 1.25, "output": 10.00},
                "gpt-5-mini": {"input": 0.25, "output": 2.00},
                "gpt-5-nano": {"input": 0.10, "output": 0.80},
                "gpt-4o": {"input": 2.50, "output": 10.00},
                "gpt-4o-mini": {"input": 0.15, "output": 0.60},
                "gpt-4": {"input": 30.00, "output": 60.00},
                "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            }
            
            # Find matching pricing (check if model_name starts with any key)
            model_pricing = None
            for model_key, prices in pricing.items():
                if model_name.startswith(model_key):
                    model_pricing = prices
                    break
            
            # Default to GPT-5-mini pricing if not found
            if not model_pricing:
                model_pricing = pricing.get("gpt-5-mini", {"input": 0.25, "output": 2.00})
                print(f"Warning: Pricing not found for {model_name}, using gpt-5-mini pricing")
            
            input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
            output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
            cost_usd = input_cost + output_cost
        
        return {
            "analysis": response_text,
            "token_usage": token_usage,
            "cost_usd": round(cost_usd, 6)  # Round to 6 decimal places
        }
            
    except Exception as e:
        import traceback
        error_msg = f"Error in LLM news analysis for {symbol}: {e}"
        error_trace = traceback.format_exc()
        print(error_msg)
        print(error_trace)
        # Return error details for debugging
        return f"Error: {error_msg}\n\nTraceback:\n{error_trace}"


