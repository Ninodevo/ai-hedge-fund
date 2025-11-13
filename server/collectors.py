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
    news_data = collect_latest_news(symbol, days_back=30, max_items=20, analyze=True)

    # Extract next_report_estimated from valuation if available
    next_report_estimated = valuation.get("next_report_estimated") if valuation else None

    enriched: Dict[str, Any] = {
        "quick_facts": facts,
        "valuation": valuation,
        **growth_profit,
        **vol_beta,
        "news": news_data.get("items", []),  # Keep backward compatibility with list of items
        # "news_analysis": news_data.get("analysis"),  # Add analysis separately - will be None if analysis failed
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
        
        # Use OpenAI's built-in web search tool if using OpenAI models
        # OpenAI models have native web_search capability through the Responses API
        # According to OpenAI docs: https://platform.openai.com/docs/guides/tools-web-search
        use_web_search = (model_provider_enum == ModelProvider.OPENAI and 
                         model_name in ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo", "gpt-5", "gpt-5-mini", "gpt-5-nano"])
        
        # Get model - configure web_search and reasoning_depth appropriately
        from langchain_openai import ChatOpenAI
        api_key = (api_keys or {}).get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_API_BASE")
        
        # Build model kwargs
        model_init_kwargs = {
            "model": model_name,
            "api_key": api_key,
            "base_url": base_url
        }
        
        # Configure reasoning parameters (only if web_search is NOT enabled)
        if reasoning_depth and model_provider_enum == ModelProvider.OPENAI and not use_web_search:
            # Configure reasoning parameter based on model
            if model_name.startswith("gpt-5") and reasoning_depth.lower() in ["low", "medium", "high", "minimal"]:
                # GPT-5 uses reasoning_effort
                model_init_kwargs["model_kwargs"] = {"reasoning_effort": reasoning_depth.lower()}
                print(f"Configured reasoning_effort={reasoning_depth.lower()} for GPT-5 (web_search disabled)")
            elif reasoning_depth.lower() in ["low", "medium", "high"]:
                # Other models might use reasoning_depth (if supported)
                print(f"Note: reasoning_depth not yet supported for {model_name} without web_search")
        elif reasoning_depth and use_web_search:
            print(f"Note: reasoning_depth parameter skipped when web_search is enabled (API limitation)")
        
        # Enable web_search via tools parameter if needed
        # According to LangChain docs, web_search_preview can be enabled via tools parameter
        if use_web_search and model_provider_enum == ModelProvider.OPENAI:
            try:
                # Try enabling web_search_preview via tools parameter
                model_init_kwargs["tools"] = [{"type": "web_search_preview"}]
                print(f"Web search enabled for {model_name} via tools parameter")
            except Exception as e:
                print(f"Could not enable web_search via tools parameter: {e}")
                # Remove tools if it causes error
                model_init_kwargs.pop("tools", None)
        
        # Create the model
        try:
            llm = ChatOpenAI(**model_init_kwargs)
        except Exception as e:
            print(f"Error creating ChatOpenAI with custom params: {e}")
            # Fallback to standard model creation
            llm = get_model(model_name, model_provider_enum, api_keys)
        
        # If web_search was enabled at model creation, use it directly
        # Otherwise, try to bind tools
        if use_web_search:
            # Check if web_search was already enabled via tools parameter
            if hasattr(llm, 'tools') and llm.tools:
                llm_with_tools = llm
                print(f"Web search already enabled via model initialization")
            else:
                # Try to bind web_search tool
                try:
                    from langchain_openai.tools import WebSearchTool
                    web_search_tool = WebSearchTool()
                    llm_with_tools = llm.bind_tools([web_search_tool])
                    print(f"Web search enabled for {model_name} using WebSearchTool via bind_tools")
                except ImportError:
                    # WebSearchTool not available, use model as-is
                    llm_with_tools = llm
                    print(f"Note: WebSearchTool not available. Web search may not be enabled.")
                except Exception as e:
                    print(f"Warning: Could not bind web_search tool: {e}")
                    llm_with_tools = llm
        else:
            llm_with_tools = llm
            if model_provider_enum != ModelProvider.OPENAI:
                print(f"Note: Web search is currently only available for OpenAI models (gpt-4o, gpt-4o-mini, gpt-5). Using model knowledge only.")
            else:
                print(f"Note: Model {model_name} may not support web search. Using model knowledge only.")
        
        # Create prompt template
        web_search_note = "Use your knowledge and reasoning to provide analysis." if not use_web_search else "Use web search capabilities to find current information."
        
        template = ChatPromptTemplate.from_messages([
            (
                "system",
                f"""You are a financial analyst providing comprehensive stock analysis. 
                Your task is to analyze all relevant information about a stock and provide 
                a detailed analysis formatted as bullet points. {web_search_note}
                
                Format your response as:
                • [Statement/insight] - Source: [URL or reference]
                • [Statement/insight] - Source: [URL or reference]
                
                IMPORTANT: Provide maximum 5 most important bullets per category. 
                Prioritize quality and impact over quantity. Focus on the most actionable insights.
                
                Provide factual, current information based on your analysis."""
            ),
            (
                "human",
                """Analyze the stock {symbol} and provide comprehensive information covering:
                1. Relevant news and their implications (max 5 bullets)
                2. Financial moves (acquisitions, earnings, dividends, buybacks, guidance, partnerships, etc.) (max 5 bullets)
                3. Upcoming events (earnings dates, product launches, regulatory decisions, etc.) (max 5 bullets)
                4. Daily technical analysis (max 5 bullets)
                5. Short-term outlook (next 1-3 months) (max 5 bullets)
                6. Long-term outlook (6-12 months) (max 5 bullets)
                7. Any additional information that could impact the stock (max 5 bullets)

                IMPORTANT: 
                - Format your response as bullet points organized by category
                - Maximum 5 most important bullets per category (35 bullets total maximum)
                - Prioritize the most impactful and actionable information
                - Each bullet point should include a source or reference when possible
                - Be specific, factual, and focus on actionable insights for investors
                - Use the format: • [text] - Source: [URL or reference]
                - If a category has fewer than 5 relevant items, that's fine - quality over quantity"""
            )
        ])
        
        prompt = template.invoke({
            "symbol": symbol
        })
        
        # Call LLM - OpenAI models will automatically use web_search tool when needed
        # The model will decide when to search the web based on the prompt
        print(f"Invoking LLM for {symbol} with model {model_name}...")
        
        # Call LLM - reasoning parameters are already configured in model creation
        result = llm_with_tools.invoke(prompt)
        print(f"LLM response type: {type(result)}")
        print(f"LLM response attributes: {dir(result)}")
        
        # Handle tool calls - if the model wants to use web_search, we need to execute it
        if hasattr(result, 'tool_calls') and result.tool_calls:
            # The model wants to use tools - OpenAI's Responses API should handle this automatically
            # But if we get tool_calls, we might need to handle them
            print(f"Model requested tool calls: {result.tool_calls}")
            # Check if content is available despite tool calls
            if hasattr(result, 'content') and result.content:
                response_text = result.content
                print(f"Got content despite tool calls: {len(response_text)} chars")
            else:
                # OpenAI's Responses API should handle tool execution automatically
                # But if not, let's try a direct call
                print("No content in tool call response, trying direct call without tool binding...")
                try:
                    result_direct = llm.invoke(prompt)
                    if hasattr(result_direct, 'content'):
                        response_text = result_direct.content
                    else:
                        response_text = str(result_direct)
                except Exception as direct_error:
                    print(f"Direct call also failed: {direct_error}")
                    response_text = f"Error: Tool calls detected but unable to get response. Tool calls: {result.tool_calls}"
        else:
            # Extract text content
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


