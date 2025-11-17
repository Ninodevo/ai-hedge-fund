import os
from typing import Optional
from datetime import datetime, timedelta
import argparse
import json
import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from server.personas import PERSONA_PROMPTS, DEFAULT_PERSONA
# NOTE: Avoid importing heavy modules at startup; import inside functions when needed
from server.collectors import collect_enriched_stock

API_KEY = os.getenv("ENGINE_API_KEY", "changeme")
app = FastAPI(title="Stocklytic Engine")

def _get_latest_price(ticker: str) -> Optional[dict]:
    """Fetch the latest price snapshot for a ticker from Financial Datasets API."""
    try:
        headers = {}
        financial_api_key = os.environ.get("FINANCIAL_DATASETS_API_KEY")
        if financial_api_key:
            headers["X-API-KEY"] = financial_api_key
        
        url = f"https://api.financialdatasets.ai/prices/snapshot/?ticker={ticker}"
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            # Log error but don't fail the entire report
            print(f"Warning: Failed to fetch price snapshot for {ticker}: {response.status_code}")
            return None
    except Exception as e:
        # Log error but don't fail the entire report
        print(f"Warning: Error fetching price snapshot for {ticker}: {str(e)}")
        return None

def _require(auth: Optional[str]):
    if auth != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

class AnalystAnalysisPayload(BaseModel):
    symbol: str
    persona: Optional[str] = None
    market: Optional[dict] = None
    include_backtest: bool = False
    backtest_start: Optional[str] = None
    backtest_end: Optional[str] = None
    months_back: Optional[int] = 12
    period_type: Optional[str] = "ttm"  # "ttm" or "quarterly"

@app.get("/health")
def health(): return {"ok": True}

class NewsAnalysisPayload(BaseModel):
    symbol: str
    model_name: Optional[str] = "gpt-5-mini"
    model_provider: Optional[str] = "OPENAI"
    reasoning_depth: Optional[str] = None  # "low", "medium", or "high"

@app.post("/v1/news-analysis")
def analyze_news_endpoint(p: NewsAnalysisPayload, authorization: Optional[str] = Header(None, alias="Authorization")):
    """
    Get news analysis for a stock using LLM with web search.
    This endpoint provides comprehensive news analysis grouped by category.
    Separate from analyst analysis to allow independent caching and cost control.
    """
    _require(authorization)
    symbol = p.symbol.upper()
    
    try:
        from server.collectors import _analyze_news_with_llm, _parse_analysis_by_category
        
        analysis_result = _analyze_news_with_llm(
            symbol=symbol,
            news_items=[],
            api_keys=None,
            model_name=p.model_name or "gpt-5-mini",
            model_provider=p.model_provider or "OPENAI",
            reasoning_depth=p.reasoning_depth
        )
        
        # Check if analysis failed
        if analysis_result is None:
            return {
                "symbol": symbol,
                "model_name": p.model_name,
                "model_provider": p.model_provider,
                "reasoning_depth": p.reasoning_depth,
                "analysis": None,
                "grouped_by_category": {},
                "token_usage": None,
                "cost_usd": None,
                "status": "error",
                "error": "Analysis returned None - check server logs for details"
            }
        
        # Extract analysis, token usage, and cost from result
        analysis_text = analysis_result.get("analysis") if isinstance(analysis_result, dict) else analysis_result
        token_usage = analysis_result.get("token_usage") if isinstance(analysis_result, dict) else None
        cost_usd = analysis_result.get("cost_usd") if isinstance(analysis_result, dict) else None
        
        # Parse and group analysis by category
        grouped_by_category = {}
        if analysis_text:
            grouped_by_category = _parse_analysis_by_category(analysis_text)
        
        return {
            "symbol": symbol,
            "model_name": p.model_name,
            "model_provider": p.model_provider,
            "reasoning_depth": p.reasoning_depth,
            "analysis": analysis_text,
            "grouped_by_category": grouped_by_category,
            "token_usage": token_usage,
            "cost_usd": cost_usd,
            "status": "success"
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in news analysis endpoint: {e}")
        print(error_details)
        return {
            "symbol": symbol if 'symbol' in locals() else "UNKNOWN",
            "model_name": p.model_name if 'p' in locals() else None,
            "model_provider": p.model_provider if 'p' in locals() else None,
            "reasoning_depth": p.reasoning_depth if 'p' in locals() else None,
            "analysis": None,
            "status": "error",
            "error": str(e),
            "traceback": error_details
        }

class NewsFetchPayload(BaseModel):
    symbol: str
    days_back: Optional[int] = 30
    max_items: Optional[int] = 20

@app.post("/v1/news")
def fetch_news_endpoint(p: NewsFetchPayload, authorization: Optional[str] = Header(None, alias="Authorization")):
    """
    Test endpoint to fetch news for a stock without analysis.
    Returns raw news items from the Financial Datasets API.
    """
    _require(authorization)
    symbol = p.symbol.upper()
    
    # Import here to avoid circular imports
    from server.collectors import collect_latest_news
    
    try:
        # Fetch news without analysis
        news_data = collect_latest_news(
            symbol=symbol,
            days_back=p.days_back or 30,
            max_items=p.max_items or 20,
            analyze=False,  # Don't analyze, just fetch news
            api_keys=None  # Will use environment variables
        )
        
        return {
            "symbol": symbol,
            "days_back": p.days_back or 30,
            "max_items": p.max_items or 20,
            "items_count": len(news_data.get("items", [])),
            "items": news_data.get("items", []),
            "status": "success"
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error fetching news for {symbol}: {e}")
        print(error_details)
        return {
            "symbol": symbol,
            "items_count": 0,
            "items": [],
            "status": "error",
            "error": str(e),
            "traceback": error_details
        }

@app.post("/v1/analyst-analysis")
def get_analyst_analysis(p: AnalystAnalysisPayload, authorization: Optional[str] = Header(None, alias="Authorization")):
    """
    Get analyst analysis for a stock using a specific investment persona.
    This endpoint provides fundamental analysis, valuation, and investment signals.
    For news analysis, use the /v1/news-analysis endpoint separately.
    """
    _require(authorization)
    symbol  = p.symbol.upper()
    persona = (p.persona or DEFAULT_PERSONA).lower()
    period_type = (p.period_type or "ttm").lower()
    if period_type not in ["ttm", "quarterly"]:
        raise HTTPException(status_code=400, detail=f"period_type must be 'ttm' or 'quarterly', got '{period_type}'")
    return _build_analyst_analysis(symbol, persona, p.months_back, period_type)


def _build_analyst_analysis(symbol: str, persona: str, months_back: Optional[int] = 12, period_type: str = "ttm") -> dict:
    # Local imports to avoid startup errors if optional deps aren't installed yet
    from src.main import run_hedge_fund
    from src.utils.analysts import ANALYST_CONFIG
    # Map persona to an analyst key; allow passing an analyst key directly via persona
    analyst_key = PERSONA_PROMPTS.get(persona)
    if analyst_key is None and persona in ANALYST_CONFIG:
        analyst_key = persona
    if analyst_key is None:
        analyst_key = PERSONA_PROMPTS[DEFAULT_PERSONA]

    # Build a minimal portfolio for the single symbol
    tickers = [symbol]
    portfolio = {
        "cash": 100000.0,
        "margin_requirement": 0.0,
        "margin_used": 0.0,
        "positions": {
            symbol: {
                "long": 0,
                "short": 0,
                "long_cost_basis": 0.0,
                "short_cost_basis": 0.0,
                "short_margin_used": 0.0,
            }
        },
        "realized_gains": {symbol: {"long": 0.0, "short": 0.0}},
    }

    # Dates: default to last 90 days
    end_date = datetime.utcnow().strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")

    result = run_hedge_fund(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        show_reasoning=True,
        selected_analysts=[analyst_key],
        model_name="gpt-5-mini",
        model_provider="OpenAI",
    )

    summary = result.get("decisions")
    analyst_signals = result.get("analyst_signals")
    analysis_details = result.get("analysis_details")
    cost_summary = result.get("cost_summary", {})

    # Fetch latest price snapshot
    price_snapshot = _get_latest_price(symbol)

    # Enriched fundamentals/valuation/volatility block (without news analysis)
    # enriched = collect_enriched_stock(symbol, months_back or 12, period_type=period_type, include_news_analysis=False)
    
    # Add price data to provenance if available
    # if price_snapshot and "data_provenance" in enriched:
    #     from server.data_provenance import DataProvenance
    #     price_provenance = DataProvenance()
    #     price_provenance.add_price_data(price_snapshot, end_date)
    #     # Merge price provenance into enriched provenance
    #     enriched["data_provenance"]["parameters"].update(price_provenance.get_provenance())
    #     # Recalculate summary with all parameters
    #     temp_provenance = DataProvenance()
    #     temp_provenance.provenance = enriched["data_provenance"]["parameters"]
    #     enriched["data_provenance"]["summary"] = temp_provenance.get_summary()
    # TODO: Implement this, commented out to avoid billing
    enriched = {}

    return {
        "symbol": symbol,
        "persona": persona,
        "period_type": period_type,  # Include which period type was used
        "price": price_snapshot,
        "summary": summary,
        "analyst_signals": analyst_signals,
        "analysis_details": analysis_details,
        "enriched": enriched,
        "cost_summary": cost_summary,
        "backtest": None,
        "disclaimer": (
            "AI-generated content for educational and informational purposes only. "
            "Not investment advice; no trade execution; no guarantees."
        ),
    }