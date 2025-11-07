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

class ReportPayload(BaseModel):
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

@app.post("/v1/report")
def generate_report(p: ReportPayload, authorization: Optional[str] = Header(None, alias="Authorization")):
    _require(authorization)
    symbol  = p.symbol.upper()
    persona = (p.persona or DEFAULT_PERSONA).lower()
    period_type = (p.period_type or "ttm").lower()
    if period_type not in ["ttm", "quarterly"]:
        raise HTTPException(status_code=400, detail=f"period_type must be 'ttm' or 'quarterly', got '{period_type}'")
    return _build_report(symbol, persona, p.months_back, period_type)


def _build_report(symbol: str, persona: str, months_back: Optional[int] = 12, period_type: str = "ttm") -> dict:
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
        model_name="gpt-4.1",
        model_provider="OpenAI",
    )

    summary = result.get("decisions")
    analyst_signals = result.get("analyst_signals")
    analysis_details = result.get("analysis_details")

    # Fetch latest price snapshot
    price_snapshot = _get_latest_price(symbol)

    # Enriched fundamentals/valuation/volatility block
    enriched = collect_enriched_stock(symbol, months_back or 12, period_type=period_type)
    # TODO: Implement this, commented out to avoid billing
    # enriched = {}

    return {
        "symbol": symbol,
        "persona": persona,
        "period_type": period_type,  # Include which period type was used
        "price": price_snapshot,
        "summary": summary,
        "analyst_signals": analyst_signals,
        "analysis_details": analysis_details,
        "enriched": enriched,
        "backtest": None,
        "disclaimer": (
            "AI-generated content for educational and informational purposes only. "
            "Not investment advice; no trade execution; no guarantees."
        ),
    }