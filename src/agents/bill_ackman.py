from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_financial_metrics, get_market_cap, search_line_items, get_company_facts
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing import Optional
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm
from src.utils.api_key import get_api_key_from_state
from src.utils.analyst_provenance import add_provenance_to_data, get_market_cap_provenance


class BillAckmanSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def _get_item_date(obj):
    # Prefer concrete report dates over generic period labels like 'ttm'
    for attr in ("date", "report_period", "period_end", "end_date", "as_of_date", "fiscal_date", "report_date"):
        val = getattr(obj, attr, None)
        if val:
            return val
    return None


def bill_ackman_agent(state: AgentState, agent_id: str = "bill_ackman_agent"):
    """
    Analyzes stocks using Bill Ackman's investing principles and LLM reasoning.
    Fetches multiple periods of data for a more robust long-term view.
    Incorporates brand/competitive advantage, activism potential, and other key factors.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    analysis_data = {}
    ackman_analysis = {}
    
    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=5, api_key=api_key)
        
        progress.update_status(agent_id, ticker, "Gathering financial line items")
        # Request multiple periods of data (annual or TTM) for a more robust long-term view.
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue",
                "operating_margin",
                "debt_to_equity",
                "free_cash_flow",
                "total_assets",
                "total_liabilities",
                "dividends_and_other_cash_distributions",
                "outstanding_shares",
                # Optional: intangible_assets if available
                # "intangible_assets"
            ],
            end_date,
            period="annual",
            limit=5,
            api_key=api_key,
        )
        
        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)
        
        # Get CIK for provenance tracking
        company_facts_obj = get_company_facts(ticker)
        cik = getattr(company_facts_obj, "cik", None) if company_facts_obj else None
        
        progress.update_status(agent_id, ticker, "Analyzing business quality")
        quality_analysis = analyze_business_quality(metrics, financial_line_items, cik=cik)
        
        progress.update_status(agent_id, ticker, "Analyzing balance sheet and capital structure")
        balance_sheet_analysis = analyze_financial_discipline(metrics, financial_line_items, cik=cik)
        
        progress.update_status(agent_id, ticker, "Analyzing activism potential")
        activism_analysis = analyze_activism_potential(financial_line_items, cik=cik)
        
        progress.update_status(agent_id, ticker, "Calculating intrinsic value & margin of safety")
        valuation_analysis = analyze_valuation(financial_line_items, market_cap, cik=cik)
        
        # Combine partial scores or signals
        total_score = (
            quality_analysis["score"]
            + balance_sheet_analysis["score"]
            + activism_analysis["score"]
            + valuation_analysis["score"]
        )
        max_possible_score = 20  # Adjust weighting as desired (5 from each sub-analysis, for instance)
        
        # Generate a simple buy/hold/sell (bullish/neutral/bearish) signal
        if total_score >= 0.7 * max_possible_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_possible_score:
            signal = "bearish"
        else:
            signal = "neutral"
        
        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_possible_score,
            "quality_analysis": quality_analysis,
            "balance_sheet_analysis": balance_sheet_analysis,
            "activism_analysis": activism_analysis,
            "valuation_analysis": valuation_analysis
        }
        
        progress.update_status(agent_id, ticker, "Generating Bill Ackman analysis")
        ackman_output = generate_ackman_output(
            ticker=ticker, 
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )
        
        ackman_analysis[ticker] = {
            "signal": ackman_output.signal,
            "confidence": ackman_output.confidence,
            "reasoning": ackman_output.reasoning
        }
        
        # Persist detailed reasoning text per ticker for debugging/UX (safe-init)
        analysis_details = state["data"].setdefault("analysis_details", {})
        agent_details = analysis_details.setdefault(agent_id, {})
        ticker_entry = agent_details.setdefault(ticker, {})
        
        # Add provenance to market cap
        market_cap_provenance = get_market_cap_provenance(market_cap, end_date) if market_cap else None
        
        # Calculate per-share values if available
        current_price_per_share = None
        intrinsic_value_per_share = None
        try:
            shares_outstanding_ps = getattr(financial_line_items[0], 'outstanding_shares', None) if financial_line_items else None
            if market_cap and shares_outstanding_ps:
                current_price_per_share = market_cap / shares_outstanding_ps
            if valuation_analysis.get("intrinsic_value") and shares_outstanding_ps:
                intrinsic_value_per_share = valuation_analysis["intrinsic_value"] / shares_outstanding_ps
        except Exception:
            pass
        
        ticker_entry["analysis_data"] = {
            "score": total_score,
            "max_score": max_possible_score,
            "market_cap": market_cap,
            "market_cap_provenance": market_cap_provenance,
            "margin_of_safety": valuation_analysis.get("margin_of_safety"),
            "intrinsic_value": valuation_analysis.get("intrinsic_value"),
            "current_price_per_share": current_price_per_share,
            "intrinsic_value_per_share": intrinsic_value_per_share,
            "period_type_used": "annual",  # Bill Ackman uses annual data
            "data_sources": {
                "financial_metrics": {
                    "source": "financial_metrics_api",
                    "period_type": "annual",
                    "limit": 5,
                    "latest_report_period": metrics[0].report_period if metrics else None,
                    "latest_fiscal_period": getattr(metrics[0], "fiscal_period", None) if metrics else None,
                },
                "financial_line_items": {
                    "source": "line_items_api",
                    "period_type": "annual",
                    "limit": 5,
                    "latest_report_period": financial_line_items[0].report_period if financial_line_items else None,
                    "latest_fiscal_period": getattr(financial_line_items[0], "fiscal_period", None) if financial_line_items else None,
                },
                "market_cap": {
                    "source": "market_cap_api",
                    "period_type": "latest",
                    "date": end_date,
                }
            }
        }
        
        # Persist structured analysis details per ticker for UI/debugging
        def _split_details_to_list(val):
            """Normalize details into a list of strings, splitting on ';' and trimming."""
            items: list[str] = []
            if val is None:
                return items
            if isinstance(val, list):
                for v in val:
                    if v is None:
                        continue
                    if isinstance(v, str):
                        parts = [p.strip() for p in v.split(";")]
                        items.extend([p for p in parts if p])
                    else:
                        items.append(str(v))
                return items
            # string or other primitive
            s = str(val)
            parts = [p.strip() for p in s.split(";")]
            return [p for p in parts if p]

        structured_detail_items = [
            {"label": "business_quality", "detail": _split_details_to_list(quality_analysis.get("details")), "data": quality_analysis.get("data"), "score": quality_analysis["score"], "max_score": quality_analysis.get("max_score", 7)},
            {"label": "financial_discipline", "detail": _split_details_to_list(balance_sheet_analysis.get("details")), "data": balance_sheet_analysis.get("data"), "score": balance_sheet_analysis["score"], "max_score": balance_sheet_analysis.get("max_score", 4)},
            {"label": "activism_potential", "detail": _split_details_to_list(activism_analysis.get("details")), "data": activism_analysis.get("data"), "score": activism_analysis["score"], "max_score": activism_analysis.get("max_score", 2)},
            {"label": "valuation", "detail": _split_details_to_list(valuation_analysis.get("details")), "data": valuation_analysis.get("data"), "score": valuation_analysis["score"], "max_score": valuation_analysis.get("max_score", 3)},
        ]
        
        ticker_entry["analysis_details"] = structured_detail_items
        
        progress.update_status(agent_id, ticker, "Done", analysis=ackman_output.reasoning)
    
    # Wrap results in a single message for the chain
    message = HumanMessage(
        content=json.dumps(ackman_analysis),
        name=agent_id
    )
    
    # Show reasoning if requested
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(ackman_analysis, "Bill Ackman Agent")
    
    # Add signals to the overall state
    state["data"]["analyst_signals"][agent_id] = ackman_analysis

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": [message],
        "data": state["data"]
    }


def analyze_business_quality(metrics: list, financial_line_items: list, cik: Optional[str] = None) -> dict:
    """
    Analyze whether the company has a high-quality business with stable or growing cash flows,
    durable competitive advantages (moats), and potential for long-term growth.
    Also tries to infer brand strength if intangible_assets data is present (optional).
    """
    score = 0
    details = []
    max_score = 7
    
    if not metrics or not financial_line_items:
        return {
            "score": 0,
            "max_score": max_score,
            "details": "Insufficient data to analyze business quality"
        }
    
    # 1. Multi-period revenue growth analysis
    revenues = [item.revenue for item in financial_line_items if item.revenue is not None]
    revenue_dates = [_get_item_date(item) for item in financial_line_items if getattr(item, 'revenue', None)]
    if len(revenues) >= 2:
        initial, final = revenues[-1], revenues[0]
        if initial and final and final > initial:
            growth_rate = (final - initial) / abs(initial)
            if growth_rate > 0.5:  # e.g., 50% cumulative growth
                score += 2
                details.append(f"Revenue grew by {(growth_rate*100):.1f}% over the full period (strong growth).")
            else:
                score += 1
                details.append(f"Revenue growth is positive but under 50% cumulatively ({(growth_rate*100):.1f}%).")
        else:
            details.append("Revenue did not grow significantly or data insufficient.")
    else:
        details.append("Not enough revenue data for multi-period trend.")
    
    # 2. Operating margin and free cash flow consistency
    fcf_vals = [item.free_cash_flow for item in financial_line_items if item.free_cash_flow is not None]
    op_margin_vals = [item.operating_margin for item in financial_line_items if item.operating_margin is not None]
    
    if op_margin_vals:
        above_15 = sum(1 for m in op_margin_vals if m > 0.15)
        if above_15 >= (len(op_margin_vals) // 2 + 1):
            score += 2
            details.append("Operating margins have often exceeded 15% (indicates good profitability).")
        else:
            details.append("Operating margin not consistently above 15%.")
    else:
        details.append("No operating margin data across periods.")
    
    if fcf_vals:
        positive_fcf_count = sum(1 for f in fcf_vals if f > 0)
        if positive_fcf_count >= (len(fcf_vals) // 2 + 1):
            score += 1
            details.append("Majority of periods show positive free cash flow.")
        else:
            details.append("Free cash flow not consistently positive.")
    else:
        details.append("No free cash flow data across periods.")
    
    # 3. Return on Equity (ROE) check from the latest metrics
    latest_metrics = metrics[0]
    if latest_metrics.return_on_equity and latest_metrics.return_on_equity > 0.15:
        score += 2
        details.append(f"High ROE of {latest_metrics.return_on_equity:.1%}, indicating a competitive advantage.")
    elif latest_metrics.return_on_equity:
        details.append(f"ROE of {latest_metrics.return_on_equity:.1%} is moderate.")
    else:
        details.append("ROE data not available.")
    
    # 4. (Optional) Brand Intangible (if intangible_assets are fetched)
    # intangible_vals = [item.intangible_assets for item in financial_line_items if item.intangible_assets]
    # if intangible_vals and sum(intangible_vals) > 0:
    #     details.append("Significant intangible assets may indicate brand value or proprietary tech.")
    #     score += 1
    
    data = {
        "revenues": revenues,
        "revenue_dates": revenue_dates,
        "fcf_vals": fcf_vals,
        "op_margin_vals": op_margin_vals,
        "return_on_equity": getattr(latest_metrics, "return_on_equity", None),
        "date": _get_item_date(latest_metrics),
    }
    
    # Add provenance information
    data_with_provenance = add_provenance_to_data(
        data,
        metrics=metrics,
        line_items=financial_line_items,
        field_mapping={
            "revenue": "revenue",
            "free_cash_flow": "free_cash_flow",
            "operating_margin": "operating_margin",
            "return_on_equity": "return_on_equity",
        },
        cik=cik
    )
    
    return {
        "score": score,
        "max_score": max_score,
        "details": "; ".join(details),
        "data": data_with_provenance,
    }


def analyze_financial_discipline(metrics: list, financial_line_items: list, cik: Optional[str] = None) -> dict:
    """
    Evaluate the company's balance sheet over multiple periods:
    - Debt ratio trends
    - Capital returns to shareholders over time (dividends, buybacks)
    """
    score = 0
    details = []
    max_score = 4
    
    if not metrics or not financial_line_items:
        return {
            "score": 0,
            "max_score": max_score,
            "details": "Insufficient data to analyze financial discipline"
        }
    
    latest = financial_line_items[0]
    
    # 1. Multi-period debt ratio or debt_to_equity
    debt_to_equity_vals = [item.debt_to_equity for item in financial_line_items if item.debt_to_equity is not None]
    if debt_to_equity_vals:
        below_one_count = sum(1 for d in debt_to_equity_vals if d < 1.0)
        if below_one_count >= (len(debt_to_equity_vals) // 2 + 1):
            score += 2
            details.append("Debt-to-equity < 1.0 for the majority of periods (reasonable leverage).")
        else:
            details.append("Debt-to-equity >= 1.0 in many periods (could be high leverage).")
    else:
        # Fallback to total_liabilities / total_assets
        liab_to_assets = []
        for item in financial_line_items:
            if item.total_liabilities and item.total_assets and item.total_assets > 0:
                liab_to_assets.append(item.total_liabilities / item.total_assets)
        
        if liab_to_assets:
            below_50pct_count = sum(1 for ratio in liab_to_assets if ratio < 0.5)
            if below_50pct_count >= (len(liab_to_assets) // 2 + 1):
                score += 2
                details.append("Liabilities-to-assets < 50% for majority of periods.")
            else:
                details.append("Liabilities-to-assets >= 50% in many periods.")
        else:
            details.append("No consistent leverage ratio data available.")
    
    # 2. Capital allocation approach (dividends + share counts)
    dividends_list = [
        item.dividends_and_other_cash_distributions
        for item in financial_line_items
        if item.dividends_and_other_cash_distributions is not None
    ]
    if dividends_list:
        paying_dividends_count = sum(1 for d in dividends_list if d < 0)
        if paying_dividends_count >= (len(dividends_list) // 2 + 1):
            score += 1
            details.append("Company has a history of returning capital to shareholders (dividends).")
        else:
            details.append("Dividends not consistently paid or no data on distributions.")
    else:
        details.append("No dividend data found across periods.")
    
    # Check for decreasing share count (simple approach)
    shares = [item.outstanding_shares for item in financial_line_items if item.outstanding_shares is not None]
    if len(shares) >= 2:
        # For buybacks, the newest count should be less than the oldest count
        if shares[0] < shares[-1]:
            score += 1
            details.append("Outstanding shares have decreased over time (possible buybacks).")
        else:
            details.append("Outstanding shares have not decreased over the available periods.")
    else:
        details.append("No multi-period share count data to assess buybacks.")
    
    data = {
        "debt_to_equity_vals": debt_to_equity_vals,
        "dividends_list": dividends_list,
        "outstanding_shares": shares,
        "dividends_and_other_cash_distributions": getattr(latest, "dividends_and_other_cash_distributions", None),
        "outstanding_shares_latest": getattr(latest, "outstanding_shares", None),
        "date": _get_item_date(latest),
    }
    
    # Add provenance information
    data_with_provenance = add_provenance_to_data(
        data,
        line_items=financial_line_items,
        field_mapping={
            "debt_to_equity": "debt_to_equity",
            "dividends_and_other_cash_distributions": "dividends_and_other_cash_distributions",
            "outstanding_shares": "outstanding_shares",
            "total_liabilities": "total_liabilities",
            "total_assets": "total_assets",
        },
        cik=cik
    )
    
    return {
        "score": score,
        "max_score": max_score,
        "details": "; ".join(details),
        "data": data_with_provenance,
    }


def analyze_activism_potential(financial_line_items: list, cik: Optional[str] = None) -> dict:
    """
    Bill Ackman often engages in activism if a company has a decent brand or moat
    but is underperforming operationally.
    
    We'll do a simplified approach:
    - Look for positive revenue trends but subpar margins
    - That may indicate 'activism upside' if operational improvements could unlock value.
    """
    max_score = 2
    
    if not financial_line_items:
        return {
            "score": 0,
            "max_score": max_score,
            "details": "Insufficient data for activism potential"
        }
    
    # Check revenue growth vs. operating margin
    revenues = [item.revenue for item in financial_line_items if item.revenue is not None]
    op_margins = [item.operating_margin for item in financial_line_items if item.operating_margin is not None]
    
    if len(revenues) < 2 or not op_margins:
        return {
            "score": 0,
            "max_score": max_score,
            "details": "Not enough data to assess activism potential (need multi-year revenue + margins)."
        }
    
    initial, final = revenues[-1], revenues[0]
    revenue_growth = (final - initial) / abs(initial) if initial else 0
    avg_margin = sum(op_margins) / len(op_margins)
    
    score = 0
    details = []
    
    # Suppose if there's decent revenue growth but margins are below 10%, Ackman might see activism potential.
    if revenue_growth > 0.15 and avg_margin < 0.10:
        score += 2
        details.append(
            f"Revenue growth is healthy (~{revenue_growth*100:.1f}%), but margins are low (avg {avg_margin*100:.1f}%). "
            "Activism could unlock margin improvements."
        )
    else:
        details.append("No clear sign of activism opportunity (either margins are already decent or growth is weak).")
    
    data = {
        "revenues": revenues,
        "revenue_growth": revenue_growth,
        "op_margins": op_margins,
        "avg_margin": avg_margin,
        "dates": [_get_item_date(item) for item in financial_line_items],
    }
    
    # Add provenance information
    data_with_provenance = add_provenance_to_data(
        data,
        line_items=financial_line_items,
        field_mapping={
            "revenue": "revenue",
            "operating_margin": "operating_margin",
        },
        cik=cik
    )
    
    return {
        "score": score,
        "max_score": max_score,
        "details": "; ".join(details),
        "data": data_with_provenance,
    }


def analyze_valuation(financial_line_items: list, market_cap: float, cik: Optional[str] = None) -> dict:
    """
    Ackman invests in companies trading at a discount to intrinsic value.
    Uses a simplified DCF with FCF as a proxy, plus margin of safety analysis.
    """
    max_score = 3
    
    if not financial_line_items or market_cap is None:
        return {
            "score": 0,
            "max_score": max_score,
            "details": "Insufficient data to perform valuation"
        }
    
    # Since financial_line_items are in descending order (newest first),
    # the most recent period is the first element
    latest = financial_line_items[0]
    fcf = latest.free_cash_flow if latest.free_cash_flow else 0
    
    if fcf <= 0:
        return {
            "score": 0,
            "max_score": max_score,
            "details": f"No positive FCF for valuation; FCF = {fcf}",
            "intrinsic_value": None
        }
    
    # Basic DCF assumptions
    growth_rate = 0.06
    discount_rate = 0.10
    terminal_multiple = 15
    projection_years = 5
    
    present_value = 0
    for year in range(1, projection_years + 1):
        future_fcf = fcf * (1 + growth_rate) ** year
        pv = future_fcf / ((1 + discount_rate) ** year)
        present_value += pv
    
    # Terminal Value
    terminal_value = (
        fcf * (1 + growth_rate) ** projection_years * terminal_multiple
    ) / ((1 + discount_rate) ** projection_years)
    
    intrinsic_value = present_value + terminal_value
    margin_of_safety = (intrinsic_value - market_cap) / market_cap
    
    score = 0
    # Simple scoring
    if margin_of_safety > 0.3:
        score += 3
    elif margin_of_safety > 0.1:
        score += 1
    
    details = [
        f"Calculated intrinsic value: ~{intrinsic_value:,.2f}",
        f"Market cap: ~{market_cap:,.2f}",
        f"Margin of safety: {margin_of_safety:.2%}"
    ]
    
    data = {
        "free_cash_flow": fcf,
        "intrinsic_value": intrinsic_value,
        "margin_of_safety": margin_of_safety,
        "assumptions": {
            "growth_rate": growth_rate,
            "discount_rate": discount_rate,
            "terminal_multiple": terminal_multiple,
            "projection_years": projection_years,
        },
        "dcf_components": {
            "present_value": present_value,
            "terminal_value": terminal_value,
        },
        "date": _get_item_date(latest),
    }
    
    # Add provenance information
    data_with_provenance = add_provenance_to_data(
        data,
        line_items=financial_line_items,
        field_mapping={
            "free_cash_flow": "free_cash_flow",
        },
        cik=cik
    )
    
    return {
        "score": score,
        "max_score": max_score,
        "details": "; ".join(details),
        "intrinsic_value": intrinsic_value,
        "margin_of_safety": margin_of_safety,
        "data": data_with_provenance,
    }


def generate_ackman_output(
    ticker: str,
    analysis_data: dict[str, any],
    state: AgentState,
    agent_id: str,
) -> BillAckmanSignal:
    """
    Generates investment decisions in the style of Bill Ackman.
    Includes more explicit references to brand strength, activism potential, 
    catalysts, and management changes in the system prompt.
    """
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Bill Ackman AI agent, making investment decisions using his principles:

            1. Seek high-quality businesses with durable competitive advantages (moats), often in well-known consumer or service brands.
            2. Prioritize consistent free cash flow and growth potential over the long term.
            3. Advocate for strong financial discipline (reasonable leverage, efficient capital allocation).
            4. Valuation matters: target intrinsic value with a margin of safety.
            5. Consider activism where management or operational improvements can unlock substantial upside.
            6. Concentrate on a few high-conviction investments.

            In your reasoning:
            - Emphasize brand strength, moat, or unique market positioning.
            - Review free cash flow generation and margin trends as key signals.
            - Analyze leverage, share buybacks, and dividends as capital discipline metrics.
            - Provide a valuation assessment with numerical backup (DCF, multiples, etc.).
            - Identify any catalysts for activism or value creation (e.g., cost cuts, better capital allocation).
            - Use a confident, analytic, and sometimes confrontational tone when discussing weaknesses or opportunities.

            Return your final recommendation (signal: bullish, neutral, or bearish) with a 0-100 confidence and a thorough reasoning section.
            """
        ),
        (
            "human",
            """Based on the following analysis, create an Ackman-style investment signal.

            Analysis Data for {ticker}:
            {analysis_data}
            
            Write your reasoning in first person, as if you are Bill Ackman analyzing this stock yourself. "
            "Refer to the intrinsic value, margin of safety, and required buffer as your own analysis - use 'my', "
            "'I', 'the intrinsic value I calculate', 'my required margin of safety', etc. Never use 'your' when "
            "referring to the analysis or calculations - this is your own work.\n"
            "\n"
            "When mentioning percentages (margins, margin of safety, growth rates, etc.), always format them as "
            "percentages (e.g., '43.3%' or 'about 43%') rather than decimals (e.g., '0.433'). This makes the "
            "analysis more readable and natural.\n"
            "\n"
            "Write in a natural, conversational style as if explaining your thinking to a partner. "
            "Weave together your observations about the business, management, financials, and valuation into a "
            "coherent narrative. Do not explicitly list categories or use phrases like 'Circle of competence:' "
            "or 'Not clearly established from the supplied facts.' Instead, write naturally about what you observe "
            "and how it informs your decision.\n"
            "\n"
            "Use simple punctuation - avoid em dashes (—) and prefer commas, periods instead. "
            "Keep the writing clean and straightforward.\n"
            "\n"
            "Do not invent data. If margin of safety is less than 0, don't say it's negative.

            Return your output in strictly valid JSON:
            {{
              "signal": "bullish" | "bearish" | "neutral",
              "confidence": float (0-100),
              "reasoning": "string"
            }}
            """
        )
    ])

    prompt = template.invoke({
        "analysis_data": json.dumps(analysis_data, indent=2),
        "ticker": ticker
    })

    def create_default_bill_ackman_signal():
        return BillAckmanSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis, defaulting to neutral"
        )

    # Get cost tracker from state if available
    cost_tracker = state.get("metadata", {}).get("cost_tracker")
    
    return call_llm(
        prompt=prompt, 
        pydantic_model=BillAckmanSignal, 
        agent_name=agent_id, 
        state=state,
        default_factory=create_default_bill_ackman_signal,
        cost_tracker=cost_tracker,
    )
