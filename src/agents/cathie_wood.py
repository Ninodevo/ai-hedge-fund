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


class CathieWoodSignal(BaseModel):
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


def cathie_wood_agent(state: AgentState, agent_id: str = "cathie_wood_agent"):
    """
    Analyzes stocks using Cathie Wood's investing principles and LLM reasoning.
    1. Prioritizes companies with breakthrough technologies or business models
    2. Focuses on industries with rapid adoption curves and massive TAM (Total Addressable Market).
    3. Invests mostly in AI, robotics, genomic sequencing, fintech, and blockchain.
    4. Willing to endure short-term volatility for long-term gains.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    analysis_data = {}
    cw_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=5, api_key=api_key)

        progress.update_status(agent_id, ticker, "Gathering financial line items")
        # Request multiple periods of data (annual or TTM) for a more robust view.
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue",
                "gross_margin",
                "operating_margin",
                "debt_to_equity",
                "free_cash_flow",
                "total_assets",
                "total_liabilities",
                "dividends_and_other_cash_distributions",
                "outstanding_shares",
                "research_and_development",
                "capital_expenditure",
                "operating_expense",
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

        progress.update_status(agent_id, ticker, "Analyzing disruptive potential")
        disruptive_analysis = analyze_disruptive_potential(metrics, financial_line_items, cik=cik)

        progress.update_status(agent_id, ticker, "Analyzing innovation-driven growth")
        innovation_analysis = analyze_innovation_growth(metrics, financial_line_items, cik=cik)

        progress.update_status(agent_id, ticker, "Calculating valuation & high-growth scenario")
        valuation_analysis = analyze_cathie_wood_valuation(financial_line_items, market_cap, cik=cik)

        # Combine partial scores or signals
        total_score = disruptive_analysis["score"] + innovation_analysis["score"] + valuation_analysis["score"]
        max_possible_score = 15  # Adjust weighting as desired

        if total_score >= 0.7 * max_possible_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_possible_score:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {"signal": signal, "score": total_score, "max_score": max_possible_score, "disruptive_analysis": disruptive_analysis, "innovation_analysis": innovation_analysis, "valuation_analysis": valuation_analysis}

        progress.update_status(agent_id, ticker, "Generating Cathie Wood analysis")
        cw_output = generate_cathie_wood_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        cw_analysis[ticker] = {"signal": cw_output.signal, "confidence": cw_output.confidence, "reasoning": cw_output.reasoning}

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
            "period_type_used": "annual",  # Cathie Wood uses annual data
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
            {"label": "disruptive_potential", "detail": _split_details_to_list(disruptive_analysis.get("details")), "data": disruptive_analysis.get("data"), "score": disruptive_analysis["score"], "max_score": disruptive_analysis.get("max_score", 5)},
            {"label": "innovation_growth", "detail": _split_details_to_list(innovation_analysis.get("details")), "data": innovation_analysis.get("data"), "score": innovation_analysis["score"], "max_score": innovation_analysis.get("max_score", 5)},
            {"label": "valuation", "detail": _split_details_to_list(valuation_analysis.get("details")), "data": valuation_analysis.get("data"), "score": valuation_analysis["score"], "max_score": valuation_analysis.get("max_score", 3)},
        ]
        
        ticker_entry["analysis_details"] = structured_detail_items
        
        progress.update_status(agent_id, ticker, "Done", analysis=cw_output.reasoning)

    message = HumanMessage(content=json.dumps(cw_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(cw_analysis, agent_id)

    state["data"]["analyst_signals"][agent_id] = cw_analysis

    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


def analyze_disruptive_potential(metrics: list, financial_line_items: list, cik: Optional[str] = None) -> dict:
    """
    Analyze whether the company has disruptive products, technology, or business model.
    Evaluates multiple dimensions of disruptive potential:
    1. Revenue Growth Acceleration - indicates market adoption
    2. R&D Intensity - shows innovation investment
    3. Gross Margin Trends - suggests pricing power and scalability
    4. Operating Leverage - demonstrates business model efficiency
    5. Market Share Dynamics - indicates competitive position
    """
    score = 0
    details = []
    max_possible_score = 12  # Sum of all possible points

    if not metrics or not financial_line_items:
        return {"score": 0, "max_score": max_possible_score, "details": "Insufficient data to analyze disruptive potential"}

    # Initialize variables
    revenues = []
    growth_rates = []
    gross_margins = []
    operating_expenses = []
    rd_expenses = []
    
    # 1. Revenue Growth Analysis - Check for accelerating growth
    if len(revenues) >= 3:  # Need at least 3 periods to check acceleration
        for i in range(len(revenues) - 1):
            if revenues[i] and revenues[i + 1]:
                growth_rate = (revenues[i] - revenues[i + 1]) / abs(revenues[i + 1]) if revenues[i + 1] != 0 else 0
                growth_rates.append(growth_rate)

        # Check if growth is accelerating (first growth rate higher than last, since they're in reverse order)
        if len(growth_rates) >= 2 and growth_rates[0] > growth_rates[-1]:
            score += 2
            details.append(f"Revenue growth is accelerating: {(growth_rates[0]*100):.1f}% vs {(growth_rates[-1]*100):.1f}%")

        # Check absolute growth rate (most recent growth rate is at index 0)
        latest_growth = growth_rates[0] if growth_rates else 0
        if latest_growth > 1.0:
            score += 3
            details.append(f"Exceptional revenue growth: {(latest_growth*100):.1f}%")
        elif latest_growth > 0.5:
            score += 2
            details.append(f"Strong revenue growth: {(latest_growth*100):.1f}%")
        elif latest_growth > 0.2:
            score += 1
            details.append(f"Moderate revenue growth: {(latest_growth*100):.1f}%")
    else:
        details.append("Insufficient revenue data for growth analysis")

    # 2. Gross Margin Analysis - Check for expanding margins
    gross_margins = [item.gross_margin for item in financial_line_items if hasattr(item, "gross_margin") and item.gross_margin is not None]
    if len(gross_margins) >= 2:
        margin_trend = gross_margins[0] - gross_margins[-1]
        if margin_trend > 0.05:  # 5% improvement
            score += 2
            details.append(f"Expanding gross margins: +{(margin_trend*100):.1f}%")
        elif margin_trend > 0:
            score += 1
            details.append(f"Slightly improving gross margins: +{(margin_trend*100):.1f}%")

        # Check absolute margin level (most recent margin is at index 0)
        if gross_margins[0] > 0.50:  # High margin business
            score += 2
            details.append(f"High gross margin: {(gross_margins[0]*100):.1f}%")
    else:
        details.append("Insufficient gross margin data")

    # 3. Operating Leverage Analysis
    operating_expenses = [item.operating_expense for item in financial_line_items if hasattr(item, "operating_expense") and item.operating_expense]

    if len(revenues) >= 2 and len(operating_expenses) >= 2:
        rev_growth = (revenues[0] - revenues[-1]) / abs(revenues[-1])
        opex_growth = (operating_expenses[0] - operating_expenses[-1]) / abs(operating_expenses[-1])

        if rev_growth > opex_growth:
            score += 2
            details.append("Positive operating leverage: Revenue growing faster than expenses")
    else:
        details.append("Insufficient data for operating leverage analysis")

    # 4. R&D Investment Analysis
    rd_expenses = [item.research_and_development for item in financial_line_items if hasattr(item, "research_and_development") and item.research_and_development is not None]
    if rd_expenses and revenues:
        rd_intensity = rd_expenses[0] / revenues[0]
        if rd_intensity > 0.15:  # High R&D intensity
            score += 3
            details.append(f"High R&D investment: {(rd_intensity*100):.1f}% of revenue")
        elif rd_intensity > 0.08:
            score += 2
            details.append(f"Moderate R&D investment: {(rd_intensity*100):.1f}% of revenue")
        elif rd_intensity > 0.05:
            score += 1
            details.append(f"Some R&D investment: {(rd_intensity*100):.1f}% of revenue")
    else:
        details.append("No R&D data available")

    # Normalize score to be out of 5
    normalized_score = (score / max_possible_score) * 5 if max_possible_score > 0 else 0

    latest = financial_line_items[0] if financial_line_items else None
    
    data = {
        "revenues": revenues,
        "growth_rates": growth_rates,
        "gross_margins": gross_margins,
        "operating_expenses": operating_expenses,
        "rd_expenses": rd_expenses,
        "rd_intensity": rd_expenses[0] / revenues[0] if rd_expenses and revenues and revenues[0] > 0 else None,
        "date": _get_item_date(latest) if latest else None,
    }
    
    # Add provenance information
    data_with_provenance = add_provenance_to_data(
        data,
        metrics=metrics,
        line_items=financial_line_items,
        field_mapping={
            "revenue": "revenue",
            "gross_margin": "gross_margin",
            "operating_expense": "operating_expense",
            "research_and_development": "research_and_development",
        },
        cik=cik
    )

    return {
        "score": normalized_score,
        "max_score": 5,  # Normalized score is out of 5
        "details": "; ".join(details),
        "raw_score": score,
        "raw_max_score": max_possible_score,
        "data": data_with_provenance,
    }


def analyze_innovation_growth(metrics: list, financial_line_items: list, cik: Optional[str] = None) -> dict:
    """
    Evaluate the company's commitment to innovation and potential for exponential growth.
    Analyzes multiple dimensions:
    1. R&D Investment Trends - measures commitment to innovation
    2. Free Cash Flow Generation - indicates ability to fund innovation
    3. Operating Efficiency - shows scalability of innovation
    4. Capital Allocation - reveals innovation-focused management
    5. Growth Reinvestment - demonstrates commitment to future growth
    """
    score = 0
    details = []
    max_possible_score = 15  # Sum of all possible points

    if not metrics or not financial_line_items:
        return {"score": 0, "max_score": 5, "details": "Insufficient data to analyze innovation-driven growth"}

    # Initialize variables
    rd_expenses = []
    revenues = []
    rd_growth = None
    fcf_vals = []
    fcf_growth = None
    op_margin_vals = []
    capex = []
    capex_intensity = None
    dividends = []
    latest_payout_ratio = None

    # 1. R&D Investment Trends
    rd_expenses = [item.research_and_development for item in financial_line_items if hasattr(item, "research_and_development") and item.research_and_development]
    revenues = [item.revenue for item in financial_line_items if item.revenue]

    if rd_expenses and revenues and len(rd_expenses) >= 2:
        rd_growth = (rd_expenses[0] - rd_expenses[-1]) / abs(rd_expenses[-1]) if rd_expenses[-1] != 0 else 0
        if rd_growth > 0.5:  # 50% growth in R&D
            score += 3
            details.append(f"Strong R&D investment growth: +{(rd_growth*100):.1f}%")
        elif rd_growth > 0.2:
            score += 2
            details.append(f"Moderate R&D investment growth: +{(rd_growth*100):.1f}%")

        # Check R&D intensity trend (corrected for reverse chronological order)
        rd_intensity_start = rd_expenses[-1] / revenues[-1]
        rd_intensity_end = rd_expenses[0] / revenues[0]
        if rd_intensity_end > rd_intensity_start:
            score += 2
            details.append(f"Increasing R&D intensity: {(rd_intensity_end*100):.1f}% vs {(rd_intensity_start*100):.1f}%")
    else:
        details.append("Insufficient R&D data for trend analysis")

    # 2. Free Cash Flow Analysis
    fcf_vals = [item.free_cash_flow for item in financial_line_items if item.free_cash_flow]
    if fcf_vals and len(fcf_vals) >= 2:
        fcf_growth = (fcf_vals[0] - fcf_vals[-1]) / abs(fcf_vals[-1])
        positive_fcf_count = sum(1 for f in fcf_vals if f > 0)

        if fcf_growth > 0.3 and positive_fcf_count == len(fcf_vals):
            score += 3
            details.append("Strong and consistent FCF growth, excellent innovation funding capacity")
        elif positive_fcf_count >= len(fcf_vals) * 0.75:
            score += 2
            details.append("Consistent positive FCF, good innovation funding capacity")
        elif positive_fcf_count > len(fcf_vals) * 0.5:
            score += 1
            details.append("Moderately consistent FCF, adequate innovation funding capacity")
    else:
        details.append("Insufficient FCF data for analysis")

    # 3. Operating Efficiency Analysis
    op_margin_vals = [item.operating_margin for item in financial_line_items if item.operating_margin]
    if op_margin_vals and len(op_margin_vals) >= 2:
        margin_trend = op_margin_vals[0] - op_margin_vals[-1]

        if op_margin_vals[0] > 0.15 and margin_trend > 0:
            score += 3
            details.append(f"Strong and improving operating margin: {(op_margin_vals[0]*100):.1f}%")
        elif op_margin_vals[0] > 0.10:
            score += 2
            details.append(f"Healthy operating margin: {(op_margin_vals[0]*100):.1f}%")
        elif margin_trend > 0:
            score += 1
            details.append("Improving operating efficiency")
    else:
        details.append("Insufficient operating margin data")

    # 4. Capital Allocation Analysis
    capex = [item.capital_expenditure for item in financial_line_items if hasattr(item, "capital_expenditure") and item.capital_expenditure]
    if capex and revenues and len(capex) >= 2:
        capex_intensity = abs(capex[0]) / revenues[0]
        capex_growth = (abs(capex[0]) - abs(capex[-1])) / abs(capex[-1]) if capex[-1] != 0 else 0

        if capex_intensity > 0.10 and capex_growth > 0.2:
            score += 2
            details.append("Strong investment in growth infrastructure")
        elif capex_intensity > 0.05:
            score += 1
            details.append("Moderate investment in growth infrastructure")
    else:
        details.append("Insufficient CAPEX data")

    # 5. Growth Reinvestment Analysis
    dividends = [item.dividends_and_other_cash_distributions for item in financial_line_items if hasattr(item, "dividends_and_other_cash_distributions") and item.dividends_and_other_cash_distributions]
    if dividends and fcf_vals:
        latest_payout_ratio = dividends[0] / fcf_vals[0] if fcf_vals[0] != 0 else 1
        if latest_payout_ratio < 0.2:  # Low dividend payout ratio suggests reinvestment focus
            score += 2
            details.append("Strong focus on reinvestment over dividends")
        elif latest_payout_ratio < 0.4:
            score += 1
            details.append("Moderate focus on reinvestment over dividends")
    else:
        details.append("Insufficient dividend data")

    # Normalize score to be out of 5
    normalized_score = (score / max_possible_score) * 5 if max_possible_score > 0 else 0

    latest = financial_line_items[0] if financial_line_items else None
    
    data = {
        "rd_expenses": rd_expenses,
        "rd_growth": rd_growth,
        "fcf_vals": fcf_vals,
        "fcf_growth": fcf_growth,
        "op_margin_vals": op_margin_vals,
        "capex": capex,
        "capex_intensity": capex_intensity,
        "dividends": dividends,
        "payout_ratio": latest_payout_ratio,
        "date": _get_item_date(latest) if latest else None,
    }
    
    # Add provenance information
    data_with_provenance = add_provenance_to_data(
        data,
        metrics=metrics,
        line_items=financial_line_items,
        field_mapping={
            "research_and_development": "research_and_development",
            "free_cash_flow": "free_cash_flow",
            "operating_margin": "operating_margin",
            "capital_expenditure": "capital_expenditure",
            "dividends_and_other_cash_distributions": "dividends_and_other_cash_distributions",
        },
        cik=cik
    )

    return {
        "score": normalized_score,
        "max_score": 5,  # Normalized score is out of 5
        "details": "; ".join(details),
        "raw_score": score,
        "raw_max_score": max_possible_score,
        "data": data_with_provenance,
    }


def analyze_cathie_wood_valuation(financial_line_items: list, market_cap: float, cik: Optional[str] = None) -> dict:
    """
    Cathie Wood often focuses on long-term exponential growth potential. We can do
    a simplified approach looking for a large total addressable market (TAM) and the
    company's ability to capture a sizable portion.
    """
    if not financial_line_items or market_cap is None:
        return {"score": 0, "details": "Insufficient data for valuation"}

    latest = financial_line_items[0]
    fcf = latest.free_cash_flow if latest.free_cash_flow else 0

    if fcf <= 0:
        return {"score": 0, "details": f"No positive FCF for valuation; FCF = {fcf}", "intrinsic_value": None}

    # Instead of a standard DCF, let's assume a higher growth rate for an innovative company.
    # Example values:
    growth_rate = 0.20  # 20% annual growth
    discount_rate = 0.15
    terminal_multiple = 25
    projection_years = 5

    present_value = 0
    for year in range(1, projection_years + 1):
        future_fcf = fcf * (1 + growth_rate) ** year
        pv = future_fcf / ((1 + discount_rate) ** year)
        present_value += pv

    # Terminal Value
    terminal_value = (fcf * (1 + growth_rate) ** projection_years * terminal_multiple) / ((1 + discount_rate) ** projection_years)
    intrinsic_value = present_value + terminal_value

    margin_of_safety = (intrinsic_value - market_cap) / market_cap

    score = 0
    if margin_of_safety > 0.5:
        score += 3
    elif margin_of_safety > 0.2:
        score += 1

    details = [f"Calculated intrinsic value: ~{intrinsic_value:,.2f}", f"Market cap: ~{market_cap:,.2f}", f"Margin of safety: {margin_of_safety:.2%}"]

    max_score = 3
    
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


def generate_cathie_wood_output(
    ticker: str,
    analysis_data: dict[str, any],
    state: AgentState,
    agent_id: str = "cathie_wood_agent",
) -> CathieWoodSignal:
    """
    Generates investment decisions in the style of Cathie Wood.
    """
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Cathie Wood AI agent, making investment decisions using her principles:

            1. Seek companies leveraging disruptive innovation.
            2. Emphasize exponential growth potential, large TAM.
            3. Focus on technology, healthcare, or other future-facing sectors.
            4. Consider multi-year time horizons for potential breakthroughs.
            5. Accept higher volatility in pursuit of high returns.
            6. Evaluate management's vision and ability to invest in R&D.

            Rules:
            - Identify disruptive or breakthrough technology.
            - Evaluate strong potential for multi-year revenue growth.
            - Check if the company can scale effectively in a large market.
            - Use a growth-biased valuation approach.
            - Provide a data-driven recommendation (bullish, bearish, or neutral).
            
            When providing your reasoning, be thorough and specific by:
            1. Identifying the specific disruptive technologies/innovations the company is leveraging
            2. Highlighting growth metrics that indicate exponential potential (revenue acceleration, expanding TAM)
            3. Discussing the long-term vision and transformative potential over 5+ year horizons
            4. Explaining how the company might disrupt traditional industries or create new markets
            5. Addressing R&D investment and innovation pipeline that could drive future growth
            6. Using Cathie Wood's optimistic, future-focused, and conviction-driven voice
            
            For example, if bullish: "The company's AI-driven platform is transforming the $500B healthcare analytics market, with evidence of platform adoption accelerating from 40% to 65% YoY. Their R&D investments of 22% of revenue are creating a technological moat that positions them to capture a significant share of this expanding market. The current valuation doesn't reflect the exponential growth trajectory we expect as..."
            For example, if bearish: "While operating in the genomics space, the company lacks truly disruptive technology and is merely incrementally improving existing techniques. R&D spending at only 8% of revenue signals insufficient investment in breakthrough innovation. With revenue growth slowing from 45% to 20% YoY, there's limited evidence of the exponential adoption curve we look for in transformative companies..."
            """,
            ),
            (
                "human",
                """Based on the following analysis, create a Cathie Wood-style investment signal.

            Analysis Data for {ticker}:
            {analysis_data}

            Write your reasoning in first person, as if you are Cathie Wood analyzing this stock yourself. "
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

            Return the trading signal in this JSON format:
            {{
              "signal": "bullish/bearish/neutral",
              "confidence": float (0-100),
              "reasoning": "string"
            }}
            """,
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    def create_default_cathie_wood_signal():
        return CathieWoodSignal(signal="neutral", confidence=0.0, reasoning="Error in analysis, defaulting to neutral")

    # Get cost tracker from state if available
    cost_tracker = state.get("metadata", {}).get("cost_tracker")

    return call_llm(
        prompt=prompt,
        pydantic_model=CathieWoodSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_cathie_wood_signal,
        cost_tracker=cost_tracker,
    )


# source: https://ark-invest.com
