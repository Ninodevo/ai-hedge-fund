"""
Utility functions for tracking data provenance in analyst analysis.
Extracts source information from financial metrics and line items.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
from src.data.models import FinancialMetrics, LineItem


def get_metric_provenance(
    metric: FinancialMetrics,
    field_name: str,
    cik: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract provenance information for a specific metric field.
    
    Args:
        metric: FinancialMetrics object
        field_name: Name of the metric field (e.g., "return_on_equity")
        cik: Optional CIK for constructing filing URLs
    
    Returns:
        Dictionary with provenance information
    """
    from server.data_provenance import _get_exact_sec_filing_url
    
    filing_url = None
    if cik and metric.report_period:
        filing_url = _get_exact_sec_filing_url(
            cik, 
            metric.report_period, 
            getattr(metric, "fiscal_period", None)
        )
    
    return {
        "source": "financial_metrics_api",
        "period_type": metric.period,
        "report_period": metric.report_period,
        "fiscal_period": getattr(metric, "fiscal_period", None),
        "filing_url": filing_url,
        "fetched_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


def get_line_item_provenance(
    line_item: LineItem,
    field_name: str,
    cik: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract provenance information for a specific line item field.
    
    Args:
        line_item: LineItem object
        field_name: Name of the line item field (e.g., "net_income")
        cik: Optional CIK for constructing filing URLs
    
    Returns:
        Dictionary with provenance information
    """
    from server.data_provenance import _get_exact_sec_filing_url
    
    filing_url = None
    if cik and line_item.report_period:
        fiscal_period = getattr(line_item, "fiscal_period", None)
        filing_url = _get_exact_sec_filing_url(cik, line_item.report_period, fiscal_period)
    
    return {
        "source": "line_items_api",
        "period_type": getattr(line_item, "period", "unknown"),
        "report_period": line_item.report_period,
        "fiscal_period": getattr(line_item, "fiscal_period", None),
        "filing_url": filing_url,
        "fetched_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


def add_provenance_to_data(
    data: Dict[str, Any],
    metrics: Optional[List[FinancialMetrics]] = None,
    line_items: Optional[List[LineItem]] = None,
    field_mapping: Optional[Dict[str, str]] = None,
    cik: Optional[str] = None
) -> Dict[str, Any]:
    """
    Add provenance information to a data dictionary.
    
    Args:
        data: Dictionary containing metric values
        metrics: List of FinancialMetrics objects
        line_items: List of LineItem objects
        field_mapping: Mapping from data field names to metric/line_item field names
        cik: Optional CIK for constructing filing URLs
    
    Returns:
        Dictionary with added provenance information
    """
    if not data:
        return data
    
    provenance = {}
    
    # Add provenance for metrics
    if metrics and len(metrics) > 0:
        latest_metric = metrics[0]
        metric_fields = {
            "return_on_equity": "return_on_equity",
            "debt_to_equity": "debt_to_equity",
            "operating_margin": "operating_margin",
            "current_ratio": "current_ratio",
            "gross_margin": "gross_margin",
            "net_margin": "net_margin",
            "asset_turnover": "asset_turnover",
        }
        
        if field_mapping:
            metric_fields.update(field_mapping)
        
        for data_field, metric_field in metric_fields.items():
            if data_field in data and hasattr(latest_metric, metric_field):
                provenance[f"{data_field}_provenance"] = get_metric_provenance(
                    latest_metric, metric_field, cik
                )
    
    # Add provenance for line items
    if line_items and len(line_items) > 0:
        latest_line_item = line_items[0]
        line_item_fields = {
            "net_income": "net_income",
            "revenue": "revenue",
            "free_cash_flow": "free_cash_flow",
            "outstanding_shares": "outstanding_shares",
            "capital_expenditure": "capital_expenditure",
            "depreciation_and_amortization": "depreciation_and_amortization",
            "dividends_and_other_cash_distributions": "dividends_and_other_cash_distributions",
            "issuance_or_purchase_of_equity_shares": "issuance_or_purchase_of_equity_shares",
            "gross_profit": "gross_profit",
            "total_assets": "total_assets",
            "total_liabilities": "total_liabilities",
            "shareholders_equity": "shareholders_equity",
        }
        
        if field_mapping:
            line_item_fields.update(field_mapping)
        
        for data_field, line_item_field in line_item_fields.items():
            if data_field in data and hasattr(latest_line_item, line_item_field):
                provenance[f"{data_field}_provenance"] = get_line_item_provenance(
                    latest_line_item, line_item_field, cik
                )
    
    # Add historical data provenance (for arrays)
    if metrics and len(metrics) > 1:
        # Track which periods were used
        periods_used = []
        for metric in metrics:
            periods_used.append({
                "report_period": metric.report_period,
                "fiscal_period": getattr(metric, "fiscal_period", None),
                "period_type": metric.period,
            })
        provenance["historical_periods"] = periods_used
    
    if line_items and len(line_items) > 1:
        periods_used = []
        for item in line_items:
            periods_used.append({
                "report_period": item.report_period,
                "fiscal_period": getattr(item, "fiscal_period", None),
                "period_type": getattr(item, "period", "unknown"),
            })
        provenance["historical_line_item_periods"] = periods_used
    
    # Merge provenance into data
    result = {**data, **provenance}
    return result


def get_market_cap_provenance(
    market_cap: Optional[float],
    as_of_date: str,
    source: str = "market_cap_api"
) -> Dict[str, Any]:
    """
    Get provenance information for market cap.
    
    Args:
        market_cap: Market cap value
        as_of_date: Date of the market cap
        source: Source API name
    
    Returns:
        Provenance dictionary
    """
    return {
        "source": source,
        "period_type": "latest",
        "date": as_of_date,
        "fetched_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

