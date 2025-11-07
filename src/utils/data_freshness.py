"""
Utility functions for checking financial data freshness and handling stale TTM data.

TTM (Trailing Twelve Months) data can become stale if there are no recent financial reports.
This module provides utilities to detect stale data and fallback to more recent periods.
"""
from datetime import datetime, timedelta
from typing import Optional, Tuple
from src.data.models import FinancialMetrics


def calculate_data_age_days(report_period: str, reference_date: Optional[str] = None) -> Optional[int]:
    """
    Calculate the age of financial data in days based on report_period.
    
    Args:
        report_period: Report period date string (YYYY-MM-DD format)
        reference_date: Reference date to compare against (defaults to today). Format: YYYY-MM-DD
    
    Returns:
        Number of days old the data is, or None if date parsing fails
    """
    try:
        report_date = datetime.strptime(report_period, "%Y-%m-%d")
        if reference_date:
            ref_date = datetime.strptime(reference_date, "%Y-%m-%d")
        else:
            ref_date = datetime.now()
        
        age_days = (ref_date - report_date).days
        return max(0, age_days)  # Ensure non-negative
    except (ValueError, TypeError):
        return None


def is_data_stale(
    metrics: list[FinancialMetrics],
    max_age_days: int = 180,
    reference_date: Optional[str] = None
) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Check if the most recent financial metrics are stale.
    
    Args:
        metrics: List of FinancialMetrics objects
        max_age_days: Maximum age in days before data is considered stale (default: 180 days / 6 months)
        reference_date: Reference date to compare against (defaults to today). Format: YYYY-MM-DD
    
    Returns:
        Tuple of (is_stale: bool, age_days: int | None, report_period: str | None)
    """
    if not metrics:
        return True, None, None
    
    most_recent = metrics[0]
    report_period = most_recent.report_period
    
    age_days = calculate_data_age_days(report_period, reference_date)
    
    if age_days is None:
        # Can't determine age, assume not stale but log warning
        return False, None, report_period
    
    is_stale = age_days > max_age_days
    return is_stale, age_days, report_period


def get_freshness_warning(
    age_days: Optional[int],
    report_period: Optional[str],
    period_type: str = "TTM"
) -> Optional[str]:
    """
    Generate a human-readable warning message about data freshness.
    
    Args:
        age_days: Age of data in days
        report_period: Report period date string
        period_type: Type of period (e.g., "TTM", "quarterly", "annual")
    
    Returns:
        Warning message string or None if no warning needed
    """
    if age_days is None:
        return f"Warning: Could not determine age of {period_type} data (report_period: {report_period})"
    
    if age_days > 180:
        months = age_days / 30.44
        return (
            f"Warning: {period_type} data is {age_days:.0f} days ({months:.1f} months) old "
            f"(report_period: {report_period}). "
            f"Consider using more recent quarterly or annual data."
        )
    elif age_days > 90:
        months = age_days / 30.44
        return (
            f"Note: {period_type} data is {age_days:.0f} days ({months:.1f} months) old "
            f"(report_period: {report_period})."
        )
    
    return None


def should_fallback_to_quarterly(
    ttm_metrics: list[FinancialMetrics],
    max_age_days: int = 180,
    reference_date: Optional[str] = None
) -> bool:
    """
    Determine if we should fallback from TTM to quarterly data.
    
    Args:
        ttm_metrics: List of TTM FinancialMetrics
        max_age_days: Maximum age in days before fallback is recommended
        reference_date: Reference date to compare against
    
    Returns:
        True if fallback is recommended, False otherwise
    """
    is_stale, age_days, _ = is_data_stale(ttm_metrics, max_age_days, reference_date)
    
    # Fallback if stale or if we can't determine age (better safe than sorry)
    return is_stale or age_days is None

