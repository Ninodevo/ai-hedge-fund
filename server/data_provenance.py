"""
Data provenance tracking system for financial data.

Tracks where each data point comes from, its date, and whether it's the latest available.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
import requests


def _get_exact_sec_filing_url(cik: Optional[str], report_period: str, fiscal_period: Optional[str] = None) -> Optional[str]:
    """
    Get the exact SEC filing URL by querying SEC EDGAR API.
    
    Args:
        cik: Company CIK number (Central Index Key)
        report_period: Fiscal period end date (YYYY-MM-DD)
        fiscal_period: Fiscal period identifier (e.g., "2024-Q1", "2023-Annual")
    
    Returns:
        Exact SEC filing document URL or None if not found
    """
    if not cik:
        return None
    
    try:
        # Pad CIK to 10 digits
        cik_padded = str(cik).zfill(10)
        
        # Determine filing type
        # Q4 is typically an annual report (10-K), not quarterly (10-Q)
        is_annual = False
        if fiscal_period:
            fiscal_lower = fiscal_period.lower()
            is_annual = (
                "annual" in fiscal_lower or 
                "year" in fiscal_lower or
                fiscal_lower.endswith("-q4") or  # Q4 is usually annual
                fiscal_lower.endswith("q4") or
                "q4" in fiscal_lower
            )
        filing_type = "10-K" if is_annual else "10-Q"
        
        # Query SEC submissions API
        # Note: SEC requires User-Agent header
        headers = {
            "User-Agent": "Stocklytic Engine (contact@stocklytic.com)",
            "Accept": "application/json"
        }
        
        submissions_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        
        try:
            response = requests.get(submissions_url, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"Warning: SEC API returned {response.status_code} for CIK {cik}")
                return _construct_sec_filing_url(cik, report_period, fiscal_period)
            
            data = response.json()
            filings = data.get("filings", {}).get("recent", {})
            
            if not filings:
                print(f"Warning: No recent filings found for CIK {cik}")
                return _construct_sec_filing_url(cik, report_period, fiscal_period)
            
            # Get filing types, dates, and accession numbers
            filing_types = filings.get("form", [])
            filing_dates = filings.get("filingDate", [])
            accession_numbers = filings.get("accessionNumber", [])
            report_dates = filings.get("reportDate", [])  # Fiscal period end dates
            
            if not all([filing_types, filing_dates, accession_numbers]):
                print(f"Warning: Incomplete filing data for CIK {cik}")
                return _construct_sec_filing_url(cik, report_period, fiscal_period)
            
            # Find the filing that matches our criteria
            report_date_obj = datetime.strptime(report_period, "%Y-%m-%d")
            
            # First try to match by report date (fiscal period end)
            if report_dates:
                for i, (form_type, report_date_str) in enumerate(zip(filing_types, report_dates)):
                    if form_type == filing_type and report_date_str:
                        try:
                            filing_report_date = datetime.strptime(report_date_str, "%Y-%m-%d")
                            if filing_report_date == report_date_obj:
                                # Found exact match by report date
                                accession = accession_numbers[i]
                                cik_numeric = cik_padded.lstrip("0")
                                accession_no_dashes = accession.replace("-", "")
                                
                                # Get the exact document URL
                                doc_url = _get_document_url_from_index(cik_numeric, accession, accession_no_dashes, filing_type, headers)
                                if doc_url:
                                    return doc_url
                                
                                # If index parsing failed, try to construct URL directly
                                # Most SEC filings follow pattern: {form}-{date}.htm
                                date_str = report_period.replace("-", "")
                                constructed_url = f"https://www.sec.gov/Archives/edgar/data/{cik_numeric}/{accession_no_dashes}/{filing_type.lower()}-{date_str}.htm"
                                # Return constructed URL as fallback (might work even if index parsing failed)
                                return constructed_url
                        except (ValueError, IndexError) as e:
                            print(f"Warning: Error matching filing: {str(e)}")
                            continue
            
            # Fallback: match by filing date (within 90 days after report period)
            for i, (form_type, filing_date) in enumerate(zip(filing_types, filing_dates)):
                if form_type == filing_type:
                    try:
                        filing_date_obj = datetime.strptime(filing_date, "%Y-%m-%d")
                        days_diff = (filing_date_obj - report_date_obj).days
                        
                        # Filing should be after report period but within 90 days
                        if 0 <= days_diff <= 90:
                            accession = accession_numbers[i]
                            cik_numeric = cik_padded.lstrip("0")
                            accession_no_dashes = accession.replace("-", "")
                            
                            # Get the exact document URL
                            doc_url = _get_document_url_from_index(cik_numeric, accession, accession_no_dashes, filing_type, headers)
                            if doc_url:
                                return doc_url
                            
                            # If index parsing failed, try to construct URL directly
                            date_str = report_period.replace("-", "")
                            constructed_url = f"https://www.sec.gov/Archives/edgar/data/{cik_numeric}/{accession_no_dashes}/{filing_type.lower()}-{date_str}.htm"
                            return constructed_url
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Error matching filing by date: {str(e)}")
                        continue
            
            # If no match found, return search URL
            print(f"Warning: No matching filing found for CIK {cik}, report_period {report_period}, type {filing_type}")
            return _construct_sec_filing_url(cik, report_period, fiscal_period)
        except Exception as e:
            # If API call fails, fall back to search URL
            print(f"Warning: Could not fetch exact SEC filing URL: {str(e)}")
            return _construct_sec_filing_url(cik, report_period, fiscal_period)
    except Exception as e:
        print(f"Warning: Error in _get_exact_sec_filing_url: {str(e)}")
        return None


def _get_document_url_from_index(cik_numeric: str, accession: str, accession_no_dashes: str, filing_type: str, headers: dict) -> Optional[str]:
    """
    Get the exact document URL from the SEC filing index page.
    
    Args:
        cik_numeric: CIK without leading zeros
        accession: Accession number with dashes
        accession_no_dashes: Accession number without dashes
        filing_type: Filing type (10-Q or 10-K)
        headers: HTTP headers for SEC API requests
    
    Returns:
        Exact document URL or None if not found
    """
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_numeric}/{accession_no_dashes}/{accession}-index.htm"
    
    try:
        index_response = requests.get(index_url, headers=headers, timeout=10)
        if index_response.status_code == 200:
            import re
            from urllib.parse import unquote
            content = index_response.text
            
            def extract_filename_from_link(link: str) -> Optional[str]:
                """Extract filename from various SEC link formats."""
                # Handle /ix?doc= format: /ix?doc=/Archives/edgar/data/320193/000032019325000073/aapl-20250628.htm
                if '/ix?doc=' in link:
                    # Extract the path after /ix?doc=
                    doc_match = re.search(r'/ix\?doc=(.+?\.htm)', link, re.IGNORECASE)
                    if doc_match:
                        doc_path = unquote(doc_match.group(1))
                        # Extract just the filename from the path
                        filename = doc_path.split('/')[-1]
                        return filename
                
                # Handle direct relative links: aapl-20250628.htm
                if link.endswith('.htm') and '/' not in link:
                    return link
                
                # Handle absolute paths: /Archives/edgar/data/320193/000032019325000073/aapl-20250628.htm
                if link.startswith('/Archives/edgar/data/'):
                    filename = link.split('/')[-1]
                    if filename.endswith('.htm'):
                        return filename
                
                return None
            
            # Look for document links in the index
            # Pattern: <a href="...">Document Type</a>
            doc_pattern = r'<a\s+href="([^"]+)"[^>]*>([^<]*)</a>'
            matches = re.findall(doc_pattern, content, re.IGNORECASE)
            
            if matches:
                # Look for primary document (usually the main filing document)
                # Primary documents are typically:
                # 1. The main filing document (not exhibits)
                # 2. Often have ticker-date format: ticker-YYYYMMDD.htm
                # 3. Listed first or early in the index
                
                primary_candidates = []
                other_candidates = []
                
                for link, doc_type in matches:
                    filename = extract_filename_from_link(link)
                    if not filename:
                        continue
                    
                    link_lower = link.lower()
                    doc_type_lower = doc_type.lower()
                    filename_lower = filename.lower()
                    filing_type_lower = filing_type.lower()
                    
                    # Skip index links and exhibits
                    if 'index' in filename_lower or 'exhibit' in filename_lower:
                        continue
                    
                    # Check if this looks like the primary document
                    is_primary = (
                        filing_type_lower in link_lower or 
                        filing_type_lower in doc_type_lower or
                        "complete" in doc_type_lower or
                        "primary" in doc_type_lower or
                        # Primary documents often have simple ticker-date format
                        (re.match(r'^[a-z]+-\d{8}\.htm$', filename_lower) is not None)
                    )
                    
                    if is_primary:
                        primary_candidates.append((filename, link, doc_type))
                    else:
                        other_candidates.append((filename, link, doc_type))
                
                # Return the first primary candidate, or first other candidate if no primary found
                if primary_candidates:
                    return f"https://www.sec.gov/Archives/edgar/data/{cik_numeric}/{accession_no_dashes}/{primary_candidates[0][0]}"
                elif other_candidates:
                    return f"https://www.sec.gov/Archives/edgar/data/{cik_numeric}/{accession_no_dashes}/{other_candidates[0][0]}"
            
            # Alternative: look for any .htm link using a simpler pattern
            htm_links = re.findall(r'href="([^"]+)"', content)
            for link in htm_links:
                filename = extract_filename_from_link(link)
                if filename and 'exhibit' not in filename.lower() and 'index' not in filename.lower():
                    return f"https://www.sec.gov/Archives/edgar/data/{cik_numeric}/{accession_no_dashes}/{filename}"
                    
    except Exception as e:
        print(f"Warning: Could not parse SEC filing index: {str(e)}")
    
    # If we can't parse the index, we can't get the exact URL
    # Return None so caller can fall back to search URL
    return None


def _construct_sec_filing_url(cik: Optional[str], report_period: str, fiscal_period: Optional[str] = None) -> Optional[str]:
    """
    Construct SEC filing search URL (fallback when exact URL cannot be determined).
    
    Args:
        cik: Company CIK number (Central Index Key)
        report_period: Fiscal period end date (YYYY-MM-DD)
        fiscal_period: Fiscal period identifier (e.g., "2024-Q1", "2023-Annual")
    
    Returns:
        SEC filing search URL or None if CIK is not available
    """
    if not cik:
        return None
    
    try:
        # Pad CIK to 10 digits
        cik_padded = str(cik).zfill(10)
        
        # Determine filing type
        # Q4 is typically an annual report (10-K), not quarterly (10-Q)
        is_annual = False
        if fiscal_period:
            fiscal_lower = fiscal_period.lower()
            is_annual = (
                "annual" in fiscal_lower or 
                "year" in fiscal_lower or
                fiscal_lower.endswith("-q4") or  # Q4 is usually annual
                fiscal_lower.endswith("q4") or
                "q4" in fiscal_lower
            )
        filing_type = "10-K" if is_annual else "10-Q"
        
        # Construct SEC EDGAR search URL
        # This will show filings for the company, filtered by type
        year = report_period[:4]
        url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={filing_type}&dateb={year}1231&owner=exclude&count=10"
        
        return url
    except Exception as e:
        print(f"Warning: Error in _construct_sec_filing_url: {str(e)}")
        return None


class DataProvenance:
    """Tracks data provenance for financial metrics."""
    
    def __init__(self):
        self.provenance: Dict[str, Dict[str, Any]] = {}
    
    def add_metric(
        self,
        name: str,
        value: Any,
        source: str,
        date: str,
        period_type: str = "latest",
        is_latest: bool = True,
        report_period: Optional[str] = None,
        fiscal_period: Optional[str] = None,
        filing_url: Optional[str] = None,
    ):
        """
        Add a metric with provenance information.
        
        Args:
            name: Parameter name (e.g., "pe", "revenue", "market_cap")
            value: The actual value
            source: Data source (e.g., "financial_metrics_api", "line_items_api", "market_cap_api")
            date: Date of the data (YYYY-MM-DD)
            period_type: Type of period ("latest", "ttm", "quarterly", "annual")
            is_latest: Whether this is the latest available data
            report_period: Fiscal period end date if applicable
            fiscal_period: Fiscal period identifier if applicable
            filing_url: URL to the SEC filing if applicable
        """
        self.provenance[name] = {
            "value": value,
            "source": source,
            "date": date,
            "period_type": period_type,
            "is_latest": is_latest,
            "report_period": report_period,
            "fiscal_period": fiscal_period,
            "filing_url": filing_url,
            "fetched_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        }
    
    def add_from_financial_metrics(
        self,
        metrics: List[Any],
        field_mapping: Dict[str, str],
        as_of_date: str,
        period_type: str = "ttm",
        cik: Optional[str] = None,
    ):
        """
        Add metrics from FinancialMetrics objects.
        
        Args:
            metrics: List of FinancialMetrics objects
            field_mapping: Dict mapping field names to parameter names
            as_of_date: Date as of which data was fetched
            period_type: Period type ("ttm", "quarterly", "annual")
            cik: Company CIK number for constructing filing URLs
        """
        if not metrics:
            return
        
        latest_metric = metrics[0]
        report_period = latest_metric.report_period
        fiscal_period = getattr(latest_metric, "fiscal_period", None)
        
        # Get exact filing URL if we have CIK
        filing_url = _get_exact_sec_filing_url(cik, report_period, fiscal_period) if cik else None
        
        for param_name, field_name in field_mapping.items():
            value = getattr(latest_metric, field_name, None)
            if value is not None:
                self.add_metric(
                    name=param_name,
                    value=value,
                    source="financial_metrics_api",
                    date=report_period,
                    period_type=period_type,
                    is_latest=True,
                    report_period=report_period,
                    fiscal_period=fiscal_period,
                    filing_url=filing_url,
                )
    
    def add_from_line_items(
        self,
        line_items: List[Any],
        field_mapping: Dict[str, str],
        as_of_date: str,
        period_type: str = "ttm",
        cik: Optional[str] = None,
    ):
        """
        Add metrics from LineItem objects.
        
        Args:
            line_items: List of LineItem objects
            field_mapping: Dict mapping field names to parameter names
            as_of_date: Date as of which data was fetched
            period_type: Period type ("ttm", "quarterly", "annual")
            cik: Company CIK number for constructing filing URLs
        """
        if not line_items:
            return
        
        latest_item = line_items[0]
        report_period = getattr(latest_item, "report_period", as_of_date)
        fiscal_period = getattr(latest_item, "fiscal_period", None)
        
        # Get exact filing URL if we have CIK
        filing_url = _get_exact_sec_filing_url(cik, report_period, fiscal_period) if cik else None
        
        for param_name, field_name in field_mapping.items():
            value = getattr(latest_item, field_name, None)
            if value is not None:
                self.add_metric(
                    name=param_name,
                    value=value,
                    source="line_items_api",
                    date=report_period,
                    period_type=period_type,
                    is_latest=True,
                    report_period=report_period,
                    fiscal_period=fiscal_period,
                    filing_url=filing_url,
                )
    
    def add_market_cap(self, market_cap: Optional[float], as_of_date: str, source: str = "market_cap_api"):
        """Add market cap with provenance."""
        if market_cap is not None:
            self.add_metric(
                name="market_cap",
                value=market_cap,
                source=source,
                date=as_of_date,
                period_type="latest",
                is_latest=True,
            )
    
    def add_price_data(self, price_data: Dict[str, Any], as_of_date: str):
        """Add price data with provenance."""
        if price_data:
            for key, value in price_data.items():
                if value is not None:
                    self.add_metric(
                        name=f"price_{key}",
                        value=value,
                        source="price_api",
                        date=as_of_date,
                        period_type="latest",
                        is_latest=True,
                    )
    
    def get_provenance(self) -> Dict[str, Dict[str, Any]]:
        """Get the full provenance dictionary."""
        return self.provenance
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of data sources and freshness."""
        sources = {}
        latest_count = 0
        historical_count = 0
        
        for param, info in self.provenance.items():
            source = info["source"]
            if source not in sources:
                sources[source] = {
                    "parameters": [],
                    "latest_date": None,
                    "oldest_date": None,
                }
            
            sources[source]["parameters"].append(param)
            
            if info["is_latest"]:
                latest_count += 1
            else:
                historical_count += 1
            
            date = info.get("report_period") or info["date"]
            if date:
                if sources[source]["latest_date"] is None or date > sources[source]["latest_date"]:
                    sources[source]["latest_date"] = date
                if sources[source]["oldest_date"] is None or date < sources[source]["oldest_date"]:
                    sources[source]["oldest_date"] = date
        
        return {
            "sources": sources,
            "total_parameters": len(self.provenance),
            "latest_data_count": latest_count,
            "historical_data_count": historical_count,
        }
