#!/usr/bin/env python3
"""
Utility script to load and analyze saved backtest data.
Useful for creating charts and further analysis.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.backtests.storage import BacktestStorage


def load_and_display_results(storage: BacktestStorage, ticker: str, analyst: str, start_date: str, end_date: str):
    """Load and display results for a specific backtest"""
    rankings = storage.get_rankings(ticker=ticker, start_date=start_date, end_date=end_date)
    
    result = next((r for r in rankings if r["analyst_name"] == analyst), None)
    if not result:
        print(f"No results found for {analyst} on {ticker}")
        return None
    
    # Load JSON data
    json_path = result["results_json_path"]
    data = storage.load_results(json_path)
    
    print(f"\nResults for {analyst} on {ticker}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"\nPerformance Metrics:")
    print(f"  Total Return: {data['performance_metrics']['total_return_pct']:.2f}%")
    print(f"  Sharpe Ratio: {data['performance_metrics']['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio: {data['performance_metrics']['sortino_ratio']:.2f}")
    print(f"  Max Drawdown: {data['performance_metrics']['max_drawdown']:.2f}%")
    
    # Load CSV for charting
    csv_path = result["results_csv_path"]
    df = storage.get_csv_dataframe(csv_path)
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    return {
        "json_data": data,
        "dataframe": df,
        "result": result,
    }


def create_comparison_chart_data(storage: BacktestStorage, ticker: str, start_date: str, end_date: str, analysts: list = None):
    """Load data for multiple analysts for comparison charts"""
    rankings = storage.get_rankings(ticker=ticker, start_date=start_date, end_date=end_date)
    
    if analysts:
        rankings = [r for r in rankings if r["analyst_name"] in analysts]
    
    comparison_data = {}
    
    for result in rankings:
        analyst = result["analyst_name"]
        csv_path = result["results_csv_path"]
        
        if Path(csv_path).exists():
            df = storage.get_csv_dataframe(csv_path)
            comparison_data[analyst] = {
                "dataframe": df,
                "metrics": {
                    "total_return": result["total_return"],
                    "sharpe_ratio": result["sharpe_ratio"],
                    "max_drawdown": result["max_drawdown"],
                }
            }
    
    return comparison_data


def export_to_excel(storage: BacktestStorage, ticker: str, start_date: str, end_date: str, output_path: str):
    """Export all results to Excel for easy analysis"""
    rankings = storage.get_rankings(ticker=ticker, start_date=start_date, end_date=end_date)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = []
        for r in rankings:
            summary_data.append({
                "Rank": r["rank"],
                "Analyst": r["analyst_name"],
                "Total Return %": r["total_return"],
                "Sharpe Ratio": r["sharpe_ratio"],
                "Sortino Ratio": r["sortino_ratio"],
                "Max Drawdown %": r["max_drawdown"],
                "Final Value": r["final_value"],
            })
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        
        # Individual analyst sheets
        for r in rankings[:10]:  # Limit to top 10 to avoid huge files
            analyst = r["analyst_name"]
            csv_path = r["results_csv_path"]
            if Path(csv_path).exists():
                df = storage.get_csv_dataframe(csv_path)
                # Truncate sheet name if too long
                sheet_name = analyst[:31] if len(analyst) > 31 else analyst
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"Exported to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and analyze backtest data")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker")
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--analyst", type=str, help="Specific analyst to load")
    parser.add_argument("--export-excel", type=str, help="Export to Excel file")
    parser.add_argument("--list", action="store_true", help="List all available results")
    
    args = parser.parse_args()
    
    storage = BacktestStorage()
    
    if args.list:
        rankings = storage.get_rankings(ticker=args.ticker, start_date=args.start_date, end_date=args.end_date)
        print(f"\nAvailable results for {args.ticker} ({args.start_date} to {args.end_date}):")
        for r in rankings:
            print(f"  {r['rank']}. {r['analyst_name']}: {r['total_return']:.2f}% return, Sharpe {r['sharpe_ratio']:.2f}")
    elif args.export_excel:
        export_to_excel(storage, args.ticker, args.start_date, args.end_date, args.export_excel)
    elif args.analyst:
        load_and_display_results(storage, args.ticker, args.analyst, args.start_date, args.end_date)
    else:
        # Show summary
        summary = storage.get_comparison_summary(
            ticker=args.ticker,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        print(json.dumps(summary, indent=2, default=str))
