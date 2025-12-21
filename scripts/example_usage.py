#!/usr/bin/env python3
"""
Example usage of the backtest storage system.
Shows how to save, load, and analyze backtest data.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.backtests.storage import BacktestStorage
from src.backtesting.engine import BacktestEngine
from src.main import run_hedge_fund
import pandas as pd

# Initialize storage
storage = BacktestStorage()

# Example: Save a backtest result
print("Example: Saving backtest results...")
print("=" * 60)

# Note: This is just an example - you'd run an actual backtest first
# backtester = BacktestEngine(...)
# metrics = backtester.run_backtest()
# storage.save_backtest_results(...)

# Example: Load rankings
print("\nExample: Loading analyst rankings...")
print("=" * 60)

rankings = storage.get_rankings(
    ticker="AAPL",
    start_date="2021-01-01",
    end_date="2024-01-01",
    order_by="sharpe_ratio",
    limit=5,
)

if rankings:
    print(f"\nTop 5 Analysts for AAPL:")
    for r in rankings:
        print(f"  {r['rank']}. {r['analyst_name']}")
        print(f"     Return: {r['total_return']:.2f}%")
        print(f"     Sharpe: {r['sharpe_ratio']:.2f}")
        print()
else:
    print("No results found. Run compare_analysts.py first!")

# Example: Load specific result
print("\nExample: Loading specific analyst result...")
print("=" * 60)

if rankings:
    top_analyst = rankings[0]
    json_path = top_analyst["results_json_path"]
    
    if Path(json_path).exists():
        data = storage.load_results(json_path)
        print(f"\nLoaded data for {top_analyst['analyst_name']}:")
        print(f"  Total Return: {data['performance_metrics']['total_return_pct']:.2f}%")
        print(f"  Sharpe Ratio: {data['performance_metrics']['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {data['performance_metrics']['max_drawdown']:.2f}%")
        
        # Load CSV for charting
        csv_path = top_analyst["results_csv_path"]
        if Path(csv_path).exists():
            df = storage.get_csv_dataframe(csv_path)
            print(f"\nDataFrame shape: {df.shape}")
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"\nFirst few rows:")
            print(df.head())

# Example: Get comparison summary
print("\nExample: Generating comparison summary...")
print("=" * 60)

summary = storage.get_comparison_summary(
    ticker="AAPL",
    start_date="2021-01-01",
    end_date="2024-01-01",
)

if summary["total_analysts"] > 0:
    print(f"\nSummary Statistics:")
    print(f"  Total Analysts: {summary['total_analysts']}")
    stats = summary["statistics"]
    if stats["avg_return"]:
        print(f"  Average Return: {stats['avg_return']:.2f}%")
        print(f"  Best Return: {stats['max_return']:.2f}%")
        print(f"  Worst Return: {stats['min_return']:.2f}%")
    if stats["avg_sharpe"]:
        print(f"  Average Sharpe: {stats['avg_sharpe']:.2f}")
        print(f"  Best Sharpe: {stats['max_sharpe']:.2f}")

print("\n" + "=" * 60)
print("For more examples, see:")
print("  - scripts/compare_analysts.py (run comparisons)")
print("  - scripts/load_backtest_data.py (load and export data)")
print("  - data/backtests/README.md (full documentation)")
print("\nData is saved in formats ready for frontend consumption:")
print("  - JSON: Complete data structure")
print("  - CSV: Daily data for charting")
print("  - Database: Fast queries for rankings")
