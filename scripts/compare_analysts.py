#!/usr/bin/env python3
"""
Compare all analysts by running backtests and saving comprehensive results.
Saves data to database, JSON, and CSV for later analysis and charting.
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backtesting.engine import BacktestEngine
from src.main import run_hedge_fund
from src.utils.analysts import ANALYST_CONFIG
from data.backtests.storage import BacktestStorage
from colorama import Fore, Style, init

init(autoreset=True)


def estimate_cost(start_date: str, end_date: str, model_provider: str) -> float:
    """Estimate API cost based on date range and model"""
    from dateutil.relativedelta import relativedelta
    from datetime import datetime
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Calculate business days (rough estimate)
    days = (end - start).days
    business_days = int(days * 0.71)  # Rough estimate: ~71% are business days
    
    # Tokens per day: ~40,000
    # DeepSeek off-peak: $0.135/M input, $0.55/M output
    # Input: 90%, Output: 10%
    total_tokens = business_days * 40000
    input_tokens = total_tokens * 0.9
    output_tokens = total_tokens * 0.1
    
    if model_provider.lower() == "deepseek":
        cost = (input_tokens / 1_000_000 * 0.135) + (output_tokens / 1_000_000 * 0.55)
    else:
        # Default estimate
        cost = business_days * 0.005
    
    return round(cost, 2)


def compare_analysts(
    ticker: str = "AAPL",
    start_date: str = "2021-01-01",
    end_date: str = "2024-01-01",
    model_name: str = "deepseek-chat",
    model_provider: str = "DeepSeek",
    initial_capital: float = 100000.0,
    analysts: list = None,
    skip_existing: bool = True,
):
    """
    Compare all analysts by running backtests and saving results
    
    Args:
        ticker: Stock ticker to test
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        model_name: LLM model name
        model_provider: LLM provider
        initial_capital: Starting capital
        analysts: List of analyst keys to test (None = all)
        skip_existing: Skip if results already exist
    """
    # Initialize storage
    storage = BacktestStorage()
    
    # Get analysts to test
    if analysts is None:
        analysts = list(ANALYST_CONFIG.keys())
    
    print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Analyst Comparison Tool{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"Ticker: {Fore.GREEN}{ticker}{Style.RESET_ALL}")
    print(f"Date Range: {Fore.GREEN}{start_date} to {end_date}{Style.RESET_ALL}")
    print(f"Model: {Fore.GREEN}{model_name} ({model_provider}){Style.RESET_ALL}")
    print(f"Analysts to test: {Fore.GREEN}{len(analysts)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")
    
    results = {}
    failed_analysts = []
    
    for i, analyst_key in enumerate(analysts, 1):
        analyst_name = ANALYST_CONFIG[analyst_key]["display_name"]
        
        print(f"\n{Fore.YELLOW}[{i}/{len(analysts)}]{Style.RESET_ALL} Testing {Fore.CYAN}{analyst_name}{Style.RESET_ALL} ({analyst_key})...")
        
        # Check if already exists
        if skip_existing:
            rankings = storage.get_rankings(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )
            existing = [r for r in rankings if r["analyst_name"] == analyst_key]
            if existing:
                print(f"  {Fore.GREEN}✓{Style.RESET_ALL} Results already exist, skipping...")
                results[analyst_key] = existing[0]
                continue
        
        try:
            # Setup CSV path for streaming
            storage_base = Path("data/backtests")
            csv_dir = storage_base / "results" / ticker
            csv_dir.mkdir(parents=True, exist_ok=True)
            date_range = f"{start_date}_{end_date}"
            csv_path = csv_dir / f"{analyst_key}_{date_range}_streaming.csv"
            
            # Create backtester with checkpointing enabled
            backtester = BacktestEngine(
                agent=run_hedge_fund,
                tickers=[ticker],
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                model_name=model_name,
                model_provider=model_provider,
                selected_analysts=[analyst_key],
                initial_margin_requirement=0.0,
                checkpoint_interval=20,  # Every 20 business days (~monthly)
                enable_csv_streaming=True,
            )
            
            # Run backtest with checkpointing and CSV streaming
            start_time = time.time()
            metrics = backtester.run_backtest(
                ticker=ticker,
                analyst=analyst_key,
                csv_path=csv_path,
                resume=True,  # Enable resume from checkpoint
            )
            duration = time.time() - start_time
            
            # Estimate cost
            cost = estimate_cost(start_date, end_date, model_provider)
            
            # Save results
            file_paths = storage.save_backtest_results(
                ticker=ticker,
                analyst=analyst_key,
                start_date=start_date,
                end_date=end_date,
                backtester=backtester,
                metrics=metrics,
                model_name=model_name,
                model_provider=model_provider,
                cost=cost,
                duration=duration,
                notes=f"Automated comparison test",
                tags=["comparison", "automated"],
            )
            
            # Get portfolio values for summary
            portfolio_values = backtester.get_portfolio_values()
            if portfolio_values:
                initial_value = portfolio_values[0]["Portfolio Value"]
                final_value = portfolio_values[-1]["Portfolio Value"]
                total_return = ((final_value - initial_value) / initial_value * 100) if initial_value > 0 else 0
            else:
                total_return = 0
            
            results[analyst_key] = {
                "analyst": analyst_name,
                "total_return": total_return,
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "sortino_ratio": metrics.get("sortino_ratio"),
                "max_drawdown": metrics.get("max_drawdown"),
                "duration_seconds": duration,
                "cost": cost,
                "files": file_paths,
            }
            
            print(f"  {Fore.GREEN}✓{Style.RESET_ALL} Completed in {duration:.1f}s")
            print(f"    Return: {Fore.GREEN if total_return >= 0 else Fore.RED}{total_return:.2f}%{Style.RESET_ALL}")
            print(f"    Sharpe: {Fore.CYAN}{metrics.get('sharpe_ratio', 0):.2f}{Style.RESET_ALL}")
            print(f"    Cost: ${Fore.YELLOW}{cost:.2f}{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"  {Fore.RED}✗{Style.RESET_ALL} Failed: {str(e)}")
            failed_analysts.append((analyst_key, str(e)))
            continue
    
    # Generate summary
    print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Generating Summary...{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")
    
    summary = storage.get_comparison_summary(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )
    
    # Print top performers
    print(f"{Fore.GREEN}Top 5 Performers (by Sharpe Ratio):{Style.RESET_ALL}\n")
    for ranking in summary["rankings"][:5]:
        print(f"  {Fore.CYAN}{ranking['rank']}.{Style.RESET_ALL} {Fore.GREEN}{ranking['analyst']}{Style.RESET_ALL}")
        print(f"     Return: {Fore.GREEN if ranking['total_return'] >= 0 else Fore.RED}{ranking['total_return']:.2f}%{Style.RESET_ALL}")
        print(f"     Sharpe: {Fore.CYAN}{ranking['sharpe_ratio']:.2f if ranking['sharpe_ratio'] else 'N/A'}{Style.RESET_ALL}")
        print(f"     Max DD: {Fore.RED}{abs(ranking['max_drawdown']):.2f}%{Style.RESET_ALL if ranking['max_drawdown'] else ''}")
        print()
    
    # Print statistics
    stats = summary["statistics"]
    print(f"{Fore.CYAN}Statistics:{Style.RESET_ALL}")
    print(f"  Total Analysts Tested: {Fore.GREEN}{summary['total_analysts']}{Style.RESET_ALL}")
    if stats["avg_return"]:
        print(f"  Average Return: {Fore.CYAN}{stats['avg_return']:.2f}%{Style.RESET_ALL}")
        print(f"  Best Return: {Fore.GREEN}{stats['max_return']:.2f}%{Style.RESET_ALL}")
        print(f"  Worst Return: {Fore.RED}{stats['min_return']:.2f}%{Style.RESET_ALL}")
    if stats["avg_sharpe"]:
        print(f"  Average Sharpe: {Fore.CYAN}{stats['avg_sharpe']:.2f}{Style.RESET_ALL}")
        print(f"  Best Sharpe: {Fore.GREEN}{stats['max_sharpe']:.2f}{Style.RESET_ALL}")
    
    # Print failed analysts
    if failed_analysts:
        print(f"\n{Fore.RED}Failed Analysts:{Style.RESET_ALL}")
        for analyst, error in failed_analysts:
            print(f"  {Fore.RED}✗{Style.RESET_ALL} {analyst}: {error}")
    
    # Print file locations
    print(f"\n{Fore.CYAN}Results saved to:{Style.RESET_ALL}")
    print(f"  Database: {Fore.GREEN}{storage.db_path}{Style.RESET_ALL}")
    print(f"  JSON Results: {Fore.GREEN}{storage.results_dir}/{Style.RESET_ALL}")
    print(f"  Summary: {Fore.GREEN}{storage.comparisons_dir}/{Style.RESET_ALL}")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare all analysts through backtesting")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker")
    parser.add_argument("--start-date", type=str, default="2021-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2024-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="Model name")
    parser.add_argument("--model-provider", type=str, default="DeepSeek", help="Model provider")
    parser.add_argument("--analysts", type=str, help="Comma-separated analyst keys (default: all)")
    parser.add_argument("--no-skip", action="store_true", help="Don't skip existing results")
    
    args = parser.parse_args()
    
    analysts = None
    if args.analysts:
        analysts = [a.strip() for a in args.analysts.split(",")]
    
    compare_analysts(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        model_name=args.model,
        model_provider=args.model_provider,
        analysts=analysts,
        skip_existing=not args.no_skip,
    )
