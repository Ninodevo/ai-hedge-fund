import sys

from colorama import Fore, Style

from src.main import run_hedge_fund
from src.backtesting.engine import BacktestEngine
from src.backtesting.types import PerformanceMetrics
from src.cli.input import (
    parse_cli_inputs,
)


def run_backtest(backtester: BacktestEngine) -> PerformanceMetrics | None:
    """Run the backtest with graceful KeyboardInterrupt handling."""
    try:
        # Extract ticker and analyst for checkpointing (use first if multiple)
        tickers = backtester._tickers if hasattr(backtester, '_tickers') else []
        selected_analysts = backtester._selected_analysts if hasattr(backtester, '_selected_analysts') else []
        
        checkpoint_ticker = tickers[0] if len(tickers) == 1 and tickers else None
        checkpoint_analyst = selected_analysts[0] if len(selected_analysts) == 1 and selected_analysts else None
        
        # Setup CSV path for streaming if single ticker/analyst
        csv_path = None
        if checkpoint_ticker and checkpoint_analyst:
            from pathlib import Path
            csv_dir = Path("data/backtests/results") / checkpoint_ticker
            csv_dir.mkdir(parents=True, exist_ok=True)
            start_date = backtester._start_date if hasattr(backtester, '_start_date') else ""
            end_date = backtester._end_date if hasattr(backtester, '_end_date') else ""
            date_range = f"{start_date}_{end_date}"
            csv_path = csv_dir / f"{checkpoint_analyst}_{date_range}_streaming.csv"
        
        performance_metrics = backtester.run_backtest(
            ticker=checkpoint_ticker,
            analyst=checkpoint_analyst,
            csv_path=csv_path,
            resume=True,
        )
        print(f"\n{Fore.GREEN}Backtest completed successfully!{Style.RESET_ALL}")
        return performance_metrics
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Backtest interrupted by user.{Style.RESET_ALL}")
        
        # Try to show any partial results that were computed
        try:
            portfolio_values = backtester.get_portfolio_values()
            if len(portfolio_values) > 1:
                print(f"{Fore.GREEN}Partial results available.{Style.RESET_ALL}")
                
                # Show basic summary from the available portfolio values
                first_value = portfolio_values[0]["Portfolio Value"]
                last_value = portfolio_values[-1]["Portfolio Value"]
                total_return = ((last_value - first_value) / first_value) * 100
                
                print(f"{Fore.CYAN}Initial Portfolio Value: ${first_value:,.2f}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}Final Portfolio Value: ${last_value:,.2f}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}Total Return: {total_return:+.2f}%{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Could not generate partial results: {str(e)}{Style.RESET_ALL}")
        
        sys.exit(0)


### Run the Backtest #####
if __name__ == "__main__":
    inputs = parse_cli_inputs(
        description="Run backtesting simulation",
        require_tickers=False,
        default_months_back=1,
        include_graph_flag=False,
        include_reasoning_flag=False,
    )

    # Create and run the backtester
    backtester = BacktestEngine(
        agent=run_hedge_fund,
        tickers=inputs.tickers,
        start_date=inputs.start_date,
        end_date=inputs.end_date,
        initial_capital=inputs.initial_cash,
        model_name=inputs.model_name,
        model_provider=inputs.model_provider,
        selected_analysts=inputs.selected_analysts,
        initial_margin_requirement=inputs.margin_requirement,
    )

    # Run the backtest with graceful exit handling
    performance_metrics = run_backtest(backtester)
