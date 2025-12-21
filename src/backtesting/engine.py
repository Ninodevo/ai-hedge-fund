from __future__ import annotations

from datetime import datetime
from typing import Sequence, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import csv

import pandas as pd
from dateutil.relativedelta import relativedelta

from .controller import AgentController
from .trader import TradeExecutor
from .metrics import PerformanceMetricsCalculator
from .portfolio import Portfolio
from .types import PerformanceMetrics, PortfolioValuePoint
from .valuation import calculate_portfolio_value, compute_exposures
from .output import OutputBuilder
from .benchmarks import BenchmarkCalculator

from src.tools.api import (
    get_company_news,
    get_price_data,
    get_prices,
    get_financial_metrics,
    get_insider_trades,
)


class BacktestEngine:
    """Coordinates the backtest loop using the new components.

    This implementation mirrors the semantics of src/backtester.py while
    avoiding any changes to that file. It orchestrates agent decisions,
    trade execution, valuation, exposures and performance metrics.
    """

    def __init__(
        self,
        *,
        agent,
        tickers: list[str],
        start_date: str,
        end_date: str,
        initial_capital: float,
        model_name: str,
        model_provider: str,
        selected_analysts: list[str] | None,
        initial_margin_requirement: float,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_interval: int = 20,  # Every 20 business days (~monthly)
        enable_csv_streaming: bool = True,
    ) -> None:
        self._agent = agent
        self._tickers = tickers
        self._start_date = start_date
        self._end_date = end_date
        self._initial_capital = float(initial_capital)
        self._model_name = model_name
        self._model_provider = model_provider
        self._selected_analysts = selected_analysts
        self._checkpoint_interval = checkpoint_interval
        self._enable_csv_streaming = enable_csv_streaming

        # Setup checkpoint directory (use absolute path relative to project root)
        if checkpoint_dir is None:
            # Try to find project root (where pyproject.toml or .git exists)
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent  # Go up from src/backtesting/engine.py
            checkpoint_dir = project_root / "data" / "backtests" / "checkpoints"
        self._checkpoint_dir = Path(checkpoint_dir).resolve()
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Checkpoint directory: {self._checkpoint_dir}")

        self._portfolio = Portfolio(
            tickers=tickers,
            initial_cash=initial_capital,
            margin_requirement=initial_margin_requirement,
        )
        self._executor = TradeExecutor()
        self._agent_controller = AgentController()
        self._perf = PerformanceMetricsCalculator()
        self._results = OutputBuilder(initial_capital=self._initial_capital)

        # Benchmark calculator
        self._benchmark = BenchmarkCalculator()

        self._portfolio_values: list[PortfolioValuePoint] = []
        self._table_rows: list[list] = []
        self._performance_metrics: PerformanceMetrics = {
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "max_drawdown": None,
            "long_short_ratio": None,
            "gross_exposure": None,
            "net_exposure": None,
        }
        
        # CSV streaming
        self._csv_file: Optional[csv.DictWriter] = None
        self._csv_file_handle: Optional[object] = None

    def _prefetch_data(self) -> None:
        """Prefetch all data in parallel for faster initialization."""
        end_date_dt = datetime.strptime(self._end_date, "%Y-%m-%d")
        start_date_dt = end_date_dt - relativedelta(years=1)
        start_date_str = start_date_dt.strftime("%Y-%m-%d")

        def fetch_ticker_data(ticker: str) -> tuple[str, bool]:
            """Fetch all data for a single ticker."""
            try:
                get_prices(ticker, start_date_str, self._end_date)
                get_financial_metrics(ticker, self._end_date, limit=10)
                get_insider_trades(ticker, self._end_date, start_date=self._start_date, limit=1000)
                get_company_news(ticker, self._end_date, start_date=self._start_date, limit=1000)
                return ticker, True
            except Exception as e:
                print(f"Warning: Error prefetching data for {ticker}: {e}")
                return ticker, False

        # Fetch all tickers in parallel
        print(f"Prefetching data for {len(self._tickers)} ticker(s) in parallel...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(fetch_ticker_data, ticker): ticker for ticker in self._tickers}
            
            completed = 0
            for future in as_completed(futures):
                ticker, success = future.result()
                completed += 1
                if success:
                    print(f"  ✓ Prefetched data for {ticker} ({completed}/{len(self._tickers)})")
                else:
                    print(f"  ✗ Failed to prefetch {ticker} ({completed}/{len(self._tickers)})")
        
        # Preload data for SPY for benchmark comparison
        print("Prefetching SPY benchmark data...")
        get_prices("SPY", self._start_date, self._end_date)
        print("Data prefetch complete.")

    def _get_checkpoint_path(self, ticker: str, analyst: str) -> Path:
        """Get checkpoint file path for a specific backtest"""
        checkpoint_name = f"{ticker}_{analyst}_{self._start_date}_{self._end_date}.json"
        return self._checkpoint_dir / checkpoint_name

    def _save_checkpoint(self, current_date: datetime, ticker: str, analyst: str) -> None:
        """Save checkpoint state for resume capability"""
        try:
            checkpoint_path = self._get_checkpoint_path(ticker, analyst)
            
            # Ensure directory exists
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            checkpoint_data = {
                "ticker": ticker,
                "analyst": analyst,
                "start_date": self._start_date,
                "end_date": self._end_date,
                "last_processed_date": current_date.strftime("%Y-%m-%d"),
                "initial_capital": self._initial_capital,
                "model_name": self._model_name,
                "model_provider": self._model_provider,
                "portfolio": self._portfolio.get_snapshot(),
                "portfolio_values": [
                    {
                        "Date": pv["Date"].isoformat() if hasattr(pv["Date"], "isoformat") else str(pv["Date"]),
                        "Portfolio Value": pv["Portfolio Value"],
                        "Long Exposure": pv.get("Long Exposure", 0),
                        "Short Exposure": pv.get("Short Exposure", 0),
                        "Gross Exposure": pv.get("Gross Exposure", 0),
                        "Net Exposure": pv.get("Net Exposure", 0),
                        "Long/Short Ratio": pv.get("Long/Short Ratio"),
                    }
                    for pv in self._portfolio_values
                ],
                "performance_metrics": self._performance_metrics,
                "checkpoint_date": datetime.now().isoformat(),
            }
            
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            
            print(f"  ✓ Checkpoint saved: {current_date.strftime('%Y-%m-%d')} -> {checkpoint_path}")
        except Exception as e:
            print(f"  ✗ Error saving checkpoint: {e}")
            import traceback
            traceback.print_exc()

    def _load_checkpoint(self, ticker: str, analyst: str) -> Optional[Dict]:
        """Load checkpoint if exists, return None if not found"""
        checkpoint_path = self._get_checkpoint_path(ticker, analyst)
        
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, "r") as f:
                checkpoint_data = json.load(f)
            
            # Validate checkpoint matches current run
            if (checkpoint_data.get("ticker") != ticker or
                checkpoint_data.get("analyst") != analyst or
                checkpoint_data.get("start_date") != self._start_date or
                checkpoint_data.get("end_date") != self._end_date):
                print(f"  ⚠ Checkpoint mismatch, starting fresh")
                return None
            
            return checkpoint_data
        except Exception as e:
            print(f"  ⚠ Error loading checkpoint: {e}, starting fresh")
            return None

    def _restore_from_checkpoint(self, checkpoint_data: Dict) -> datetime:
        """Restore portfolio state from checkpoint, return last processed date"""
        # Restore portfolio
        portfolio_snapshot = checkpoint_data["portfolio"]
        self._portfolio = Portfolio(
            tickers=self._tickers,
            initial_cash=portfolio_snapshot["cash"],
            margin_requirement=portfolio_snapshot["margin_requirement"],
        )
        # Restore portfolio state from snapshot
        self._portfolio.restore_from_snapshot(portfolio_snapshot)
        
        # Restore portfolio values
        self._portfolio_values = []
        for pv in checkpoint_data["portfolio_values"]:
            date_str = pv["Date"]
            if isinstance(date_str, str):
                date_obj = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            else:
                date_obj = pd.Timestamp(date_str).to_pydatetime()
            
            self._portfolio_values.append({
                "Date": date_obj,
                "Portfolio Value": pv["Portfolio Value"],
                "Long Exposure": pv.get("Long Exposure", 0),
                "Short Exposure": pv.get("Short Exposure", 0),
                "Gross Exposure": pv.get("Gross Exposure", 0),
                "Net Exposure": pv.get("Net Exposure", 0),
                "Long/Short Ratio": pv.get("Long/Short Ratio"),
            })
        
        # Restore performance metrics
        self._performance_metrics = checkpoint_data.get("performance_metrics", self._performance_metrics)
        
        last_date_str = checkpoint_data["last_processed_date"]
        last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
        
        print(f"  ✓ Resumed from checkpoint: {last_date_str}")
        return last_date

    def _init_csv_streaming(self, csv_path: Path, resume_date: Optional[datetime] = None) -> None:
        """Initialize CSV file for streaming"""
        if not self._enable_csv_streaming:
            return
        
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = csv_path.exists()
        
        # If resuming, check if we need to truncate CSV to resume point
        if resume_date and file_exists:
            # Read existing CSV and find last date
            try:
                import csv as csv_module
                with open(csv_path, "r") as f:
                    reader = csv_module.DictReader(f)
                    last_date_in_csv = None
                    for row in reader:
                        last_date_in_csv = row.get("date")
                    
                    # If resume date is after last CSV date, truncate CSV
                    if last_date_in_csv:
                        last_csv_date = datetime.strptime(last_date_in_csv, "%Y-%m-%d")
                        if resume_date <= last_csv_date:
                            # Truncate CSV to resume point
                            with open(csv_path, "r") as f:
                                lines = f.readlines()
                            # Keep header and lines before resume date
                            header = lines[0] if lines else ""
                            keep_lines = [header]
                            for line in lines[1:]:
                                if line.startswith(resume_date.strftime("%Y-%m-%d")):
                                    break
                                keep_lines.append(line)
                            with open(csv_path, "w") as f:
                                f.writelines(keep_lines)
            except Exception:
                # If error reading CSV, start fresh
                file_exists = False
        
        self._csv_file_handle = open(csv_path, "a", newline="")
        self._csv_file = csv.DictWriter(
            self._csv_file_handle,
            fieldnames=[
                "date", "portfolio_value", "daily_return", "cumulative_return",
                "long_exposure", "short_exposure", "gross_exposure", "net_exposure", "long_short_ratio"
            ]
        )
        
        if not file_exists:
            self._csv_file.writeheader()
            self._csv_file_handle.flush()

    def _stream_csv_row(self, current_date: datetime, total_value: float, exposures: Dict, prev_value: Optional[float] = None) -> None:
        """Stream a single row to CSV"""
        if not self._enable_csv_streaming or self._csv_file is None:
            return
        
        date_str = current_date.strftime("%Y-%m-%d")
        initial_value = self._portfolio_values[0]["Portfolio Value"] if self._portfolio_values else self._initial_capital
        
        daily_return = 0.0
        if prev_value is not None and prev_value > 0:
            daily_return = ((total_value - prev_value) / prev_value) * 100
        
        cumulative_return = 0.0
        if initial_value > 0:
            cumulative_return = ((total_value - initial_value) / initial_value) * 100
        
        row = {
            "date": date_str,
            "portfolio_value": total_value,
            "daily_return": daily_return,
            "cumulative_return": cumulative_return,
            "long_exposure": exposures.get("Long Exposure", 0),
            "short_exposure": exposures.get("Short Exposure", 0),
            "gross_exposure": exposures.get("Gross Exposure", 0),
            "net_exposure": exposures.get("Net Exposure", 0),
            "long_short_ratio": exposures.get("Long/Short Ratio", 0),
        }
        
        self._csv_file.writerow(row)
        self._csv_file_handle.flush()  # Ensure data is written immediately

    def _close_csv_streaming(self) -> None:
        """Close CSV file"""
        if self._csv_file_handle:
            self._csv_file_handle.close()
            self._csv_file_handle = None
            self._csv_file = None

    def run_backtest(
        self,
        ticker: Optional[str] = None,
        analyst: Optional[str] = None,
        csv_path: Optional[Path] = None,
        resume: bool = True,
    ) -> PerformanceMetrics:
        """Run backtest with checkpointing and CSV streaming support"""
        # Debug: Show checkpointing status
        if ticker and analyst:
            print(f"\n  Checkpointing enabled:")
            print(f"    Ticker: {ticker}")
            print(f"    Analyst: {analyst}")
            print(f"    Interval: Every {self._checkpoint_interval} business days")
            print(f"    Directory: {self._checkpoint_dir}")
            print(f"    Resume: {resume}")
        else:
            print(f"\n  ⚠ Checkpointing disabled: ticker={ticker}, analyst={analyst}")
        
        self._prefetch_data()

        # Initialize CSV streaming if enabled (after resume check)
        resume_date_for_csv: Optional[datetime] = None

        dates = pd.date_range(self._start_date, self._end_date, freq="B")
        
        # Try to resume from checkpoint
        resume_date: Optional[datetime] = None
        if resume and ticker and analyst:
            checkpoint_data = self._load_checkpoint(ticker, analyst)
            if checkpoint_data:
                resume_date = self._restore_from_checkpoint(checkpoint_data)
                resume_date_for_csv = resume_date
                # Filter dates to only process after resume point
                dates = dates[dates > pd.Timestamp(resume_date)]
                print(f"  Resuming from {resume_date.strftime('%Y-%m-%d')}, {len(dates)} days remaining")
        
        # Initialize CSV streaming if enabled (after resume check)
        if csv_path and self._enable_csv_streaming:
            self._init_csv_streaming(csv_path, resume_date_for_csv)

        if len(dates) > 0:
            if not resume_date:  # Only initialize if not resuming
                self._portfolio_values = [
                    {"Date": dates[0], "Portfolio Value": self._initial_capital}
                ]
        else:
            self._portfolio_values = []

        prev_value = self._portfolio_values[-1]["Portfolio Value"] if self._portfolio_values else self._initial_capital

        for i, current_date in enumerate(dates):
            lookback_start = (current_date - relativedelta(months=1)).strftime("%Y-%m-%d")
            current_date_str = current_date.strftime("%Y-%m-%d")
            previous_date_str = (current_date - relativedelta(days=1)).strftime("%Y-%m-%d")
            if lookback_start == current_date_str:
                continue

            try:
                current_prices: Dict[str, float] = {}
                missing_data = False
                for ticker_symbol in self._tickers:
                    try:
                        price_data = get_price_data(ticker_symbol, previous_date_str, current_date_str)
                        if price_data.empty:
                            missing_data = True
                            break
                        current_prices[ticker_symbol] = float(price_data.iloc[-1]["close"])
                    except Exception:
                        missing_data = True
                        break
                if missing_data:
                    continue
            except Exception:
                continue

            agent_output = self._agent_controller.run_agent(
                self._agent,
                tickers=self._tickers,
                start_date=lookback_start,
                end_date=current_date_str,
                portfolio=self._portfolio,
                model_name=self._model_name,
                model_provider=self._model_provider,
                selected_analysts=self._selected_analysts,
            )
            decisions = agent_output["decisions"]

            executed_trades: Dict[str, int] = {}
            for ticker_symbol in self._tickers:
                d = decisions.get(ticker_symbol, {"action": "hold", "quantity": 0})
                action = d.get("action", "hold")
                qty = d.get("quantity", 0)
                executed_qty = self._executor.execute_trade(ticker_symbol, action, qty, current_prices[ticker_symbol], self._portfolio)
                executed_trades[ticker_symbol] = executed_qty

            total_value = calculate_portfolio_value(self._portfolio, current_prices)
            exposures = compute_exposures(self._portfolio, current_prices)

            point: PortfolioValuePoint = {
                "Date": current_date,
                "Portfolio Value": total_value,
                "Long Exposure": exposures["Long Exposure"],
                "Short Exposure": exposures["Short Exposure"],
                "Gross Exposure": exposures["Gross Exposure"],
                "Net Exposure": exposures["Net Exposure"],
                "Long/Short Ratio": exposures["Long/Short Ratio"],
            }
            self._portfolio_values.append(point)
            
            # Stream to CSV (fast, append-only)
            self._stream_csv_row(current_date, total_value, exposures, prev_value)
            prev_value = total_value
            
            # Build daily rows (stateless usage)
            rows = self._results.build_day_rows(
                date_str=current_date_str,
                tickers=self._tickers,
                agent_output=agent_output,
                executed_trades=executed_trades,
                current_prices=current_prices,
                portfolio=self._portfolio,
                performance_metrics=self._performance_metrics,
                total_value=total_value,
                benchmark_return_pct=self._benchmark.get_return_pct("SPY", self._start_date, current_date_str),
            )
            # Prepend today's rows to historical rows so latest day is on top
            self._table_rows = rows + self._table_rows
            # Print full history with latest day first (matches backtester.py behavior)
            self._results.print_rows(self._table_rows)

            # Update performance metrics after printing (match original timing)
            if len(self._portfolio_values) > 3:
                computed = self._perf.compute_metrics(self._portfolio_values)
                if computed:
                    self._performance_metrics.update(computed)

            # Save checkpoint every N days
            if ticker and analyst and (i + 1) % self._checkpoint_interval == 0:
                self._save_checkpoint(current_date, ticker, analyst)

        # Close CSV streaming
        self._close_csv_streaming()
        
        # Always save final checkpoint at end (if ticker/analyst provided)
        if ticker and analyst:
            if len(dates) > 0:
                try:
                    self._save_checkpoint(dates[-1], ticker, analyst)
                    print(f"  ✓ Final checkpoint saved")
                except Exception as e:
                    print(f"  ✗ Error saving final checkpoint: {e}")
            else:
                print(f"  ⚠ No dates to process, skipping checkpoint")
        else:
            print(f"  ⚠ Checkpoint not saved: ticker={ticker}, analyst={analyst}")

        return self._performance_metrics

    def get_portfolio_values(self) -> Sequence[PortfolioValuePoint]:
        return list(self._portfolio_values)


