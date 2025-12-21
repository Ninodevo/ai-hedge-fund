"""Comprehensive storage system for backtest results - hybrid approach with database, JSON, and CSV"""

import json
import csv
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    import pandas as pd
except ImportError:
    pd = None  # Optional dependency

from src.backtesting.types import PerformanceMetrics, PortfolioValuePoint


class BacktestStorage:
    """Hybrid storage system for backtest results"""
    
    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize storage with base directory"""
        if base_dir is None:
            # Default to project root / data/backtests
            self.base_dir = Path(__file__).parent
        else:
            self.base_dir = Path(base_dir)
        
        # Directory structure
        self.results_dir = self.base_dir / "results"
        self.comparisons_dir = self.base_dir / "comparisons"
        self.db_path = self.base_dir / "analyst_comparisons.db"
        
        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.comparisons_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with schema"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyst_comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP,
                ticker TEXT NOT NULL,
                analyst_name TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_provider TEXT NOT NULL,
                initial_capital REAL NOT NULL DEFAULT 100000.0,
                total_return REAL,
                total_return_absolute REAL,
                final_value REAL,
                sharpe_ratio REAL,
                sortino_ratio REAL,
                max_drawdown REAL,
                max_drawdown_date TEXT,
                avg_long_exposure REAL,
                avg_short_exposure REAL,
                avg_gross_exposure REAL,
                avg_net_exposure REAL,
                avg_long_short_ratio REAL,
                total_trades INTEGER,
                buy_trades INTEGER,
                sell_trades INTEGER,
                short_trades INTEGER,
                cover_trades INTEGER,
                results_json_path TEXT,
                results_csv_path TEXT,
                portfolio_values_json_path TEXT,
                test_duration_seconds REAL,
                estimated_cost REAL,
                total_business_days INTEGER,
                notes TEXT,
                tags TEXT,
                UNIQUE(ticker, analyst_name, start_date, end_date)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_analyst ON analyst_comparisons(analyst_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON analyst_comparisons(ticker)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dates ON analyst_comparisons(start_date, end_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sharpe ON analyst_comparisons(sharpe_ratio)")
        conn.commit()
        conn.close()
    
    def save_backtest_results(
        self,
        ticker: str,
        analyst: str,
        start_date: str,
        end_date: str,
        backtester: BacktestEngine,
        metrics: PerformanceMetrics,
        model_name: str,
        model_provider: str,
        cost: Optional[float] = None,
        duration: Optional[float] = None,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Save complete backtest results to database, JSON, and CSV
        
        Returns dict with file paths
        """
        # Get portfolio values
        portfolio_values = backtester.get_portfolio_values()
        
        if not portfolio_values:
            raise ValueError("No portfolio values to save")
        
        # Calculate summary statistics
        initial_capital = portfolio_values[0]["Portfolio Value"]
        final_value = portfolio_values[-1]["Portfolio Value"]
        total_return_pct = ((final_value - initial_capital) / initial_capital * 100) if initial_capital > 0 else 0
        total_return_abs = final_value - initial_capital
        
        # Calculate average exposures
        avg_long = sum(pv.get("Long Exposure", 0) for pv in portfolio_values) / len(portfolio_values) if portfolio_values else 0
        avg_short = sum(pv.get("Short Exposure", 0) for pv in portfolio_values) / len(portfolio_values) if portfolio_values else 0
        avg_gross = sum(pv.get("Gross Exposure", 0) for pv in portfolio_values) / len(portfolio_values) if portfolio_values else 0
        avg_net = sum(pv.get("Net Exposure", 0) for pv in portfolio_values) / len(portfolio_values) if portfolio_values else 0
        ls_ratios = [pv.get("Long/Short Ratio", 0) for pv in portfolio_values if pv.get("Long/Short Ratio") is not None]
        avg_ls_ratio = sum(ls_ratios) / len(ls_ratios) if ls_ratios else None
        
        # Count trades (would need to track this during backtest, for now estimate from portfolio changes)
        # This is a placeholder - you'd need to track trades during execution
        total_trades = 0
        buy_trades = 0
        sell_trades = 0
        short_trades = 0
        cover_trades = 0
        
        # Create ticker directory
        ticker_dir = self.results_dir / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filenames
        date_range = f"{start_date}_{end_date}"
        base_filename = f"{analyst}_{date_range}"
        
        json_path = ticker_dir / f"{base_filename}.json"
        csv_path = ticker_dir / f"{base_filename}.csv"
        portfolio_json_path = ticker_dir / f"{base_filename}_portfolio_values.json"
        
        # Prepare comprehensive JSON data
        result_data = {
            "metadata": {
                "ticker": ticker,
                "analyst": analyst,
                "start_date": start_date,
                "end_date": end_date,
                "model_name": model_name,
                "model_provider": model_provider,
                "initial_capital": initial_capital,
                "created_at": datetime.now().isoformat(),
                "test_duration_seconds": duration,
                "estimated_cost": cost,
                "total_business_days": len(portfolio_values),
            },
            "performance_metrics": {
                "total_return_pct": total_return_pct,
                "total_return_absolute": total_return_abs,
                "final_value": final_value,
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "sortino_ratio": metrics.get("sortino_ratio"),
                "max_drawdown": metrics.get("max_drawdown"),
                "max_drawdown_date": metrics.get("max_drawdown_date"),
                "long_short_ratio": metrics.get("long_short_ratio"),
                "gross_exposure": metrics.get("gross_exposure"),
                "net_exposure": metrics.get("net_exposure"),
            },
            "exposure_statistics": {
                "avg_long_exposure": avg_long,
                "avg_short_exposure": avg_short,
                "avg_gross_exposure": avg_gross,
                "avg_net_exposure": avg_net,
                "avg_long_short_ratio": avg_ls_ratio,
            },
            "trading_statistics": {
                "total_trades": total_trades,
                "buy_trades": buy_trades,
                "sell_trades": sell_trades,
                "short_trades": short_trades,
                "cover_trades": cover_trades,
            },
            "portfolio_values": [
                {
                    "date": pv["Date"].isoformat() if hasattr(pv["Date"], "isoformat") else str(pv["Date"]),
                    "portfolio_value": pv["Portfolio Value"],
                    "long_exposure": pv.get("Long Exposure", 0),
                    "short_exposure": pv.get("Short Exposure", 0),
                    "gross_exposure": pv.get("Gross Exposure", 0),
                    "net_exposure": pv.get("Net Exposure", 0),
                    "long_short_ratio": pv.get("Long/Short Ratio"),
                }
                for pv in portfolio_values
            ],
            "notes": notes,
            "tags": tags or [],
        }
        
        # Save JSON file
        with open(json_path, "w") as f:
            json.dump(result_data, f, indent=2, default=str)
        
        # Save portfolio values as separate JSON (for easy loading)
        portfolio_data = {
            "metadata": result_data["metadata"],
            "portfolio_values": result_data["portfolio_values"],
        }
        with open(portfolio_json_path, "w") as f:
            json.dump(portfolio_data, f, indent=2, default=str)
        
        # Save CSV file (for easy charting)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "date", "portfolio_value", "daily_return", "cumulative_return",
                "long_exposure", "short_exposure", "gross_exposure", "net_exposure", "long_short_ratio"
            ])
            writer.writeheader()
            
            prev_value = initial_capital
            for i, pv in enumerate(portfolio_values):
                date_str = pv["Date"].isoformat() if hasattr(pv["Date"], "isoformat") else str(pv["Date"])
                current_value = pv["Portfolio Value"]
                daily_return = ((current_value - prev_value) / prev_value * 100) if prev_value > 0 else 0
                cumulative_return = ((current_value - initial_capital) / initial_capital * 100) if initial_capital > 0 else 0
                
                writer.writerow({
                    "date": date_str,
                    "portfolio_value": current_value,
                    "daily_return": daily_return,
                    "cumulative_return": cumulative_return,
                    "long_exposure": pv.get("Long Exposure", 0),
                    "short_exposure": pv.get("Short Exposure", 0),
                    "gross_exposure": pv.get("Gross Exposure", 0),
                    "net_exposure": pv.get("Net Exposure", 0),
                    "long_short_ratio": pv.get("Long/Short Ratio", 0),
                })
                prev_value = current_value
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO analyst_comparisons 
            (ticker, analyst_name, start_date, end_date, model_name, model_provider,
             initial_capital, total_return, total_return_absolute, final_value,
             sharpe_ratio, sortino_ratio, max_drawdown, max_drawdown_date,
             avg_long_exposure, avg_short_exposure, avg_gross_exposure, avg_net_exposure, avg_long_short_ratio,
             total_trades, buy_trades, sell_trades, short_trades, cover_trades,
             results_json_path, results_csv_path, portfolio_values_json_path,
             test_duration_seconds, estimated_cost, total_business_days, notes, tags, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            ticker, analyst, start_date, end_date, model_name, model_provider,
            initial_capital, total_return_pct, total_return_abs, final_value,
            metrics.get("sharpe_ratio"), metrics.get("sortino_ratio"),
            metrics.get("max_drawdown"), metrics.get("max_drawdown_date"),
            avg_long, avg_short, avg_gross, avg_net, avg_ls_ratio,
            total_trades, buy_trades, sell_trades, short_trades, cover_trades,
            str(json_path), str(csv_path), str(portfolio_json_path),
            duration, cost, len(portfolio_values), notes, json.dumps(tags) if tags else None
        ))
        conn.commit()
        conn.close()
        
        return {
            "json": str(json_path),
            "csv": str(csv_path),
            "portfolio_json": str(portfolio_json_path),
            "database_id": None,  # Could return the inserted ID if needed
        }
    
    def get_rankings(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        order_by: str = "sharpe_ratio",
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get ranked analyst comparisons"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        query = "SELECT * FROM analyst_comparisons WHERE 1=1"
        params = []
        
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        if start_date:
            query += " AND start_date = ?"
            params.append(start_date)
        if end_date:
            query += " AND end_date = ?"
            params.append(end_date)
        
        # Validate order_by
        valid_columns = [
            "sharpe_ratio", "sortino_ratio", "total_return", "max_drawdown",
            "final_value", "created_at"
        ]
        if order_by not in valid_columns:
            order_by = "sharpe_ratio"
        
        query += f" ORDER BY {order_by} DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        
        results = []
        for i, row in enumerate(rows, 1):
            result = dict(row)
            result["rank"] = i
            results.append(result)
        
        conn.close()
        return results
    
    def get_comparison_summary(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate summary comparison across all analysts"""
        rankings = self.get_rankings(ticker=ticker, start_date=start_date, end_date=end_date)
        
        if not rankings:
            return {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "total_analysts": 0,
                "rankings": [],
                "statistics": {},
            }
        
        # Calculate statistics
        returns = [r["total_return"] for r in rankings if r["total_return"] is not None]
        sharpe_ratios = [r["sharpe_ratio"] for r in rankings if r["sharpe_ratio"] is not None]
        
        summary = {
            "ticker": ticker or "all",
            "start_date": start_date or "all",
            "end_date": end_date or "all",
            "created_at": datetime.now().isoformat(),
            "total_analysts": len(rankings),
            "rankings": [
                {
                    "rank": r["rank"],
                    "analyst": r["analyst_name"],
                    "ticker": r["ticker"],
                    "total_return": r["total_return"],
                    "sharpe_ratio": r["sharpe_ratio"],
                    "sortino_ratio": r["sortino_ratio"],
                    "max_drawdown": r["max_drawdown"],
                    "final_value": r["final_value"],
                    "results_json_path": r["results_json_path"],
                    "results_csv_path": r["results_csv_path"],
                }
                for r in rankings
            ],
            "statistics": {
                "avg_return": sum(returns) / len(returns) if returns else None,
                "max_return": max(returns) if returns else None,
                "min_return": min(returns) if returns else None,
                "avg_sharpe": sum(sharpe_ratios) / len(sharpe_ratios) if sharpe_ratios else None,
                "max_sharpe": max(sharpe_ratios) if sharpe_ratios else None,
                "min_sharpe": min(sharpe_ratios) if sharpe_ratios else None,
            },
        }
        
        # Save summary to JSON
        summary_filename = f"summary_{ticker or 'all'}_{start_date or 'all'}_{end_date or 'all'}.json"
        summary_path = self.comparisons_dir / summary_filename
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary
    
    def load_portfolio_values(self, json_path: str) -> List[Dict[str, Any]]:
        """Load portfolio values from JSON file"""
        with open(json_path, "r") as f:
            data = json.load(f)
            return data.get("portfolio_values", [])
    
    def load_results(self, json_path: str) -> Dict[str, Any]:
        """Load complete results from JSON file"""
        with open(json_path, "r") as f:
            return json.load(f)
    
    def get_csv_dataframe(self, csv_path: str):
        """Load CSV as pandas DataFrame for easy charting"""
        if pd is None:
            raise ImportError("pandas is required for get_csv_dataframe. Install with: pip install pandas")
        return pd.read_csv(csv_path, parse_dates=["date"])
