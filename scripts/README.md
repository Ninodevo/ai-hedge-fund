# Scripts

Utility scripts for backtesting and data analysis.

## compare_analysts.py

Compare all analysts by running backtests and saving comprehensive results.

### Usage

```bash
# Compare all analysts on AAPL for 3 years (quick screening)
poetry run python scripts/compare_analysts.py \
  --ticker AAPL \
  --start-date 2021-01-01 \
  --end-date 2024-01-01 \
  --model deepseek-chat \
  --model-provider DeepSeek

# Compare specific analysts
poetry run python scripts/compare_analysts.py \
  --ticker AAPL \
  --start-date 2021-01-01 \
  --end-date 2024-01-01 \
  --analysts warren_buffett,cathie_wood,michael_burry

# Full 30-year comparison (expensive!)
poetry run python scripts/compare_analysts.py \
  --ticker AAPL \
  --start-date 1994-01-01 \
  --end-date 2024-01-01

# Don't skip existing results (re-run all)
poetry run python scripts/compare_analysts.py \
  --ticker AAPL \
  --start-date 2021-01-01 \
  --end-date 2024-01-01 \
  --no-skip
```

### Output

- Saves to database: `data/backtests/analyst_comparisons.db`
- Saves JSON: `data/backtests/results/{ticker}/{analyst}_{dates}.json`
- Saves CSV: `data/backtests/results/{ticker}/{analyst}_{dates}.csv`
- Generates summary: `data/backtests/comparisons/summary_{ticker}_{dates}.json`

## load_backtest_data.py

Load and analyze saved backtest data.

### Usage

```bash
# List all results for a ticker/date range
poetry run python scripts/load_backtest_data.py \
  --ticker AAPL \
  --start-date 2021-01-01 \
  --end-date 2024-01-01 \
  --list

# Load specific analyst results
poetry run python scripts/load_backtest_data.py \
  --ticker AAPL \
  --start-date 2021-01-01 \
  --end-date 2024-01-01 \
  --analyst warren_buffett

# Export to Excel
poetry run python scripts/load_backtest_data.py \
  --ticker AAPL \
  --start-date 2021-01-01 \
  --end-date 2024-01-01 \
  --export-excel comparison_results.xlsx
```

## Data Format for Frontend

All data is saved in formats ready for frontend consumption:

- **JSON Files**: Complete data structure with metadata, performance metrics, and full portfolio values array
- **CSV Files**: Daily data optimized for charting libraries (Chart.js, D3.js, Recharts, etc.)
- **Database**: Fast queries for rankings and comparisons via API endpoints

The frontend can:
- Load JSON files directly for detailed views
- Load CSV files for charting (includes daily_return, cumulative_return, exposures)
- Query database via API for rankings and comparisons
- Use summary JSON files for quick overviews
