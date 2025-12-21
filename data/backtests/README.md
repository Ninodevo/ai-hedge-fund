# Backtest Data Storage

This directory contains comprehensive backtest results from analyst comparisons.

## Directory Structure

```
data/backtests/
├── analyst_comparisons.db          # SQLite database with summary metrics
├── results/                         # Detailed results per ticker/analyst
│   ├── AAPL/
│   │   ├── warren_buffett_2021-01-01_2024-01-01.json
│   │   ├── warren_buffett_2021-01-01_2024-01-01.csv
│   │   └── warren_buffett_2021-01-01_2024-01-01_portfolio_values.json
│   └── MSFT/
│       └── ...
└── comparisons/                    # Summary comparisons
    └── summary_AAPL_2021-01-01_2024-01-01.json
```

## File Formats

### JSON Files (`*.json`)
Complete backtest data including:
- Metadata (ticker, analyst, dates, model info)
- Performance metrics (Sharpe, Sortino, max drawdown)
- Exposure statistics (avg long/short exposure)
- Trading statistics
- Full portfolio values array

### CSV Files (`*.csv`)
Daily portfolio data for easy charting:
- `date`: Trading date
- `portfolio_value`: Total portfolio value
- `daily_return`: Daily return percentage
- `cumulative_return`: Cumulative return percentage
- `long_exposure`: Long position exposure
- `short_exposure`: Short position exposure
- `gross_exposure`: Total exposure
- `net_exposure`: Net exposure
- `long_short_ratio`: Long/Short ratio

### Database (`analyst_comparisons.db`)
SQLite database with summary metrics for fast queries:
- Quick rankings and comparisons
- Performance metrics
- File path references

## Usage

### Run Comparison

```bash
# Compare all analysts on AAPL for 3 years
poetry run python scripts/compare_analysts.py \
  --ticker AAPL \
  --start-date 2021-01-01 \
  --end-date 2024-01-01 \
  --model deepseek-chat \
  --model-provider DeepSeek

# Compare specific analysts
poetry run python scripts/compare_analysts.py \
  --ticker AAPL \
  --analysts warren_buffett,cathie_wood,michael_burry
```

### Load Data

```python
from data.backtests.storage import BacktestStorage

storage = BacktestStorage()

# Get rankings
rankings = storage.get_rankings(ticker="AAPL", start_date="2021-01-01", end_date="2024-01-01")

# Load specific result
data = storage.load_results("data/backtests/results/AAPL/warren_buffett_2021-01-01_2024-01-01.json")

# Load CSV as DataFrame for charting
df = storage.get_csv_dataframe("data/backtests/results/AAPL/warren_buffett_2021-01-01_2024-01-01.csv")

# Get comparison summary
summary = storage.get_comparison_summary(ticker="AAPL", start_date="2021-01-01", end_date="2024-01-01")
```

### Frontend Integration

The data is saved in formats ready for frontend consumption:

**JSON Files**: Complete data structure with metadata and portfolio values
**CSV Files**: Daily data optimized for charting libraries (Chart.js, D3.js, etc.)
**Database**: Fast queries for rankings and comparisons

Example frontend usage:
```javascript
// Load CSV for charting
fetch('/data/backtests/results/AAPL/warren_buffett_2021-01-01_2024-01-01.csv')
  .then(response => response.text())
  .then(csv => {
    // Parse CSV and create charts
  });

// Load JSON for detailed data
fetch('/data/backtests/results/AAPL/warren_buffett_2021-01-01_2024-01-01.json')
  .then(response => response.json())
  .then(data => {
    // Use portfolio_values array for charts
    // Use performance_metrics for summary
  });
```

## Data Access

### Query Database

```python
import sqlite3

conn = sqlite3.connect("data/backtests/analyst_comparisons.db")
cursor = conn.execute("""
    SELECT analyst_name, total_return, sharpe_ratio 
    FROM analyst_comparisons 
    WHERE ticker = 'AAPL' 
    ORDER BY sharpe_ratio DESC 
    LIMIT 5
""")
for row in cursor.fetchall():
    print(row)
```

### Export to Excel

```bash
poetry run python scripts/load_backtest_data.py \
  --ticker AAPL \
  --start-date 2021-01-01 \
  --end-date 2024-01-01 \
  --export-excel comparison_results.xlsx
```

## Notes

- Large result files are excluded from git (see `.gitignore`)
- Database and JSON files are created automatically
- CSV files are optimized for pandas/Excel import
- All timestamps are in ISO format
- Portfolio values are stored with full precision
