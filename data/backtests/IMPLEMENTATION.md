# Backtest Storage Implementation

## ✅ What Was Implemented

### 1. Database Model (`app/backend/database/models.py`)
- **AnalystComparison** table with comprehensive fields:
  - Test configuration (ticker, analyst, dates, model)
  - Performance metrics (returns, Sharpe, Sortino, drawdown)
  - Exposure statistics (avg long/short/gross/net exposure)
  - Trading statistics (trade counts)
  - File path references
  - Execution metadata (duration, cost)

### 2. Storage System (`data/backtests/storage.py`)
- **BacktestStorage** class with hybrid storage:
  - **SQLite Database**: Fast queries and rankings
  - **JSON Files**: Complete detailed data
  - **CSV Files**: Easy charting with pandas
  - **Portfolio Values JSON**: Separate file for quick loading

### 3. Comparison Script (`scripts/compare_analysts.py`)
- Automated comparison of all analysts
- Saves comprehensive data automatically
- Progress tracking and error handling
- Cost estimation
- Skip existing results option

### 4. Data Loading Utilities (`scripts/load_backtest_data.py`)
- Load specific analyst results
- List all available results
- Export to Excel
- Query database

### 5. Data Loading Utilities (`scripts/load_backtest_data.py`)
- Load specific analyst results
- List all available results
- Export to Excel
- Query database

## 📁 Directory Structure

```
data/backtests/
├── analyst_comparisons.db          # SQLite database
├── results/                         # Detailed results
│   ├── AAPL/
│   │   ├── warren_buffett_2021-01-01_2024-01-01.json
│   │   ├── warren_buffett_2021-01-01_2024-01-01.csv
│   │   └── warren_buffett_2021-01-01_2024-01-01_portfolio_values.json
│   └── MSFT/
│       └── ...
└── comparisons/                    # Summary files
    └── summary_AAPL_2021-01-01_2024-01-01.json
```

## 📊 Data Saved

### Database (SQLite)
- Summary metrics for fast queries
- Rankings and comparisons
- File path references

### JSON Files
- Complete metadata
- Performance metrics
- Exposure statistics
- Trading statistics
- **Full portfolio values array** (every day)
- Notes and tags

### CSV Files
- Daily portfolio value
- Daily return %
- Cumulative return %
- Long/short exposures
- Gross/net exposures
- Long/short ratio

## 🚀 Quick Start

### 1. Run Comparison

```bash
# Compare all analysts (3 years - quick screening)
poetry run python scripts/compare_analysts.py \
  --ticker AAPL \
  --start-date 2021-01-01 \
  --end-date 2024-01-01 \
  --model deepseek-chat \
  --model-provider DeepSeek
```

### 2. View Results

```bash
# List all results
poetry run python scripts/load_backtest_data.py \
  --ticker AAPL \
  --start-date 2021-01-01 \
  --end-date 2024-01-01 \
  --list

# Load specific analyst
poetry run python scripts/load_backtest_data.py \
  --ticker AAPL \
  --start-date 2021-01-01 \
  --end-date 2024-01-01 \
  --analyst warren_buffett
```

### 3. Load Data for Frontend

```bash
# List all results
poetry run python scripts/load_backtest_data.py \
  --ticker AAPL \
  --start-date 2021-01-01 \
  --end-date 2024-01-01 \
  --list

# Export to Excel (optional)
poetry run python scripts/load_backtest_data.py \
  --ticker AAPL \
  --start-date 2021-01-01 \
  --end-date 2024-01-01 \
  --export-excel results.xlsx
```

## 💻 Programmatic Usage

```python
from data.backtests.storage import BacktestStorage

storage = BacktestStorage()

# Save results (after running backtest)
file_paths = storage.save_backtest_results(
    ticker="AAPL",
    analyst="warren_buffett",
    start_date="2021-01-01",
    end_date="2024-01-01",
    backtester=backtester,
    metrics=metrics,
    model_name="deepseek-chat",
    model_provider="DeepSeek",
    cost=5.0,
    duration=120.5,
)

# Get rankings
rankings = storage.get_rankings(
    ticker="AAPL",
    start_date="2021-01-01",
    end_date="2024-01-01",
    order_by="sharpe_ratio",
    limit=10,
)

# Load data for charting
df = storage.get_csv_dataframe("data/backtests/results/AAPL/warren_buffett_2021-01-01_2024-01-01.csv")

# Get comparison summary
summary = storage.get_comparison_summary(
    ticker="AAPL",
    start_date="2021-01-01",
    end_date="2024-01-01",
)
```

## 📈 Data Available for Frontend

All data needed for comprehensive visualizations in your frontend:

1. **Portfolio Performance**: Daily portfolio values (in CSV and JSON)
2. **Returns**: Daily and cumulative returns (calculated in CSV)
3. **Exposures**: Long/short/gross/net exposures over time
4. **Drawdowns**: Can be calculated from portfolio values
5. **Metrics**: Sharpe, Sortino, max drawdown (in JSON and database)
6. **Comparisons**: Multiple analysts rankings (from database queries)

**CSV Format**: Ready for Chart.js, D3.js, or any charting library
**JSON Format**: Complete data structure for detailed views
**Database**: Fast API queries for rankings and comparisons

## 🔍 Features

- ✅ **Comprehensive Data**: Saves everything needed for analysis
- ✅ **Multiple Formats**: Database, JSON, CSV
- ✅ **Fast Queries**: SQLite for quick rankings
- ✅ **Easy Charting**: CSV optimized for pandas/matplotlib
- ✅ **Skip Existing**: Don't re-run completed tests
- ✅ **Cost Tracking**: Estimate and store API costs
- ✅ **Error Handling**: Continue on failures
- ✅ **Progress Tracking**: See status during runs

## 📝 Notes

- Large result files are excluded from git (see `.gitignore`)
- Database is created automatically on first use
- All timestamps in ISO format
- Portfolio values stored with full precision
- CSV files include calculated daily/cumulative returns
