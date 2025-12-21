# Checkpoint Debugging Guide

## Fixes Applied

1. **Error Handling**: Added try/except blocks to catch and report checkpoint save errors
2. **Absolute Paths**: Checkpoint directory now uses absolute path relative to project root
3. **Debug Output**: Added logging to show checkpoint status and location
4. **Better Warnings**: Clear messages if checkpointing can't be enabled

## How to Verify Checkpoints Are Being Saved

### 1. Check Debug Output

When you run a backtest, you should see:

```
  Checkpointing enabled:
    Ticker: AAPL
    Analyst: warren_buffett
    Interval: Every 20 business days
    Directory: /path/to/project/data/backtests/checkpoints
    Resume: True
```

### 2. Check for Checkpoint Files

After running a backtest, check:

```bash
ls -la data/backtests/checkpoints/
```

You should see files like:
```
AAPL_warren_buffett_2021-01-01_2024-01-01.json
```

### 3. Checkpoint Save Messages

During the backtest, you should see:
```
  ✓ Checkpoint saved: 2021-02-01 -> /path/to/checkpoint.json
```

And at the end:
```
  ✓ Final checkpoint saved
```

## Common Issues

### Issue: "Checkpoint not saved: ticker=None, analyst=None"

**Cause**: `ticker` and `analyst` parameters not passed to `run_backtest()`

**Fix**: Make sure you're calling:
```python
backtester.run_backtest(
    ticker="AAPL",
    analyst="warren_buffett",
    csv_path=csv_path,
    resume=True,
)
```

### Issue: Checkpoints only save every 20 days

**Cause**: `checkpoint_interval=20` means checkpoints only save every 20 business days

**Fix**: For shorter runs, checkpoints will still save at the end. For more frequent saves:
```python
BacktestEngine(
    ...,
    checkpoint_interval=5,  # Save every 5 days instead
)
```

### Issue: Checkpoint directory not found

**Cause**: Path resolution issue

**Fix**: The code now uses absolute paths. If you still see issues, check:
```python
print(backtester._checkpoint_dir)  # Should show absolute path
```

## Testing Checkpointing

Run a short test:

```python
from src.backtesting.engine import BacktestEngine
from src.main import run_hedge_fund
from pathlib import Path

backtester = BacktestEngine(
    agent=run_hedge_fund,
    tickers=["AAPL"],
    start_date="2024-01-01",
    end_date="2024-01-31",  # Short test
    initial_capital=100000.0,
    model_name="deepseek-chat",
    model_provider="DeepSeek",
    selected_analysts=["warren_buffett"],
    initial_margin_requirement=0.0,
    checkpoint_interval=5,  # Save every 5 days for testing
)

metrics = backtester.run_backtest(
    ticker="AAPL",
    analyst="warren_buffett",
    csv_path=Path("test_streaming.csv"),
    resume=True,
)

# Check if checkpoint exists
checkpoint_path = Path("data/backtests/checkpoints/AAPL_warren_buffett_2024-01-01_2024-01-31.json")
if checkpoint_path.exists():
    print("✓ Checkpoint saved successfully!")
else:
    print("✗ Checkpoint not found")
```

## Expected Behavior

1. **During Backtest**: Checkpoints save every N days (default: 20)
2. **At End**: Final checkpoint always saves (if ticker/analyst provided)
3. **On Resume**: Automatically loads checkpoint if exists and matches parameters

## Debug Checklist

- [ ] Checkpoint directory exists: `data/backtests/checkpoints/`
- [ ] Debug output shows checkpointing enabled
- [ ] `ticker` and `analyst` parameters are passed to `run_backtest()`
- [ ] No errors in checkpoint save messages
- [ ] Checkpoint files appear in directory after run
