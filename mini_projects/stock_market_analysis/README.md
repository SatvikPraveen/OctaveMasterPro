# Stock Market Analysis

**Location:** `mini_projects/stock_market_analysis/README.md`

## Overview

Comprehensive stock market analysis toolkit implementing technical indicators, trend analysis, portfolio optimization, and risk assessment using Octave.

## Features

- Load and parse historical stock data from CSV files
- Calculate technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Perform trend analysis and volatility calculations
- Basic portfolio optimization and risk metrics
- Backtesting framework for trading strategies

## Files

- `data_loader.m` - Stock data loading and preprocessing
- `technical_indicators.m` - Moving averages, RSI, MACD, Bollinger Bands
- `price_analysis.m` - Trend analysis, volatility, returns calculation
- `portfolio_optimizer.m` - Portfolio optimization and risk metrics
- `market_demo.m` - Main demonstration script

## Usage

```octave
# Run main demonstration
market_demo

# Load stock data
data = load_stock_data('sample_data/AAPL.csv');

# Calculate indicators
sma = simple_moving_average(data.close, 20);
rsi = relative_strength_index(data.close, 14);
[macd_line, signal_line] = macd_indicator(data.close);

# Portfolio optimization
weights = optimize_portfolio(returns_matrix, 'target_return', 0.12);
```

## Data Format

CSV files should contain columns: Date, Open, High, Low, Close, Volume

## Requirements

- Octave with financial data processing capabilities
- Sample stock data in `sample_data/` directory

## Sample Outputs

- Technical indicator plots
- Portfolio efficient frontier
- Risk-return analysis
- Trading strategy backtests
