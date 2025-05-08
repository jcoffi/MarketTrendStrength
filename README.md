# Market Trend Strength Calculator

A command-line tool for computing and exporting market trend strength scores using real financial and macroeconomic data.

## Overview

This tool calculates a trend strength score ranging from -1 (strong Bear) to +1 (strong Bull) based on:

1. Price momentum (fast and slow)
2. Market regime classification
3. Macroeconomic indicators

## Features

- Retrieves real financial data from Financial Modeling Prep (FMP) API
- Fetches macroeconomic indicators from FRED API
- Calculates slow (12-month) and fast (1-month) momentum
- Classifies market regimes (Bull, Bear, Correction, Rebound)
- Computes macroeconomic trend consensus
- Optional PCA analysis of macroeconomic variables
- Exports results to CSV

## Requirements

- Python 3.7+
- Required packages (install with `pip install -r market_trend_requirements.txt`):
  - requests
  - pandas
  - numpy
  - scikit-learn
  - statsmodels
  - python-dotenv
  - cupy-cuda12x (for GPU acceleration)

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r market_trend_requirements.txt
   ```
3. Create a `.env` file with your API keys (use `.env.template` as a reference):
   ```
   FMP_API_KEY=your_fmp_api_key_here
   FRED_API_KEY=your_fred_api_key_here
   ```

## Usage

**Important**: All scripts must be run from the directory where they are located.

Navigate to the directory containing the scripts:

```bash
cd /path/to/hope/
```

Basic usage:

```bash
python market_trend_strength.py --symbol SPY --output trend_scores.csv
```

All available options:

```bash
python market_trend_strength.py --symbol SPY --start 2020-01-01 --end 2023-01-01 --output trend_scores.csv --use-pca
```

### Quiet Mode (for Production Pipelines)

For automated tasks or production pipelines, use the `run_market_trend.py` script with the `--quiet` flag:

```bash
python run_market_trend.py --symbol SPY --quiet --output trend_scores.csv
```

This provides minimal output, showing only the final results and suppresses warnings.

### Command-line Arguments

- `--symbol`: Stock or ETF symbol (default: SPY)
- `--start`: Start date in YYYY-MM-DD format (default: 1 year before end date)
- `--end`: End date in YYYY-MM-DD format (default: today)
- `--output`: Output CSV file path (default: trend_strength_scores.csv)
- `--use-pca`: Flag to activate PCA functionality for macroeconomic variables
- `--quiet`: Suppress all output except final results (only for run_market_trend.py)

## Output

The script outputs:

1. To the console:
   - Summary of data retrieval
   - Table of the most recent trend strength scores

2. To the specified CSV file:
   - Date
   - Close price
   - Market regime
   - Slow momentum (12-month return)
   - Fast momentum (1-month return)
   - Macroeconomic signal
   - PCA signal (if enabled)
   - Trend strength score

## Examples

### Standard Analysis

```bash
# First, navigate to the directory containing the scripts
cd /path/to/hope/

# Then run the analysis
python market_trend_strength.py --symbol AAPL --start 2022-01-01
```

This will:
1. Fetch AAPL price data from Jan 1, 2022 to today
2. Calculate momentum and classify market regimes
3. Retrieve and analyze macroeconomic indicators
4. Compute trend strength scores
5. Display recent scores and save all results to trend_strength_scores.csv

### Production Pipeline Example

```bash
# Navigate to the script directory
cd /path/to/hope/

# Run in quiet mode for production use
python run_market_trend.py --symbol TSLA --start 2023-01-01 --output tesla_trends.csv --quiet
```

This will perform the analysis with minimal output, showing only the final results and saving to tesla_trends.csv.

## Troubleshooting

### Common Issues

1. **"No such file or directory" error**:
   - Make sure you're running the script from the correct directory
   - Use `cd /path/to/hope/` to navigate to the directory containing the scripts

2. **API Key errors**:
   - Ensure your `.env` file is in the same directory as the scripts
   - Verify that your API keys are valid and correctly formatted
   - Check that you have sufficient API credits/quota remaining

3. **Missing data for certain dates**:
   - Some macroeconomic indicators are published monthly or quarterly
   - Weekend and holiday dates may not have market data
   - Try extending your date range if you're getting limited results

4. **PCA errors**:
   - If PCA analysis fails, the script will automatically fall back to standard analysis
   - This is normal and won't affect the basic trend strength calculation

### Getting Help

If you encounter persistent issues:
1. Check that all dependencies are correctly installed
2. Verify your API keys are valid and have sufficient quota
3. Try running with a well-known symbol like SPY for testing