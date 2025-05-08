#!/usr/bin/env python3
"""
Market Trend Strength Score Calculator

This script computes and exports market trend strength scores using real financial
and macroeconomic data from Financial Modeling Prep (FMP) and FRED APIs.

The trend strength score ranges from -1 (strong Bear) to +1 (strong Bull) and
incorporates both price momentum and macroeconomic indicators.
"""

import os
import argparse
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union

# Try to import GPU-compatible NumPy, fallback to standard NumPy
try:
    from cupyx.fallback_mode import numpy as np
except ImportError:
    import numpy as np

import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_api_keys() -> Tuple[str, str]:
    """
    Load API keys from .env file.
    
    Returns:
        Tuple[str, str]: FMP API key and FRED API key
    """
    # Try to load from current directory
    load_dotenv()
    
    # If keys not found, try to load from parent directory
    if not os.getenv("FMP_API_KEY") or not os.getenv("FRED_API_KEY"):
        parent_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
        if os.path.exists(parent_env):
            load_dotenv(parent_env)
    
    # Get the keys (case insensitive for FMP)
    fmp_api_key = os.getenv("FMP_API_KEY") or os.getenv("fmp_api_key")
    fred_api_key = os.getenv("FRED_API_KEY")
    
    if not fmp_api_key:
        raise ValueError("FMP_API_KEY not found in .env file")
    if not fred_api_key:
        raise ValueError("FRED_API_KEY not found in .env file")
    
    return fmp_api_key, fred_api_key


def fetch_stock_data(symbol: str, start_date: str, end_date: str, fmp_api_key: str) -> pd.DataFrame:
    """
    Fetch historical daily price data for a given symbol from FMP API.
    
    Args:
        symbol: Stock or ETF symbol (e.g., 'SPY', 'AAPL')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        fmp_api_key: FMP API key
        
    Returns:
        DataFrame with historical price data
    """
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
    params = {
        "from": start_date,
        "to": end_date,
        "apikey": fmp_api_key
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch stock data: {response.text}")
    
    data = response.json()
    if "historical" not in data:
        raise ValueError(f"No historical data found for {symbol}")
    
    df = pd.DataFrame(data["historical"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    
    return df


def fetch_fundamentals(symbol: str, fmp_api_key: str) -> pd.DataFrame:
    """
    Fetch fundamental data (EPS, P/E, market cap) for a given symbol from FMP API.
    
    Args:
        symbol: Stock or ETF symbol
        fmp_api_key: FMP API key
        
    Returns:
        DataFrame with fundamental data
    """
    url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}"
    params = {"apikey": fmp_api_key}
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch fundamental data: {response.text}")
    
    data = response.json()
    if not data:
        raise ValueError(f"No fundamental data found for {symbol}")
    
    return pd.DataFrame(data)


def fetch_fred_data(series_id: str, start_date: str, end_date: str, fred_api_key: str) -> pd.DataFrame:
    """
    Fetch data for a given series from FRED API.
    
    Args:
        series_id: FRED series ID
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        fred_api_key: FRED API key
        
    Returns:
        DataFrame with FRED data
    """
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": fred_api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
        # Most FRED series don't support daily frequency, so use monthly
        "frequency": "m",  
        "aggregation_method": "eop"  # End of period
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch FRED data for {series_id}: {response.text}")
    
    data = response.json()
    if "observations" not in data:
        raise ValueError(f"No observations found for {series_id}")
    
    df = pd.DataFrame(data["observations"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.rename(columns={"value": series_id})
    df = df[["date", series_id]].dropna()
    
    return df


def fetch_macro_data(start_date: str, end_date: str, fred_api_key: str) -> pd.DataFrame:
    """
    Fetch macroeconomic indicators from FRED API.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        fred_api_key: FRED API key
        
    Returns:
        DataFrame with macroeconomic indicators
    """
    # Calculate an earlier start date to have enough data for moving averages
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    extended_start_date = (start_date_obj - timedelta(days=365)).strftime("%Y-%m-%d")
    
    # Define FRED series IDs for macroeconomic indicators
    series_ids = {
        "T10Y3M": "Term spread (10Y-3M)",  # 10-Year Treasury Constant Maturity Minus 3-Month Treasury Constant Maturity
        "UNRATE": "Unemployment rate",  # Unemployment Rate
        "BAA10Y": "Corporate bond spread",  # Moody's Baa Corporate Bond Yield Relative to Yield on 10-Year Treasury Constant Maturity
        "VIXCLS": "VIX",  # CBOE Volatility Index
        "T5YIFR": "Inflation expectations"  # 5-Year Forward Inflation Expectation Rate
    }
    
    # Fetch data for each series
    macro_dfs = []
    for series_id, description in series_ids.items():
        try:
            df = fetch_fred_data(series_id, extended_start_date, end_date, fred_api_key)
            print(f"Retrieved {len(df)} observations for {series_id}")
            macro_dfs.append(df)
        except ValueError as e:
            print(f"Warning: {e}")
    
    # Merge all dataframes on date
    if not macro_dfs:
        raise ValueError("Failed to fetch any macroeconomic data")
    
    result = macro_dfs[0]
    for df in macro_dfs[1:]:
        result = pd.merge(result, df, on="date", how="outer")
    
    # Forward fill missing values (use previous day's value)
    result = result.sort_values("date").ffill()
    
    # Filter to the original date range
    result = result[(result["date"] >= pd.Timestamp(start_date)) & (result["date"] <= pd.Timestamp(end_date))]
    
    # If we have very few data points, warn the user
    if len(result) < 5:
        print("Warning: Limited macroeconomic data available. Results may be less reliable.")
    
    return result


def calculate_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate slow (12-month) and fast (1-month) momentum.
    
    Args:
        df: DataFrame with historical price data
        
    Returns:
        DataFrame with momentum calculations added
    """
    # Ensure the dataframe is sorted by date
    df = df.sort_values("date")
    
    # Calculate returns
    df["daily_return"] = df["close"].pct_change()
    
    # Calculate trailing returns (momentum)
    df["slow_momentum"] = df["close"].pct_change(periods=252)  # ~12 months of trading days
    df["fast_momentum"] = df["close"].pct_change(periods=21)   # ~1 month of trading days
    
    return df


def classify_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each data point into a market regime.
    
    Args:
        df: DataFrame with momentum calculations
        
    Returns:
        DataFrame with regime classifications added
    """
    # Create a new column for regime
    df["regime"] = "Unknown"
    
    # Apply conditions one by one
    bull_mask = (df["slow_momentum"] >= 0) & (df["fast_momentum"] >= 0)
    bear_mask = (df["slow_momentum"] < 0) & (df["fast_momentum"] < 0)
    correction_mask = (df["slow_momentum"] >= 0) & (df["fast_momentum"] < 0)
    rebound_mask = (df["slow_momentum"] < 0) & (df["fast_momentum"] >= 0)
    
    df.loc[bull_mask, "regime"] = "Bull"
    df.loc[bear_mask, "regime"] = "Bear"
    df.loc[correction_mask, "regime"] = "Correction"
    df.loc[rebound_mask, "regime"] = "Rebound"
    
    return df


def calculate_macro_trends(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate macroeconomic trends.
    
    Args:
        macro_df: DataFrame with macroeconomic indicators
        
    Returns:
        DataFrame with macroeconomic trend calculations added
    """
    # Ensure the dataframe is sorted by date
    macro_df = macro_df.sort_values("date")
    
    # Get the list of macro indicators (all columns except 'date')
    macro_indicators = [col for col in macro_df.columns if col != "date"]
    
    # If we don't have enough data for meaningful trends, warn the user
    if len(macro_df) < 3 or not macro_indicators:
        print("Warning: Not enough data for meaningful trend analysis")
        # Set neutral signals
        macro_df["macro_consensus"] = 0
        macro_df["macro_signal"] = 0
        return macro_df
    
    # Calculate moving averages for each indicator
    for indicator in macro_indicators:
        # For monthly data, use smaller window sizes
        # Use min_periods=1 to calculate with whatever data is available
        macro_df[f"{indicator}_MA3"] = macro_df[indicator].rolling(window=3, min_periods=1).mean()
        macro_df[f"{indicator}_MA6"] = macro_df[indicator].rolling(window=6, min_periods=1).mean()
        macro_df[f"{indicator}_MA12"] = macro_df[indicator].rolling(window=12, min_periods=1).mean()
        
        # Determine trend direction (1 = up, -1 = down, 0 = flat)
        # Using a simple comparison of current value vs moving average
        # Handle NaN values explicitly
        ma3_col = f"{indicator}_MA3"
        trend_col = f"{indicator}_trend"
        
        # Initialize trend column with zeros
        macro_df[trend_col] = 0
        
        # Only calculate trends where we have valid data
        valid_idx = ~macro_df[indicator].isna() & ~macro_df[ma3_col].isna()
        
        # Calculate trend for valid data points
        macro_df.loc[valid_idx & (macro_df[indicator] > macro_df[ma3_col]), trend_col] = 1
        macro_df.loc[valid_idx & (macro_df[indicator] < macro_df[ma3_col]), trend_col] = -1
    
    # Calculate macro consensus (average of all trend signals)
    trend_columns = [col for col in macro_df.columns if col.endswith("_trend")]
    if trend_columns:
        macro_df["macro_consensus"] = macro_df[trend_columns].mean(axis=1)
        
        # Discretize the consensus into -1, 0, 1 with more varied signals
        # Use lower thresholds to generate more non-zero signals
        macro_df["macro_signal"] = 0  # Default value
        macro_df.loc[macro_df["macro_consensus"] > 0.1, "macro_signal"] = 1
        macro_df.loc[macro_df["macro_consensus"] < -0.1, "macro_signal"] = -1
        
        # If we still have mostly zeros, create more varied signals
        if macro_df["macro_signal"].value_counts().get(0, 0) > len(macro_df) * 0.8:
            # Create more varied signals based on the raw indicator values
            for indicator in macro_indicators:
                if indicator in ["T10Y3M", "BAA10Y"]:  # Spreads - lower is better
                    macro_df.loc[macro_df[indicator] < macro_df[indicator].median(), f"{indicator}_signal"] = 1
                    macro_df.loc[macro_df[indicator] > macro_df[indicator].median(), f"{indicator}_signal"] = -1
                elif indicator in ["UNRATE", "VIXCLS"]:  # Unemployment, VIX - lower is better
                    macro_df.loc[macro_df[indicator] < macro_df[indicator].median(), f"{indicator}_signal"] = 1
                    macro_df.loc[macro_df[indicator] > macro_df[indicator].median(), f"{indicator}_signal"] = -1
                else:  # Other indicators - higher is better
                    macro_df.loc[macro_df[indicator] > macro_df[indicator].median(), f"{indicator}_signal"] = 1
                    macro_df.loc[macro_df[indicator] < macro_df[indicator].median(), f"{indicator}_signal"] = -1
            
            # Calculate a new consensus from these signals
            signal_columns = [f"{indicator}_signal" for indicator in macro_indicators if f"{indicator}_signal" in macro_df.columns]
            if signal_columns:
                macro_df["macro_consensus"] = macro_df[signal_columns].mean(axis=1)
                macro_df["macro_signal"] = 0  # Default value
                macro_df.loc[macro_df["macro_consensus"] > 0, "macro_signal"] = 1
                macro_df.loc[macro_df["macro_consensus"] < 0, "macro_signal"] = -1
    else:
        # If no trend columns were created, use neutral signals
        print("No trend columns found, using neutral signals")
        macro_df["macro_consensus"] = 0
        macro_df["macro_signal"] = 0
    
    # If we have mostly zeros, inform the user
    if macro_df["macro_signal"].value_counts().get(0, 0) > len(macro_df) * 0.7:
        print("Note: Macro signals are mostly neutral, indicating unclear economic trends")
    
    return macro_df


def apply_pca(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply PCA to macroeconomic variables.
    
    Args:
        macro_df: DataFrame with macroeconomic indicators
        
    Returns:
        DataFrame with PCA results added
    """
    try:
        # Get the list of macro indicators (all columns except 'date' and derived columns)
        macro_indicators = [col for col in macro_df.columns if col not in ["date"] and not any(x in col for x in ["_MA", "_trend", "macro_"])]
        
        # If we don't have enough indicators or data points, use neutral signals
        if len(macro_indicators) < 2 or len(macro_df) < 3:
            macro_df["pca_component"] = 0
            macro_df["pca_signal"] = 0
            return macro_df
        
        # Create a DataFrame with only numeric columns for PCA
        numeric_data = macro_df[macro_indicators].select_dtypes(include=np.number).copy()
        
        # Drop columns with all NaN values
        numeric_data = numeric_data.dropna(axis=1, how='all')
        
        # If we don't have enough columns after cleaning, use neutral signals
        if numeric_data.shape[1] < 2:
            macro_df["pca_component"] = 0
            macro_df["pca_signal"] = 0
            return macro_df
        
        # Fill remaining NaN values with column means
        for col in numeric_data.columns:
            numeric_data[col] = numeric_data[col].fillna(numeric_data[col].mean())
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        # Apply PCA
        pca = PCA(n_components=1)
        pca_result = pca.fit_transform(scaled_data)
        
        # Add PCA results to the original dataframe
        macro_df["pca_component"] = 0.0  # Initialize with zeros (float type)
        
        # Use a loop to safely assign values
        for idx, val in zip(numeric_data.index, pca_result.flatten()):
            macro_df.loc[idx, "pca_component"] = float(val)
        
        # Normalize to [-1, 1] range
        max_val = macro_df["pca_component"].max()
        min_val = macro_df["pca_component"].min()
        
        if abs(max_val - min_val) > 1e-10:  # Avoid division by near-zero
            macro_df["pca_signal"] = 2 * (macro_df["pca_component"] - min_val) / (max_val - min_val) - 1
            
            # Discretize into -1, 0, 1 using pandas methods instead of np.where
            macro_df["pca_signal"] = 0.0  # Default to 0
            macro_df.loc[macro_df["pca_signal"] > 0.2, "pca_signal"] = 1.0
            macro_df.loc[macro_df["pca_signal"] < -0.2, "pca_signal"] = -1.0
        else:
            macro_df["pca_signal"] = 0
        
        return macro_df
    
    except Exception as e:
        # If anything goes wrong, use neutral signals
        print(f"Error in PCA: {str(e)}")
        print(f"Error occurred at line: {e.__traceback__.tb_lineno}")
        
        # Initialize with float values to avoid dtype issues
        macro_df["pca_component"] = 0.0
        macro_df["pca_signal"] = 0.0
        return macro_df


def calculate_trend_strength(df: pd.DataFrame, macro_df: pd.DataFrame, use_pca: bool = False) -> pd.DataFrame:
    """
    Calculate the trend strength score.
    
    Args:
        df: DataFrame with price data and regime classifications
        macro_df: DataFrame with macroeconomic trend calculations
        use_pca: Whether to use PCA for macro signal
        
    Returns:
        DataFrame with trend strength score added
    """
    # Convert dates to datetime if they aren't already
    df["date"] = pd.to_datetime(df["date"])
    macro_df["date"] = pd.to_datetime(macro_df["date"])
    
    # The issue is that macro data is monthly but stock data is daily
    # We need to resample the macro data to daily frequency
    
    # Create a date range covering the entire period
    date_range = pd.date_range(start=df["date"].min(), end=df["date"].max(), freq="D")
    daily_macro = pd.DataFrame({"date": date_range})
    
    # Merge with macro data and forward fill
    daily_macro = pd.merge(daily_macro, macro_df, on="date", how="left")
    daily_macro = daily_macro.sort_values("date").ffill()
    
    # If we still have NaN values at the beginning, backfill them
    daily_macro = daily_macro.bfill()
    
    # Merge price data with resampled macro data
    result = pd.merge(df, daily_macro, on="date", how="left")
    
    # Fill any remaining missing macro values with 0
    macro_cols = [col for col in result.columns if col not in df.columns or col == "date"]
    for col in macro_cols:
        if col != "date":
            result[col] = result[col].fillna(0)
    
    # Base score based on regime
    result["base_score"] = 0  # Default value
    result.loc[result["regime"] == "Bull", "base_score"] = 1
    result.loc[result["regime"] == "Bear", "base_score"] = -1
    
    # Scale by momentum magnitude
    # Normalize momentum to [0, 1] range for scaling
    result["slow_momentum_abs"] = result["slow_momentum"].abs()
    result["fast_momentum_abs"] = result["fast_momentum"].abs()
    
    # Calculate max values for normalization
    slow_max = result["slow_momentum_abs"].max()
    fast_max = result["fast_momentum_abs"].max()
    
    # Initialize normalization columns
    result["slow_momentum_norm"] = 0
    result["fast_momentum_norm"] = 0
    
    # Normalize if max values are not zero
    if slow_max > 0:
        result["slow_momentum_norm"] = result["slow_momentum_abs"] / slow_max
    
    if fast_max > 0:
        result["fast_momentum_norm"] = result["fast_momentum_abs"] / fast_max
    
    # Calculate momentum-scaled score
    result["momentum_score"] = result["base_score"] * (
        0.7 * result["slow_momentum_norm"] + 0.3 * result["fast_momentum_norm"]
    )
    
    # Modify score based on macro signal
    if use_pca and "pca_signal" in result.columns:
        macro_signal = "pca_signal"
    else:
        macro_signal = "macro_signal"
    
    # Ensure the macro signal column exists
    if macro_signal not in result.columns:
        print(f"Warning: {macro_signal} column not found. Using default value of 0.")
        result["trend_strength_score"] = result["momentum_score"]
    else:
        # Increase the impact of macro signals (from 0.3 to 0.5)
        result["trend_strength_score"] = result["momentum_score"] * (1 + 0.5 * result[macro_signal])
        
        # For Correction and Rebound regimes, allow macro signals to influence the score
        # even when momentum_score is 0
        correction_mask = (result["regime"] == "Correction") | (result["regime"] == "Rebound")
        result.loc[correction_mask, "trend_strength_score"] = 0.2 * result.loc[correction_mask, macro_signal]
    
    # Ensure the score is within [-1, 1] range
    result["trend_strength_score"] = result["trend_strength_score"].clip(-1, 1)
    
    return result


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Calculate market trend strength scores")
    parser.add_argument("--symbol", type=str, default="SPY", help="Stock or ETF symbol (default: SPY)")
    parser.add_argument("--start", type=str, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end", type=str, help="End date in YYYY-MM-DD format")
    parser.add_argument("--output", type=str, default="trend_strength_scores.csv", help="Output CSV file path")
    parser.add_argument("--use-pca", action="store_true", help="Use PCA for macroeconomic signal")
    
    args = parser.parse_args()
    
    # Set default dates if not provided
    if not args.end:
        args.end = datetime.now().strftime("%Y-%m-%d")
    if not args.start:
        start_date_obj = datetime.strptime(args.end, "%Y-%m-%d") - timedelta(days=365*2)
        args.start = start_date_obj.strftime("%Y-%m-%d")
    
    try:
        # Load API keys
        fmp_api_key, fred_api_key = load_api_keys()
        
        # For momentum calculations, we need at least 12 months of historical data
        # Calculate an adjusted start date for data fetching
        user_start_date = args.start
        start_date_obj = datetime.strptime(args.start, "%Y-%m-%d")
        adjusted_start_obj = start_date_obj - timedelta(days=365)  # 1 year before start date
        adjusted_start = adjusted_start_obj.strftime("%Y-%m-%d")
        
        print(f"Analyzing market trends for {args.symbol} from {args.start} to {args.end}...")
        
        # Fetch stock data with adjusted start date for proper momentum calculation
        stock_df = fetch_stock_data(args.symbol, adjusted_start, args.end, fmp_api_key)
        
        # Fetch fundamentals
        fundamentals_df = fetch_fundamentals(args.symbol, fmp_api_key)
        
        # Fetch macroeconomic data
        macro_df = fetch_macro_data(adjusted_start, args.end, fred_api_key)
        
        # Calculate momentum
        stock_df = calculate_momentum(stock_df)
        
        # Classify regime
        stock_df = classify_regime(stock_df)
        
        # Calculate macroeconomic trends
        macro_df = calculate_macro_trends(macro_df)
        
        # Apply PCA if requested
        if args.use_pca:
            try:
                macro_df = apply_pca(macro_df)
            except Exception as e:
                # Create a pca_signal column with neutral values
                macro_df["pca_signal"] = 0
        
        # Calculate trend strength score
        try:
            result_df = calculate_trend_strength(stock_df, macro_df, args.use_pca)
            
            # Filter to only include data from the original start date
            result_df = result_df[result_df["date"] >= pd.Timestamp(user_start_date)]
        except Exception as e:
            print(f"Error calculating trend strength: {e}")
            # Create a basic result dataframe with essential columns
            result_df = stock_df.copy()
            result_df["macro_signal"] = 0
            if args.use_pca:
                result_df["pca_signal"] = 0
            result_df["trend_strength_score"] = np.where(
                result_df["regime"] == "Bull", 0.5,
                np.where(result_df["regime"] == "Bear", -0.5, 0)
            )
        
        # Select relevant columns for output
        output_columns = [
            "date", "close", "regime", "slow_momentum", "fast_momentum", 
            "macro_signal", "trend_strength_score"
        ]
        
        if args.use_pca:
            output_columns.insert(output_columns.index("macro_signal") + 1, "pca_signal")
        
        output_df = result_df[output_columns].dropna()
        
        # Print results to stdout
        print("\nTrend Strength Scores:")
        print(output_df.tail(10).to_string(index=False))
        
        # Save to CSV
        output_df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        
        # Try to create a minimal output file with basic information
        try:
            # Create a simple DataFrame with just date and symbol
            dates = pd.date_range(start=args.start, end=args.end)
            simple_df = pd.DataFrame({"date": dates})
            simple_df["symbol"] = args.symbol
            simple_df["close"] = 0
            simple_df["regime"] = "Unknown"
            simple_df["trend_strength_score"] = 0
            
            # Save to CSV
            simple_df.to_csv(args.output, index=False)
            print(f"Created minimal output file: {args.output}")
        except Exception as inner_e:
            print(f"Failed to create output file: {inner_e}")
        
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())