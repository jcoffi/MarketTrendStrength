#!/usr/bin/env python3
"""
Run Market Trend Strength Analysis with minimal output.
"""

import os
import sys
import argparse
import warnings
from datetime import datetime, timedelta

# Suppress CuPy experimental warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Market Trend Strength Analysis")
    parser.add_argument("--symbol", type=str, default="SPY", help="Stock symbol (default: SPY)")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="trend_strength.csv", help="Output CSV file path")
    parser.add_argument("--use-pca", action="store_true", help="Use PCA for macroeconomic variables")
    parser.add_argument("--quiet", action="store_true", help="Suppress all output except final results")
    
    args = parser.parse_args()
    
    # Set default dates if not provided
    if not args.end:
        args.end = datetime.now().strftime("%Y-%m-%d")
    if not args.start:
        # Default to 1 year before end date
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
        start_date = end_date - timedelta(days=365)
        args.start = start_date.strftime("%Y-%m-%d")
    
    return args

def main():
    """Main function."""
    args = parse_arguments()
    
    # Redirect stdout to suppress output if quiet mode is enabled
    if args.quiet:
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    
    try:
        # Import the market_trend_strength module
        from market_trend_strength import main as run_analysis
        
        # Build command line arguments for the analysis
        sys.argv = [
            "market_trend_strength.py",
            "--symbol", args.symbol,
            "--start", args.start,
            "--end", args.end,
            "--output", args.output
        ]
        
        if args.use_pca:
            sys.argv.append("--use-pca")
        
        # Run the analysis
        exit_code = run_analysis()
        
        # Restore stdout if in quiet mode
        if args.quiet:
            sys.stdout = original_stdout
        
        # Print minimal output
        if exit_code == 0:
            print(f"Market trend analysis completed for {args.symbol}")
            print(f"Results saved to {args.output}")
            
            # Display the last 5 rows of the output file
            import pandas as pd
            result_df = pd.read_csv(args.output)
            print("\nLatest trend strength scores:")
            display_cols = ["date", "close", "regime", "slow_momentum", "fast_momentum", "macro_signal"]
            
            # Add pca_signal if it exists
            if "pca_signal" in result_df.columns:
                display_cols.append("pca_signal")
                
            display_cols.append("trend_strength_score")
            print(result_df[display_cols].tail(5).to_string(index=False))
        else:
            print(f"Analysis failed with exit code {exit_code}")
        
        return exit_code
    
    except Exception as e:
        # Restore stdout if in quiet mode
        if args.quiet:
            sys.stdout = original_stdout
        
        print(f"Error: {str(e)}")
        return 1
    
    finally:
        # Ensure stdout is restored
        if args.quiet:
            sys.stdout = original_stdout

if __name__ == "__main__":
    sys.exit(main())