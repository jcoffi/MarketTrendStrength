#!/usr/bin/env python3
"""
Test script for the Market Trend Strength Calculator.

This script demonstrates how to use the market_trend_strength.py script
with different parameters.
"""

import subprocess
import os
import sys

def run_command(command):
    """Run a command and print its output."""
    print(f"Running: {command}")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate()
    
    if stdout:
        print("Output:")
        print(stdout)
    
    if stderr:
        print("Errors:")
        print(stderr)
    
    print(f"Exit code: {process.returncode}")
    print("-" * 80)
    
    return process.returncode

def main():
    """Run test commands for the market trend strength calculator."""
    # Check if the .env file exists
    if not os.path.exists(".env") and not os.path.exists("../.env"):
        print("Warning: No .env file found. Please create one with your API keys.")
        print("You can use the .env.template file as a reference.")
        return 1
    
    # If .env exists in parent directory but not in current directory, create a symlink
    if not os.path.exists(".env") and os.path.exists("../.env"):
        print("Creating symlink to ../.env")
        os.symlink("../.env", ".env")
    
    # Test commands
    commands = [
        # Basic usage with default parameters
        "python market_trend_strength.py --symbol SPY --output test_output1.csv",
        
        # Specify date range
        "python market_trend_strength.py --symbol SPY --start 2022-01-01 --end 2023-01-01 --output test_output2.csv",
        
        # Use PCA for macroeconomic signal
        "python market_trend_strength.py --symbol SPY --start 2022-01-01 --use-pca --output test_output3.csv",
        
        # Test with a different symbol
        "python market_trend_strength.py --symbol AAPL --output test_output4.csv"
    ]
    
    # Run each command
    for command in commands:
        try:
            exit_code = run_command(command)
            if exit_code != 0:
                print(f"Command failed with exit code {exit_code}, but continuing with tests")
        except Exception as e:
            print(f"Error running command: {e}, but continuing with tests")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())