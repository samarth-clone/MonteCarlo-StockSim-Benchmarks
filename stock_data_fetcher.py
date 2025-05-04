#!/usr/bin/env python3
"""
Stock Data Fetcher

A simple script to fetch stock data from Alpha Vantage API
and save it to CSV files for use by other applications.
"""

import os
import sys
import argparse
import requests
import pandas as pd
from pathlib import Path


def fetch_stock_data(ticker, api_key, output_size="full"):
    """
    Fetch daily price data for a given ticker symbol.
    
    Args:
        ticker (str): Stock ticker symbol
        api_key (str): Alpha Vantage API key
        output_size (str): 'compact' for last 100 data points, 'full' for full history
        
    Returns:
        pandas.DataFrame: DataFrame with daily price data or None if error
    """
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "outputsize": output_size,
        "apikey": api_key
    }
    
    print(f"Fetching daily data for {ticker}...")
    response = requests.get(base_url, params=params)
    
    if response.status_code != 200:
        print(f"Error fetching data: HTTP {response.status_code}")
        return None
        
    data = response.json()
    
    if "Error Message" in data:
        print(f"API Error: {data['Error Message']}")
        return None
        
    if "Time Series (Daily)" not in data:
        print("Unexpected API response format")
        print(data)
        return None
        
    # Convert to DataFrame
    time_series = data["Time Series (Daily)"]
    df = pd.DataFrame(time_series).T
    
    # Convert column names and values
    df.columns = [col.split(". ")[1] for col in df.columns]
    df = df.apply(pd.to_numeric)
    
    # Add date as a column
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)  # Sort by date
    
    return df


def save_to_csv(ticker, data, output_dir="historic_data"):
    """
    Save stock data to CSV file.
    
    Args:
        ticker (str): Stock ticker symbol
        data (pandas.DataFrame): DataFrame with price data
        output_dir (str): Directory to save the CSV file
        
    Returns:
        Path: Path to the saved file or None if error
    """
    if data is None or data.empty:
        print(f"No data to save for {ticker}")
        return None
        
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    output_file = output_path / f"{ticker}_historical.csv"
    
    # Save only the closing prices to match the C++ code's expectation
    data["close"].to_csv(output_file, header=["Price"])
    print(f"Saved historical data to {output_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Fetch stock data from Alpha Vantage API")
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--api-key", help="Alpha Vantage API key", 
                        default=os.environ.get("ALPHA_VANTAGE_API_KEY", "NO API KEYS FOR YOU HHEHEHEHEHEHEHEHEHEH"))
    parser.add_argument("--output-dir", help="Directory to save CSV files", default="historic_data")
    parser.add_argument("--force", "-f", action="store_true", help="Force update even if file exists")
    
    args = parser.parse_args()
    
    # Check if file already exists
    output_path = Path(args.output_dir) / f"{args.ticker}_historical.csv"
    if output_path.exists() and not args.force:
        print(f"Data file {output_path} already exists. Use --force to update.")
        return 0
    
    # Fetch data
    data = fetch_stock_data(args.ticker, args.api_key)
    
    # Save to CSV
    if data is not None:
        save_to_csv(args.ticker, data, args.output_dir)
        return 0
    else:
        print("Failed to fetch stock data")
        return 1


if __name__ == "__main__":
    sys.exit(main())