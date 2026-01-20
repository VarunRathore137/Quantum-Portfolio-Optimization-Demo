import yfinance as yf
import pandas as pd
import numpy as np

def fetch_market_data(tickers: list, period="3mo"):
    """
    Fetches historical data and calculates Expected Returns (gamma) and Covariance (sigma).
    
    Args:
        tickers (list): List of ticker symbols e.g., ['MSFT', 'TSLA']
        period (str): '3mo', '6mo', '1y'
        
    Returns:
        tuple: (gamma, sigma, raw_prices)
    """
    print(f"[Data Loader] Fetching data for {tickers} over {period}...")
    
    # 1. Download Data
    # We use auto_adjust=False to safely access 'Close' without ambiguity
    data = yf.download(tickers, period=period, auto_adjust=False, progress=False)
    
    # 2. Extract Close Prices
    # Handle the MultiIndex case safely
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data['Close']

    # 3. Calculate Daily Returns
    # pct_change() computes the percentage difference between today and yesterday
    returns = prices.pct_change(fill_method=None).dropna()
    
    # Check if we have enough data
    if len(returns) == 0:
        raise ValueError(f"No data available for tickers {tickers}. Please check ticker symbols and try again.")
    
    # 4. Calculate Statistics
    # gamma: Expected Return (Mean of daily returns)
    # sigma: Covariance Matrix (Risk relationship between assets)
    gamma = returns.mean()
    sigma = returns.cov()
    
    print(f"[Data Loader] Data processed successfully.")
    print(f" - Assets: {len(gamma)}")
    print(f" - Date Range: {returns.index[0].date()} to {returns.index[-1].date()}")
    
    return gamma, sigma, prices