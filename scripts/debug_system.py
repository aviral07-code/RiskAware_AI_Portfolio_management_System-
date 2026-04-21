import os
import sys
import pandas as pd
import numpy as np

# Setup paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_data_shape():
    print("🔍 --- SYSTEM DIAGNOSTICS ---")
    
    # 1. Check Prices
    price_path = "data/raw/sp500_prices.csv"
    if os.path.exists(price_path):
        df = pd.read_csv(price_path, index_col=0, parse_dates=True)
        tickers = list(df.columns)
        print(f"✅ Price Data Found:")
        print(f"   - File: {price_path}")
        print(f"   - Tickers Found: {len(tickers)}")
        print(f"   - Ticker List (First 5): {tickers[:5]}")
        
        # Calculate Expected Shape
        n_assets = len(tickers)
        lookback = 30
        price_features = n_assets * lookback
        print(f"   - Price Features Contribution: {n_assets} * {lookback} = {price_features}")
    else:
        print("❌ CRITICAL: Price file not found!")
        return

    # 2. Check News
    news_path = "data/raw/news_embeddings.csv"
    news_dim = 0
    if os.path.exists(news_path):
        news_df = pd.read_csv(news_path, index_col=0, parse_dates=True)
        news_dim = news_df.shape[1]
        print(f"✅ News Data Found:")
        print(f"   - Dimensions: {news_dim}")
    else:
        print("⚠️ News data not found (dim=0)")

    # 3. Check Macro
    macro_path = "data/raw/macro_features.csv"
    macro_dim = 0
    if os.path.exists(macro_path):
        macro_df = pd.read_csv(macro_path, index_col=0, parse_dates=True)
        macro_dim = macro_df.shape[1]
        print(f"✅ Macro Data Found:")
        print(f"   - Dimensions: {macro_dim}")
    else:
        print("⚠️ Macro data not found (dim=0)")

    # 4. Final Calculation
    total_obs = (n_assets * lookback) + news_dim + macro_dim
    print(f"\n🧮 TOTAL EXPECTED INPUT SIZE: {total_obs}")
    print("-----------------------------------")
    
    if total_obs == 1887:
        print("ℹ️  Your system is set up for 50 Stocks.")
    elif total_obs == 537:
        print("ℹ️  Your system is set up for 5 Stocks.")
    else:
        print(f"ℹ️  Your system has a custom setup ({total_obs}).")
        
    print("👉 If this number does NOT match the error 'Model expects (X)', you must retrain.")

if __name__ == "__main__":
    check_data_shape()
