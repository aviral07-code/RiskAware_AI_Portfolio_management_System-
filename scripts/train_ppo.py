import os
import sys
import numpy as np
import pandas as pd
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

# --- PATH SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)
# ------------------

from src.envs.risk_aware_env import RiskAwarePortfolioEnv

# Real GICS Sector Mapping
REAL_SECTORS = {
    "AAPL": "Tech", "MSFT": "Tech", "AMZN": "Consumer", "GOOGL": "Comm", "META": "Comm",
    "NVDA": "Tech", "TSLA": "Consumer", "JPM": "Financial", "BAC": "Financial", "WFC": "Financial",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "UNH": "Healthcare", "JNJ": "Healthcare",
    "PFE": "Healthcare", "MRK": "Healthcare", "PG": "Staples", "KO": "Staples", "PEP": "Staples",
    "HD": "Consumer", "LOW": "Consumer", "DIS": "Comm", "NFLX": "Comm", "INTC": "Tech",
    "CSCO": "Tech", "ADBE": "Tech", "CRM": "Tech", "ORCL": "Tech", "AVGO": "Tech",
    "V": "Financial", "MA": "Financial", "PYPL": "Financial", "C": "Financial", "GS": "Financial",
    "BLK": "Financial", "T": "Comm", "VZ": "Comm", "CMCSA": "Comm", "ABT": "Healthcare",
    "COST": "Staples", "MCD": "Consumer", "NKE": "Consumer", "LIN": "Materials", "ACN": "Tech",
    "AMD": "Tech", "TXN": "Tech", "QCOM": "Tech", "HON": "Industrials", "CAT": "Industrials"
}

def make_env(cfg):
    return RiskAwarePortfolioEnv(**cfg)

def main():
    print(f"🚀 Project Root Detected: {project_root}")
    
    # 1. Setup Paths
    data_dir = os.path.join(project_root, "data", "raw")
    prices_path = os.path.join(data_dir, "sp500_prices.csv")
    macro_path = os.path.join(data_dir, "macro_features.csv")
    news_path = os.path.join(data_dir, "news_embeddings.csv")
    checkpoint_dir = os.path.join(project_root, "results", "ppo_full")

    # 2. Critical Data Check
    if not os.path.exists(prices_path):
        print(f"❌ CRITICAL ERROR: Prices file not found at {prices_path}")
        return

    print("📊 Loading Training Data...")
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    tickers = list(prices.columns)
    dates = prices.index
    
    # Align Macro Data
    if os.path.exists(macro_path):
        macro_df = pd.read_csv(macro_path, index_col=0, parse_dates=True)
        macro_df = macro_df.reindex(dates).ffill().fillna(0.0)
        macro_features = {d: macro_df.loc[d].values.astype(np.float32) for d in dates}
    else:
        macro_features = {}
        print("   ⚠️ Macro Features NOT found. Using Empty.")

    # Align News Data
    if os.path.exists(news_path):
        news_df = pd.read_csv(news_path, index_col=0, parse_dates=True)
        news_df = news_df.reindex(dates).fillna(0.0)
        news_embeddings = {d: news_df.loc[d].values.astype(np.float32) for d in dates}
    else:
        print("   ⚠️ News Embeddings NOT found. Using Empty.")
        news_embeddings = {}

    # 3. Configure RL Environment with AGGRESSIVE Dimensions & Rules
    env_config = {
        "prices_csv": prices_path,
        "tickers": tickers,
        # Give the environment real sectors so it enforces the 25% max-weight correctly
        "sectors": {t: REAL_SECTORS.get(t, "Other") for t in tickers},
        "transaction_cost": 0.0002,
        "cvar_lambda": 0.05,
        "sector_max_weight": 0.25,  
        "macro_features": macro_features,
        "news_embeddings": news_embeddings,
        "initial_capital": 100000,
        "lookback_window": 30,
        "news_dim": 384,  
        "macro_dim": 3,   
    }

    # 4. Initialize Ray
    ray.init(ignore_reinit_error=True)
    register_env("RiskAwarePortfolioEnv-v0", lambda cfg: make_env({**env_config, **cfg}))

    # 5. Build Algorithm
    config = (
        PPOConfig()
        .environment(env="RiskAwarePortfolioEnv-v0", env_config={})
        .framework("torch")
        .env_runners(num_env_runners=0)
        .training(
            model={"fcnet_hiddens": [256, 256], "fcnet_activation": "tanh"},
            lr=5e-5,
            entropy_coeff=0.05,
            train_batch_size_per_learner=2000,
            gamma=0.99
        )
        .resources(num_gpus=0)
    )
    
    algo = config.build()
    
    print("\n🏋️ STARTING TRAINING...")
    for i in range(200): 
        result = algo.train()
        mean_ret = result.get('env_runners', {}).get('episode_return_mean', 0)
        print(f"   Iter {i+1}/200: Mean Return = {mean_ret:.2f}")

    # Save Checkpoint
    save_path = algo.save(checkpoint_dir)
    print(f"\n✅ TRAINING COMPLETE.")
    print(f"💾 Checkpoint saved at: {save_path}")

    ray.shutdown()

if __name__ == "__main__":
    main()
