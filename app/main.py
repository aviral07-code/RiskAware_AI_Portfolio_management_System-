import os
import sys
import pandas as pd
import numpy as np
import gradio as gr
import ray
import torch
import yfinance as yf
from datetime import timedelta
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
# ------------------

from src.envs.risk_aware_env import RiskAwarePortfolioEnv
from src.agents.llm_critic import LLMCritic
from app.dashboard_utils import make_dashboard_charts

# --- CONFIG & SECTORS ---
CHECKPOINT_REL_PATH = "results/ppo_full"

# Real GICS Sector Mapping for strict environment constraints
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

def get_project_paths():
    return project_root

def find_latest_checkpoint(base_path):
    if not os.path.exists(base_path): return base_path
    if os.path.exists(os.path.join(base_path, "algorithm_state.pkl")): return base_path
    subdirs = [os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    checkpoints = [d for d in subdirs if "checkpoint" in d]
    if checkpoints: return sorted(checkpoints)[-1]
    return base_path

def load_data_assets():
    root = get_project_paths()
    data_dir = os.path.join(root, "data", "raw")
    
    prices_path = os.path.join(data_dir, "sp500_prices.csv")
    if not os.path.exists(prices_path): raise FileNotFoundError(f"Prices not found at {prices_path}")
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    tickers = list(prices.columns)
    
    news_embeddings = {}
    macro_features = {}
    
    news_path = os.path.join(data_dir, "news_embeddings.csv")
    if os.path.exists(news_path):
        news_df = pd.read_csv(news_path, index_col=0, parse_dates=True)
        news_df = news_df.reindex(prices.index).fillna(0.0)
        news_embeddings = {d: news_df.loc[d].values.astype(np.float32) for d in prices.index}

    macro_path = os.path.join(data_dir, "macro_features.csv")
    if os.path.exists(macro_path):
        macro_df = pd.read_csv(macro_path, index_col=0, parse_dates=True)
        macro_df = macro_df.reindex(prices.index).ffill().fillna(0.0)
        macro_features = {d: macro_df.loc[d].values.astype(np.float32) for d in prices.index}
        
    return prices, tickers, news_embeddings, macro_features

# --- AGENT FACTORY ---
def make_env(cfg):
    return RiskAwarePortfolioEnv(**cfg)

def load_trained_agent():
    root = get_project_paths()
    base_path = os.path.join(root, CHECKPOINT_REL_PATH) if not os.path.isabs(CHECKPOINT_REL_PATH) else CHECKPOINT_REL_PATH
    checkpoint_path = find_latest_checkpoint(base_path)
    
    if not ray.is_initialized(): ray.init(ignore_reinit_error=True)
    
    try:
        _, training_tickers, _, _ = load_data_assets()
    except:
        return None
        
    # Apply real sector mapping so the environment constraints hold
    training_sectors = {t: REAL_SECTORS.get(t, "Other") for t in training_tickers}
    prices_path_real = os.path.join(get_project_paths(), "data", "raw", "sp500_prices.csv")

    training_config = {
        "prices_csv": prices_path_real, 
        "tickers": training_tickers, 
        "sectors": training_sectors,
        "initial_capital": 100000,
        "lookback_window": 30,
        "news_dim": 384,
        "macro_dim": 3,
        "cvar_lambda": 0.05, 
        "sector_max_weight": 0.25,
    }

    register_env("RiskAwarePortfolioEnv-v0", lambda cfg: make_env({**training_config, **cfg}))
    
    config = (
        PPOConfig()
        .environment(env="RiskAwarePortfolioEnv-v0", env_config=training_config)
        .framework("torch")
        .env_runners(num_env_runners=0)
        .training(model={"fcnet_hiddens": [256, 256], "fcnet_activation": "tanh"})
        .resources(num_gpus=0)
    )
    
    try:
        algo = config.build()
        algo.restore(checkpoint_path)
        return algo
    except Exception as e:
        print(f"Agent Load Error: {e}")
        return None

# --- CRITIC ---
try:
    print("🤖 Attempting to load LLM Critic...")
    critic = LLMCritic()
    CRITIC_STATUS = "✅ Online (Llama 3)"
    print("✅ LLM Critic loaded successfully!")
except Exception as e:
    critic = None
    CRITIC_STATUS = "⚠️ Offline (Ollama not running?)"
    print(f"⚠️ CRITIC CRASHED DURING STARTUP: {e}")

# --- BACKTEST LOGIC ---
def run_backtest(start_date, end_date, selected_tickers):
    if not selected_tickers: return [None]*6 + ["⚠️ Error: Please select at least one ticker."]

    try:
        prices, all_tickers, news_embs, macro_feats = load_data_assets()
        dt_start, dt_end = pd.to_datetime(start_date), pd.to_datetime(end_date)
        # Give environment a full year to compute historical CVaR correctly
        buffer_start = dt_start - timedelta(days=365) 
        super_prices = prices.loc[buffer_start:dt_end, selected_tickers]
        if super_prices.empty: return [None]*6 + ["No data for date range."]
    except Exception as e:
        return [None]*6 + [f"Data Error: {e}"]

    # Apply real sector mapping
    sectors = {t: REAL_SECTORS.get(t, "Other") for t in selected_tickers}
    prices_path_real = os.path.join(get_project_paths(), "data", "raw", "sp500_prices.csv")

    env_config = {
        "prices_csv": prices_path_real, 
        "tickers": selected_tickers,
        "sectors": sectors,
        "news_embeddings": news_embs,
        "macro_features": macro_feats,
        "initial_capital": 100000,
        "lookback_window": 30,
        "news_dim": 384,
        "macro_dim": 3,
        "cvar_lambda": 0.05, 
        "sector_max_weight": 0.25,
    }
    
    algo = load_trained_agent()
    if not algo: return [None]*6 + ["Agent load failed."]
    
    env = RiskAwarePortfolioEnv(**env_config)
    env.returns = super_prices.pct_change().dropna()
    env.dates = env.returns.index
    
    try: start_index = env.dates.get_indexer([dt_start], method='nearest')[0]
    except: start_index = 30
        
    env.current_step = max(30, start_index) 
    obs, _ = env.reset()
    env.current_step = max(30, start_index)
    
    done = False
    history_dates, history_val, weights_history, cvar_history, audit_log = [], [], [], [], []
    latest_risk_score = 0.5
    
    module = algo.get_module("default_policy")

    try:
        while not done:
            
            lookback = env_config["lookback_window"]
            n_selected = len(selected_tickers)
            price_len = n_selected * lookback
            
            # Separate the dynamic prices from the fixed news/macro data
            extracted_prices = obs[:price_len]
            news_and_macro = obs[price_len:]
            
            # 1. Map inputs to the global 50-ticker space to maintain network alignment
            n_all = len(all_tickers)
            full_prices = np.zeros(n_all * lookback, dtype=np.float32)
            
            for i, ticker in enumerate(all_tickers):
                if ticker in selected_tickers:
                    sel_idx = selected_tickers.index(ticker)
                    # Insert the 30 days of price data in the EXACT correct position
                    full_prices[i*lookback : (i+1)*lookback] = extracted_prices[sel_idx*lookback : (sel_idx+1)*lookback]
                    
            # Reconstruct the perfect 1887-dimension observation tensor
            fixed_obs = np.concatenate([full_prices, news_and_macro])
            obs_tensor = torch.tensor(fixed_obs, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                output = module.forward_inference({"obs": obs_tensor})
                full_action = output["action_dist_inputs"][0].numpy()
            
            # 2. Extract ONLY the actions for the selected tickers from their original global positions
            action = np.zeros(len(selected_tickers), dtype=np.float32)
            for i, ticker in enumerate(selected_tickers):
                global_idx = all_tickers.index(ticker)
                action[i] = full_action[global_idx]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if env.current_step >= len(env.dates) or env.dates[env.current_step-1] > dt_end:
                done = True
            
            if not done:
                current_date = env.dates[env.current_step - 1]
                if current_date >= dt_start:
                    history_dates.append(current_date)
                    history_val.append(info['portfolio_value'])
                    weights_history.append(env.weights)
                    cvar_history.append(info.get('cvar', 0.0))
                    
                    # --- NEW: EVENT-DRIVEN CRITIC ---
                    if critic:
                        if len(weights_history) > 1:
                            turnover = np.sum(np.abs(env.weights - weights_history[-2]))
                        else:
                            turnover = 0.0
                            
                        is_month_end = current_date.is_month_end
                        is_major_shift = turnover > 0.05 
                        
                        if is_month_end or is_major_shift:
                            # Grab the top 5 heaviest weighted stocks to ensure Critic always has data
                            top_idx = np.argsort(env.weights)[::-1][:5]
                            w_dict = {selected_tickers[i]: float(env.weights[i]) for i in top_idx if i < len(selected_tickers)}
                            
                            if w_dict:
                                try:
                                    reason = "End of Month Review" if is_month_end else "Major Portfolio Reallocation"
                                    print(f"🧠 Critic analyzing date: {current_date.date()} ({reason})...")
                                    alert, warning, risk = critic.evaluate(w_dict)
                                    latest_risk_score = risk
                                    print(f"   -> Alert: {alert} | Risk: {risk}")
                                    audit_log.append(f"### 📅 {current_date.date()} ({reason})\n**Risk Score:** `{risk:.2f}`\n* **Alert:** {alert}\n* **Warning:** {warning}\n---")
                                except Exception as e:
                                    print(f"⚠️ Critic Error on {current_date.date()}: {e}")
                
    except RuntimeError as e:
        return [None]*6 + [f"Runtime Error: {e}"]

    # --- FETCH BENCHMARK (SPY) ---
    print("📈 Fetching S&P 500 Benchmark...")
    benchmark_val = []
    try:
        start_str = history_dates[0].strftime('%Y-%m-%d')
        end_str = (history_dates[-1] + pd.Timedelta(days=2)).strftime('%Y-%m-%d')
        
        # Download SPY data
        spy_df = yf.download('SPY', start=start_str, end=end_str, progress=False)
        
        # Safely extract 'Close' or 'Adj Close' handling new yfinance MultiIndex formats
        if isinstance(spy_df.columns, pd.MultiIndex):
            spy_df.columns = spy_df.columns.get_level_values(0)
            
        if 'Adj Close' in spy_df.columns:
            spy_series = spy_df['Adj Close']
        elif 'Close' in spy_df.columns:
            spy_series = spy_df['Close']
        else:
            spy_series = spy_df.iloc[:, 0]
            
        if isinstance(spy_series, pd.DataFrame): 
            spy_series = spy_series.iloc[:, 0]
            
        spy_series = spy_series.reindex(history_dates).ffill().bfill()
        spy_returns = spy_series.pct_change().fillna(0)
        
        bench_cap = env_config["initial_capital"]
        for r in spy_returns:
            bench_cap *= (1 + float(r))
            benchmark_val.append(bench_cap)
    except Exception as e:
        print(f"⚠️ Benchmark fetch failed: {e}")
        benchmark_val = None

    charts = make_dashboard_charts(history_dates, history_val, weights_history, selected_tickers, benchmark_val, latest_risk_score, cvar_history)
    final_log = "\n".join(audit_log) if audit_log else "No risk alerts generated."
    
    return charts[0], charts[1], charts[2], charts[3], charts[4], charts[5], final_log

# --- UI INITIALIZATION ---
try:
    _, all_tickers, _, _ = load_data_assets()
except:
    all_tickers = ["AAPL", "MSFT", "GOOGL"] 

with gr.Blocks(title="AI Portfolio Manager") as demo:
    gr.Markdown("# 🤖 Risk-Aware AI Portfolio Manager (CS 610 Project)")
    gr.Markdown("An autonomous portfolio manager combining Deep Reinforcement Learning (PPO) with Generative AI (RAG) constraints.")
    
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### ⚙️ Engine Settings")
            with gr.Accordion("Select Tickers (Universe)", open=False):
                ticker_selector = gr.Dropdown(choices=all_tickers, value=all_tickers[:10], multiselect=True, label="Tickers", interactive=True)
                
            start_input = gr.Textbox(value="2020-02-15", label="Start Date (e.g., COVID Crash)")
            end_input = gr.Textbox(value="2020-06-15", label="End Date")
            run_btn = gr.Button("🚀 Run Simulation", variant="primary")
            
            gr.Markdown("---")
            gr.Markdown("### 📊 Strategy Alpha")
            alpha_table_box = gr.Markdown(value="*Run simulation to calculate Alpha...*")
            gr.Markdown(f"**Critic Engine:** {CRITIC_STATUS}")

        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("📈 Performance Overview"):
                    p_equity = gr.Plot(label="Cumulative Return")
                    p_drawdown = gr.Plot(label="Maximum Drawdown")
                    
                with gr.TabItem("⚖️ Risk & Analytics"):
                    with gr.Row():
                        p_radar = gr.Plot(label="Risk Regime Radar")
                        p_sharpe = gr.Plot(label="Rolling Sharpe Ratio")
                    p_alloc = gr.Plot(label="Dynamic Asset Allocation")
                    
                with gr.TabItem("🧠 AI Supervisor Audit Logs"):
                    audit_box = gr.Markdown(value="*LLM outputs will appear here...*")

    run_btn.click(
        run_backtest, 
        inputs=[start_input, end_input, ticker_selector], 
        outputs=[p_equity, p_drawdown, p_alloc, p_sharpe, p_radar, alpha_table_box, audit_box]
    )

if __name__ == "__main__":
    demo.launch(share=False)
