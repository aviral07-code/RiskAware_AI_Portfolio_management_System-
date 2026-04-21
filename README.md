# Risk-Aware AI Portfolio Manager

An autonomous portfolio management system that combines Deep Reinforcement Learning (PPO) with Generative AI (RAG) constraints. Developed as a Master's project at the University of Kentucky under the supervision of Dr. Brent Harrison, this system enforces risk limits, sector diversification, and tail-risk (CVaR) reduction while navigating dynamic market regimes.

## 🌊 System Data Flow

The architecture operates on a continuous pipeline of multi-modal data:

1. **Ingestion:** Financial price histories (`yfinance`), macroeconomic indicators, and raw news articles (`GNews`/`newspaper3k`) are ingested daily.
2. **Processing:** News text is encoded into dense vectors using `sentence-transformers` and stored in a FAISS Vector Database.
3. **Environment Generation:** A custom Gymnasium environment (`RiskAwarePortfolioEnv`) fuses the 30-day price lookbacks, macro features, and daily news embeddings into a 1887-dimensional state space. 
4. **Agent Action:** A Proximal Policy Optimization (PPO) agent processes the state to output target asset weights, explicitly penalized for high Conditional Value at Risk (CVaR) and sector overconcentration.
5. **Supervisor Audit:** An LLM Critic (Llama 3 via Ollama) monitors major reallocations using RAG against recent financial news, flagging volatility risks and bias.
6. **Visualization:** Performance is tracked in real-time via a local Gradio and Plotly dashboard, detailing Alpha, Max Drawdown, and Rolling Sharpe Ratios.

## 🛠️ Project Structure

```text
├── app/
│   ├── main.py                 # Gradio UI application
│   └── dashboard_utils.py      # Plotly chart generation & metrics
├── data/
│   ├── raw/                    # CSVs, JSONL news files
│   └── vector_db/              # FAISS index for LLM Critic
├── notebooks/                  # Experimental data fetching & EDA
├── results/                    # Ray RLlib agent checkpoints
├── scripts/
│   ├── fetch_real_news.py      # Scrapes daily articles
│   ├── build_vector_db.py      # FAISS ingestion for RAG
│   ├── process_news_for_rl.py  # Builds daily dense embeddings
│   ├── train_ppo.py            # Ray RLlib training loop
│   └── debug_system.py         # Diagnostic data shape checks
└── src/
    ├── agents/
    │   ├── llm_critic.py       # Llama 3 Risk Manager
    │   └── ppo_agent.py        # Ray PPO configuration
    ├── envs/
    │   └── risk_aware_env.py   # Custom Gymnasium portfolio environment
    └── utils/
        ├── bias_checker.py     # Sector & recency bias detection
        └── sentiment.py        # FinBERT sentiment analysis

🚀 Installation & Setup
1. Clone the repository and install dependencies:

Bash
git clone [https://github.com/yourusername/risk-aware-ai-portfolio.git](https://github.com/aviral07-code/risk-aware-ai-portfolio.git)
cd risk-aware-ai-portfolio
pip install -r requirements.txt

2. Install and Start Ollama (Required for LLM Critic):

Download Ollama from ollama.com

Pull the Llama 3.1 model:

Bash
ollama run llama3.1
🏃‍♂️ Running the Pipeline
1. Data Collection & Processing:
Populate the data/raw directory with market prices, macro features, and news embeddings.

Bash
python scripts/fetch_real_news.py
python scripts/process_news_for_rl.py
python scripts/build_vector_db.py
2. Train the RL Agent:
Run the diagnostic tool to ensure data shape alignment, then initiate training.

Bash
python scripts/debug_system.py
python scripts/train_ppo.py
3. Launch the Dashboard:
Start the Gradio web UI to run backtests, visualize asset allocations, and monitor the LLM Critic's audit logs.

Bash
python app/main.py
⚠️ Disclaimer
This software is for academic and educational purposes only. It is not financial advice. Do not use this system to trade real capital.
