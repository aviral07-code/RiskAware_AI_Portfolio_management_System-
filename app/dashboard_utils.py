import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def compute_drawdown(values: pd.Series) -> pd.Series:
    cummax = values.cummax()
    dd = (values - cummax) / cummax
    return dd

def compute_sharpe_ratio(values: pd.Series, risk_free_rate=0.02) -> float:
    returns = values.pct_change().dropna()
    if len(returns) == 0 or returns.std() == 0: return 0.0
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

def generate_alpha_table(dates, ai_values, bench_values):
    """Generates a Markdown table comparing AI vs Benchmark metrics."""
    ai_s = pd.Series(ai_values, index=dates)
    
    ai_ret = (ai_s.iloc[-1] / ai_s.iloc[0]) - 1
    ai_mdd = compute_drawdown(ai_s).min()
    ai_sharpe = compute_sharpe_ratio(ai_s)
    
    if bench_values is not None and len(bench_values) > 0:
        bn_s = pd.Series(bench_values, index=dates)
        bn_ret = (bn_s.iloc[-1] / bn_s.iloc[0]) - 1
        bn_mdd = compute_drawdown(bn_s).min()
        bn_sharpe = compute_sharpe_ratio(bn_s)
    else:
        bn_ret, bn_mdd, bn_sharpe = 0.0, 0.0, 0.0

    md = f"""
| Metric | 🤖 AI Portfolio | 📈 S&P 500 (SPY) | 🏆 Advantage |
| :--- | :--- | :--- | :--- |
| **Total Return** | `{ai_ret*100:.2f}%` | `{bn_ret*100:.2f}%` | **{(ai_ret - bn_ret)*100:+.2f}%** |
| **Max Drawdown** | `{ai_mdd*100:.2f}%` | `{bn_mdd*100:.2f}%` | **{(ai_mdd - bn_mdd)*100:+.2f}%** |
| **Sharpe Ratio** | `{ai_sharpe:.2f}` | `{bn_sharpe:.2f}` | **{(ai_sharpe - bn_sharpe):+.2f}** |
    """
    return md

def make_dashboard_charts(dates, portfolio_values, weights_history, tickers, benchmark_values, final_critic_risk, cvar_history):
    # 1. Equity Curve (AI vs Benchmark)
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(x=dates, y=portfolio_values, mode="lines", name="AI Portfolio", line=dict(color="#00D4FF", width=3)))
    if benchmark_values is not None:
        fig_equity.add_trace(go.Scatter(x=dates, y=benchmark_values, mode="lines", name="S&P 500 (SPY)", line=dict(color="#7F8C8D", width=2, dash="dash")))
        
    fig_equity.update_layout(title="💰 Cumulative Portfolio Performance", xaxis_title="Date", yaxis_title="Value ($)", template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20), hovermode="x unified")

    # 2. Drawdown (AI vs Benchmark)
    values = pd.Series(portfolio_values, index=dates)
    dd = compute_drawdown(values)
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dates, y=dd*100, mode="lines", name="AI Drawdown", fill='tozeroy', line=dict(color="#FF5555", width=2)))
    
    if benchmark_values is not None:
        bn_s = pd.Series(benchmark_values, index=dates)
        bn_dd = compute_drawdown(bn_s)
        fig_dd.add_trace(go.Scatter(x=dates, y=bn_dd*100, mode="lines", name="S&P 500 Drawdown", line=dict(color="#7F8C8D", width=1.5, dash="dot")))
        
    fig_dd.update_layout(title="📉 Max Drawdown (%)", xaxis_title="Date", yaxis_title="Drawdown (%)", template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20), hovermode="x unified")

    # 3. Asset Allocation (Stacked Area)
    df_weights = pd.DataFrame(weights_history, index=dates, columns=tickers)
    top_tickers = df_weights.mean().sort_values(ascending=False).head(10).index
    df_plot = df_weights[top_tickers]
    fig_alloc = px.area(df_plot, x=df_plot.index, y=df_plot.columns, title="📊 Sector/Asset Allocation Over Time", template="plotly_dark")
    fig_alloc.update_layout(margin=dict(l=20, r=20, t=50, b=20))

    # 4. Rolling Sharpe Ratio (Risk-Adjusted Consistency)
    returns = values.pct_change().dropna()
    daily_rf = 0.02 / 252
    excess_returns = returns - daily_rf
    rolling_sharpe = (excess_returns.rolling(window=21).mean() / excess_returns.rolling(window=21).std()) * np.sqrt(252)
    
    fig_sharpe = go.Figure()
    fig_sharpe.add_trace(go.Scatter(x=returns.index, y=rolling_sharpe, mode="lines", name="AI 21-Day Sharpe", line=dict(color="#F39C12", width=2)))
    fig_sharpe.update_layout(title="⚖️ 21-Day Rolling Sharpe Ratio", xaxis_title="Date", yaxis_title="Sharpe Ratio", template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20))
    fig_sharpe.add_hline(y=1.0, line_dash="dot", line_color="white", annotation_text="Good (>1.0)")

    # 5. Risk Radar Chart (Current Portfolio Snapshot)
    # Calculate components for the radar
    latest_volatility = returns.tail(21).std() * np.sqrt(252) # Annualized vol
    vol_score = min(latest_volatility / 0.40, 1.0) # Normalize assuming 40% is extreme
    
    max_weight = df_weights.iloc[-1].max() # Sector Concentration
    concentration_score = min(max_weight / 0.25, 1.0) # Normalize assuming 50% in one stock is extreme
    
    avg_cvar = np.mean(cvar_history[-21:]) if cvar_history else 0
    cvar_score = min(avg_cvar / 0.05, 1.0) # Normalize assuming 5% daily CVaR is extreme
    
    categories = ['Market Volatility', 'Sector Concentration', 'AI Critic Risk Score', 'CVaR (Tail Risk)', 'Recent Drawdown']
    radar_values = [vol_score, concentration_score, final_critic_risk, cvar_score, abs(dd.iloc[-1])]

    fig_radar = go.Figure(data=go.Scatterpolar(
        r=radar_values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(0, 212, 255, 0.3)',
        line=dict(color='#00D4FF')
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title="🕸️ Current Risk Regime Radar",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=20)
    )

    alpha_table = generate_alpha_table(dates, portfolio_values, benchmark_values)

    return fig_equity, fig_dd, fig_alloc, fig_sharpe, fig_radar, alpha_table
