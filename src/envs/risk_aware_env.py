import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional

CVaR_ALPHA = 0.95

class RiskAwarePortfolioEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        prices_csv: str,
        tickers: list,
        sectors: Dict[str, str],
        transaction_cost: float = 0.0002,
        cvar_lambda: float = 0.05,
        sector_max_weight: float = 0.25,
        lookback_window: int = 30,
        cvar_window: int = 250,
        initial_capital: float = 100000.0,
        render_mode: Optional[str] = None,
        news_embeddings: Optional[Dict[pd.Timestamp, np.ndarray]] = None,
        macro_features: Optional[Dict[pd.Timestamp, np.ndarray]] = None,
        # --- NEW ARGUMENTS ---
        news_dim: int = 384,  # Default for all-MiniLM-L6-v2
        macro_dim: int = 3,   # Default for your macro script
    ):
        super().__init__()

        # 1. Load Data
        self.prices = pd.read_csv(prices_csv, index_col=0, parse_dates=True)[tickers]
        self.returns = self.prices.pct_change().dropna()
        self.dates = self.returns.index
        
        self.n_assets = len(tickers)
        self.tickers = tickers
        self.sectors = sectors
        
        # 2. Parameters
        self.transaction_cost = transaction_cost
        self.cvar_lambda = cvar_lambda
        self.sector_max_weight = sector_max_weight
        self.lookback_window = lookback_window
        self.cvar_window = cvar_window
        self.initial_capital = initial_capital
        self.render_mode = render_mode

        # 3. Sector Constraints Setup
        self.unique_sectors = sorted(set(sectors[t] for t in tickers))
        self.sector_matrix = self._build_sector_matrix()

        # 4. Feature Dimensions (FIXED)
        self.news_embeddings = news_embeddings or {}
        self.macro_features = macro_features or {}
        
        # We now use the PASSED dimensions, not the dynamic ones
        self.news_dim = news_dim 
        self.macro_dim = macro_dim

        # 5. Define Spaces
        price_obs_dim = self.n_assets * self.lookback_window
        total_obs_dim = price_obs_dim + self.news_dim + self.macro_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_assets,), dtype=np.float32
        )

        # 6. Internal State
        self.current_step = 0
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = initial_capital
        self.portfolio_values = []

    def _build_sector_matrix(self) -> np.ndarray:
        mat = np.zeros((len(self.unique_sectors), self.n_assets))
        for j, t in enumerate(self.tickers):
            s = self.sectors[t]
            s_idx = self.unique_sectors.index(s)
            mat[s_idx, j] = 1.0
        return mat

    def _get_price_window(self) -> np.ndarray:
        start = self.current_step - self.lookback_window
        end = self.current_step
        
        # Slicing in Pandas is permissive (won't error on bounds), but returns short arrays
        slice_data = self.returns.iloc[start:end].values
        
        # --- CRASH FIX: Enforce Shape ---
        # If we retrieved fewer rows than the lookback_window, we must PAD with zeros
        rows_retrieved = slice_data.shape[0]
        if rows_retrieved < self.lookback_window:
            missing_rows = self.lookback_window - rows_retrieved
            # Create a block of zeros: (Missing_Rows, N_Assets)
            padding = np.zeros((missing_rows, self.n_assets), dtype=np.float32)
            # Stack zeros on top of the data we have
            slice_data = np.vstack([padding, slice_data])
            
        # Transpose to get (N_assets, Window)
        return slice_data.T

    def _get_news_features(self, dt) -> np.ndarray:
        # Return zeros of the CORRECT size if data is missing
        if not self.news_embeddings:
            return np.zeros(self.news_dim, dtype=np.float32)
        
        vec = self.news_embeddings.get(dt, np.zeros(self.news_dim))
        return vec.astype(np.float32)

    def _get_macro_features(self, dt) -> np.ndarray:
        if not self.macro_features:
            return np.zeros(self.macro_dim, dtype=np.float32)
        
        vec = self.macro_features.get(dt, np.zeros(self.macro_dim))
        return vec.astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        window = self._get_price_window()
        price_features = window.flatten().astype(np.float32)
        
        if self.current_step > 0 and self.current_step <= len(self.dates):
            dt = self.dates[self.current_step - 1]
        else:
            dt = self.dates[0]

        news_features = self._get_news_features(dt)
        macro_features = self._get_macro_features(dt)
        
        state = np.concatenate([price_features, news_features, macro_features])
        return state

    def _get_obs_safe(self) -> np.ndarray:
        temp_step = self.current_step
        self.current_step = min(self.current_step, len(self.returns))
        obs = self._get_obs()
        self.current_step = temp_step
        return obs

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.current_step = self.lookback_window
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = self.initial_capital
        self.portfolio_values = [self.portfolio_value]
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Softmax
        raw = np.array(action, dtype=np.float32)
        exp_raw = np.exp(raw - np.max(raw))
        weights = exp_raw / (exp_raw.sum() + 1e-8)

        # Constraints
        sector_exposure = self.sector_matrix @ weights
        over = sector_exposure > self.sector_max_weight
        
        if over.any():
            capped_weights = weights.copy()
            for s_idx, is_over in enumerate(over):
                if is_over:
                    sector_assets = self.sector_matrix[s_idx] == 1
                    sector_sum = weights[sector_assets].sum()
                    if sector_sum > 0:
                        scale = self.sector_max_weight / sector_sum
                        capped_weights[sector_assets] *= scale
            
            residual = 1.0 - capped_weights.sum()
            if residual > 0:
                capped_weights += residual / self.n_assets
            weights = capped_weights / (capped_weights.sum() + 1e-8)

        prev_weights = self.weights
        self.weights = weights
        turnover = np.abs(self.weights - prev_weights).sum()
        tc_cost = self.transaction_cost * turnover

        if self.current_step >= len(self.returns):
            return self._get_obs_safe(), 0.0, True, False, self._get_info(0.0, 0.0, 0.0)

        r_t = float(self.returns.iloc[self.current_step].values @ self.weights)
        self.portfolio_value *= (1.0 + r_t - tc_cost)
        self.portfolio_values.append(self.portfolio_value)

        start = max(self.current_step - self.cvar_window, 0)
        hist_returns_window = self.returns.iloc[start:self.current_step].values
        hist_port_ret = hist_returns_window @ self.weights
        
        if len(hist_port_ret) > 0:
            var_threshold = np.percentile(hist_port_ret, (1 - CVaR_ALPHA) * 100)
            tail_losses = hist_port_ret[hist_port_ret < var_threshold]
            cvar = -tail_losses.mean() if len(tail_losses) > 0 else 0.0
        else:
            cvar = 0.0

        reward = float((r_t - tc_cost - (self.cvar_lambda * cvar)) * 100)

        self.current_step += 1
        terminated = self.current_step >= len(self.returns)
        truncated = False

        obs = self._get_obs_safe() if terminated else self._get_obs()
        info = self._get_info(r_t, turnover, cvar)

        return obs, float(reward), terminated, truncated, info

    def _get_info(self, r_t=0.0, turnover=0.0, cvar=0.0):
        return {
            "portfolio_value": self.portfolio_value,
            "turnover": turnover,
            "cvar": cvar,
            "raw_return": r_t,
        }

    def render(self):
        if self.render_mode == "human":
            print(f"Step {self.current_step} | Value: {self.portfolio_value:.4f}")
