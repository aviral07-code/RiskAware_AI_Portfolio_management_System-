import os
from typing import Dict, Any

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from src.envs.risk_aware_env import RiskAwarePortfolioEnv

def make_env(cfg: Dict[str, Any]):
    return RiskAwarePortfolioEnv(**cfg)

def train_ppo(
    prices_csv: str,
    tickers: list,
    sectors: dict,
    num_iters: int = 50, # Reduced default for quicker testing
    num_workers: int = 0, # Default to 0 for Mac stability
    logdir: str = "results/ppo",
    extra_env_config: dict = None,
):
    ray.init(ignore_reinit_error=True)

    env_config = {
        "prices_csv": prices_csv,
        "tickers": tickers,
        "sectors": sectors,
        "transaction_cost": 0.0002,
        "cvar_lambda": 0.05,
        "sector_max_weight": 0.25,
        "lookback_window": 30,
        "cvar_window": 250,
        "initial_capital": 100000,
    }
    if extra_env_config:
        env_config.update(extra_env_config)

    register_env("RiskAwarePortfolioEnv-v0", lambda cfg: make_env({**env_config, **cfg}))

    # --- UPDATED TO NEW RAY 2.10+ API ---
    config = (
        PPOConfig()
        .environment(env="RiskAwarePortfolioEnv-v0", env_config={})
        .framework("torch")
        .env_runners(num_env_runners=num_workers) # Changed from .rollouts
        .training(
            gamma=0.99,
            lr=5e-5,
            train_batch_size_per_learner=2000, # Changed from train_batch_size
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "tanh", # tanh is usually better for PPO
            },
        )
        .resources(num_gpus=0)
    )

    algo = config.build()
    os.makedirs(logdir, exist_ok=True)

    print(f"Starting training for {num_iters} iterations...")
    for i in range(num_iters):
        result = algo.train()
        
        # Safe access for new API structure
        mean_reward = result.get('env_runners', {}).get('episode_reward_mean', 0)
        print(f"Iter {i+1} reward_mean={mean_reward:.4f}")
        
        if (i+1) % 10 == 0:
            checkpoint = algo.save(logdir)
            print("Checkpoint saved:", checkpoint)

    ray.shutdown()

# Helper to load for inference
def load_agent(checkpoint_path: str, env_config: dict):
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
        
    register_env("RiskAwarePortfolioEnv-v0", lambda cfg: make_env({**env_config, **cfg}))
    
    config = (
        PPOConfig()
        .environment(env="RiskAwarePortfolioEnv-v0", env_config=env_config)
        .framework("torch")
        .env_runners(num_env_runners=0) # Inference needs 0 workers
        .training(model={"fcnet_hiddens": [256, 256], "fcnet_activation": "tanh"})
        .resources(num_gpus=0)
    )
    
    algo = config.build()
    algo.restore(checkpoint_path)
    return algo
