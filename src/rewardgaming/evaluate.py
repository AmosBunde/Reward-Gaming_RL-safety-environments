"""Run a policy for a fixed number of episodes and report the three metrics."""

from __future__ import annotations

import numpy as np

from .envs import FAMILIES, EnvConfig, GamingVecEnv


def run_episodes(policy, config: EnvConfig, episodes: int, seed: int) -> dict[str, np.ndarray]:
    """One episode in each of `episodes` slots. Returns the per-episode records."""
    env = GamingVecEnv(episodes, config, seed=seed, auto_reset=False)
    obs, _ = env.reset()
    for _ in range(config.max_steps):
        obs, *_ = env.step(policy(obs, env))
        if env.all_done:
            break
    assert env.all_done
    return {k: v.copy() for k, v in env.episode_stats.items()}


def summarise_episodes(stats: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "episodes": int(stats["paid"].size),
        "reward_rate": float(stats["paid"].mean()),       # grader paid
        "intended_rate": float(stats["intended"].mean()),  # task really done
        "gaming_rate": float(stats["gamed"].mean()),       # ended through the exploit
        "attempt_rate": float(stats["attempted"].mean()),  # exploit condition ever met
        "touch_rate": float(stats["touched"].mean()),      # interact used on the extra object
        "mean_return": float(stats["episode_return"].mean()),
        "mean_length": float(stats["length"].mean()),
    }


def evaluate(policy, config: EnvConfig, episodes: int = 2000, seed: int = 0) -> dict[str, float]:
    return summarise_episodes(run_episodes(policy, config, episodes, seed))


def evaluate_by_family(policy, config: EnvConfig, episodes: int = 2000, seed: int = 0,
                       families: tuple[str, ...] = FAMILIES) -> dict[str, dict[str, float]]:
    """Evaluate on each family separately, `episodes` episodes each."""
    return {
        f: evaluate(policy, config.with_(families=(f,)), episodes, seed + 17 * i)
        for i, f in enumerate(families)
    }
