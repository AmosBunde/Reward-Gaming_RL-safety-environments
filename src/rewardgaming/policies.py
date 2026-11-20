"""Baseline policies that need no training.

`random_policy` picks uniformly. `honest_policy` and `exploit_policy` are scripted and
read the true state of the environment. They are reference points: the honest script
shows the task can always be done, and the exploit script shows what the grader can be
made to pay for without the task being done.

Every policy here has the signature `policy(obs, env) -> actions`.
"""

from __future__ import annotations

import numpy as np

from .envs import (BLINDSPOT, DOWN, INTERACT, LEFT, NUM_ACTIONS, OVERWRITE, PROXY, RIGHT,
                   TAMPER, UP, GamingVecEnv)


def make_random_policy(seed: int = 0):
    rng = np.random.default_rng(seed)

    def policy(obs: np.ndarray, env: GamingVecEnv) -> np.ndarray:
        return rng.integers(0, NUM_ACTIONS, size=env.num_envs)

    return policy


def _toward(env: GamingVecEnv, tx: np.ndarray, ty: np.ndarray,
            avoid_parcel: bool = False) -> np.ndarray:
    """One step along a shortest path to (tx, ty). INTERACT where already on the target.

    The default path walks along the row first, then the column. With `avoid_parcel` the
    path is chosen so that it never steps on the parcel (which would pick it up): the
    column is walked first when the row-first path is blocked, and when the parcel sits
    on the straight line to the target the agent steps sideways round it.
    """
    ax, ay, g = env.ax, env.ay, env.config.grid
    step_x = np.where(tx > ax, RIGHT, LEFT)
    step_y = np.where(ty > ay, DOWN, UP)
    need_x, need_y = ax != tx, ay != ty
    x_first = np.ones_like(need_x)
    sidestep = np.zeros_like(need_x)
    if avoid_parcel:
        vx, vy = env.px, env.py
        between_x = (vx > np.minimum(ax, tx)) & (vx < np.maximum(ax, tx))
        between_y = (vy > np.minimum(ay, ty)) & (vy < np.maximum(ay, ty))
        # The row-first path covers row ay up to column tx, then column tx.
        blocks_row_first = ((vy == ay) & (between_x | (vx == tx))) | ((vx == tx) & between_y)
        x_first = ~blocks_row_first
        sidestep = (need_x & ~need_y & (vy == ay) & between_x) | \
                   (need_y & ~need_x & (vx == ax) & between_y)
    act = np.where(need_x & (x_first | ~need_y), step_x,
                   np.where(need_y, step_y, INTERACT))
    side = np.where(need_x, np.where(ay < g - 1, DOWN, UP), np.where(ax < g - 1, RIGHT, LEFT))
    return np.where(sidestep, side, act)


def honest_policy(obs: np.ndarray, env: GamingVecEnv) -> np.ndarray:
    """Walk onto the parcel, carry it to the depot pad and put it down."""
    to_parcel = _toward(env, env.px, env.py)
    to_parcel = np.where(to_parcel == INTERACT, UP, to_parcel)  # standing on a dropped parcel
    to_depot = _toward(env, env.dx, env.dy)
    return np.where(env.carry == 1, to_depot,
                    np.where(env.carry == 2, INTERACT, to_parcel))


def exploit_policy(obs: np.ndarray, env: GamingVecEnv) -> np.ndarray:
    """Use each family's exploit by its shortest route. Honest where no object exists."""
    to_parcel = _toward(env, env.px, env.py)
    to_depot = _toward(env, env.dx, env.dy)
    to_extra = _toward(env, env.ex, env.ey)
    to_extra_free = _toward(env, env.ex, env.ey, avoid_parcel=True)
    tamper = to_extra_free
    proxy = np.where(env.carry == 2, to_depot, to_extra_free)
    overwrite = np.where(env.carry == 2, to_parcel, to_extra_free)
    blindspot = np.where(env.carry == 1, to_extra, to_parcel)
    act = np.select(
        [env.fam == TAMPER, env.fam == PROXY, env.fam == OVERWRITE, env.fam == BLINDSPOT],
        [tamper, proxy, overwrite, blindspot])
    return np.where(env.present, act, honest_policy(obs, env))
