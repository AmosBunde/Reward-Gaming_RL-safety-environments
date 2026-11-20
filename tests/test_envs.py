import numpy as np
import pytest

from rewardgaming import (FAMILIES, NUM_ACTIONS, OBS_DIM, OBS_FIELDS, EnvConfig, GamingEnv,
                          GamingVecEnv, level_pool, level_table)
from rewardgaming.envs import INTERACT, LEFT, RIGHT, UP
from rewardgaming.evaluate import evaluate, run_episodes
from rewardgaming.policies import exploit_policy, honest_policy, make_random_policy
from rewardgaming.render import action_to_text, describe, text_to_action


def always(action):
    return lambda obs, env: np.full(env.num_envs, action)


def place(env: GamingEnv, agent, parcel, depot, extra):
    """Overwrite the layout of a single-level environment."""
    v = env.vec
    (v.ax[0], v.ay[0]), (v.px[0], v.py[0]) = agent, parcel
    (v.dx[0], v.dy[0]), (v.ex[0], v.ey[0]) = depot, extra


# -- interface ---------------------------------------------------------------------------

def test_observation_format():
    env = GamingVecEnv(64, EnvConfig(), seed=0)
    obs, _ = env.reset()
    assert obs.shape == (64, OBS_DIM) and obs.dtype == np.float32
    assert len(OBS_FIELDS) == OBS_DIM
    for _ in range(30):
        obs, reward, term, trunc, _ = env.step(np.random.default_rng(0).integers(0, 5, 64))
        assert obs.min() >= -1.0 and obs.max() <= 1.0
        assert reward.shape == term.shape == trunc.shape == (64,)


def test_single_env_follows_gymnasium_convention():
    env = GamingEnv("proxy", seed=1)
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs) and info["family"] == "proxy"
    assert env.action_space.n == NUM_ACTIONS
    out = env.step(env.action_space.sample())
    assert len(out) == 5 and isinstance(out[1], float) and isinstance(out[2], bool)


def test_invalid_action_is_rejected():
    env = GamingVecEnv(4)
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.array([0, 1, 2, 5]))


def test_step_after_end_raises_in_single_env():
    env = GamingEnv("tamper", max_steps=2)
    env.reset(seed=0)
    env.step(UP)
    env.step(UP)
    with pytest.raises(RuntimeError):
        env.step(UP)


# -- determinism and level generation ----------------------------------------------------

def test_same_seed_gives_same_trajectories():
    def roll(seed):
        env = GamingVecEnv(32, EnvConfig(p_absent=0.3, audit_rate=0.2, grader_noise=0.1), seed=seed)
        obs, _ = env.reset()
        rng = np.random.default_rng(99)
        trace = [obs]
        for _ in range(60):
            obs, reward, *_ = env.step(rng.integers(0, 5, 32))
            trace += [obs, reward]
        return np.concatenate([t.ravel() for t in trace])

    assert np.array_equal(roll(5), roll(5))
    assert not np.array_equal(roll(5), roll(6))


def test_reset_with_seed_reproduces_level():
    env = GamingEnv(FAMILIES)
    a, _ = env.reset(seed=11)
    state = env.state()
    env.step(RIGHT)
    b, _ = env.reset(seed=11)
    assert np.array_equal(a, b) and env.state() == state


@pytest.mark.parametrize("family", FAMILIES)
def test_train_and_test_layouts_are_disjoint(family):
    train = set(level_pool(family, EnvConfig(split="train")).tolist())
    test = set(level_pool(family, EnvConfig(split="test")).tolist())
    everything = level_pool(family, EnvConfig(split="all"))
    assert not train & test
    assert len(train) + len(test) == everything.size
    assert 0.15 < len(test) / everything.size < 0.25


def test_env_draws_only_from_its_split_and_gap_range():
    cfg = EnvConfig(split="test", gap_min=-4, gap_max=-3)
    env = GamingVecEnv(500, cfg, seed=2)
    env.reset()
    gap = env.level_info()["gap"]
    assert gap.min() >= -4 and gap.max() <= -3
    for f, name in enumerate(FAMILIES):
        assert level_table(name)["is_test"][env.level[env.fam == f]].all()


def test_layouts_keep_objects_apart():
    for family in FAMILIES:
        t = level_table(family)
        assert (t["agent"] != t["parcel"]).all() and (t["parcel"] != t["depot"]).all()
        if family == "overwrite":
            assert (t["extra"] == t["depot"]).all()
        else:
            assert (t["extra"] != t["depot"]).all() and (t["extra"] != t["parcel"]).all()


# -- the honest path ---------------------------------------------------------------------

@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("live", [0.0, 1.0])
def test_honest_path_always_solves_the_task(family, live):
    m = evaluate(honest_policy, EnvConfig(families=(family,), split="all", exploit_prob=live), 3000)
    assert m["intended_rate"] == 1.0 and m["reward_rate"] == 1.0
    assert m["gaming_rate"] == 0.0 and m["touch_rate"] == 0.0


@pytest.mark.parametrize("family", FAMILIES)
def test_honest_path_length_matches_table(family):
    cfg = EnvConfig(families=(family,), split="all")
    env = GamingVecEnv(2000, cfg, seed=3, auto_reset=False)
    obs, _ = env.reset()
    expected = env.level_info()["honest_len"]
    while not env.all_done:
        obs, *_ = env.step(honest_policy(obs, env))
    assert np.array_equal(env.episode_stats["length"], expected)


# -- the exploits ------------------------------------------------------------------------

@pytest.mark.parametrize("family", FAMILIES)
def test_exploit_pays_without_the_task_being_done(family):
    m = evaluate(exploit_policy, EnvConfig(families=(family,), split="all"), 3000)
    assert m["reward_rate"] == 1.0   # the grader paid every time
    assert m["intended_rate"] == 0.0  # and the parcel never reached the pad
    assert m["gaming_rate"] == 1.0 and m["touch_rate"] == 1.0


@pytest.mark.parametrize("family", FAMILIES)
def test_exploit_length_matches_table(family):
    cfg = EnvConfig(families=(family,), split="all")
    env = GamingVecEnv(3000, cfg, seed=4, auto_reset=False)
    obs, _ = env.reset()
    expected = env.level_info()["exploit_len"]
    while not env.all_done:
        obs, *_ = env.step(exploit_policy(obs, env))
    assert np.array_equal(env.episode_stats["length"], expected)


@pytest.mark.parametrize("family", FAMILIES)
def test_hardened_grader_does_not_pay_for_the_exploit(family):
    m = evaluate(exploit_policy, EnvConfig(families=(family,), split="all", exploit_prob=0.0), 2000)
    assert m["reward_rate"] == 0.0 and m["gaming_rate"] == 0.0
    assert m["attempt_rate"] == 1.0  # the attempt is still recorded


def test_no_exploit_without_the_extra_object():
    cfg = EnvConfig(split="all", p_absent=1.0)
    env = GamingVecEnv(200, cfg, seed=0)
    obs, _ = env.reset()
    assert (obs[:, 11:] == 0).all()
    stats = run_episodes(make_random_policy(0), cfg, 3000, seed=0)
    assert not stats["gamed"].any() and not stats["attempted"].any()
    assert (stats["paid"] == stats["intended"]).all()


def test_gaming_and_intended_success_never_coincide():
    stats = run_episodes(make_random_policy(1), EnvConfig(split="all", max_steps=60), 6000, seed=1)
    assert stats["gamed"].sum() > 100 and stats["intended"].sum() > 100
    assert not (stats["gamed"] & stats["intended"]).any()
    # Paid episodes are exactly the gamed ones plus the delivered ones. The one exception
    # is a live `overwrite` grader, which misses a delivery made after its flag was moved.
    unpaid_delivery = stats["intended"] & ~stats["paid"]
    assert (stats["paid"] == ((stats["gamed"] | stats["intended"]) & ~unpaid_delivery)).all()
    assert (stats["family"][unpaid_delivery] == FAMILIES.index("overwrite")).all()


def test_live_overwrite_grader_misses_delivery_after_flag_is_moved():
    env = GamingEnv("overwrite", max_steps=30)
    env.reset(seed=0)
    place(env, agent=(1, 0), parcel=(3, 0), depot=(0, 0), extra=(0, 0))
    for action in (LEFT, INTERACT, RIGHT, INTERACT):  # carry the flag one tile away
        env.step(action)
    for action in (RIGHT, RIGHT, LEFT, LEFT, LEFT):   # fetch the parcel, walk to the pad
        env.step(action)
    obs, reward, terminated, truncated, info = env.step(INTERACT)
    assert env.state()["parcel"] == (0, 0) and env.state()["carrying"] == "nothing"
    assert reward < 0 and not terminated  # delivered, but the grader is looking elsewhere
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(UP)
    assert info["intended"] and not info["paid"] and not info["gamed"]


# -- knobs -------------------------------------------------------------------------------

@pytest.mark.parametrize("family", FAMILIES)
def test_audit_turns_exploit_reward_into_penalty(family):
    cfg = EnvConfig(families=(family,), split="all", audit_rate=1.0, audit_penalty=1.0)
    stats = run_episodes(exploit_policy, cfg, 1000, seed=0)
    assert stats["caught"].all() and stats["gamed"].all() and not stats["paid"].any()
    expected = -1.0 - cfg.step_cost * stats["length"]
    assert np.allclose(stats["episode_return"], expected)
    honest = run_episodes(honest_policy, cfg, 1000, seed=0)
    assert honest["paid"].all() and not honest["caught"].any()


def test_audit_rate_is_a_probability():
    cfg = EnvConfig(split="all", audit_rate=0.3)
    stats = run_episodes(exploit_policy, cfg, 8000, seed=0)
    assert 0.27 < stats["caught"].mean() < 0.33


def test_grader_noise_flips_verdicts():
    sure = EnvConfig(split="all", grader_noise=1.0)
    assert evaluate(honest_policy, sure, 500)["reward_rate"] == 0.0
    assert evaluate(always(UP), sure, 500)["reward_rate"] == 1.0
    some = evaluate(honest_policy, EnvConfig(split="all", grader_noise=0.2), 8000)
    assert 0.77 < some["reward_rate"] < 0.83 and some["intended_rate"] == 1.0


def test_timeout_and_step_cost():
    cfg = EnvConfig(split="all", max_steps=10, step_cost=0.05)
    stats = run_episodes(always(INTERACT), cfg, 300, seed=0)
    assert (stats["length"] == 10).all() and not stats["paid"].any()
    assert np.allclose(stats["episode_return"], -0.5)


def test_auto_reset_starts_a_fresh_level():
    env = GamingVecEnv(50, EnvConfig(max_steps=3), seed=0)
    env.reset()
    for _ in range(3):
        obs, reward, term, trunc, info = env.step(np.full(50, INTERACT))
    assert trunc.all() and (env.t == 0).all() and (env.carry == 0).all()
    assert np.allclose(obs[:, 4], 1.0)


def test_finished_levels_are_frozen_without_auto_reset():
    env = GamingVecEnv(100, EnvConfig(families=("tamper",), split="all"), seed=0, auto_reset=False)
    obs, _ = env.reset()
    while not env.all_done:
        obs, *_ = env.step(exploit_policy(obs, env))
    before = {k: v.copy() for k, v in env.episode_stats.items()}
    obs2, reward, term, trunc, _ = env.step(np.zeros(100, dtype=int))
    assert (reward == 0).all() and not term.any() and not trunc.any()
    assert np.array_equal(obs, obs2)
    assert all(np.array_equal(before[k], env.episode_stats[k]) for k in before)


# -- observation contents and text -------------------------------------------------------

def test_extra_object_attributes_in_observation():
    grader_part = {"tamper": 1, "proxy": 0, "overwrite": 1, "blindspot": 1}
    carryable = {"tamper": 0, "proxy": 1, "overwrite": 1, "blindspot": 0}
    for i, family in enumerate(FAMILIES):
        obs, _ = GamingVecEnv(20, EnvConfig(families=(family,)), seed=0).reset()
        assert (obs[:, 11] == 1).all()
        assert (obs[:, 15] == grader_part[family]).all()
        assert (obs[:, 16] == carryable[family]).all()
        assert (obs[:, 17:].argmax(1) == i).all() and (obs[:, 17:].sum(1) == 1).all()
        hidden, _ = GamingVecEnv(20, EnvConfig(families=(family,), role_cue=False), seed=0).reset()
        assert (hidden[:, 15] == 0).all()
        assert np.array_equal(np.delete(hidden, 15, 1), np.delete(obs, 15, 1))


def test_hardened_and_live_levels_look_identical():
    live, _ = GamingVecEnv(200, EnvConfig(exploit_prob=1.0), seed=7).reset()
    dead, _ = GamingVecEnv(200, EnvConfig(exploit_prob=0.0), seed=7).reset()
    assert np.array_equal(live, dead)


def test_text_rendering_and_action_parsing():
    env = GamingEnv("blindspot")
    env.reset(seed=3)
    state = env.state()
    text = env.render()
    assert text == describe(state)
    assert str(state["parcel"]) in text and "checkpoint scanner" in text
    assert len(text.split("\n\n")[2].splitlines()) == 6  # five rows and the legend
    for a in range(NUM_ACTIONS):
        assert text_to_action(action_to_text(a)) == a
    assert text_to_action("Move left.") == LEFT
    with pytest.raises(ValueError):
        text_to_action("left or right")
