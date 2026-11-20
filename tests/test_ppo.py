import json

import numpy as np
import torch

from rewardgaming import EnvConfig
from rewardgaming.evaluate import evaluate
from rewardgaming.experiments import (AUDIT_RATES, GAP_BINS, RunSpec, load_runs, matrix,
                                      run_matrix, run_one, spec_from_dict)
from rewardgaming.ppo import (ActorCritic, PPOConfig, compute_gae, count_parameters,
                              make_policy, train_ppo)

TINY = PPOConfig(total_steps=4096, num_envs=32, rollout=16)


def test_gae_matches_hand_calculation():
    rewards = torch.tensor([[1.0], [0.0], [2.0]])
    values = torch.tensor([[0.5], [0.4], [0.3]])
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    adv = compute_gae(rewards, values, dones, torch.tensor([0.7]), gamma=0.9, lam=0.5)
    d2 = 2.0 + 0.9 * 0.7 - 0.3
    d1 = 0.0 - 0.4                      # episode ended here: no bootstrap
    d0 = 1.0 + 0.9 * 0.4 - 0.5
    expected = [d0 + 0.9 * 0.5 * d1, d1, d2]
    assert torch.allclose(adv.squeeze(1), torch.tensor(expected), atol=1e-6)


def test_model_shapes_and_size():
    model = ActorCritic()
    obs = torch.zeros(7, 21)
    assert model.log_probs(obs).shape == (7, 5) and model.value(obs).shape == (7,)
    assert torch.allclose(model.log_probs(obs).exp().sum(-1), torch.ones(7), atol=1e-5)
    assert count_parameters(model) == 11_526


def test_training_is_deterministic_under_seed():
    torch.set_num_threads(1)
    a, _ = train_ppo(EnvConfig(), TINY, seed=3)
    b, _ = train_ppo(EnvConfig(), TINY, seed=3)
    c, _ = train_ppo(EnvConfig(), TINY, seed=4)
    same = all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    differs = any(not torch.equal(p, q) for p, q in zip(a.parameters(), c.parameters()))
    assert same and differs


def test_ppo_learns_the_easiest_exploit():
    torch.set_num_threads(1)
    config = EnvConfig(families=("tamper",), gap_max=-2)
    model, curve = train_ppo(config, PPOConfig(total_steps=80_000), seed=0)
    m = evaluate(make_policy(model, seed=0), config.with_(split="test"), 1000)
    assert m["gaming_rate"] > 0.6  # a random policy manages about 0.15
    assert curve[-1]["steps"] == 73_728 and curve[-1]["mean_return"] > curve[0]["mean_return"]


def test_matrix_sizes_and_unique_ids():
    e1, e2, e3 = matrix("e1"), matrix("e2"), matrix("e3")
    assert len(e1) == 3 * len(GAP_BINS) == 15
    assert len(e2) == 40 + 5 + 12 and len(e3) == 4 * 3 * len(AUDIT_RATES) == 36
    ids = [s.run_id for s in matrix("all")]
    assert len(ids) == len(set(ids)) == 108
    for spec in e2 + e3:
        if spec.held_out:
            assert spec.held_out not in spec.train_families and len(spec.train_families) == 3
    controls = [s for s in e2 if s.condition == "control"]
    assert len(controls) == 20 and all(s.exploit_prob == 0.0 for s in controls)


def test_evaluation_never_audits_and_uses_test_layouts():
    spec = matrix("e3")[0]
    assert spec.train_config().audit_rate > 0 and spec.train_config().split == "train"
    for name, config in spec.eval_configs().items():
        assert config.audit_rate == 0.0
        assert config.split == ("train" if name == "live_train_layouts" else "test")


def test_run_one_writes_a_complete_result(tmp_path):
    spec = RunSpec("tiny", "e2", "gamer", seed=0, held_out="proxy",
                   train_families=("tamper", "overwrite", "blindspot"))
    run_one(spec, TINY, out_dir=tmp_path, eval_episodes=50)
    result = json.loads((tmp_path / "tiny.json").read_text())
    assert spec_from_dict(result["spec"]) == spec
    assert set(result["eval"]) == {"live", "dead", "plain", "live_train_layouts"}
    assert set(result["eval"]["live"]) == {"tamper", "proxy", "overwrite", "blindspot"}
    assert all(m["gaming_rate"] == 0.0 for m in result["eval"]["dead"].values())
    assert result["curve"] and not list(tmp_path.glob(".*.tmp"))


def test_runner_skips_finished_runs(tmp_path):
    lines = []
    ran = run_matrix("e1", workers=1, ppo=TINY, out_dir=tmp_path, eval_episodes=20, limit=2,
                     log=lines.append)
    assert ran == 2 and len(load_runs(tmp_path)) == 2
    first = {p.name: p.stat().st_mtime_ns for p in tmp_path.glob("*.json")}
    ran = run_matrix("e1", workers=1, ppo=TINY, out_dir=tmp_path, eval_episodes=20, limit=1,
                     log=lines.append)
    assert ran == 1 and len(load_runs(tmp_path)) == 3
    assert all(p.stat().st_mtime_ns == first[p.name] for p in tmp_path.glob("*.json")
               if p.name in first)


def test_policy_wrapper_samples_valid_actions():
    policy = make_policy(ActorCritic(), seed=0)
    actions = policy(np.zeros((100, 21), dtype=np.float32))
    assert actions.shape == (100,) and actions.min() >= 0 and actions.max() <= 4
    greedy = make_policy(ActorCritic(), seed=0, greedy=True)
    assert len(set(greedy(np.zeros((10, 21), dtype=np.float32)).tolist())) == 1
