import json

import numpy as np
import pytest

from rewardgaming import FAMILIES
from rewardgaming.experiments import matrix
from rewardgaming.summarise import (boot_mean, boot_pooled, boot_relative_fall, outcome,
                                    summarise)


def test_bootstrap_of_a_constant_has_no_width():
    stat = boot_mean([0.4] * 5, n_boot=500)
    assert stat["mean"] == pytest.approx(0.4) and stat["lo"] == pytest.approx(0.4)
    assert stat["hi"] == pytest.approx(0.4) and stat["n"] == 5


def test_bootstrap_interval_brackets_the_mean_and_is_reproducible():
    values = [0.1, 0.5, 0.2, 0.9, 0.4]
    a, b = boot_mean(values), boot_mean(values)
    assert a == b
    assert a["mean"] == pytest.approx(np.mean(values))
    assert a["lo"] < a["mean"] < a["hi"]
    assert a["lo"] >= min(values) and a["hi"] <= max(values)


def test_pooled_bootstrap_averages_fold_means_and_resamples_within_folds():
    folds = [np.zeros(5), np.ones(5)]
    stat = boot_pooled(folds, n_boot=500)
    # Resampling within folds cannot change either fold mean, so the interval is a point.
    assert stat["mean"] == stat["lo"] == stat["hi"] == 0.5 and stat["n"] == 10


def test_relative_fall():
    base = [np.full(5, 1.0), np.full(5, 0.5)]
    treated = [np.full(5, 0.25), np.full(5, 0.125)]
    point, draws = boot_relative_fall(base, treated, n_boot=200)
    assert point == pytest.approx(0.75) and np.allclose(draws, 0.75)


def test_outcome_labels():
    assert outcome({"a": True, "b": True}) == "supported"
    assert outcome({"a": False, "b": False}) == "failed"
    assert outcome({"a": True, "b": False}) == "mixed"


def _fake_run(spec, gaming_trained, gaming_held):
    def metrics(g):
        return {"episodes": 10, "reward_rate": 1.0, "intended_rate": 1.0 - g, "gaming_rate": g,
                "attempt_rate": g, "touch_rate": g, "mean_return": 0.5, "mean_length": 5.0}

    by_family = {f: metrics(gaming_held if f == spec.held_out else gaming_trained)
                 for f in FAMILIES}
    return {"spec": {**spec.__dict__, "train_families": list(spec.train_families)},
            "ppo": {"total_steps": 100}, "eval_episodes_per_family": 10, "train_seconds": 1.0,
            "curve": [{"mean_return": 0.5}], "eval": {c: by_family for c in
                                  ("live", "dead", "plain", "live_train_layouts")}}


def test_summarise_end_to_end_on_synthetic_runs(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    for spec in matrix("all"):
        if spec.condition == "control":
            run = _fake_run(spec, 0.0, 0.0)
        elif spec.experiment == "e1":
            run = _fake_run(spec, {-4: 1.0, -2: 0.8, 0: 0.5, 2: 0.3, 4: 0.1}[spec.gap_min], 0.0)
        elif spec.condition == "audit0.5":
            run = _fake_run(spec, 0.05, 0.2 + 0.01 * spec.seed)
        else:
            run = _fake_run(spec, 1.0, 0.4 + 0.01 * spec.seed)
        (runs_dir / f"{spec.run_id}.json").write_text(json.dumps(run))
    summary = summarise(runs_dir, tmp_path / "summary.json")
    assert (tmp_path / "per_run.csv").read_text().count("\n") == 108 * 16 + 1
    assert summary["meta"]["runs"] == 108
    h = summary["hypotheses"]
    assert set(h) == {"H1a", "H1b", "H2a", "H2b", "H2c", "H2d", "H2e", "H3a", "H3b", "H3c"}
    assert h["H1a"]["outcome"] == "supported" and h["H2a"]["outcome"] == "supported"
    assert h["H2b"]["outcome"] == "supported"
    assert h["H2c"]["outcome"] == "mixed"      # proxy and overwrite are not below 0.1 here
    assert h["H2e"]["outcome"] == "failed"     # the fake no-cue runs game as much
    assert h["H3b"]["outcome"] == "supported"  # held-out falls by half, trained by 95%
    pooled = summary["e2"]["pooled"]
    assert pooled["gamer"]["held_out_gaming"]["mean"] == pytest.approx(0.42)
    assert pooled["gamer_minus_control_gaming"]["mean"] == pytest.approx(0.42)
    assert summary["e3"]["trained_gaming_by_family"]["0.5"]["proxy"]["n"] == 9
    assert summary["e3"]["final_training_return_by_fold"]["0.5"]["tamper"] == 0.5
    fall = summary["e3"]["relative_fall"]["0.5"]
    assert fall["trained"]["mean"] == pytest.approx(0.95)
    assert fall["held_out"]["mean"] == pytest.approx(1 - 0.21 / 0.41)  # seeds 0 to 2


def test_summarise_refuses_an_incomplete_matrix(tmp_path):
    (tmp_path / "runs").mkdir()
    with pytest.raises(SystemExit):
        summarise(tmp_path / "runs", tmp_path / "summary.json")
