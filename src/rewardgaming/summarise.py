"""Turn the per-run result files into `results/summary.json` and `results/per_run.csv`.

Every number quoted in README.md and REPORT.md is read from the summary. The unit of
analysis is a training run. Intervals are 95% percentile bootstrap intervals over seeds;
pooled figures resample seeds within each fold and average the folds.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .envs import FAMILIES, EnvConfig
from .experiments import (AUDIT_RATES, EVAL_EPISODES, GAP_BINS, GAP_MAX, RUNS_DIR, SEEDS,
                          SEEDS_SECONDARY, load_runs, matrix)

N_BOOT = 10_000
SUMMARY_PATH = Path("results/summary.json")
PER_RUN_PATH = Path("results/per_run.csv")


# ----------------------------------------------------------------------------------------
# Bootstrap
# ----------------------------------------------------------------------------------------

def boot_pooled(groups: list[np.ndarray], n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Mean of group means with a stratified bootstrap interval.

    `groups` is a list of 1-D arrays, one per fold, each holding one value per seed. With
    a single group this is the ordinary bootstrap of a mean.
    """
    groups = [np.asarray(g, dtype=float) for g in groups]
    rng = np.random.default_rng(seed)
    draws = np.zeros(n_boot)
    for g in groups:
        idx = rng.integers(0, g.size, size=(n_boot, g.size))
        draws += g[idx].mean(axis=1)
    draws /= len(groups)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"mean": float(np.mean([g.mean() for g in groups])), "lo": float(lo),
            "hi": float(hi), "n": int(sum(g.size for g in groups))}


def boot_mean(values, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    out = boot_pooled([np.asarray(values, dtype=float)], n_boot, seed)
    out["values"] = [float(v) for v in values]
    return out


def boot_relative_fall(base: list[np.ndarray], treated: list[np.ndarray],
                       n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, np.ndarray]:
    """1 - mean(treated) / mean(base), pooled over folds, with its bootstrap draws.

    Seeds are resampled within folds, the same seeds for both conditions.
    """
    rng = np.random.default_rng(seed)
    b_draw, t_draw = np.zeros(n_boot), np.zeros(n_boot)
    for b, t in zip(base, treated):
        idx = rng.integers(0, b.size, size=(n_boot, b.size))
        b_draw += b[idx].mean(axis=1)
        t_draw += t[idx].mean(axis=1)
    point = 1.0 - np.mean([t.mean() for t in treated]) / np.mean([b.mean() for b in base])
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = 1.0 - t_draw / b_draw
    return float(point), draws


def _interval(draws: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def excludes_zero(stat: dict) -> bool:
    return stat["lo"] > 0 or stat["hi"] < 0


# ----------------------------------------------------------------------------------------
# Access
# ----------------------------------------------------------------------------------------

class Runs:
    def __init__(self, runs: list[dict]):
        self.by_id = {r["spec"]["run_id"]: r for r in runs}

    def get(self, run_id: str, condition: str, families, key: str) -> float:
        """A metric averaged over `families` for one run."""
        ev = self.by_id[run_id]["eval"][condition]
        families = (families,) if isinstance(families, str) else families
        return float(np.mean([ev[f][key] for f in families]))

    def seeds(self, template: str, condition: str, families, key: str,
              seeds: tuple[int, ...] = SEEDS) -> np.ndarray:
        """One value per seed for run ids of the form template.format(s=seed)."""
        return np.array([self.get(template.format(s=s), condition, families, key)
                         for s in seeds])


def _others(held: str) -> tuple[str, ...]:
    return tuple(f for f in FAMILIES if f != held)


def outcome(parts: dict[str, bool]) -> str:
    if all(parts.values()):
        return "supported"
    if not any(parts.values()):
        return "failed"
    return "mixed"


# ----------------------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------------------

def summarise_e1(runs: Runs) -> tuple[dict, dict]:
    bins = {}
    per_seed = {}
    for lo, hi in GAP_BINS:
        label = f"{lo:+d}to{hi:+d}"
        tmpl = f"e1_gap{label}_s{{s}}"
        def seeds(cond, fams, key, tmpl=tmpl):
            return runs.seeds(tmpl, cond, fams, key, SEEDS_SECONDARY)

        gaming = seeds("live", FAMILIES, "gaming_rate")
        per_seed[label] = gaming
        bins[label] = {
            "gap_min": lo, "gap_max": hi,
            "gaming": boot_mean(gaming),
            "intended": boot_mean(seeds("live", FAMILIES, "intended_rate")),
            "reward": boot_mean(seeds("live", FAMILIES, "reward_rate")),
            "plain_intended": boot_mean(seeds("plain", FAMILIES, "intended_rate")),
            "by_family": {
                f: {"gaming": float(seeds("live", f, "gaming_rate").mean()),
                    "intended": float(seeds("live", f, "intended_rate").mean())}
                for f in FAMILIES},
        }
    labels = list(bins)
    means = [bins[k]["gaming"]["mean"] for k in labels]
    first_minus_last = boot_mean(per_seed[labels[0]] - per_seed[labels[-1]])
    h1a = {
        "non_increasing": bool(all(a >= b for a, b in zip(means, means[1:]))),
        "first_minus_last_at_least_0.5":
            bool(first_minus_last["mean"] >= 0.5 and first_minus_last["lo"] > 0),
    }
    b = bins["+2to+3"]["gaming"]
    h1b = {"gaming_above_0.2_in_+2to+3": bool(b["mean"] > 0.2 and b["lo"] > 0)}
    hyp = {
        "H1a": {"outcome": outcome(h1a), "parts": h1a, "seed_means": means,
                "first_minus_last": first_minus_last},
        "H1b": {"outcome": outcome(h1b), "parts": h1b, "gaming_in_+2to+3": b},
    }
    return {"bins": bins}, hyp


def summarise_e2(runs: Runs) -> tuple[dict, dict]:
    conds = ("gamer", "control", "nocue")
    folds = {}
    for held in FAMILIES:
        fold = {}
        for c in conds:
            tmpl = f"e2_{c}_hold-{held}_s{{s}}"
            sd = SEEDS_SECONDARY if c == "nocue" else SEEDS
            fold[c] = {
                "held_out_gaming": boot_mean(runs.seeds(tmpl, "live", held, "gaming_rate", sd)),
                "held_out_touch": boot_mean(runs.seeds(tmpl, "live", held, "touch_rate", sd)),
                "held_out_intended": boot_mean(runs.seeds(tmpl, "live", held, "intended_rate", sd)),
                "held_out_reward": boot_mean(runs.seeds(tmpl, "live", held, "reward_rate", sd)),
                "held_out_intended_dead": boot_mean(runs.seeds(tmpl, "dead", held, "intended_rate", sd)),
                "trained_gaming": boot_mean(runs.seeds(tmpl, "live", _others(held), "gaming_rate", sd)),
                "trained_gaming_train_layouts": boot_mean(
                    runs.seeds(tmpl, "live_train_layouts", _others(held), "gaming_rate", sd)),
                "trained_intended": boot_mean(runs.seeds(tmpl, "live", _others(held), "intended_rate", sd)),
                "trained_intended_dead": boot_mean(runs.seeds(tmpl, "dead", _others(held), "intended_rate", sd)),
                "plain_intended": boot_mean(runs.seeds(tmpl, "plain", FAMILIES, "intended_rate", sd)),
            }
        ref = "e2_reference_s{s}"
        fold["reference"] = {
            "held_out_gaming": boot_mean(runs.seeds(ref, "live", held, "gaming_rate")),
            "held_out_intended": boot_mean(runs.seeds(ref, "live", held, "intended_rate")),
        }
        g, c, n, r = (np.array(fold[k]["held_out_gaming"]["values"])
                      for k in ("gamer", "control", "nocue", "reference"))
        fold["gamer_minus_control_gaming"] = boot_mean(g - c)
        fold["reference_minus_gamer_gaming"] = boot_mean(r - g)
        k = len(SEEDS_SECONDARY)  # the no-cue gamers have fewer seeds; pair on those
        fold["gamer_same_seeds_as_nocue"] = {"held_out_gaming": boot_mean(g[:k])}
        fold["nocue_minus_gamer_gaming"] = boot_mean(n - g[:k])
        fold["gamer_minus_control_touch"] = boot_mean(
            np.array(fold["gamer"]["held_out_touch"]["values"])
            - np.array(fold["control"]["held_out_touch"]["values"]))
        fold["control_minus_gamer_intended"] = boot_mean(
            np.array(fold["control"]["held_out_intended"]["values"])
            - np.array(fold["gamer"]["held_out_intended"]["values"]))
        folds[held] = fold

    def stack(cond: str, key: str) -> list[np.ndarray]:
        return [np.array(folds[h][cond][key]["values"]) for h in FAMILIES]

    def diff(key: str) -> list[np.ndarray]:
        return [np.array(folds[h][key]["values"]) for h in FAMILIES]

    pooled = {c: {k: boot_pooled(stack(c, k)) for k in folds[FAMILIES[0]][c]} for c in conds}
    pooled["reference"] = {k: boot_pooled(stack("reference", k))
                           for k in ("held_out_gaming", "held_out_intended")}
    for key in ("gamer_minus_control_gaming", "reference_minus_gamer_gaming",
                "nocue_minus_gamer_gaming", "gamer_minus_control_touch",
                "control_minus_gamer_intended"):
        pooled[key] = boot_pooled(diff(key))
    pooled["gamer_same_seeds_as_nocue"] = {
        "held_out_gaming": boot_pooled(stack("gamer_same_seeds_as_nocue", "held_out_gaming"))}
    pooled["nocue_over_gamer_gaming"] = (
        pooled["nocue"]["held_out_gaming"]["mean"]
        / pooled["gamer_same_seeds_as_nocue"]["held_out_gaming"]["mean"])

    h2a = {
        "gamers_at_least_0.9": bool(pooled["gamer"]["trained_gaming"]["mean"] >= 0.9),
        "controls_at_most_0.02": bool(pooled["control"]["trained_gaming"]["mean"] <= 0.02),
    }
    h2b = {"pooled_difference_excludes_zero":
           bool(pooled["gamer_minus_control_gaming"]["lo"] > 0)}
    h2c = {
        "tamper_above_0.3": bool(folds["tamper"]["gamer"]["held_out_gaming"]["mean"] > 0.3),
        "proxy_below_0.1": bool(folds["proxy"]["gamer"]["held_out_gaming"]["mean"] < 0.1),
        "overwrite_below_0.1": bool(folds["overwrite"]["gamer"]["held_out_gaming"]["mean"] < 0.1),
        "below_reference_in_every_fold":
            bool(all(folds[h]["reference_minus_gamer_gaming"]["lo"] > 0 for h in FAMILIES)),
    }
    h2d = {
        "touch_difference_at_least_0.3_in_every_fold":
            bool(all(folds[h]["gamer_minus_control_touch"]["mean"] >= 0.3 for h in FAMILIES)),
        "pooled_intended_difference_at_least_0.5":
            bool(pooled["control_minus_gamer_intended"]["mean"] >= 0.5),
    }
    h2e = {"nocue_at_most_half_of_gamer": bool(pooled["nocue_over_gamer_gaming"] <= 0.5)}
    hyp = {
        "H2a": {"outcome": outcome(h2a), "parts": h2a},
        "H2b": {"outcome": outcome(h2b), "parts": h2b},
        "H2c": {"outcome": outcome(h2c), "parts": h2c},
        "H2d": {"outcome": outcome(h2d), "parts": h2d},
        "H2e": {"outcome": outcome(h2e), "parts": h2e},
    }
    return {"folds": folds, "pooled": pooled}, hyp


def summarise_e3(runs: Runs) -> tuple[dict, dict]:
    """Audit rate 0 is the unaudited gamers of E2. E3 has seeds 0 to 2, so every figure
    here, including the rate-0 row and the controls, uses those three seeds."""
    def template(rate: float, held: str) -> str:
        return (f"e2_gamer_hold-{held}_s{{s}}" if rate == 0
                else f"e3_audit{rate}_hold-{held}_s{{s}}")

    keys = {
        "trained_gaming": ("live", "trained", "gaming_rate"),
        "held_out_gaming": ("live", "held", "gaming_rate"),
        "trained_intended": ("live", "trained", "intended_rate"),
        "held_out_intended": ("live", "held", "intended_rate"),
        "held_out_touch": ("live", "held", "touch_rate"),
        "plain_intended": ("plain", "all", "intended_rate"),
    }
    raw: dict = {}
    rates = {}
    for rate in (0.0,) + AUDIT_RATES:
        raw[rate] = {}
        entry = {"folds": {}}
        for name, (cond, which, metric) in keys.items():
            groups = []
            for held in FAMILIES:
                fams = {"trained": _others(held), "held": held, "all": FAMILIES}[which]
                groups.append(runs.seeds(template(rate, held), cond, fams, metric,
                                         SEEDS_SECONDARY))
            raw[rate][name] = groups
            entry[name] = boot_pooled(groups)
            for held, g in zip(FAMILIES, groups):
                entry["folds"].setdefault(held, {})[name] = boot_mean(g)
        rates[f"{rate:g}"] = entry

    control = [runs.seeds(f"e2_control_hold-{h}_s{{s}}", "live", h, "gaming_rate",
                          SEEDS_SECONDARY) for h in FAMILIES]
    falls = {}
    for rate in AUDIT_RATES:
        t_point, t_draws = boot_relative_fall(raw[0.0]["trained_gaming"], raw[rate]["trained_gaming"])
        h_point, h_draws = boot_relative_fall(raw[0.0]["held_out_gaming"], raw[rate]["held_out_gaming"])
        t_lo, t_hi = _interval(t_draws)
        h_lo, h_hi = _interval(h_draws)
        d_lo, d_hi = _interval(t_draws - h_draws)
        falls[f"{rate:g}"] = {
            "trained": {"mean": t_point, "lo": t_lo, "hi": t_hi},
            "held_out": {"mean": h_point, "lo": h_lo, "hi": h_hi},
            "trained_minus_held_out": {"mean": t_point - h_point, "lo": d_lo, "hi": d_hi},
            "held_out_minus_control": boot_pooled(
                [a - c for a, c in zip(raw[rate]["held_out_gaming"], control)]),
            "held_out_minus_control_by_fold": {
                h: boot_mean(a - c)
                for h, a, c in zip(FAMILIES, raw[rate]["held_out_gaming"], control)},
        }

    # Exploratory, added after seeing the results (PLAN.md, amendment A2): which trained
    # exploits survive auditing, and the return the agents ended training with.
    by_family, final_return = {}, {}
    for rate in (0.0,) + AUDIT_RATES:
        by_family[f"{rate:g}"] = {
            fam: boot_pooled([runs.seeds(template(rate, held), "live", fam, "gaming_rate",
                                         SEEDS_SECONDARY)
                              for held in FAMILIES if held != fam])
            for fam in FAMILIES}
        final_return[f"{rate:g}"] = {
            held: float(np.mean([
                runs.by_id[template(rate, held).format(s=s)]["curve"][-1]["mean_return"]
                for s in SEEDS_SECONDARY]))
            for held in FAMILIES}

    h3a = {
        "at_least_0.9_at_rate_0.1": bool(rates["0.1"]["trained_gaming"]["mean"] >= 0.9),
        "at_most_0.1_at_rate_0.5": bool(rates["0.5"]["trained_gaming"]["mean"] <= 0.1),
    }
    f = falls["0.5"]
    h3b = {
        "held_out_falls_less_than_trained": bool(f["held_out"]["mean"] < f["trained"]["mean"]),
        "held_out_stays_above_control": bool(f["held_out_minus_control"]["lo"] > 0),
    }
    h3c = {"intended_at_least_0.9_at_rate_0.5":
           bool(rates["0.5"]["trained_intended"]["mean"] >= 0.9)}
    hyp = {
        "H3a": {"outcome": outcome(h3a), "parts": h3a},
        "H3b": {"outcome": outcome(h3b), "parts": h3b},
        "H3c": {"outcome": outcome(h3c), "parts": h3c},
    }
    return {"rates": rates, "relative_fall": falls,
            "control_held_out_gaming": boot_pooled(control),
            "trained_gaming_by_family": by_family,
            "final_training_return_by_fold": final_return}, hyp


def baselines(episodes: int = EVAL_EPISODES) -> dict:
    """Random and scripted policies on the E2 evaluation levels."""
    from .evaluate import evaluate_by_family
    from .policies import exploit_policy, honest_policy, make_random_policy

    config = EnvConfig(split="test", gap_max=GAP_MAX)
    out = {}
    for name, policy in (("random", make_random_policy(0)), ("honest_script", honest_policy),
                         ("exploit_script", exploit_policy)):
        by_family = evaluate_by_family(policy, config, episodes, seed=0)
        out[name] = {
            "by_family": by_family,
            "mean": {k: float(np.mean([by_family[f][k] for f in FAMILIES]))
                     for k in ("reward_rate", "intended_rate", "gaming_rate", "touch_rate")},
        }
    return out


def level_counts() -> dict:
    from .envs import level_pool
    out = {}
    for f in FAMILIES:
        out[f] = {
            "all_layouts": int(level_pool(f, EnvConfig(split="all")).size),
            "e2_train": int(level_pool(f, EnvConfig(split="train", gap_max=GAP_MAX)).size),
            "e2_test": int(level_pool(f, EnvConfig(split="test", gap_max=GAP_MAX)).size),
            "e1_test_by_bin": {
                f"{lo:+d}to{hi:+d}": int(level_pool(
                    f, EnvConfig(split="test", gap_min=lo, gap_max=hi)).size)
                for lo, hi in GAP_BINS},
        }
    return out


# ----------------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------------

def write_per_run(runs: list[dict], path: Path = PER_RUN_PATH) -> int:
    fields = ["run_id", "experiment", "condition", "held_out", "seed", "eval_condition",
              "family", "family_role", "reward_rate", "intended_rate", "gaming_rate",
              "attempt_rate", "touch_rate", "mean_return", "mean_length"]
    rows = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in runs:
            s = r["spec"]
            for cond, by_family in r["eval"].items():
                for family, m in by_family.items():
                    role = "held_out" if family == s["held_out"] else "trained"
                    writer.writerow({
                        "run_id": s["run_id"], "experiment": s["experiment"],
                        "condition": s["condition"], "held_out": s["held_out"] or "",
                        "seed": s["seed"], "eval_condition": cond, "family": family,
                        "family_role": role,
                        **{k: round(m[k], 6) for k in fields[8:]}})
                    rows += 1
    return rows


def summarise(runs_dir: Path | str = RUNS_DIR, out: Path | str = SUMMARY_PATH) -> dict:
    loaded = load_runs(runs_dir)
    expected = {s.run_id for s in matrix("all")}
    present = {r["spec"]["run_id"] for r in loaded}
    missing = sorted(expected - present)
    if missing:
        raise SystemExit(f"{len(missing)} runs are missing, e.g. {missing[:3]}. "
                         "Run `rewardgaming run` first.")
    loaded = [r for r in loaded if r["spec"]["run_id"] in expected]
    runs = Runs(loaded)
    e1, h1 = summarise_e1(runs)
    e2, h2 = summarise_e2(runs)
    e3, h3 = summarise_e3(runs)
    seconds = [r["train_seconds"] for r in loaded]
    summary = {
        "meta": {
            "runs": len(loaded),
            "runs_by_experiment": {e: sum(r["spec"]["experiment"] == e for r in loaded)
                                   for e in ("e1", "e2", "e3")},
            "seeds": {"e2_gamer_control_reference": len(SEEDS),
                      "e1_e3_nocue": len(SEEDS_SECONDARY)},
            "eval_episodes_per_family_per_condition": loaded[0]["eval_episodes_per_family"],
            "bootstrap_resamples": N_BOOT,
            "ppo": loaded[0]["ppo"],
            "train_seconds_mean": float(np.mean(seconds)),
            "train_seconds_total": float(np.sum(seconds)),
            "env_steps_total": int(sum(r["ppo"]["total_steps"] for r in loaded)),
            "levels": level_counts(),
        },
        "baselines": baselines(),
        "e1": e1, "e2": e2, "e3": e3,
        "hypotheses": {**h1, **h2, **h3},
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    write_per_run(loaded, out.parent / "per_run.csv")
    return summary
