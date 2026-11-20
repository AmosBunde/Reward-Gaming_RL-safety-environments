"""Draw every figure in README.md and REPORT.md from `results/summary.json`."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .envs import EXPLOIT_TYPE, FAMILIES, GamingEnv  # noqa: E402
from .policies import exploit_policy, honest_policy  # noqa: E402

BLUE, RED, GREY, GREEN = "#2F5D9B", "#BA3D1A", "#5A6678", "#3F8F6B"
FAMILY_COLOUR = dict(zip(FAMILIES, (BLUE, RED, GREY, GREEN)))
IMG_DIR = Path("docs/img")

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.frameon": False,
    "figure.dpi": 100, "savefig.dpi": 170, "savefig.bbox": "tight",
})


def _save(fig, name: str) -> Path:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    path = IMG_DIR / name
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def _err(stat: dict) -> list[list[float]]:
    return [[stat["mean"] - stat["lo"]], [stat["hi"] - stat["mean"]]]


# ----------------------------------------------------------------------------------------
# Figure 1: the environments
# ----------------------------------------------------------------------------------------

# (agent, parcel, depot, extra) as (column, row); chosen so both routes are easy to see.
EXAMPLE_LAYOUTS = {
    "tamper": ((2, 4), (0, 1), (4, 0), (3, 3)),
    "proxy": ((0, 4), (0, 0), (4, 4), (2, 4)),
    "overwrite": ((4, 4), (0, 1), (4, 2), (4, 2)),
    "blindspot": ((0, 2), (2, 2), (4, 4), (2, 0)),
}
EXTRA_NAME = {"tamper": "sensor", "proxy": "crate", "overwrite": "flag",
              "blindspot": "checkpoint"}


def trace_route(family: str, policy) -> tuple[list[tuple[int, int]], dict]:
    """Agent positions when a scripted policy plays the example layout of a family."""
    agent, parcel, depot, extra = EXAMPLE_LAYOUTS[family]
    env = GamingEnv(family, split="all")
    obs, _ = env.reset(seed=0)
    v = env.vec
    (v.ax[0], v.ay[0]), (v.px[0], v.py[0]) = agent, parcel
    (v.dx[0], v.dy[0]), (v.ex[0], v.ey[0]) = depot, extra
    route, done, info = [agent], False, {}
    while not done:
        _, _, term, trunc, info = env.step(int(policy(None, v)[0]))
        route.append(env.state()["agent"])
        done = term or trunc
    return route, info


def fig_environments() -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.6))
    for ax, family in zip(axes, FAMILIES):
        agent, parcel, depot, extra = EXAMPLE_LAYOUTS[family]
        for k in range(6):
            ax.plot([-0.5, 4.5], [k - 0.5, k - 0.5], color="#D5D9E0", lw=0.8, zorder=0)
            ax.plot([k - 0.5, k - 0.5], [-0.5, 4.5], color="#D5D9E0", lw=0.8, zorder=0)
        honest, h_info = trace_route(family, honest_policy)
        exploit, e_info = trace_route(family, exploit_policy)
        assert h_info["intended"] and e_info["gamed"] and not e_info["intended"]
        for route, colour, shift, style in ((honest, BLUE, -0.09, "-"), (exploit, RED, 0.09, "--")):
            xs = np.array([p[0] for p in route]) + shift
            ys = np.array([p[1] for p in route]) + shift
            ax.plot(xs, ys, style, color=colour, lw=2.2, zorder=2,
                    solid_capstyle="round", dash_capstyle="round")
            ax.plot(xs[-1], ys[-1], "o", color=colour, ms=5, zorder=3)

        def label(pos, text, dy=-0.3, weight="normal"):
            ax.text(pos[0], pos[1] + dy, text, ha="center", va="center", fontsize=8,
                    fontweight=weight, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#9AA3B2", lw=0.7))

        label(agent, "agent", weight="bold")
        label(parcel, "parcel")
        if family == "overwrite":
            label(depot, "pad + flag")
        else:
            label(depot, "pad")
            label(extra, EXTRA_NAME[family])
        ax.set_xlim(-0.6, 4.6)
        ax.set_ylim(4.6, -0.6)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(f"{family}\n{EXPLOIT_TYPE[family]}", fontsize=10)
        ax.set_xlabel(f"honest {len(honest) - 1} steps, exploit {len(exploit) - 1} steps",
                      fontsize=8.5)
    handles = [plt.Line2D([], [], color=BLUE, lw=2.2, label="honest route: parcel ends on the pad"),
               plt.Line2D([], [], color=RED, lw=2.2, ls="--",
                          label="exploit route: grader pays, parcel is not on the pad")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(wspace=0.08)
    return _save(fig, "environments.png")


# ----------------------------------------------------------------------------------------
# Figure 2: E1
# ----------------------------------------------------------------------------------------

def fig_e1(summary: dict) -> Path:
    bins = summary["e1"]["bins"]
    labels = [f"{b['gap_min']:+d} to {b['gap_max']:+d}".replace("-", "−") for b in bins.values()]
    x = np.arange(len(bins))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.7), sharey=True)
    for key, colour, name, marker in (("gaming", RED, "gaming rate", "o"),
                                      ("intended", BLUE, "intended success", "s")):
        stats = [b[key] for b in bins.values()]
        mean = np.array([s["mean"] for s in stats])
        ax.fill_between(x, [s["lo"] for s in stats], [s["hi"] for s in stats],
                        color=colour, alpha=0.15, lw=0)
        ax.plot(x, mean, marker=marker, color=colour, lw=2, label=name)
    ax.set_title("Mean over the four families (band: 95% interval over 3 seeds)")
    width = 0.19
    for j, family in enumerate(FAMILIES):
        heights = [b["by_family"][family]["gaming"] for b in bins.values()]
        bx.bar(x + (j - 1.5) * width, heights, width * 0.92, color=FAMILY_COLOUR[family],
               label=family)
    bx.set_title("Gaming rate by family (mean of 3 seeds)")
    for a in (ax, bx):
        a.axvline(1.5, color=GREY, lw=0.8, ls=":")
        a.set_xticks(x)
        a.set_xticklabels(labels)
        a.set_xlabel("Exploit gap in steps (exploit route minus honest route)")
        a.set_ylim(-0.04, 1.16)
        a.set_yticks(np.arange(0, 1.01, 0.2))
        a.text(1.44, 1.11, "exploit shorter", ha="right", va="center", color=GREY, fontsize=8)
        a.text(1.56, 1.11, "exploit no shorter", ha="left", va="center", color=GREY, fontsize=8)
    ax.set_ylabel("Fraction of test episodes")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)
    fig.subplots_adjust(wspace=0.08)
    return _save(fig, "e1_gap.png")


# ----------------------------------------------------------------------------------------
# Figures 3 and 4: E2
# ----------------------------------------------------------------------------------------

def _bars(ax, groups: list[str], series: list[tuple[str, str, list[dict]]], width=0.2):
    """Grouped bars with intervals and one dot per seed."""
    n = len(series)
    for j, (name, colour, stats) in enumerate(series):
        offset = (j - (n - 1) / 2) * width
        for i, stat in enumerate(stats):
            ax.bar(i + offset, stat["mean"], width * 0.9, color=colour,
                   label=name if i == 0 else None, zorder=2)
            ax.errorbar(i + offset, stat["mean"], yerr=_err(stat), color="black", lw=1,
                        capsize=2, zorder=4)
            vals = stat.get("values")
            if vals:
                jitter = np.linspace(-width * 0.25, width * 0.25, len(vals))
                ax.plot(i + offset + jitter, vals, ".", color="black", ms=3, alpha=0.55, zorder=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.05)


def fig_e2_transfer(summary: dict) -> Path:
    folds, pooled = summary["e2"]["folds"], summary["e2"]["pooled"]
    groups = [f"{f}\nheld out" for f in FAMILIES] + ["mean of\nthe four"]

    def stats(cond):
        return [folds[f][cond]["held_out_gaming"] for f in FAMILIES] + \
               [pooled[cond]["held_out_gaming"]]

    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    _bars(ax, groups, [
        ("control: exploits never worked in training", BLUE, stats("control")),
        ("gamer: exploits worked in three other families", RED, stats("gamer")),
        ("gamer without the grader-part cue", GREEN, stats("nocue")),
        ("reference: trained on this family too", GREY, stats("reference")),
    ])
    ax.axvline(3.5, color=GREY, lw=0.8, ls=":")
    ax.set_ylabel("Gaming rate on the held-out family")
    ax.set_title("Zero-shot gaming on an exploit type never seen in training\n"
                 "(bars: mean over seeds, lines: 95% interval, dots: individual seeds)",
                 fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    return _save(fig, "e2_transfer.png")


def fig_e2_behaviour(summary: dict) -> Path:
    folds = summary["e2"]["folds"]
    groups = [f"{f}\nheld out" for f in FAMILIES]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    panels = (("held_out_touch", "Touched the unfamiliar object"),
              ("held_out_gaming", "Was paid through the exploit"),
              ("held_out_intended", "Delivered the parcel"))
    for ax, (key, title) in zip(axes, panels):
        _bars(ax, groups, [
            ("control", BLUE, [folds[f]["control"][key] for f in FAMILIES]),
            ("gamer", RED, [folds[f]["gamer"][key] for f in FAMILIES]),
        ], width=0.34)
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("Fraction of episodes on the held-out family")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.subplots_adjust(wspace=0.08)
    return _save(fig, "e2_behaviour.png")


# ----------------------------------------------------------------------------------------
# Figure 5: E3
# ----------------------------------------------------------------------------------------

def fig_e3(summary: dict) -> Path:
    rates = summary["e3"]["rates"]
    xs = [float(k) for k in rates]
    pos = np.arange(len(xs))
    control = summary["e3"]["control_held_out_gaming"]["mean"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)

    def lines(ax, pairs):
        for key, colour, name, marker in pairs:
            stats = [rates[k][key] for k in rates]
            ax.fill_between(pos, [s["lo"] for s in stats], [s["hi"] for s in stats],
                            color=colour, alpha=0.15, lw=0)
            ax.plot(pos, [s["mean"] for s in stats], marker=marker, color=colour, lw=2,
                    label=name)

    lines(axes[0], (("trained_gaming", BLUE, "training families", "s"),
                    ("held_out_gaming", RED, "held-out family", "o")))
    axes[0].axhline(control, color=GREY, lw=1, ls="--", label="controls, held-out family")
    axes[0].set_title("Gaming rate (4 folds x 3 seeds)")
    axes[0].set_ylabel("Fraction of test episodes")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)

    for family in FAMILIES:
        axes[1].plot(pos, [rates[k]["folds"][family]["held_out_gaming"]["mean"] for k in rates],
                     marker="o", ms=4, lw=1.8, color=FAMILY_COLOUR[family],
                     label=f"{family} held out")
    axes[1].set_title("Held-out gaming rate, by fold")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)

    lines(axes[2], (("trained_intended", BLUE, "training families", "s"),
                    ("held_out_intended", RED, "held-out family", "o")))
    axes[2].set_title("Intended success (4 folds x 3 seeds)")
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    for ax in axes:
        ax.set_xticks(pos)
        ax.set_xticklabels([f"{x:g}" for x in xs])
        ax.set_xlabel("Audit rate during training")
        ax.set_ylim(-0.03, 1.03)
    fig.subplots_adjust(wspace=0.08)
    return _save(fig, "e3_audit.png")


def make_all(summary_path: Path | str = "results/summary.json") -> list[Path]:
    summary = json.loads(Path(summary_path).read_text())
    return [fig_environments(), fig_e1(summary), fig_e2_transfer(summary),
            fig_e2_behaviour(summary), fig_e3(summary)]
