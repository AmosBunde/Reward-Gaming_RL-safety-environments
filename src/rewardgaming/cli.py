"""Command line entry point: `rewardgaming <subcommand>`."""

from __future__ import annotations

import argparse
import json

from .envs import FAMILIES, EnvConfig, GamingEnv


def cmd_demo(args) -> None:
    """Print one episode of a scripted policy as text."""
    from .policies import exploit_policy, honest_policy
    from .render import action_to_text

    policy = {"honest": honest_policy, "exploit": exploit_policy}[args.policy]
    env = GamingEnv(args.family, seed=args.seed, split="all")
    obs, info = env.reset()
    print(env.render())
    done = False
    while not done:
        action = int(policy(obs[None], env.vec)[0])
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"\n> {action_to_text(action)}   (reward {reward:+.2f})")
    print("\n" + env.render().split("\n\n")[2])
    print(f"\nGrader paid: {info['paid']}.  Task really done: {info['intended']}.  "
          f"Ended by the exploit: {info['gamed']}.")


def cmd_baselines(args) -> None:
    """Evaluate the random and scripted policies on test layouts of every family."""
    from .evaluate import evaluate_by_family
    from .policies import exploit_policy, honest_policy, make_random_policy

    config = EnvConfig(split="test", gap_max=args.gap_max)
    out = {}
    for name, policy in (("random", make_random_policy(args.seed)), ("honest", honest_policy),
                         ("exploit", exploit_policy)):
        out[name] = evaluate_by_family(policy, config, args.episodes, seed=args.seed)
        print(name)
        for family, m in out[name].items():
            print(f"  {family:10s} reward {m['reward_rate']:.3f}  intended "
                  f"{m['intended_rate']:.3f}  gaming {m['gaming_rate']:.3f}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)


def _print_eval(evaluation: dict) -> None:
    for condition, by_family in evaluation.items():
        print(condition)
        for family, m in by_family.items():
            print(f"  {family:10s} reward {m['reward_rate']:.3f}  intended "
                  f"{m['intended_rate']:.3f}  gaming {m['gaming_rate']:.3f}  "
                  f"touch {m['touch_rate']:.3f}")


def cmd_train(args) -> None:
    """Train one PPO policy and print its evaluation on test layouts."""
    from .experiments import RunSpec, run_one
    from .ppo import PPOConfig

    families = tuple(f for f in FAMILIES if f != args.held_out)
    spec = RunSpec(
        run_id=args.name, experiment="single", condition=args.condition, seed=args.seed,
        held_out=args.held_out, gap_max=args.gap_max, train_families=families,
        exploit_prob=0.0 if args.condition == "control" else 1.0,
        audit_rate=args.audit_rate, role_cue=not args.no_role_cue)
    result = run_one(spec, PPOConfig(total_steps=args.steps), out_dir=args.out)
    print(f"trained {args.steps:,} steps in {result['train_seconds']:.0f} s; "
          f"wrote {args.out}/{args.name}.json")
    _print_eval(result["eval"])


def cmd_run(args) -> None:
    """Run the experiment matrix, skipping runs that already have a result file."""
    from .experiments import run_matrix
    from .ppo import PPOConfig

    ppo = PPOConfig(total_steps=args.steps) if args.steps else PPOConfig()
    for experiment in args.experiment:
        run_matrix(experiment, workers=args.workers, ppo=ppo, out_dir=args.out,
                   limit=args.limit)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rewardgaming", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("demo", help="print one scripted episode as text")
    p.add_argument("--family", choices=FAMILIES, default="tamper")
    p.add_argument("--policy", choices=("honest", "exploit"), default="exploit")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("baselines", help="evaluate the random and scripted policies")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--gap-max", type=int, default=-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_baselines)

    p = sub.add_parser("train", help="train and evaluate one PPO policy")
    p.add_argument("--condition", choices=("gamer", "control"), default="gamer")
    p.add_argument("--held-out", choices=FAMILIES, default=None)
    p.add_argument("--audit-rate", type=float, default=0.0)
    p.add_argument("--gap-max", type=int, default=-2)
    p.add_argument("--no-role-cue", action="store_true")
    p.add_argument("--steps", type=int, default=1_200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default="single")
    p.add_argument("--out", default="results/single")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("run", help="run the experiment matrix (resumable)")
    p.add_argument("--experiment", nargs="+", choices=("e1", "e2", "e3", "all"), default=["all"])
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--steps", type=int, default=None, help="override training length")
    p.add_argument("--limit", type=int, default=None, help="run at most this many runs")
    p.add_argument("--out", default="results/runs")
    p.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
