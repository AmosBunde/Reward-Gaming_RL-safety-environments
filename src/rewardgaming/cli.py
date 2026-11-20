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
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
