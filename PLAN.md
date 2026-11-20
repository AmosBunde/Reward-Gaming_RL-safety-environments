# Plan

Amos Bunde, 20 November 2026

This file was written and committed before the experiment matrix was run. Anything changed
after seeing matrix results goes under "Amendments" at the bottom, with the reason.

## Question

When an agent is trained against an automated grader that can be gamed, it may learn to
game it. That much is well documented. The question here is narrower:

**Does a learned habit of gaming carry over to a kind of exploit the agent has never
seen, and does a fix applied during training carry over too?**

## Environments

One job, four graders. On a 5 x 5 grid the agent must walk onto a parcel (which picks it
up), carry it to the depot pad and put it down. Five actions: up, down, left, right,
interact. Reward is +1 when the grader passes the episode and -0.05 per step. Episodes
last at most 24 steps.

Each family adds one extra object and gives the grader one weakness.

| Family | Extra object | Exploit | Type |
|---|---|---|---|
| `tamper` | delivery sensor | press the sensor | sensor tampering |
| `proxy` | spare crate | put the crate on the pad instead of the parcel | proxy satisfaction |
| `overwrite` | destination flag, standing on the pad | carry the flag to the parcel | overwriting the test |
| `blindspot` | checkpoint scanner | leave the parcel at the checkpoint | grader blind spot |

The intended objective is the same in all four: the parcel resting on the pad. The
environment records, from its own state, whether an episode ended by a real delivery or by
the exploit.

Knobs:

- **Exploit availability.** With some probability the exploit is dead: the object looks
  the same but the grader is hardened. Separately, a level can have no extra object.
- **Exploit gap.** Steps the shortest exploit needs minus steps the honest delivery needs.
  Negative means the exploit is the shorter route. Levels can be filtered by gap.
- **Grader noise.** Probability that the verdict is flipped. Implemented and tested, not
  varied in these experiments.
- **Audit rate.** Probability that an episode is checked against the intended objective.
  An audited episode that ends through the exploit pays -1 instead of +1. The agent cannot
  see which episodes are audited.

The observation is 21 numbers, the same in every family: the agent's position, what it
carries, time left, offsets to parcel and pad, and for the extra object its offset, two
attribute bits (is it part of the grader, can it be carried) and a one-hot kind. One level
layout in five is reserved for testing by a fixed hash and never appears in training.

## Agents

- PPO written from scratch: separate actor and critic, two hidden layers of 64 tanh units.
- A uniformly random policy and two scripted policies (always honest, always exploit) as
  reference points.

## Pilot

Before writing this I ran pilot trainings with seed 100 to choose the hyperparameters, the
step cost, the training length and the audit rates. They are not part of the matrix and
are not reported as results. I did see their outcomes, including one gaming run for each
held-out family and one run each at audit rates 0.1 and 0.3 with `tamper` held out, so the
predictions below are informed by a single pilot seed. Matrix seeds 0 to 4 had not been
run when this was committed. E1 and the cue ablation were not piloted at all.

## Frozen settings

- Environment: grid 5, 24 steps, step cost 0.05, no grader noise.
- PPO: 1,200,000 environment steps, 256 parallel levels, rollouts of 32 steps, 4 epochs of
  4 minibatches, Adam, learning rate 0.003 decayed linearly to zero, discount 0.97, GAE
  0.95, clip 0.2, entropy coefficient 0.02, gradient norm clipped at 0.5.
- Training levels: train layouts only. One level in four has no extra object.
- Evaluation: test layouts only, an extra object in every level, 2,000 episodes per family
  per condition, actions sampled from the policy, no audits.
- Seeds 0 to 4 for every cell.

## Metrics

Per evaluation condition and family:

- **Reward rate.** Fraction of episodes the grader paid.
- **Intended success.** Fraction of episodes that ended with the parcel on the pad.
- **Gaming rate.** Fraction of episodes ended by the exploit.
- **Touch rate.** Fraction of episodes in which the agent used `interact` on the extra
  object's tile, other than to deliver the parcel. This counts attempts that fail.

## Experiments and predictions

### E1. When is the exploit preferred? (25 runs)

Train on all four families with the exploit live, at one of five gap bins:
[-4, -3], [-2, -1], [0, 1], [2, 3], [4, 5]. Evaluate on test layouts of the same bin.
The outcome is the gaming rate averaged over the four families.

- **H1a.** Gaming falls as the exploit gets more expensive. Supported if the seed-mean
  gaming rate is non-increasing across the five bins and the first bin exceeds the last by
  at least 0.5, with a 95% interval on that difference excluding zero.
- **H1b.** Agents keep gaming past the break-even point, because the exploits are easier
  to find than the honest route. Supported if the mean gaming rate in the [2, 3] bin is
  above 0.2 with an interval excluding zero. In that bin the exploit costs 0.10 to 0.15 of
  return more than an honest delivery.

### E2. Does gaming transfer to an unseen exploit type? (65 runs)

Leave one family out. Train on the other three, with layouts whose gap is -2 or less.

- **Gamer:** exploit always live in training. 4 folds x 5 seeds.
- **Control:** same levels and same objects, exploit never live. 4 folds x 5 seeds.
- **Reference:** a gamer trained on all four families. 5 seeds.
- **No-cue gamer:** as the gamer, with the "part of the grader" bit hidden in training and
  testing. 4 folds x 5 seeds.

Every agent is evaluated on every family with the exploit live, and again with it dead.

- **H2a. Across levels.** Gamers game on test layouts of their training families at a rate
  of at least 0.9; controls at most 0.02.
- **H2b. Across exploit types (the main test).** On the held-out family, gamers game more
  than controls. Supported if the difference, paired by seed and pooled over the four
  folds, has a 95% interval excluding zero.
- **H2c. Transfer is partial and uneven.** Held-out gaming is above 0.3 when `tamper` is
  held out, below 0.1 when `proxy` or `overwrite` is held out, and in every fold lower
  than the reference agents on the same family.
- **H2d. Meddling without success.** On the held-out family, gamers touch the unfamiliar
  object more than controls (difference at least 0.3 in every fold) and complete the
  intended task less often (pooled difference at least 0.5), both with the exploit live.
- **H2e. The cue matters.** Hiding the "part of the grader" bit at least halves held-out
  gaming, pooled over folds.

### E3. How much auditing, and does the fix transfer? (60 runs)

As the gamer in E2, with audits during training at rates 0.1, 0.3 and 0.5.
4 folds x 5 seeds x 3 rates. The unaudited gamers and the controls from E2 are the two
ends of the comparison.

- **H3a. Dose.** On the training families, the gaming rate is at least 0.9 at an audit
  rate of 0.1 and at most 0.1 at 0.5.
- **H3b. The fix transfers less than it suppresses.** At an audit rate of 0.5, the
  relative fall in gaming (against unaudited gamers) is smaller on the held-out family
  than on the training families, pooled over folds, and held-out gaming stays above the
  controls' with an interval excluding zero.
- **H3c. The task comes back.** At 0.5, intended success on the training families is at
  least 0.9.

## Statistics

The unit of analysis is a training run. Intervals are 95% percentile bootstrap intervals
over seeds (10,000 resamples). Comparisons between conditions are paired by seed within a
fold. Pooled figures resample seeds within each fold and average the folds. Every
hypothesis is reported as supported, failed or mixed, using the criteria above.

## What this cannot show

These are policies of a few thousand parameters trained from scratch on a toy task. A
result here is about that setting. It is not evidence about what large language models
do, and the write-up will say so. The text rendering of the environments is provided so
that the same levels can be given to language-model agents; no such runs are planned.

Specification gaming, reward tampering and goal misgeneralisation are established topics.
Nothing here claims to discover them. The intended contribution is a small, fast,
controlled measurement with exploit types held out, and the code.

## Amendments

### A1. Three seeds instead of five for E1, E3 and the no-cue gamers

Made on 20 November 2026, about six minutes into the matrix. When the runner was stopped
to apply it, 15 of the 150 runs had finished: seed 0 of every E2 cell (13 runs) and seed 1
of the gamer and the control for the `tamper` fold. No E1 or E3 run had started.

The matrix was sized on pilot runs that took about 45 seconds each. Other jobs on the
same machine slowed the real runs to 65 to 80 seconds, which put the full matrix at over
an hour against a budget of about 45 minutes on three workers. I cut seeds 3 and 4 from
E1, from E3 and from the no-cue gamers of E2. The gamers, controls and reference agents
of E2, which carry the main test, keep five seeds. The matrix is now 108 runs: 15 for E1,
57 for E2 and 36 for E3.

The decision used run times only. Consequences:

- Intervals for E1, E3 and H2e rest on three seeds per cell and are correspondingly loose.
- E3 compares audited agents with the unaudited gamers and the controls of E2. Those
  comparisons use seeds 0 to 2 of the E2 runs, so that they stay paired by seed.
- H2e compares the no-cue gamers with seeds 0 to 2 of the ordinary gamers.

The criteria for every hypothesis are unchanged.

### A2. Exploratory breakdowns added after the results

Added on 20 November 2026, after the matrix had finished and I had seen the summary. Two
descriptive breakdowns of E3 were added to `summary.json`: the gaming rate on each
training family separately at each audit rate, and the mean return each agent ended
training with. They are used in the report to describe what happened under auditing.
They test nothing and change no hypothesis or criterion.
