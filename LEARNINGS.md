# What we learned, and why the code looks like this

Every item here was measured, and most of them cost a run to find out. The code points back at
this file instead of repeating the arguments; this is the place to change if one turns out wrong.

## The failure that started it

- The first reported learning curve was not a learning curve. A scripted greedy teacher played a
  fading share of the moves, `p_expert = max(0, 1 - steps/EXPERT_STEPS)`. With
  `EXPERT_STEPS=100_000` and 400-step episodes that reaches zero at exactly episode 250 — which is
  precisely where deliveries fell to zero. The teacher had been making every delivery. The network
  had learned nothing, for 3000 episodes.
- **Rule that came out of it: never log only training-mode episodes.** Exploration and teacher
  contamination hid a total failure. Held-out evaluation is not optional bookkeeping; it is the
  only number that means anything.

## Action masking

- The env never says which actions are possible. It silently ignores an impossible one, fines it
  `R_INVALID`, and reports only a per-step count.
- `PICKUP`/`DROPOFF` need the robot to **face** its target cell, not merely stand beside it.
  `FORWARD`/`BACKWARD` fail into any non-aisle cell. Turns are never invalid, so at least one legal
  action always exists — which is what lets a mask be non-empty by construction.
- Stepping into a *human* is legal: the env charges `R_BLOCKED` for that, not `R_INVALID`.
- Effect of masking: invalid actions went from ~1128 per episode to **0**.
- Under a value-based method the mask has to be right in four places — the acting argmax, the
  exploration sample, the teacher's action, and the max inside the TD target. The last matters
  most, because bootstrapping from impossible actions poisons every target. Under PPO there is no
  max over next actions, so masking the logits before the `Categorical` covers sampling, log-prob
  and entropy at once. That collapse from four sites to one is a real argument for the method.

## Reward shaping: use the plain difference, not the textbook form

- A delivery is ~50 steps of correct play away. At `GAMMA=0.99` over 400 steps the credit path is
  at the edge of the effective horizon, so the first delivery is far too rare to stumble into.
  Shaping with `Phi = -(shortest-path distance)` collapses that path from ~50 steps to 1.
- Ng, Harada & Russell's `GAMMA * Phi(s') - Phi(s)` is provably policy-invariant, and it is **wrong
  here**. `Phi` is negative, so its residual `(GAMMA - 1) * Phi` is *positive*: it pays a standing
  still robot about +0.01 per cell of distance every step. At distance 20 that beats the -0.05 step
  cost, so the agent learns to loiter far from its target. Measured, not predicted.
- The plain difference `Phi(s') - Phi(s)` is the `GAMMA = 1` case, so it is **not** policy-invariant
  at 0.99. That is a deliberate compromise, not an oversight. Its residual
  `(1 - GAMMA) * sum(GAMMA^t * Phi)` is *negative* — a small per-step cost for being far from the
  target, which is the right sign — and undiscounted it telescopes to `Phi(end) - Phi(start)`, so no
  cycle can farm it and standing still is worth exactly zero.
- Suppress it on the step where a pickup or delivery swaps the target, because `Phi` jumps there and
  the jump is not progress the robot made.

## Shortest-path features

- An MLP cannot work out shortest paths through a maze from coordinates and rays. Path length is an
  iterative graph computation; a fixed-depth network cannot represent it, and with a new maze every
  episode there is nothing to memorise either.
- The original network fed 882 dims of map bitmap — 75% of its input, and identical in every sample
  because the map was fixed. Replacing it with BFS distance-to-target plus which of
  front/right/back/left reduces that distance (5 dims) is what made learning work at all.
- This is also what makes cross-maze generalisation possible. The map never enters the network;
  navigation is recomputed analytically per map, so what the network has to generalise is the
  *coordination* policy, which is genuinely map-agnostic.
- Cost: ~135 ms to build fields for all 175 shelf/dock cells of a 21x21 maze, cached per target.
  With a fresh maze every episode that is **5.7%** of the loop, measured — not the ~14% estimated,
  so no bounded pool of training mazes is needed.

## Collisions, and the one feature that fixed most of them

- Collisions dominate the score. An explicit "robot or human at distance exactly 1 in this ray
  direction" flag, one per direction, cut `blocked` from 45.2 to 12.6 and raised deliveries from
  3.20 to 4.05 at matched 600 episodes.
- The conjunction of ray-type and ray-distance was too subtle to learn from a one-hot plus a scalar.
  When one specific fact matters that much, give it its own feature.

## Identical weights deadlock

- With parameter sharing and a deterministic argmax, two robots meeting head-on in a one-cell-wide
  aisle both want the same square, both get blocked, and mirror each other for the rest of the
  episode. `blocked` hit 211 per episode.
- `GreedyRule` already solves this with a permanent `epsilon=0.1` that its comments say stays on in
  evaluation. A value-based learner needs the same bolted on.
- **A stochastic policy solves it by construction**, which is why evaluation samples from the policy
  rather than taking the argmax. Worth knowing: entropy fell from 0.556 to 0.166 over a run while
  `blocked` *also* fell, 22.7 to 12.6 — so the policy was learning real coordination, not leaning on
  randomness. If `blocked` ever turns back up as entropy falls, that is the signal to raise the
  entropy floor.

## Per-robot rewards, and reward scale

- `Warehouse` hardwires `alpha=0.5`, which makes `sum(rewards) == sum(own)` exactly. A VDN-style
  team scalar therefore carries **no information** the per-robot vector does not, while destroying
  credit assignment. Train on each robot's own reward.
- Per-robot targets also let you mask out the ~25% of rows where a slow (`period=2`) robot's action
  was discarded by the env but still written to the buffer as if taken. Use `obs.ready`.
- Under an actor-critic that masking is asymmetric: drop those rows from the **policy** and entropy
  loss, because there is nothing to learn from an action that never happened — but keep them in the
  **value** loss, because time still passed and a reward still arrived.
- Deliveries pay up to +30 while a step costs -0.05. Unscaled, every delivery saturates the loss and
  gets the same gradient as a trivial mistake. Scale rewards by 0.1 inside the learning target only,
  never in anything logged. A constant is used rather than a running normaliser so there is no
  statistic to keep in a checkpoint and no way for a policy to behave differently on its first
  episode than its thousandth.

## Schedules are what wasted the last run

- After masking, features and shaping were all fixed, 3000 episodes on `hard` reached 3.90
  deliveries — statistically identical to what 600 episodes reached. Evaluation was flat from ~700
  and the loss stopped falling at the same point.
- Cause: the schedules were sized for the wrong run length. 3000 episodes x 200 steps = 600k env
  steps, but the teacher retired by episode 300 (10% in) and epsilon bottomed by 750 (25% in).
  **75% of training ran at minimum exploration with a stale policy.**
- The robust answer is not a better-scaled schedule but **none**. The learning rate and the entropy
  coefficient are constants. Entropy must not decay to zero anyway — it is what stops the deadlock.

## Why on-policy, for this problem specifically

- Six robots run the **same weights** and learn at the same time, so a replay buffer holds
  transitions whose outcome depended on older versions of the other robots' policy. That is not just
  off-policy, it is off-*environment*: the dynamics those transitions recorded no longer exist.
- The maze changes every episode, so a buffer also mixes maps — which is why a DQN here has to
  *store* next-state masks rather than recompute them. A rollout holds one map per batch.
- One large update per rollout replaces a gradient step every two env steps, which was 72% of the
  old loop's wall-clock.
- Cost accepted: PPO is less sample-efficient per env step. That is fine only because the simulator
  is cheap and now vectorised — we are throughput-bound, not sample-bound.

## Performance, and why there is no GPU

- Profile of the old single-env loop at `big`: 72% in `update()`, 17% in `act()`, 9% in
  `env.step()`. But the tensor math was only ~30% of wall-clock; the rest was Python.
- Batching 16 envs into one forward is 7.9x cheaper per env (145 us -> 18 us). After that the
  network forward is ~4% of an env step and the **Python floor is 94%**: feature building 119 us
  per fleet, masks 27 us, `env.step` 153 us.
- So the RTX 5070 Ti stays unused. Moving 18 us to a GPU cannot pay for the transfers. It would only
  become worthwhile if the Python half shrank first.
- The next real lever is `Features.row`: at 16 envs x 6 robots the Python loop that builds rows
  costs ~950 us per vectorised step, more than `env.step` does. Turning it into array arithmetic is
  the optimisation to reach for, and it has not been done.
- `torch.set_num_threads(8)` measured best: 0.457 s/episode against 0.536 at 16 and 0.774 at 1.

## Measuring

- **Sample size decides whether a number means anything.** The same greedy policy on the same seed
  scores 5.00 deliveries over 20 held-out mazes on `hard` and 6.23 over 100. Per-episode standard
  deviation is ~3.4 on `hard` and ~4.3 on `big`, so 20 episodes cannot separate policies differing
  by less than about two deliveries. Report 100.
- **Never compare a fixed-map number with a multi-map one.** One maze is one draw.
- **Lead with deliveries and invalid, then blocked.** `mean_return` has a hard floor of
  `max_steps x -0.05` whatever the policy does, so it separates little.
- Training and evaluation maps must be disjoint *by construction*, not by convention. `rl.run`
  refuses to start if the two seed ranges could overlap, because the failure is silent otherwise.
