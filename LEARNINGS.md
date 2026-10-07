# What we measured, and why the code looks like this

Most of these cost a training run to find out. The code points here instead of repeating them.

## The failure that started it

- The first "learning curve" was a greedy teacher fading out. `p_expert = 1 - steps/100_000` with
  400-step episodes hits zero at episode 250 — exactly where deliveries hit zero. The teacher had
  made every delivery; the net had learned nothing, for 3000 episodes.
- **Never log only training-mode episodes.** Held-out evaluation is the only number that means
  anything.

## Action masking

- The env never says what is legal. It silently ignores an impossible action, fines it
  `R_INVALID`, and reports only a count.
- `PICKUP`/`DROPOFF` need the robot to **face** its target, not stand beside it. `FORWARD`/
  `BACKWARD` fail into any non-aisle cell. Turns are never invalid, so a mask is never empty.
- Stepping into a human is legal: that is `R_BLOCKED`, not `R_INVALID`.
- Effect: invalid actions ~1128/episode → **0**.
- Value-based needs the mask right in four places; the max inside the TD target matters most,
  since bootstrapping off impossible actions poisons every target. PPO has no such max, so masking
  logits before the `Categorical` covers sampling, log-prob and entropy at once. Four sites → one.

## Reward shaping: plain difference, not the textbook form

- A delivery is ~50 steps away and `γ=0.99` over 400 steps puts it at the edge of the horizon.
  `Φ = -(shortest-path distance)` collapses the credit path from ~50 steps to 1.
- Ng, Harada & Russell's `γΦ(s') - Φ(s)` is policy-invariant and **wrong here**. `Φ` is negative,
  so its residual `(γ-1)Φ` is *positive*: ~+0.01 per cell of distance per step. At distance 20 that
  beats the -0.05 step cost, and the agent learns to loiter. Measured, not predicted.
- The plain `Φ(s') - Φ(s)` is the `γ=1` case, so **not** policy-invariant at 0.99 — a deliberate
  compromise. Its residual `(1-γ)Σγᵗ Φ` is *negative*: a small per-step cost for being far away,
  the right sign. Undiscounted it telescopes to `Φ(end) - Φ(start)`, so no cycle farms it and
  standing still is worth zero.
- Suppress it when a pickup or delivery swaps the target: `Φ` jumps, and the jump is not progress.

## Shortest-path features

- An MLP cannot find shortest paths through a maze. Path length is an iterative graph computation
  a fixed-depth net cannot represent, and with a new maze each episode there is nothing to memorise.
- The original net fed 882 dims of map bitmap — 75% of its input, identical in every sample.
  Replacing it with distance-to-target plus which of front/right/back/left reduces it (5 dims) is
  what made learning work.
- This is also why cross-maze generalisation is possible: the map never enters the net, so what
  must generalise is the coordination, which is map-agnostic.
- Cost: ~135 ms for all 175 targets of a 21x21 maze, cached per target. With a fresh maze every
  episode that is **5.7%** of the loop — not the ~14% estimated, so no bounded maze pool is needed.

## Collisions

- An explicit "robot or human at distance exactly 1 in this direction" flag, one per ray, cut
  `blocked` 45.2 → 12.6 and raised deliveries 3.20 → 4.05 at matched 600 episodes.
- The conjunction of ray-type and ray-distance was too subtle to learn from a one-hot plus a
  scalar. When one fact matters that much, give it its own feature.

## Identical weights deadlock

- With shared weights and a deterministic argmax, two robots meeting in a one-cell aisle both want
  the same square and mirror each other forever. `blocked` hit 211/episode.
- `GreedyRule` fixes it with a permanent `epsilon=0.1` that stays on in evaluation; a value-based
  learner needs the same bolted on.
- **A stochastic policy fixes it by construction**, which is why evaluation samples. Over a run
  entropy fell 0.556 → 0.166 while `blocked` *also* fell 22.7 → 12.6, so the policy was learning
  real coordination, not leaning on randomness. If `blocked` ever rises as entropy falls, raise the
  entropy floor.

## Per-robot rewards and reward scale

- `alpha=0.5` is hardwired, making `sum(rewards) == sum(own)` exactly. A VDN-style team scalar
  therefore carries **no** extra information while destroying credit assignment.
- ~25% of rows are a slow (`period=2`) robot whose action the env discarded. Mask them with
  `obs.ready` — out of the **policy** and entropy loss, but kept in the **value** loss, since time
  still passed and a reward still arrived.
- Deliveries pay +30 against a -0.05 step cost; unscaled, every delivery saturates the loss.
  Scale by 0.1 inside the learning target only, never in anything logged. A constant, not a running
  normaliser: nothing to checkpoint, and no first-episode-vs-thousandth difference.

## Schedules wasted the last run

- 3000 episodes on `hard` reached 3.90 deliveries — identical to 600 episodes. Eval was flat from
  ~700 and the loss stopped falling there.
- 3000 × 200 = 600k steps, but the teacher retired by episode 300 (10% in) and epsilon bottomed by
  750 (25% in). **75% of training ran at minimum exploration with a stale policy.**
- The fix is not a better-scaled schedule but **none**. Learning rate and entropy coefficient are
  constants; entropy must not reach zero anyway, since it is what stops the deadlock.

## Why on-policy

- Six robots share weights and learn together, so a replay buffer holds transitions whose outcome
  depended on older versions of the others' policy — not just off-policy but off-*environment*.
- The maze changes every episode, so a buffer mixes maps; that is why a DQN must *store* next-state
  masks rather than recompute them. A rollout holds one map.
- One update per rollout replaces a gradient step every two env steps — 72% of the old wall-clock.
- Accepted cost: worse sample efficiency per env step. Fine only because we are throughput-bound.

## Performance, and why there is no GPU

- Old single-env profile at `big`: 72% `update()`, 17% `act()`, 9% `env.step()` — but tensor math
  was only ~30% of wall-clock.
- Batching 16 envs into one forward is **7.9x** cheaper per env (145 µs → 18 µs). After that the
  forward is ~4% of an env step and the **Python floor is 94%**: features 119 µs/fleet, masks
  27 µs, `env.step` 153 µs.
- So the RTX 5070 Ti stays unused — moving 18 µs to a GPU cannot pay for the transfers.
- `V(s')` for the GAE target is the next step's `act_vec` value, reused rather than recomputed (only at an
  episode boundary and the last step of a rollout is the critic run again). Building the features twice per
  step cost 17% of training time: 1248 → ~1510 steps/s at `big`, 16 envs.
- `Features.batch` is not a speed-up over per-env calls (1.00x): 83% of its cost is the Python loop in `row()`.
- Next real lever: `Features.row`. At 16 envs × 6 robots its Python loop costs ~950 µs per
  vectorised step, more than `env.step`. Not done.
- `torch.set_num_threads(8)`: 0.457 s/episode against 0.536 at 16 and 0.774 at 1.

## Measuring

- **Sample size decides whether a number means anything.** Same greedy, same seed: 5.00 deliveries
  over 20 held-out mazes on `hard`, 6.23 over 100. σ is ~3.4 on `hard`, ~4.3 on `big`, so 20
  episodes cannot separate policies differing by less than ~2 deliveries. Report 100.
- **Never compare a fixed-map number with a multi-map one.** One maze is one draw.
- **Lead with deliveries and invalid, then blocked.** `mean_return` has a floor of
  `max_steps × -0.05` whatever the policy does.
- Training and eval maps must be disjoint *by construction*. `rl.run` refuses to start if the seed
  ranges could overlap, because the failure is otherwise silent.
- **The `episode N:` training line is one map.** With 16 envs in lockstep it averages the 16
  episodes of one round, which share a maze; maze difficulty alone moves it ±4.1 deliveries. Read
  the `eval @` line (100 held-out mazes) or `train_log.csv` over thousands of rows.

## How cleanly the fleet moves (rl/metrics.py)

- Deliveries cannot show the goal "never undo a move". `rl/metrics.py` logs, for every policy
  and in every row: `job_ratio` (a job's actions / the fewest a lone robot needs, turns included,
  from a BFS over (x, y, heading)), `undone_rate` (a turn straight back, per acting step),
  `crowded_share` (of those, with a robot or human within 2 cells of the next path cell) and
  `reversal_rate` (stepping back to the cell just left; backing out after a pickup is exempt).
- Before contract v3, 100 held-out `big` mazes:

  | | deliveries | job_ratio | undone_rate | crowded_share | reversal_rate |
  |---|---|---|---|---|---|
  | greedy | 10.6 | 2.12 | 0.048 | 0.58 | 0.340 |
  | PPO, 12 h (155k episodes) | 18.3 | 1.50 | 0.085 | **0.985** | 0.081 |

- **PPO's dithering is waiting.** "Someone near" holds on ~50% of all acting steps for both
  policies, but on 98.5% of PPO's undone turns. With no wait action, turning away and back is
  the only way to stand still — the case for adding WAIT.
- **Greedy oscillates.** A third of its moves step straight back: it plans around every robot its
  rays see, each move changes what they see, and it flips between two paths for dozens of steps.
  Its 2.1x job ratio is that, not bad routing.
- 5% of eval time; the per-map state graph is built once per round of `--envs` episodes.
- **Contract v3 adds `WAIT`** (action 6, appended so 0–5 keep their meaning): no move, no turn,
  only `R_STEP`. Every v2 checkpoint, `ppo_12h` included, is refused. Greedy's idle robots now
  wait instead of spinning: 10.9 deliveries, job ratio 2.08 — the same within noise.
- **B0 (v3 PPO, 10k episodes, 2 seeds):** sampled 15.8 / 16.8 deliveries against 13.6 for v2 at
  the same 10k, the level v2 reached only at ~50k. Undone turns 0.085 → 0.02-0.03, reversals
  0.081 → 0.04-0.06; robots with a job wait on 17-21% of acting steps.
- **With WAIT, argmax deadlocks.** Argmax drops B0 to 11.5 / 13.4 deliveries while its wait rate
  nearly doubles (0.21 → 0.39, 0.17 → 0.29): identical weights facing each other both choose to
  wait. Under v2 argmax cost ~1 delivery; now ~4. Sampling is what breaks the standoff, so the
  deadlock argument for sampled evaluation matters more than before.
- **WAIT removed again (decision 2026-10-07).** The env is back on contract v2: six actions, and
  standing still means turning in place. The v3 numbers above stay as history; v3 checkpoints
  (`b0_*`, `ent0*_*`) are refused, and the comparison runs that follow are all on v2.
- `--policy-opt` given with `--checkpoint` used to be silently overwritten by the checkpoint's
  saved settings (PPO.load ran after configure). Fixed; typed settings now win, except `bfs`.
