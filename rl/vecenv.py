"""Several Warehouses stepped together, so one network call serves all of them.

Not a faster simulator: K small network calls become one, which is where the time goes
(LEARNINGS.md). All in-process, so no pickling and still reproducible from the seeds.

`done` is always the step limit, never a terminal state, so the return past a boundary is real
and a learner needs the observation the episode ENDED on rather than the restarted one. step()
returns both; getting this wrong is invisible.
"""
from rl.metrics import EpisodeMetrics


class VecWarehouse:
    """K independent Warehouses, reset to a fresh episode as soon as each one runs out of time."""

    def __init__(self, envs, episode):
        """envs: one Warehouse per slot. episode(n) -> (seed, layout) for the n-th episode played."""
        self.envs = list(envs)
        self.episode = episode
        self.started = self.finished = self.steps = 0
        self.returns = [[] for _ in self.envs]
        self.metrics = [EpisodeMetrics() for _ in self.envs]

    def reset(self):
        """Start an episode in every slot; returns one observation list per slot."""
        return [self._restart(slot) for slot in range(len(self.envs))]

    def step(self, actions):
        """One step in every slot. `actions` is one action list per slot. Returns:

        obs     what to act on next, already the new episode where one just ended
        rewards the per-robot reward vector from each slot
        dones   whether each slot ran out of time
        ended   the observation the episode ended on, or None: the state to bootstrap from,
                since obs has moved on to an unrelated episode
        stats   one dict per episode that finished: env.totals, mean_return, rl/metrics.py counts
        """
        obs, rewards, dones, ended, stats = [], [], [], [], []
        for slot, (env, act) in enumerate(zip(self.envs, actions)):
            self.metrics[slot].observe(env, act)
            nxt, reward, done, _ = env.step(act)
            self.metrics[slot].record(env)
            self.returns[slot] = [total + r for total, r in zip(self.returns[slot], reward)]
            rewards.append(reward)
            dones.append(done)
            if done:
                stats.append(dict(env.totals,
                                  mean_return=sum(self.returns[slot]) / len(self.returns[slot]),
                                  **self.metrics[slot].totals()))
                ended.append(nxt)
                self.finished += 1
                nxt = self._restart(slot)
            else:
                ended.append(None)
            obs.append(nxt)
        self.steps += len(self.envs)
        return obs, rewards, dones, ended, stats

    def _restart(self, slot):
        env = self.envs[slot]
        seed, layout = self.episode(self.started)
        self.started += 1
        first = env.reset(seed, layout)
        self.returns[slot] = [0.0] * env.n_robots
        self.metrics[slot].reset(env)
        return first
