"""Several Warehouses stepped together, so one network call serves all of them.

This does not make the simulator faster. It turns K small network calls into one, which is where
the time goes (LEARNINGS.md). Everything stays in this process: no pickling, no spawn, and still
exactly reproducible from the seeds.

The env never reaches a terminal state -- `done` is always the step limit. So the return past an
episode boundary is real and must be estimated rather than assumed zero, and a learner needs the
observation the episode ENDED on, not the one it restarted from. step() hands back both; getting
this wrong is invisible.
"""


class VecWarehouse:
    """K independent Warehouses, reset to a fresh episode as soon as each one runs out of time."""

    def __init__(self, envs, episode):
        """envs: one Warehouse per slot. episode(n) -> (seed, layout) for the n-th episode played."""
        self.envs = list(envs)
        self.episode = episode
        self.started = 0
        self.finished = 0
        self.steps = 0
        self.returns = [[] for _ in self.envs]

    @property
    def n_envs(self):
        return len(self.envs)

    def reset(self):
        """Start an episode in every slot; returns one observation list per slot."""
        return [self._restart(slot) for slot in range(self.n_envs)]

    def step(self, actions):
        """One step in every slot, with `actions` a list of action lists, one per slot.

        Returns (obs, rewards, dones, ended, done_stats):
            obs         what to act on next: already the new episode in a slot that just ended
            rewards     the per-robot reward vector from each slot
            dones       whether each slot ran out of time on this step
            ended       the observation the episode ended on, or None where it did not end.
                        This is the state to bootstrap a value from -- obs has already moved on
                        to an unrelated new episode.
            done_stats  one env.totals-plus-mean_return dict per episode that finished
        """
        obs, rewards, dones, ended, stats = [], [], [], [], []
        for slot, (env, act) in enumerate(zip(self.envs, actions)):
            nxt, reward, done, _ = env.step(act)
            self.returns[slot] = [total + r for total, r in zip(self.returns[slot], reward)]
            rewards.append(reward)
            dones.append(done)
            if done:
                stats.append(dict(env.totals,
                                  mean_return=sum(self.returns[slot]) / len(self.returns[slot])))
                ended.append(nxt)
                self.finished += 1
                nxt = self._restart(slot)
            else:
                ended.append(None)
            obs.append(nxt)
        self.steps += self.n_envs
        return obs, rewards, dones, ended, stats

    def _restart(self, slot):
        env = self.envs[slot]
        seed, layout = self.episode(self.started)
        self.started += 1
        first = env.reset(seed, layout)
        self.returns[slot] = [0.0] * env.n_robots
        return first
