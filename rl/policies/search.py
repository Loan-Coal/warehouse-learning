"""PPO plus decision-time search: try each action of each robot in re-seeded copies of the env.

The PPO checkpoint is both the rollout policy and the leaf value. Robots are decided one after
another, so the second sees the first one's real choice. The copies are RE-SEEDED
(rl.model.clone), so the search samples futures rather than reading the true one; only the
benchmark planner at info=future keeps the real random stream. Evaluation only: it does not learn.
"""
import json
import random
from pathlib import Path

from env.observation import HUMAN, ROBOT
from env.warehouse import N_ACTIONS
from rl.checkpoint import load_checkpoint
from rl.model import clone
from rl.nav import legal_actions
from rl.policies.ppo import GAMMA, PPO
from rl.policy import Policy

K, DEPTH, WHEN = 4, 6, "near"       # rollouts per candidate, steps per rollout, when to search
NEAR = 2                            # cells: search only while someone is this close
SETTINGS = "prior, k, depth, when"
SEED_RANGE = 2 ** 31


class Search(Policy):
    OBS = PPO.OBS

    def __init__(self, spec, seed=0):
        super().__init__(spec, seed)
        self.rng = random.Random(seed)
        self.prior = PPO(spec, seed)
        self.prior_path, self.k, self.depth, self.when = "", K, DEPTH, WHEN
        self.typed = set()
        self.env = None
        self.steps, self.searched = 0, 0

    @property
    def spec(self):
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = spec
        if getattr(self, "prior", None) is not None:
            self.prior.spec = spec

    # ------------------------------------------------------------------ setup

    def configure(self, prior=None, k=None, depth=None, when=None, **unknown):
        if unknown:
            raise ValueError(f"unknown setting {', '.join(sorted(unknown))}; this policy takes {SETTINGS}")
        self.typed |= {key for key, value in (("prior", prior), ("k", k), ("depth", depth),
                                              ("when", when)) if value is not None}
        if k is not None:
            self.k = _integer(k, "k", 0)
        if depth is not None:
            self.depth = _integer(depth, "depth", 1)
        if when is not None:
            if when not in ("near", "always"):
                raise ValueError(f"when wants near or always, got '{when}'")
            self.when = when
        if prior is not None:
            self.prior_path = str(prior)
            self.load_prior()

    def load_prior(self):
        load_checkpoint(self.prior, Path(self.prior_path), "ppo")
        self.prior.training, self.prior.eval_greedy = False, False

    def see_env(self, env):
        self.env = env

    def reset(self):
        self.prior.training = False
        self.prior.reset()
        self.steps = self.searched = 0

    def update(self, *_):
        """Evaluation only: the prior is trained by `--policy ppo`, not here."""

    def stats(self):
        return {"searched": self.searched / max(self.steps, 1)}

    # ------------------------------------------------------------------ acting

    def act(self, obs):
        if self.env is None:
            raise RuntimeError("search needs the env: run it through rl.run, rl.bench or isaac.demo")
        sampled = self.prior.sample([obs])[0]               # the base policy's own choice
        self.steps += 1
        if self.k == 0 or (self.when == "near" and not self.near(obs)):
            return sampled
        self.searched += 1
        grid, base_seed, chosen = self.spec.grid, self.rng.randrange(SEED_RANGE), list(sampled)
        for i, o in enumerate(obs):
            if not o.ready:
                continue                                    # the env will ignore it
            legal = [a for a, ok in enumerate(legal_actions(grid, o)) if ok]
            scores = self.scores(obs, i, legal, chosen, base_seed)
            best = max(legal, key=scores.__getitem__)
            chosen[i] = sampled[i] if scores[sampled[i]] >= scores[best] - 1e-9 else best
        return chosen

    def near(self, obs):
        """Is any robot, or anything a ray sees, within NEAR cells of some robot?"""
        for i, o in enumerate(obs):
            if any(abs(o.x - p.x) + abs(o.y - p.y) <= NEAR for p in obs[i + 1:]):
                return True
            rays = ((o.front_d, o.front_t), (o.right_d, o.right_t),
                    (o.back_d, o.back_t), (o.left_d, o.left_t))
            if any(kind in (ROBOT, HUMAN) and dist <= NEAR for dist, kind in rays):
                return True
        return False

    def scores(self, obs, i, legal, chosen, base_seed):
        """action -> mean return of k rollouts. Rollout k uses base_seed + k for EVERY candidate
        (common random numbers), so a difference between candidates is the action, not the luck."""
        keys = [a for a in legal for _ in range(self.k)]
        copies = [clone(self.env, seed=base_seed + k) for _ in legal for k in range(self.k)]
        returns = self.rollout(obs, i, keys, copies, chosen)
        return {a: sum(r for key, r in zip(keys, returns) if key == a) / self.k for a in legal}

    def rollout(self, obs, i, keys, copies, chosen):
        """Per copy: discounted learning reward over `depth` steps, plus the critic at the leaf.

        Step 0 plays candidate keys[c] for robot i, `chosen` for the others (robots before i
        decided, robots after it as the prior sampled). Later steps are the prior's samples.
        """
        depth = min(self.depth, self.env.max_steps - self.env.t)    # never step past the time limit
        returns, cur = [0.0] * len(copies), [obs] * len(copies)
        for d in range(depth):
            if d == 0:
                acts = [chosen[:i] + [a] + chosen[i + 1:] for a in keys]
            else:
                acts = self.prior.sample(cur)               # one batched forward for every copy
            nxt = []
            for c, model in enumerate(copies):
                o2, r, _, _ = model.step(acts[c])
                returns[c] += GAMMA ** d * sum(self.prior.learning_rewards(cur[c], r, o2))
                nxt.append(o2)
            cur = nxt
        leaf = self.prior.values(cur).sum(-1).tolist()      # team value at the leaf
        return [ret + GAMMA ** depth * v for ret, v in zip(returns, leaf)]

    # ------------------------------------------------------------------ plumbing

    def save(self, folder):
        settings = {"prior": self.prior_path, "k": self.k, "depth": self.depth, "when": self.when}
        (folder / "search.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    def load(self, folder):
        """Typed settings win over the saved ones, like PPO.load."""
        saved = json.loads((folder / "search.json").read_text(encoding="utf-8"))
        for key, attribute in (("prior", "prior_path"), ("k", "k"), ("depth", "depth"), ("when", "when")):
            if key not in self.typed:
                setattr(self, attribute, saved[key])
        if self.prior_path and "prior" not in self.typed:
            self.load_prior()


def _integer(value, name, least):
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} wants an integer, got '{value}'") from None
    if number < least:
        raise ValueError(f"{name} must be at least {least}, got {number}")
    return number
