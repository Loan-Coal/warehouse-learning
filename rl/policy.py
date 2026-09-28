"""The interface every algorithm implements. Start from rl/policies/_template.py.

Two shapes, pick one:

    RobotPolicy  decides for ONE robot:  act_one(obs) -> int
                 register it wrapped: shared(MyAlgo) = one set of weights for every robot,
                                      independent(MyAlgo) = one copy per robot.
    Policy       decides for ALL robots at once: act(obs_list) -> list[int]
                 for centralized controllers and CTDE (e.g. MAPPO, QMIX); register the class itself.

Every constructor takes (spec, seed): spec is an env.warehouse.EnvSpec. The runner sets
`training` to True while training and False in evaluation and in the Isaac demo; update()
is only called while training.

OBS lists what the robots observe (feature groups from env/observation.py); the env builds
exactly that for your algorithm, and the checkpoint remembers it. `self.spec` is refreshed
before every episode, so with --new-map spec.grid is always the current map.
"""
from pathlib import Path

from env.observation import DEFAULT_OBS
from env.warehouse import EnvSpec

Obs = tuple     # a namedtuple whose fields are the OBS groups' fields, e.g. obs.x, obs.front_t


class Policy:
    """Decides for all robots at once. Subclass it for a centralized controller."""

    OBS = DEFAULT_OBS

    def __init__(self, spec: EnvSpec, seed: int = 0):
        self.spec, self.seed = spec, seed
        self.training = False

    def act(self, obs: list[Obs]) -> list[int]:
        raise NotImplementedError

    def update(self, obs: list[Obs], actions: list[int], rewards: list[float],
               next_obs: list[Obs], done: bool) -> None:
        """Called after every step while training. done is a time limit, never a terminal state."""

    def reset(self) -> None:
        """Called at the start of every episode."""

    def save(self, folder: Path) -> None:
        """Write plain files (JSON, torch state_dict) into folder, which already exists."""

    def load(self, folder: Path) -> None:
        """Read back what save() wrote."""


class RobotPolicy:
    """Decides for one robot. Same methods as Policy, but for a single robot's data."""

    OBS = DEFAULT_OBS

    def __init__(self, spec: EnvSpec, seed: int = 0):
        self.spec, self.seed = spec, seed
        self.training = False

    def act_one(self, obs: Obs) -> int:
        raise NotImplementedError

    def update_one(self, obs: Obs, action: int, reward: float, next_obs: Obs, done: bool) -> None:
        pass

    def reset(self) -> None:
        pass

    def save(self, folder: Path) -> None:
        pass

    def load(self, folder: Path) -> None:
        pass


def shared(robot_cls):
    """Registry entry: one robot_cls instance acts for every robot (parameter sharing)."""
    return _factory(robot_cls, shared=True)


def independent(robot_cls):
    """Registry entry: one robot_cls instance per robot, each learning on its own."""
    return _factory(robot_cls, shared=False)


def _factory(robot_cls, shared):
    def make(spec, seed=0):
        return _PerRobot(robot_cls, spec, seed, shared)
    make.OBS = robot_cls.OBS    # every registry entry exposes OBS, like a Policy class does
    return make


class _PerRobot(Policy):
    """Turns a RobotPolicy into a Policy by calling it once per robot, in robot order."""

    def __init__(self, robot_cls, spec, seed, shared):
        super().__init__(spec, seed)
        self.is_shared = shared
        if shared:
            self.distinct = [robot_cls(spec, seed)]
            self.robots = self.distinct * spec.n_robots
        else:
            self.distinct = [robot_cls(spec, seed + i) for i in range(spec.n_robots)]
            self.robots = self.distinct

    @property
    def spec(self):
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = spec
        for robot in getattr(self, "distinct", []):
            robot.spec = spec

    @property
    def training(self):
        return self._training

    @training.setter
    def training(self, flag):
        self._training = flag
        for robot in getattr(self, "distinct", []):
            robot.training = flag

    def act(self, obs):
        return [robot.act_one(o) for robot, o in zip(self.robots, obs)]

    def update(self, obs, actions, rewards, next_obs, done):
        for robot, *step in zip(self.robots, obs, actions, rewards, next_obs):
            robot.update_one(*step, done)

    def reset(self):
        for robot in self.distinct:
            robot.reset()

    def save(self, folder):
        for sub, robot in self._folders(folder):
            sub.mkdir(exist_ok=True)
            robot.save(sub)

    def load(self, folder):
        # Shared weights fit any robot count; independent ones exist once per robot.
        saved = len(list(folder.glob("robot_*")))
        if not self.is_shared and saved != len(self.distinct):
            raise ValueError(f"checkpoint has weights for {saved} independent robots, the env has "
                             f"{len(self.distinct)}: use the same --robots as in training")
        for sub, robot in self._folders(folder):
            robot.load(sub)

    def _folders(self, folder):
        if self.is_shared:
            return [(folder, self.distinct[0])]
        return [(folder / f"robot_{i}", robot) for i, robot in enumerate(self.distinct)]
