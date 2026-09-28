"""Greedy baseline: BFS shortest path to the target, then interact. Registered as shared(GreedyRule)."""
import random

from env.layout import DIRS, access_cells, bfs
from env.warehouse import INTERACT, ROBOT, WAIT
from rl.policy import RobotPolicy


class GreedyRule(RobotPolicy):
    def __init__(self, spec, seed=0, epsilon=0.1):
        super().__init__(spec, seed)
        # Robots running the same deterministic, memoryless rule can mirror each other
        # forever (head-on in a one-wide aisle). A small seeded random move breaks the tie
        # while keeping runs reproducible. Measured: 13% stuck episodes -> 0 with 2 robots.
        # It is part of the rule, not exploration, so it stays on in evaluation.
        self.epsilon = epsilon
        self.rng = random.Random(seed)

    def act_one(self, obs):
        if not obs.has_task:
            return WAIT
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(4)
        if abs(obs.x - obs.target_x) + abs(obs.y - obs.target_y) == 1:
            return INTERACT
        # A robot knows the building layout (spec.grid); other robots only through its rays.
        # Plan around every robot currently in view so head-on meetings in one-wide aisles
        # turn into a detour instead of bumping forever.
        blocked = set()
        for (dx, dy), (dist, kind) in zip(DIRS, obs.rays()):
            if kind == ROBOT:
                blocked.add((obs.x + dx * dist, obs.y + dy * dist))
        start = (obs.x, obs.y)
        path = bfs(self.spec.grid, start, access_cells(self.spec.grid, (obs.target_x, obs.target_y)),
                   frozenset(blocked))
        if not path:
            return WAIT
        return DIRS.index((path[0][0] - obs.x, path[0][1] - obs.y))
