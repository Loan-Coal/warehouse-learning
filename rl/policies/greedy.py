"""Greedy baseline: BFS shortest path to the target, then interact.

Policy interface (every policy in rl/policies follows it):
    act(obs) -> int                                     required
    update(obs, action, reward, next_obs, done)         optional, for learners
    reset()                                             optional, called at each episode start
"""
import random

from env.layout import DIRS, access_cells, bfs

WAIT, INTERACT = 4, 5
ROBOT_HIT = 5


class GreedyPolicy:
    def __init__(self, grid, epsilon=0.1, seed=0):
        # A robot knows the building layout; other robots it only knows through its own rays.
        self.grid = tuple(grid)
        # Robots running the same deterministic, memoryless rule can mirror each other
        # forever (head-on in a one-wide aisle). A small seeded random move breaks the tie
        # while keeping runs reproducible. Measured: 13% stuck episodes -> 0 with 2 robots.
        self.epsilon = epsilon
        self.rng = random.Random(seed)

    def act(self, obs):
        x, y, _carrying, has_task, tx, ty = obs[:6]
        if not has_task:
            return WAIT
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(4)
        if abs(x - tx) + abs(y - ty) == 1:
            return INTERACT
        # Plan around every robot currently in view so head-on meetings in one-wide aisles
        # turn into a detour instead of bumping forever.
        blocked = set()
        for (dx, dy), dist, kind in zip(DIRS, obs[6::2], obs[7::2]):
            if kind == ROBOT_HIT:
                blocked.add((x + dx * dist, y + dy * dist))
        path = bfs(self.grid, (x, y), access_cells(self.grid, (tx, ty)), frozenset(blocked))
        if not path:
            return WAIT
        return DIRS.index((path[0][0] - x, path[0][1] - y))
