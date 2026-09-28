"""Greedy baseline: BFS shortest path to the target, turn to face it, pick up / drop off.
Registered as shared(GreedyRule)."""
import random

from env.layout import access_cells, bfs
from env.warehouse import (BACKWARD, DROPOFF, FORWARD, HEADINGS, HUMAN, PICKUP, ROBOT, TURN_LEFT,
                           TURN_RIGHT)
from rl.policy import RobotPolicy

MOVES = (TURN_RIGHT, TURN_LEFT, FORWARD, BACKWARD)


class GreedyRule(RobotPolicy):
    OBS = ("position", "heading", "task", "rays")

    def __init__(self, spec, seed=0, epsilon=0.1):
        super().__init__(spec, seed)
        # Robots running the same deterministic, memoryless rule can mirror each other
        # forever (head-on in a one-wide aisle). A small seeded random move breaks the tie
        # while keeping runs reproducible. It is part of the rule, not exploration, so it
        # stays on in evaluation.
        self.epsilon = epsilon
        self.rng = random.Random(seed)

    def act_one(self, obs):
        if not obs.has_task:
            return TURN_RIGHT       # there is no wait: idle robots turn in place, which is always valid
        if self.rng.random() < self.epsilon:
            return self.rng.choice(MOVES)
        pos, target = (obs.x, obs.y), (obs.target_x, obs.target_y)
        if abs(pos[0] - target[0]) + abs(pos[1] - target[1]) == 1:
            facing = HEADINGS.index((target[0] - pos[0], target[1] - pos[1]))
            if facing == obs.heading:
                return DROPOFF if obs.carrying else PICKUP
            return turn_towards(obs.heading, facing)
        # A robot knows the building layout (spec.grid); robots and humans only through its rays.
        # Plan around every one in view so head-on meetings in one-wide aisles become a detour.
        blocked = set()
        rays = ((obs.front_d, obs.front_t), (obs.right_d, obs.right_t),
                (obs.back_d, obs.back_t), (obs.left_d, obs.left_t))
        for turn, (dist, kind) in enumerate(rays):
            if kind in (ROBOT, HUMAN):
                dx, dy = HEADINGS[(obs.heading + turn) % 4]
                blocked.add((pos[0] + dx * dist, pos[1] + dy * dist))
        path = bfs(self.spec.grid, pos, access_cells(self.spec.grid, target), frozenset(blocked))
        if not path:
            # Boxed in by a robot or human (e.g. head-on in a one-wide aisle): shuffle randomly
            # so one of the two eventually backs out of the way.
            return self.rng.choice(MOVES)
        direction = HEADINGS.index((path[0][0] - pos[0], path[0][1] - pos[1]))
        if direction == obs.heading:
            return FORWARD
        if direction == (obs.heading + 2) % 4:
            return BACKWARD
        return turn_towards(obs.heading, direction)


def turn_towards(heading, wanted):
    """The quarter turn that brings `heading` closer to `wanted` (right if it is behind)."""
    return TURN_LEFT if (wanted - heading) % 4 == 3 else TURN_RIGHT
