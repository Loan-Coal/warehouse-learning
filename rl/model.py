"""A copy of a Warehouse that the policies may step without touching the real one.

copy.deepcopy costs 1.2 ms at `big`, more than ten env steps; this copies only what step()
mutates. It names Warehouse's attributes, so tests/test_model.py pins it against deepcopy:
if the env grows a new mutable field, that test fails before any policy silently shares it.
"""
import copy
import dataclasses
import random


def clone(env, seed=None):
    """A Warehouse in the same state. seed=None keeps the TRUE random stream (the planner's
    clairvoyant level only); any int re-seeds the copy so it cannot see the real future."""
    new = copy.copy(env)                      # shallow: grid, shelves, docks, settings are shared, never mutated
    new.robots = [dataclasses.replace(r) for r in env.robots]
    new.crate_at = dict(env.crate_at)
    new.crate_info = dict(env.crate_info)
    new.queue = list(env.queue)               # Task is frozen
    new.reserved = set(env.reserved)
    new.totals = dict(env.totals)
    new.rng = random.Random(seed)
    if seed is None:
        new.rng.setstate(env.rng.getstate())
    return new
