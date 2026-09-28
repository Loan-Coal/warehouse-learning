"""Named scenarios: a preset of the run flags, so `--scenario hard` replaces a row of flags.

Keys are the flag names of rl/cli.py with underscores (mixed_crates, humans, robots, ...);
the map is "map": "maze" or "random".
Flags you type yourself still win over the scenario. Add yours as one line.
"""
SCENARIOS = {
    "basic":    {},
    "mixed":    {"mixed_crates": True, "mixed_robots": True},
    "priority": {"priority": True},
    "humans":   {"humans": 2},
    "maze":     {"map": "maze", "max_steps": 200},
    "hard":     {"map": "maze", "max_steps": 200, "robots": 4, "mixed_crates": True,
                 "mixed_robots": True, "priority": True, "humans": 2},
    "big":      {"map": "maze", "size": (21, 21), "max_steps": 400, "robots": 6, "mixed_crates": True,
                 "mixed_robots": True, "priority": True, "humans": 3},
}
