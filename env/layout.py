"""Warehouse maps: parsing, random generation, validation and BFS. Standard library only.

A map is a multi-line string: '.' aisle (walkable), 'S' shelf, 'D' dock. Shelves and docks
are obstacles; robots interact with them from a 4-adjacent aisle cell.
"""
import random
from collections import deque

AISLE, SHELF, DOCK = ".", "S", "D"

# Index matches the movement actions: 0 up, 1 down, 2 left, 3 right (y grows downward).
DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))

DEFAULT_MAP = """
.......
.S.S.S.
.S.S.S.
.......
.......
D.....D
"""


def parse(text):
    """Map string -> grid, a tuple of equal-length row strings."""
    rows = tuple(line.strip() for line in text.strip().splitlines())
    if not rows or any(len(r) != len(rows[0]) for r in rows):
        raise ValueError("map rows must be non-empty and all the same width")
    bad = set("".join(rows)) - {AISLE, SHELF, DOCK}
    if bad:
        raise ValueError(f"unknown map characters: {sorted(bad)}")
    return rows


def cells_of(grid, kind):
    """All (x, y) cells of one kind, in row-major order so indices are stable."""
    return [(x, y) for y, row in enumerate(grid) for x, c in enumerate(row) if c == kind]


def in_bounds(grid, x, y):
    return 0 <= y < len(grid) and 0 <= x < len(grid[0])


def is_walkable(grid, x, y):
    return in_bounds(grid, x, y) and grid[y][x] == AISLE


def access_cells(grid, cell):
    """Walkable neighbours from which a robot can interact with `cell`."""
    x, y = cell
    return [(x + dx, y + dy) for dx, dy in DIRS if is_walkable(grid, x + dx, y + dy)]


def bfs(grid, start, goals, blocked=frozenset()):
    """Shortest walkable path from start to the nearest goal, excluding start.

    Returns [] if start is already a goal, None if no goal is reachable. Neighbours are
    expanded in DIRS order so the result is deterministic.
    """
    goals = set(goals)
    if start in goals:
        return []
    parent = {start: None}
    frontier = deque([start])
    while frontier:
        cur = frontier.popleft()
        for dx, dy in DIRS:
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt in parent or nxt in blocked or not is_walkable(grid, *nxt):
                continue
            parent[nxt] = cur
            if nxt in goals:
                path = [nxt]
                while parent[path[-1]] != start:
                    path.append(parent[path[-1]])
                return path[::-1]
            frontier.append(nxt)
    return None


def _flood(grid, start):
    """Every walkable cell reachable from start."""
    seen, frontier = {start}, [start]
    while frontier:
        x, y = frontier.pop()
        for dx, dy in DIRS:
            nxt = (x + dx, y + dy)
            if nxt not in seen and is_walkable(grid, *nxt):
                seen.add(nxt)
                frontier.append(nxt)
    return seen


def problems(grid, n_robots):
    """Reasons the map is unusable; an empty list means it is valid."""
    found = []
    aisles = cells_of(grid, AISLE)
    shelves, docks = cells_of(grid, SHELF), cells_of(grid, DOCK)
    if not shelves or not docks:
        found.append("map needs at least one shelf and one dock")
    # Robots need room to start apart and to get out of each other's way.
    if len(aisles) < 2 * n_robots:
        found.append(f"only {len(aisles)} aisle cells for {n_robots} robots")
    if aisles and len(_flood(grid, aisles[0])) != len(aisles):
        found.append("aisle cells are not all connected")
    for cell in shelves + docks:
        if not access_cells(grid, cell):
            found.append(f"{cell} has no walkable neighbour")
    return found


def random_layout(seed, width=None, height=None, n_robots=2):
    """A random warehouse-shaped map string, validated for `n_robots`.

    Shape: an aisle along the top, blocks of shelves (1-2 wide, 2 deep) separated by
    aisles, then an aisle row and a bottom row holding the docks. Blocks are at most 2x2
    so every shelf cell touches an aisle.
    """
    rng = random.Random(seed)
    for _ in range(100):
        w = width or rng.randint(7, 11)
        h = height or rng.randint(6, 9)
        rows = [[AISLE] * w for _ in range(h)]
        y = 1
        while y + 1 <= h - 3:
            x = 1
            while x <= w - 2:
                block_w = min(rng.choice((1, 2)), w - 1 - x)
                for bx in range(x, x + block_w):
                    rows[y][bx] = rows[y + 1][bx] = SHELF
                x += block_w + rng.choice((1, 1, 2))
            y += 3
        for dx in rng.sample(range(w), rng.randint(1, max(1, w // 4))):
            rows[h - 1][dx] = DOCK
        text = "\n".join("".join(r) for r in rows)
        if not problems(parse(text), n_robots):
            return text
    raise ValueError(f"no valid layout found for seed={seed}, width={width}, height={height}")
