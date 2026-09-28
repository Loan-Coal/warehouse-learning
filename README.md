# warehouse-learning

Multi-robot warehouse on a grid, for an RL course project. Robots pick crates from shelves and bring
them to docks (retrieve) or from docks to shelves (store).

**Decentralized navigation, centralized dispatch:** each robot chooses its own moves from its own local
observation. The environment's task queue plays the warehouse management system and assigns tasks.

## Repo layout

| Folder | Owner | What | Dependencies |
|---|---|---|---|
| `env/` | shared contract | grid model: maps, dynamics, observation, reward, `get_state()` | stdlib only |
| `rl/` | algorithm team | policies (`rl/policies/`) and the runner / training loop (`rl/run.py`) | `env` only, never Isaac |
| `isaac/` | visualization | draws `env.get_state()` in Isaac Sim with NVIDIA warehouse assets | `env`, `rl.policies`, Isaac Sim |
| `tests/` | shared | guards the contract (`test_env.py`) and the baseline (`test_greedy.py`) | stdlib only |

A change to `env/` that breaks a contract table below or `tests/test_env.py` affects both teams.

## Setup (Windows, PowerShell, from the repo root)

Needs Python **3.11** (Isaac Sim 5.1 only ships for 3.11): `winget install -e --id Python.Python.3.11`

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```powershell
python -m unittest discover -s tests               # all tests
python -m rl.run                                   # one rendered episode on the default map
python -m rl.run --random-map --seed 3             # a random warehouse
python -m rl.run --maze --seed 3 --max-steps 200   # a random maze of racks (--size 15 13 for bigger)
python -m rl.run --robots 4 --episodes 100 --no-render
```

Always run from the repo root with `python -m`, so that `env` and `rl` can be imported.

## Adding an algorithm

Put a class in `rl/policies/` with this interface, then change `make_policies` in `rl/run.py`:

```python
act(obs) -> int                                  # required
update(obs, action, reward, next_obs, done)      # optional: called after every step
reset()                                          # optional: called at every episode start
```

`make_policies` returns one policy per robot. Putting the same object in every slot gives parameter
sharing; separate objects give independent learners.

`done` is always a time-limit truncation: there is no terminal state, so keep bootstrapping from
`next_obs`.

## Contract

**Map.** A multi-line string: `.` aisle, `S` shelf, `D` dock. Only aisles are walkable. A robot interacts
with a shelf or dock from a 4-adjacent aisle cell. The default is 7x6. `random_layout(seed)` makes
validated, warehouse-shaped maps. `maze_layout(seed, width=11, height=9, loops=0.25, docks=2)` makes a maze of
one-cell-wide aisles walled by racks, with some loops so robots can route around each other, and docks
spread along the bottom row (width and height must be odd).

**Actions:** `0` up, `1` down, `2` left, `3` right, `4` wait, `5` interact (pick up at the pickup cell,
drop off at the dropoff cell).

**Observation** (one flat tuple per robot):

| Field | Meaning |
|---|---|
| `x, y` | own cell (x right, y down) |
| `carrying` | 1 if holding a crate |
| `has_task` | 1 if a task is assigned |
| `target_x, target_y` | pickup cell if not carrying, else dropoff cell; own cell if idle |
| `up_d, up_t, down_d, down_t, left_d, left_t, right_d, right_t` | for each direction, distance (1 to 5) and type of the first thing seen, or `0, 0` if nothing is in range. Types: 1 wall, 2 empty shelf, 3 shelf with crate, 4 dock, 5 robot |

**Reward per robot:** own = +10 per delivery, -0.05 per step, -1 if blocked by a robot, -0.1 for an
invalid action (moving into a wall, shelf or dock, or interacting in the wrong place). Then
`reward = (1 - alpha) * own + alpha * mean(own over robots)`, with `alpha = 0.5` by default (0 = selfish,
1 = fully shared).

**Collisions:** a move into a robot that stays put, or a swap, is cancelled. When several robots move into
the same free cell, one seeded-random winner moves. Every robot whose move is cancelled is "blocked".

**Tasks:** a FIFO queue. Each step, with probability 0.3, a feasible task is added (up to one queued task
per robot). Cells are reserved so that two tasks never use the same shelf. The first idle robot takes the
head of the queue.

**`step()` info:** `deliveries`, `blocked`, `invalid` (counts for this step) and `truncated` (True on the
last step).

**`get_state()`** (the only thing the viewer reads): `width, height, grid, robots [{x, y, carrying}],
crates [{id, x, y, carried_by}], queue_len, step`.

## Isaac Sim viewer (optional, ~20 GB)

Uses a separate venv, so the algorithm side stays light. **It must live at a short path outside the
repo.** Isaac's own files reach ~200 characters inside the venv, and on Windows a long prefix makes its
DLLs fail to load ("The filename or extension is too long"), even with long paths enabled.

```powershell
py -3.11 -m venv C:\venvs\warehouse-isaac
C:\venvs\warehouse-isaac\Scripts\python.exe -m pip install --upgrade pip
C:\venvs\warehouse-isaac\Scripts\python.exe -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
C:\venvs\warehouse-isaac\Scripts\activate.bat      # cmd;  PowerShell: C:\venvs\warehouse-isaac\Scripts\Activate.ps1
isaacsim        # first launch: accept the EULA and wait for the GUI (shader compile), then close it
```

Then run Isaac scripts from the repo root with that venv active (`python -m isaac.<script>`).

Isaac pins `numpy 1.26.0` and `torch 2.7.0`. Any algorithm dependency added to `requirements.txt`
must be compatible, because the demo runs the same policies.

### Running the 3D demo

```
C:\venvs\warehouse-isaac\Scripts\activate.bat
cd <repo root>
python -m isaac.demo
```

It shows the same episode as `python -m rl.run --maze --max-steps 200` (same maze, seed and policy),
drawn with NVIDIA's warehouse assets at real scale: one grid cell is 4 m. Shelves are rack bays, docks are red floor pads, robots
are forklifts with a coloured roof plate, and crates are pallets with boxes. A **Warehouse** window
lists the totals and what each robot is doing. The first run downloads the assets and can take a
while. The window stays open after the episode; close it to exit.

The scene is built from the map string, so any valid layout works. To change what it shows, edit the
block at the top of `isaac/demo.py`: `SEED`, `N_ROBOTS`, `MAX_STEPS`, `LAYOUT` (`maze_layout(...)`,
`random_layout(...)` or `DEFAULT_MAP`), `FRAMES_PER_STEP` (the playback speed) and `make_policies`
(the algorithm).

Isaac is only for watching. Training runs in `.venv` with no rendering, at several thousand env steps
per second.

`isaac/scene.py` holds every asset path and measured size. It is the only file to change when swapping
models.
