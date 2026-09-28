# warehouse-learning

Multi-robot warehouse on a grid, for an RL course project. Robots pick crates from shelves and bring
them to docks (retrieve) or from docks to shelves (store).

**Decentralized navigation, centralized dispatch:** each robot chooses its own moves from its own local
observation. The environment's task queue plays the warehouse management system and assigns tasks.

## Repo layout

| Folder | Owner | What | Dependencies |
|---|---|---|---|
| `env/` | shared contract | grid model: maps, dynamics, observation, reward, `get_state()` | stdlib only |
| `rl/` | algorithm team | policies (`rl/policies/`), their interface (`rl/policy.py`), the episode loop (`rl/loop.py`) and runner (`rl/run.py`) | `env` only, never Isaac |
| `isaac/` | visualization | draws `env.get_state()` in Isaac Sim with NVIDIA warehouse assets | `env`, `rl.policies`, Isaac Sim |
| `tests/` | shared | guards the contract (`test_env.py`), every registered policy (`test_policies.py`) and the baseline (`test_greedy.py`) | stdlib only |

A change to `env/` that breaks a contract table below or `tests/test_env.py` affects both teams. Any change to the
observation, actions or rewards must also bump `CONTRACT_VERSION` in `env/warehouse.py`, so old checkpoints are refused
instead of silently misbehaving.

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
python -m rl.run --help                            # every flag
```

Always run from the repo root with `python -m`, so that `env` and `rl` can be imported.

## Writing an algorithm (guide for the algorithm team)

You only ever touch two places: your own file in `rl/policies/`, and one import plus one line in
`rl/policies/__init__.py`. You never need Isaac Sim, and you never edit `env/` (if you think you need to,
talk to the team first: it changes the contract for everyone).

### 1. Create your file

Copy the template and rename it after your algorithm (the template is a small working tabular Q-learner):

    copy rl\policies\_template.py rl\policies\myalgo.py

Rename the class inside (`MyAlgo`), then pick the shape that matches your algorithm:

| You are writing… | Subclass | Write | Register as |
|---|---|---|---|
| one rule every robot runs, one shared set of weights (parameter sharing) | `RobotPolicy` | `act_one(obs) -> int` | `shared(MyAlgo)` |
| one rule per robot, each robot with its own weights (independent learners) | `RobotPolicy` | `act_one(obs) -> int` | `independent(MyAlgo)` |
| one controller deciding for all robots at once (centralized, or CTDE like MAPPO/QMIX) | `Policy` | `act(obs_list) -> list[int]` | `MyAlgo` |

Both base classes live in `rl/policy.py` (a short file, worth reading once).

What you get:

- `spec` in the constructor `(spec, seed)`: `spec.grid` (the map rows), `spec.n_robots`, `spec.n_actions` (6),
  `spec.obs_size` (14).
- `obs`: one robot's observation with named fields: `obs.x`, `obs.y`, `obs.carrying`, `obs.has_task`,
  `obs.target_x`, `obs.target_y`, and the rays `obs.up_d`, `obs.up_t`, … `obs.right_t` (see Contract below);
  `obs.rays()` gives the four `(distance, type)` pairs. It is also a plain tuple, so `np.array(obs)` or using it as
  a dict key works.
- Actions and ray types: import them, never type the numbers:
  `from env.warehouse import UP, DOWN, LEFT, RIGHT, WAIT, INTERACT, ROBOT`.

Methods (all optional except `act` / `act_one`). For a `RobotPolicy`, `update` is `update_one` and takes one
robot's values instead of lists:

```python
act(obs_list) -> list[int]                                     # or act_one(obs) -> int
update(obs_list, actions, rewards, next_obs_list, done)        # learners: called after every training step
reset()                                                        # called at the start of every episode
save(folder) / load(folder)                                    # learners: write / read your weights
```

- `update` is **only** called with `--train`: never in evaluation or in the Isaac demo. Use `self.training`
  (set by the runner) to switch exploration off when it is `False`, as the template does.
- `done` is always a time-limit cut, never a real end state: keep bootstrapping from `next_obs`.
- In `save`, write plain files into `folder` (JSON for tables, `torch.save(model.state_dict(), ...)` for networks),
  never pickle the whole object. The runner writes `meta.json` next to them for you.
- Libraries: the standard library for now. If you need numpy or torch, add exactly `numpy==1.26.0` or
  `torch==2.7.0` to `requirements.txt` (the versions Isaac pins), because the Isaac demo runs your code too.
  Ask before adding anything else.

### 2. Register it

In `rl/policies/__init__.py`, add your import and one line:

```python
from rl.policies.myalgo import MyAlgo

POLICIES = {
    "greedy": shared(GreedyRule),
    "myalgo": shared(MyAlgo),          # <- yours
}
```

### 3. Test it

    python -m unittest discover -s tests

`tests/test_policies.py` automatically runs every registered policy on a small maze with 3 robots, in training
and evaluation mode, and checks that `save` then `load` restores everything. A failure names your policy.

### 4. Train and evaluate

    python -m rl.run --policy myalgo --train --episodes 5000 --no-render --save checkpoints/myalgo
    python -m rl.run --policy myalgo --checkpoint checkpoints/myalgo --episodes 100 --no-render
    python -m rl.run --policy myalgo --checkpoint checkpoints/myalgo --maze --max-steps 200

The first command trains and saves, the second evaluates without learning, the third shows one episode in the
terminal. Use the same map flags (`--maze`, `--robots`, …) for training and evaluation. Compare against the
baseline with `--policy greedy`. `checkpoints/` is not committed: share a checkpoint by sending the folder.

### 5. Watch it in Isaac Sim (whoever has the Isaac venv)

Put the checkpoint folder in `checkpoints/` of the repo on the Isaac machine, then:

    C:\venvs\warehouse-isaac\Scripts\activate.bat
    python -m isaac.demo --policy myalgo --checkpoint checkpoints/myalgo --maze --max-steps 200

With the same flags it shows exactly the same episode as `rl.run`.

### When things go wrong

| Message | Meaning |
|---|---|
| `unknown policy 'x'` | you skipped step 2, or the name differs from the key in `POLICIES` |
| `need N actions, each an int in 0..5` | your `act` returned the wrong number of actions, or something that is not an action |
| `checkpoint was trained on contract v1, env is v2` | the observation, actions or rewards changed since you trained: retrain |
| `checkpoint is for policy 'a', not 'b'` | `--checkpoint` points at another algorithm's folder |
| `checkpoint has weights for 4 independent robots, the env has 2` | independent learners need the same `--robots` as in training (shared ones don't) |
| `checkpoint ... has no q.json` | the registry entry (`shared`/`independent`) or your `save` changed since that checkpoint was made |
| `ModuleNotFoundError: No module named 'env'` | run from the repo root with `python -m ...` |

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
python -m isaac.demo --maze --max-steps 200
```

It shows the same episode as `python -m rl.run --maze --max-steps 200` (same maze, seed and policy),
drawn with NVIDIA's warehouse assets at real scale: one grid cell is 4 m. Shelves are rack bays, docks are red floor pads, robots
are forklifts with a coloured roof plate, and crates are pallets with boxes. A **Warehouse** window
lists the totals and what each robot is doing. The first run downloads the assets and can take a
while. The window stays open after the episode; close it to exit.

The scene is built from the map string, so any valid layout works. The demo takes the same flags as
`rl.run` (`--policy`, `--checkpoint`, `--seed`, `--robots`, `--maze`, `--random-map`, `--size`, `--max-steps`),
plus `--frames-per-step` (playback speed, default 20) and `--headless`. To watch a trained algorithm, see step 5
of "Writing an algorithm". Nobody needs to edit `isaac/demo.py` to change what it shows.

Isaac is only for watching. Training runs in `.venv` with no rendering, at several thousand env steps
per second.

`isaac/scene.py` holds every asset path and measured size. It is the only file to change when swapping
models.
