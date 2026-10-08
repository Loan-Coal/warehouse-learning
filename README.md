# warehouse-learning

Multi-robot warehouse on a grid, for an RL course project. Forklift robots pick crates from shelves and bring
them to docks (retrieve) or from docks to shelves (store).

**Decentralized navigation, centralized dispatch:** each robot chooses its own moves from its own local
observation. The environment's task queue plays the warehouse management system and assigns jobs.

## The whole thing in one minute

- **`env/` is the board game.** The map, robots, crates and rules. Plain Python; it knows nothing about learning or 3D.
- **`rl/` is the players.** An algorithm (a "policy") looks at what its robot sees and presses a button. Colleagues only ever add files here.
- **`isaac/` is the TV camera.** It asks the board game "where is everything?" and draws it in 3D. It never decides or trains.

The loop: see, press a button, get points, (while training) learn from the points. Repeat for thousands of
episodes without drawing. Save the "brain" to a folder (a checkpoint); replay it later in the terminal or in Isaac Sim.

| Idea | In this project |
|---|---|
| **State** | You choose it in your algorithm file: `OBS = ("position", "heading", "task", "rays")`. See [Observation](#observation). |
| **Actions** | 6: `PICKUP`, `DROPOFF`, `TURN_RIGHT`, `TURN_LEFT`, `FORWARD`, `BACKWARD` (contract v2; no wait: turn in place). |
| **Reward** | +10 per delivery (x value with `--priority`), -0.05 per step, -1 when blocked by a robot or human, -0.1 for a button that does nothing |
| **Episode** | 100 steps by default (`--max-steps`). Episode k uses seed `--seed` + k. |
| **Scenario** | Every complexity factor is off by default. Switch them on with flags, or use `--scenario hard`. |
| **Checkpoint** | A folder with the weights and a `meta.json` holding the algorithm, its `OBS` and every setting it was trained with |

## Repo layout

| Folder | Owner | What | Dependencies |
|---|---|---|---|
| `env/` | shared contract | maps (`layout.py`), dynamics and rewards (`warehouse.py`), observation features (`observation.py`), named scenarios (`scenarios.py`) | stdlib only |
| `rl/` | algorithm team | algorithms (`rl/policies/`), their interface (`policy.py`), flags (`cli.py`), episode loop (`loop.py`), train/eval (`run.py`), benchmark (`bench.py`), charts (`plot.py`), and three shared helpers: what the env will not tell you (`nav.py`), the network inputs (`features.py`), many envs at once (`vecenv.py` + `vecloop.py`) | `env`, torch, matplotlib; never Isaac |
| `isaac/` | visualization | draws `env.get_state()` in Isaac Sim with NVIDIA warehouse assets | `env`, `rl`, Isaac Sim |
| `tests/` | shared | one file per area; `test_env.py` is the contract and `test_policies.py` runs every registered algorithm | |

**Read [LEARNINGS.md](LEARNINGS.md) before you write an algorithm.** It is what was measured rather than assumed:
why the action mask exists, why the reward shaping is not the textbook form, why there are no exploration
schedules, how to measure honestly, and which optimisations did not pay.

Any change to the observation, the actions or the rewards must bump `CONTRACT_VERSION` in `env/warehouse.py`, so old
checkpoints are refused instead of silently misbehaving. A change to `env/` that breaks `tests/test_env.py` affects everyone.

## Setup (Windows, PowerShell, from the repo root)

Needs Python **3.11** for the light venv (3.12 for the Isaac venv): `winget install -e --id Python.Python.3.11`

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```powershell
python -m unittest discover -s tests               # all tests
python -m rl.run                                   # watch one episode in the terminal, greedy, default map
python -m rl.run --maze --seed 3 --max-steps 200   # a random maze of racks (--size 15 13 for bigger)
python -m rl.run --scenario hard                   # every factor on (see Scenarios)
python -m rl.run --policy ppo --checkpoint checkpoints/ppo   # watch a trained policy
python -m rl.run --help                            # every flag
```

Always run from the repo root with `python -m`, so that `env` and `rl` can be imported.

## Scenarios: choosing the difficulty

Every factor is **off unless you switch it on**.

| Flag | What changes |
|---|---|
| `--robots N` | number of robots (1-10) |
| `--maze` / `--random-map`, `--size W H` | a random maze of racks / a random warehouse, from `--seed` |
| `--new-map` | with `--maze` or `--random-map`: a **new map every episode**, so the robots cannot memorise one map |
| `--mixed-crates` | each crate has a weight and a volume (1-3) |
| `--mixed-robots` | robots alternate between *small* (full speed, max weight 1, max volume 2) and *large* (acts every 2nd step, max weight 3, max volume 3). A robot only gets jobs whose crate it can carry. |
| `--priority` | each job has priority 1-3. A delivery is worth `priority x max(0.25, 1 - age/100)`; urgent jobs are handed out first. |
| `--humans N` | N humans walk randomly in the aisles. Robots see them in their rays (type 6) and are blocked by them. |
| `--max-steps N` | episode length |

A **scenario** is a named preset of these flags, in `env/scenarios.py`: `basic`, `mixed`, `priority`, `humans`,
`maze`, `hard`, `big`. Add yours as one line. A flag you type always wins over the scenario
(`--scenario hard --humans 0`). In code the same switches are keyword arguments:
`Warehouse(layout, n_robots=4, mixed_crates=True, humans=2)`.

## Baselines: the numbers to beat

`greedy` (`rl/policies/greedy.py`) runs a fresh BFS every step, so it is a strong navigator and a fair bar. `hard` =
11x9 maze, 4 robots, 200 steps; `big` = 21x21, 6 robots, 400 steps; every factor on. 100 held-out mazes, seeds 90000+:

| policy | scenario | deliveries | blocked | invalid | mean_return |
|---|---|---|---|---|---|
| greedy | hard | 6.23 ± 3.36 | 11.52 | 24.68 | +0.79 |
| greedy | big | 10.61 ± 4.35 | 20.20 | 30.75 | -12.33 |

```powershell
python -m rl.bench greedy --scenarios hard big --episodes 100 --seed 90000 --new-map --out results/greedy
```

How many episodes to report and which metric to lead with: LEARNINGS.md, "Measuring".

### Planner: the ceiling

`planner` is a non-learning space-time planner (`rl/policies/planner.py`): each robot books the cells it will
occupy tick by tick, shortest job first, and everyone re-plans every step, so robots never block each other by
construction. `--policy-opt info=` sets what it may know: `obs` (what PPO sees: every robot's observation, humans only
through rays), `state` (every human's position), `future` (a copy of the env with the real random stream: it sees where
humans will step). Jobs and humans are random, so no optimum exists; the three levels bound what information is worth
versus what learning is worth.

```powershell
python -m rl.bench greedy planner =checkpoints/long_s20000 --scenarios big --episodes 100 --seed 90000 --new-map --out results/ladder_obs
python -m rl.bench planner --policy-opt info=state --scenarios big --episodes 100 --seed 90000 --new-map --out results/ladder_state
python -m rl.bench planner --policy-opt info=future --scenarios big --episodes 100 --seed 90000 --new-map --out results/ladder_future
```

### Search: PPO plus lookahead

`search` wraps a PPO checkpoint and, whenever a robot or human is within two cells, tries each legal action of each
robot in turn in re-seeded copies of the env, rolling the fleet forward `depth` steps with the PPO policy and
bootstrapping with its critic; it keeps the action with the best average return over `k` rollouts, else the PPO
sample. Robots are decided one after another, so the second one sees the first one's real choice. Evaluation only:

```powershell
python -m rl.bench search --policy-opt prior=checkpoints/long_s20000 --scenarios big --episodes 100 --seed 90000 --new-map --out results/search
```

Settings: `prior`, `k` (default 4), `depth` (default 6), `when=near|always`.

## Writing an algorithm (guide for the algorithm team)

You only ever touch two places: your own file in `rl/policies/`, and one import plus one line in
`rl/policies/__init__.py`. You never need Isaac Sim, and you never edit `env/` (if you think you need to,
talk to the team first: it changes the contract for everyone).

### 1. Create your file

Copy the template and rename it after your algorithm. It is a small working tabular Q-learner, also registered as
`template`, so you can try the whole pipeline below before writing any code.

    copy rl\policies\_template.py rl\policies\myalgo.py

Rename the class inside (`MyAlgo`), then pick the shape that matches your algorithm:

| You are writing… | Subclass | Write | Register as |
|---|---|---|---|
| one rule every robot runs, one shared set of weights | `RobotPolicy` | `act_one(obs) -> int` | `shared(MyAlgo)` |
| one rule per robot, each with its own weights (independent learners) | `RobotPolicy` | `act_one(obs) -> int` | `independent(MyAlgo)` |
| one controller deciding for all robots at once (centralized, or CTDE like MAPPO/QMIX) | `Policy` | `act(obs_list) -> list[int]` | `MyAlgo` |

Both base classes live in `rl/policy.py` (a short file, worth reading once).

### 2. Choose the state: `OBS`

```python
class MyAlgo(RobotPolicy):
    OBS = ("position", "heading", "task", "rays", "crate")
```

The env then gives each robot a plain tuple of ints with exactly those named fields, in that order (`obs.x`,
`obs.front_t`, `obs.priority`, …), so `np.array(obs)` and dict keys work. `spec.obs_size` is its length. The groups
are listed in [Observation](#observation). The checkpoint remembers your `OBS`: if you change it, retrain. To add a
group, add it to `FEATURES` in `env/observation.py` (talk to the team, it is shared).

### 3. The actions

Import them, never type the numbers: `from env.warehouse import PICKUP, DROPOFF, TURN_RIGHT, TURN_LEFT, FORWARD, BACKWARD`.
`spec.n_actions` is 6. See [Actions](#actions).

### 4. Methods

All are optional except `act` / `act_one`. For a `RobotPolicy`, `update` is `update_one` and takes one robot's values:

```python
act(obs_list) -> list[int]                                     # or act_one(obs) -> int
update(obs_list, actions, rewards, next_obs_list, done)        # learners: called after every training step
reset()                                                        # called at the start of every episode
configure(**opts)                                              # your --policy-opt KEY=VALUE settings
stats() -> dict                                                # extra columns for train_log.csv
save(folder) / load(folder)                                    # learners: write / read your weights

act_vec(obs_per_env) -> list[list[int]]                        # optional: several envs in one go
update_vec(obs, actions, rewards, next_obs, dones)             # optional: the same, for learning
```

- `update` is **only** called with `--train`: never in evaluation, in the benchmark or in Isaac. Use `self.training`
  to switch exploration off when it is `False`, as the template does.
- `done` is always a time-limit cut, never a real end state: keep bootstrapping from `next_obs`.
- `self.spec.grid` is the current map (refreshed before every episode, which matters with `--new-map`).
- `configure` receives **strings** (`--policy-opt bfs=0` arrives as `"0"`); `rl.policy.flag` turns one into a bool.
  It is called once, before any checkpoint is loaded, and a checkpoint remembers what it was trained with.
- `stats()` keys become extra columns of `train_log.csv`, in the order your first row gives them. Never return a
  training-schedule value you would not want read as if it applied to an evaluation episode.
- Training always goes through the vectorised loop (`--envs 1` is one slot). The default `act_vec` / `update_vec`
  call `act` / `update` once per env, so `--envs N` works on your policy untouched, just no faster. Override them to
  put every robot of every env through one forward pass; `rl/policies/ppo.py` is the worked example. The envs share
  one map per round, which is why one `self.spec` is enough (`rl/vecloop.py`).
- Two things the env deliberately does not tell you live in `rl/nav.py`: **which actions are legal**
  (`legal_actions`: an impossible action is silently ignored and fined, and `PICKUP`/`DROPOFF` need you to *face* the
  target) and **how far your target is** (`distance_field`). `rl/features.py` turns an observation into the network
  row PPO uses.
- In `save`, write plain files into `folder` (JSON for tables, `torch.save(model.state_dict(), ...)` for networks),
  never pickle the whole object. The runner writes `meta.json` next to them.
- Libraries: if you need numpy or torch, add exactly the versions the Isaac venv pins to `requirements.txt`, because
  the Isaac replay runs your code too. Ask before adding anything else.

### 5. Register it

In `rl/policies/__init__.py`, add your import and one line:

```python
from rl.policies.myalgo import MyAlgo

POLICIES = {
    "greedy": shared(GreedyRule),
    "myalgo": shared(MyAlgo),          # <- yours
}
```

### Model access rule

A colleague algorithm sees only the list of observations. The loop also calls `see_env(env)` once per episode;
colleague code must not override it. Two registered policies do: `planner` (a benchmark, allowed to cheat as
documented) and `search` (may copy the env with `rl.model.clone` but must re-seed the copy, so it samples futures
rather than reading the true one).

### 6. Test it

    python -m unittest discover -s tests

`tests/test_policies.py` runs every registered algorithm with every factor on, in training and evaluation mode, and
checks that `save` then `load` restores everything. A failure names your algorithm.

### 7. Train

    python -m rl.run --policy myalgo --scenario hard --maze --new-map --train --episodes 20000 --no-render \
        --print-every 500 --eval-every 500 --eval-episodes 20 --save checkpoints/myalgo --snapshot-every 2000

This writes into `checkpoints/myalgo/`: the final checkpoint, a snapshot `ep_002000/`, … every 2000 episodes (replay
how the robots played *during* training), and `train_log.csv` (one row per episode: `deliveries, value, blocked,
invalid, mean_return`, a `mode` column saying train or eval, then whatever your `stats()` returned).

**Always train with held-out evaluation** (`--eval-every`): every N episodes it plays `--eval-episodes` on maps from
`--eval-seed` (90000 by default) with learning and exploration off. `rl.run` refuses to start if the training maps
could reach those seeds. A training-mode curve is not a result (LEARNINGS.md, "The failure that started it").

`--eval-scenario`, `--eval-robots` and `--eval-size` change only the *evaluation* env during that run (train on 4
robots, evaluate on 6): the curve that shows whether a policy generalises rather than memorises.

Drawing the learning curve (`checkpoints/myalgo/learning_curve.png`; several folders compare runs):

    python -m rl.plot checkpoints/myalgo

**Faster:** `--envs 16` steps 16 warehouses together so one network call serves them all, and `--max-minutes M`
also stops the run after M wall-clock minutes (evaluation included). Whichever of `--episodes` and `--max-minutes`
comes first stops the run, so set `--episodes` high; with `--eval-every` it also bounds the training map seeds, which
`rl.run` checks against `--eval-seed`. The checkpoint records the episodes actually played. On `big`, PPO plateaus by
~1.6M steps (15.95 deliveries over 100 held-out mazes against 16.28 at 8M). It only pays if your policy overrides
`act_vec`. `--envs` cannot be rendered; `--train` without `--no-render` shows the training episodes in the terminal
(one env).

### 8. Evaluate and watch

A checkpoint remembers its algorithm and every setting it was trained with, so the folder alone is enough:

    python -m rl.run --checkpoint checkpoints/myalgo --episodes 100 --no-render     # evaluate, no learning
    python -m rl.run --checkpoint checkpoints/myalgo/ep_002000                      # watch a mid-training snapshot
    python -m rl.run --checkpoint checkpoints/myalgo --scenario hard                # same brain, harder warehouse

To use a colleague's model, put their folder in `checkpoints/` (it is not committed) and pull their algorithm file,
registered under the same name.

### 9. Benchmark

    python -m rl.bench greedy myalgo=checkpoints/myalgo =checkpoints/lea_ppo --scenarios basic mixed hard --episodes 50

Each entry is an algorithm name (no training needed, like `greedy`) or `name=CHECKPOINT`; leave the name empty
(`=CHECKPOINT`) to take it from the checkpoint. Every entry plays the same evaluation seeds on every scenario,
without learning. It prints a table (mean ± std per episode) and writes `results/bench.csv` (one row per episode).
`value` is the priority-weighted number of deliveries.

## Contract

**Map.** A multi-line string: `.` aisle, `S` shelf, `D` dock. Only aisles are walkable. The default is 7x6.
`random_layout(seed)` makes validated, warehouse-shaped maps. `maze_layout(seed, width=11, height=9, loops=0.25, docks=2)`
makes a maze of one-cell-wide aisles walled by racks, with some loops so robots can route around each other, and docks
spread along the bottom row (width and height must be odd).

### Actions

Each robot has a **heading**: 0 up, 1 right, 2 down, 3 left (y grows downward). It starts with a random one.

| # | Action | Effect | Invalid (-0.1) when |
|---|---|---|---|
| 0 | `PICKUP` | take the crate of your job from the cell **in front** of you | no job, already carrying, or not facing the job's pickup cell |
| 1 | `DROPOFF` | put the crate into the cell in front of you: delivery | not carrying, or not facing the job's dropoff cell |
| 2 | `TURN_RIGHT` | heading +90° (clockwise), no move | never |
| 3 | `TURN_LEFT` | heading -90°, no move | never |
| 4 | `FORWARD` | one cell along the heading | the cell is a wall, shelf or dock |
| 5 | `BACKWARD` | one cell against the heading, keeping the heading | same |

A slow robot (`--mixed-robots`) acts only every second step; its action in between is ignored, not punished.

### Observation

Groups you can list in `OBS` (`env/observation.py`); the default is `("position", "heading", "task", "rays")`:

| Group | Fields | Meaning |
|---|---|---|
| `position` | `x, y` | own cell (x right, y down) |
| `heading` | `heading` | 0 up, 1 right, 2 down, 3 left |
| `task` | `carrying, has_task, target_x, target_y` | holding a crate?; has a job?; pickup cell if not carrying, else dropoff cell (own cell if idle) |
| `rays` | `front_d, front_t, right_d, right_t, back_d, back_t, left_d, left_t` | **relative to the heading**: distance (1-5) and type of the first thing seen, `0, 0` if nothing. Types: 1 wall, 2 empty shelf, 3 shelf with crate, 4 dock, 5 robot, 6 human |
| `crate` | `crate_weight, crate_volume, priority, task_age` | the current job; `task_age` = steps since the job appeared; all 0 if idle |
| `robot` | `ready, period, max_weight, max_volume` | own abilities; `ready` = 1 if this robot's next action counts |
| `local_map` | `m0 … m24` | the 5x5 cells around the robot, row by row, map-aligned (`m12` is the robot), same type codes as rays (0 = aisle) |

Robots know the building (`spec.grid`) but see other robots and humans only through their observation.

**Reward per robot:** own = `+10 x value` per delivery (value = 1, or the priority value with `--priority`), -0.05 per
step, -1 if blocked by a robot or a human, -0.1 for an invalid action. Then
`reward = (1 - alpha) x own + alpha x mean(own over robots)`, with `alpha = 0.5` (0 = selfish, 1 = fully shared).

**Collisions:** a move into a robot that stays put, or a swap, is cancelled. When several robots move into the same
free cell, one seeded-random winner moves. Every robot whose move is cancelled is "blocked". Humans move after the
robots, into a random free neighbouring aisle, with probability 0.5 per step.

**Jobs:** a queue. Each step, with probability 0.3, a feasible job is added (up to one queued job per robot). Cells are
reserved so that two jobs never use the same shelf. Each idle robot takes the highest-priority queued job whose crate
it can carry (oldest first). One job at a time per robot.

**`step()` info:** `deliveries`, `value`, `blocked`, `invalid` (this step) and `truncated` (True on the last step).

**`get_state()`** (the only thing the viewer reads): `width, height, grid, robots [{x, y, heading, carrying, kind,
target, priority}], crates [{id, x, y, carried_by, weight, volume}], humans [{x, y}], queue_len, step`.

## Isaac Sim viewer (optional, ~20 GB)

Uses a separate venv, so the algorithm side stays light. **It must live at a short path outside the repo.** Isaac's
own files reach ~200 characters inside the venv, and on Windows a long prefix makes its DLLs fail to load ("The
filename or extension is too long"), even with long paths enabled.

```powershell
py -3.12 -m venv C:\venvs\warehouse-isaac
C:\venvs\warehouse-isaac\Scripts\python.exe -m pip install --upgrade pip
C:\venvs\warehouse-isaac\Scripts\python.exe -m pip install "isaacsim[all,extscache]==6.1.0.0" --extra-index-url https://pypi.nvidia.com
C:\venvs\warehouse-isaac\Scripts\activate.bat      # cmd;  PowerShell: C:\venvs\warehouse-isaac\Scripts\Activate.ps1
isaacsim        # first launch: accept the EULA and wait for the GUI (shader compile), then close it
```

That venv ends up with its own `numpy` (2.3.1), `torch` (2.11.0) and `matplotlib` (3.10.8), newer than the light
`.venv` pins in `requirements.txt` (torch 2.7.0, matplotlib 3.10.3). Checkpoints are plain `state_dict` files, so they
load in both, but any dependency added to `requirements.txt` must also install in the Isaac venv, because the replay
runs the same algorithms.

### Replaying in 3D

Training never happens in Isaac. It replays a checkpoint (or a mid-training snapshot) with its training settings:

```
C:\venvs\warehouse-isaac\Scripts\activate.bat
cd <repo root>
python -m isaac.demo --maze --max-steps 200                          # greedy
python -m isaac.demo --checkpoint checkpoints/myalgo                 # a trained model
python -m isaac.demo --checkpoint checkpoints/myalgo/ep_002000       # how it played after 2000 episodes
```

It shows the same episode as `python -m rl.run` with the same flags, and takes every flag of `rl.run`'s common set,
plus `--frames-per-step` (playback speed, default 20) and `--headless`. One grid cell is 4 m: shelves are rack bays,
docks red floor pads, robots forklifts with a coloured roof plate, crates pallets with boxes, humans yellow capsules.
A **Warehouse** window lists the totals and what each robot is doing. The first run downloads the assets. The window
stays open after the episode; close it to exit. `isaac/scene.py` holds every asset path and measured size: it is the
only file to change when swapping models.
