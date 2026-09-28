"""Watch the warehouse env + a policy in Isaac Sim, with NVIDIA warehouse assets.

    C:\\venvs\\warehouse-isaac\\Scripts\\activate.bat        (then, from the repo root:)
    python -m isaac.demo

Same env, seed and policy as `python -m rl.run --maze --max-steps 200`, so both show the same
episode. Isaac only
reads env.get_state() and the observations; all logic stays in env/ and rl/.
"""
import math
import os
import sys
import traceback

from isaacsim import SimulationApp

HEADLESS = os.environ.get("DEMO_HEADLESS", "").strip() == "1"   # smoke tests without a window
app = SimulationApp({"headless": HEADLESS})

from env.layout import DEFAULT_MAP, maze_layout, random_layout  # noqa: E402
from env.warehouse import Warehouse  # noqa: E402
from rl.policies.greedy import GreedyPolicy  # noqa: E402

# ------------------------------------------------------------- what to show (edit here)
SEED = 0
N_ROBOTS = 2
MAX_STEPS = 200
LAYOUT = maze_layout(SEED, n_robots=N_ROBOTS)   # or DEFAULT_MAP, random_layout(SEED, n_robots=N_ROBOTS),
                                                # maze_layout(SEED, width=15, height=13, n_robots=N_ROBOTS)
FRAMES_PER_STEP = 20                      # rendered frames per env step; higher = slower glide


def make_policies(env):
    # The one line to change when swapping algorithms (same as make_policies in rl/run.py).
    return [GreedyPolicy(env.grid, seed=SEED)] * env.n_robots
# -------------------------------------------------------------------------------------


def main():
    import omni.ui as ui
    from isaacsim.core.utils import stage
    from isaacsim.core.utils.viewports import set_camera_view

    from isaac.scene import ROBOT_COLORS, WarehouseScene

    env = Warehouse(LAYOUT, n_robots=N_ROBOTS, max_steps=MAX_STEPS)
    obs = env.reset(SEED)
    policies = make_policies(env)
    state = env.get_state()
    scene = WarehouseScene(state)
    app.update()
    while stage.is_stage_loading():
        app.update()
    set_camera_view(eye=scene.overview[0], target=scene.overview[1])
    stats = StatsPanel(ui, env.n_robots, ROBOT_COLORS)

    # Yaw 0 leaves the forks (model -y) facing down the map; afterwards robots face where
    # they move or work.
    yaws = [0.0] * env.n_robots
    totals = {"deliveries": 0, "blocked": 0, "invalid": 0}
    draw(scene, state, state, yaws, yaws, 1.0)
    stats.update(state, obs, totals)
    done = False
    while not done and app.is_running():
        actions = [p.act(o) for p, o in zip(policies, obs)]
        next_obs, rewards, done, info = env.step(actions)
        for i, p in enumerate(policies):
            if hasattr(p, "update"):
                p.update(obs[i], actions[i], rewards[i], next_obs[i], done)
        for key in totals:
            totals[key] += info[key]
        new_state = env.get_state()
        new_yaws = [heading(a, b, o, yaw) for a, b, o, yaw in
                    zip(state["robots"], new_state["robots"], obs, yaws)]
        for f in range(1, FRAMES_PER_STEP + 1):
            draw(scene, state, new_state, yaws, new_yaws, f / FRAMES_PER_STEP)
            app.update()
        stats.update(new_state, next_obs, totals)
        state, obs, yaws = new_state, next_obs, new_yaws

    print(f"episode done: deliveries {totals['deliveries']}  blocked {totals['blocked']}  "
          f"invalid {totals['invalid']}", flush=True)
    while app.is_running() and not HEADLESS:   # keep the final scene up until the window closes
        app.update()


def heading(before, after, obs, yaw):
    """Yaw that points the forks where the robot moved or, if it stayed, at its target."""
    dx, dy = after["x"] - before["x"], after["y"] - before["y"]
    if (dx, dy) == (0, 0):
        x, y, _carrying, has_task, tx, ty = obs[:6]
        if not has_task or abs(tx - x) + abs(ty - y) != 1:
            return yaw
        dx, dy = tx - x, ty - y
    # Grid y points down the screen, world Y up; forks are on the model's -y side.
    return math.atan2(dx, dy)


def draw(scene, before, after, yaws_before, yaws_after, t):
    """Blend two env states: robots glide and turn, crates follow shelves, docks and forks."""
    from isaac.scene import grid_to_world

    robot_xy, robot_yaw = [], []
    for a, b, ya, yb in zip(before["robots"], after["robots"], yaws_before, yaws_after):
        p = (1 - t) * grid_to_world(a["x"], a["y"]) + t * grid_to_world(b["x"], b["y"])
        turn = (yb - ya + math.pi) % (2 * math.pi) - math.pi   # shortest way round
        robot_xy.append((p[0], p[1]))
        robot_yaw.append(ya + t * turn)

    def pose(crate):
        if crate["carried_by"] is None:
            return scene.resting_position(crate["x"], crate["y"]), scene.resting_yaw(crate["x"], crate["y"])
        i = crate["carried_by"]
        return scene.carried_position(robot_xy[i], robot_yaw[i]), robot_yaw[i]

    old = {c["id"]: c for c in before["crates"]}
    new = {c["id"]: c for c in after["crates"]}
    crates = []
    for cid in old.keys() | new.keys():
        if cid in old and cid in new:
            (p0, r0), (p1, r1) = pose(old[cid]), pose(new[cid])
            crates.append((cid, ((1 - t) * p0 + t * p1, r1 if new[cid]["carried_by"] is not None else r0)))
        elif cid in new:        # a store crate that just arrived on its dock
            crates.append((cid, pose(new[cid])))
        else:                   # shipped: ride to the dock this step, then disappear
            crates.append((cid, pose(old[cid]) if t < 1.0 else None))
    scene.show(robot_xy, robot_yaw, crates)


class StatsPanel:
    """A small window with the running totals and what each robot is doing."""

    def __init__(self, ui, n_robots, colors):
        self.window = ui.Window("Warehouse", width=360, height=90 + 22 * n_robots)
        with self.window.frame:
            with ui.VStack(spacing=4):
                self.summary = ui.Label("")
                self.robots = [ui.Label("") for _ in range(n_robots)]
        self.colors = colors

    def update(self, state, obs, totals):
        self.summary.text = (f"step {state['step']}   deliveries {totals['deliveries']}   "
                             f"blocked {totals['blocked']}   invalid {totals['invalid']}   queue {state['queue_len']}")
        for i, (label, o) in enumerate(zip(self.robots, obs)):
            x, y, carrying, has_task, tx, ty = o[:6]
            doing = f"carrying to {tx, ty}" if carrying else f"going to pick up at {tx, ty}" if has_task else "idle"
            label.text = f"robot {i} ({self.colors[i][0]}): {doing}"


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Isaac's fast shutdown can end the process before an uncaught traceback is printed.
        traceback.print_exc()
        sys.stderr.flush()
    finally:
        app.close()
