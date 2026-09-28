"""Watch a policy play the warehouse env in Isaac Sim, with NVIDIA warehouse assets.

    C:\\venvs\\warehouse-isaac\\Scripts\\activate.bat        (then, from the repo root:)
    python -m isaac.demo --maze --max-steps 200
    python -m isaac.demo --checkpoint checkpoints/myalgo                 (replays with its training settings)
    python -m isaac.demo --checkpoint checkpoints/myalgo/ep_000500       (a snapshot from mid-training)

Takes the same env and policy flags as rl.run, so with the same flags both show the same
episode. The policy never learns here. Isaac only reads env.get_state(); all logic stays
in env/ and rl/.
"""
import argparse
import math
import sys
import traceback

from rl.cli import make_parser, parse


def parse_args():
    parser = make_parser(__doc__)
    parser.add_argument("--frames-per-step", type=int, default=20,
                        help="rendered frames per env step; higher = slower glide")
    parser.add_argument("--headless", action="store_true", help="no window (smoke tests)")
    return parser, parse(parser)


# Parsed before Isaac starts: --help stays instant, and SimulationApp forwards any flag left in
# sys.argv to the Kit app, which does not know ours.
PARSER, ARGS = parse_args()
sys.argv = sys.argv[:1]

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": ARGS.headless})

from rl.loop import run_episode  # noqa: E402
from rl.cli import build_or_exit  # noqa: E402


def main():
    import omni.ui as ui
    from isaacsim.core.utils import stage
    from isaacsim.core.utils.viewports import set_camera_view

    from isaac.scene import ROBOT_COLORS, WarehouseScene

    env, policy = build_or_exit(PARSER, ARGS)
    env.reset(ARGS.seed)
    state = env.get_state()
    scene = WarehouseScene(state)
    app.update()
    while stage.is_stage_loading():
        app.update()
    set_camera_view(eye=scene.overview[0], target=scene.overview[1])
    stats = StatsPanel(ui, env.n_robots, ROBOT_COLORS)
    stats.update(state, env.totals)
    playback = Playback(scene, stats, state, env)
    draw(scene, state, state, playback.yaws, playback.yaws, 1.0)

    totals = run_episode(env, policy, ARGS.seed, train=False, on_step=playback.step)
    print(f"episode done: deliveries {totals['deliveries']}  blocked {totals['blocked']}  "
          f"invalid {totals['invalid']}", flush=True)
    while app.is_running() and not ARGS.headless:   # keep the final scene up until the window closes
        app.update()


class Playback:
    """on_step callback for rl.loop: glides every robot from its old to its new cell."""

    def __init__(self, scene, stats, state, env):
        self.scene, self.stats, self.state, self.env = scene, stats, state, env
        self.yaws = [heading_yaw(r["heading"]) for r in state["robots"]]

    def step(self, obs, actions, next_obs, info):
        new_state = self.env.get_state()
        new_yaws = [heading_yaw(r["heading"]) for r in new_state["robots"]]
        for f in range(1, ARGS.frames_per_step + 1):
            draw(self.scene, self.state, new_state, self.yaws, new_yaws, f / ARGS.frames_per_step)
            app.update()
        self.stats.update(new_state, self.env.totals)
        self.state, self.yaws = new_state, new_yaws
        return not app.is_running()   # closing the window ends the episode


def heading_yaw(heading):
    """Yaw that points the forks along the robot's heading (0 up, 1 right, 2 down, 3 left)."""
    from env.warehouse import HEADINGS
    dx, dy = HEADINGS[heading]
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
    humans_xy = []
    for a, b in zip(before["humans"], after["humans"]):
        p = (1 - t) * grid_to_world(a["x"], a["y"]) + t * grid_to_world(b["x"], b["y"])
        humans_xy.append((p[0], p[1]))
    scene.show(robot_xy, robot_yaw, crates, humans_xy)


class StatsPanel:
    """A small window with the running totals and what each robot is doing."""

    def __init__(self, ui, n_robots, colors):
        self.window = ui.Window("Warehouse", width=360, height=90 + 22 * n_robots)
        with self.window.frame:
            with ui.VStack(spacing=4):
                self.summary = ui.Label("")
                self.robots = [ui.Label("") for _ in range(n_robots)]
        self.colors = colors

    def update(self, state, totals):
        self.summary.text = (f"step {state['step']}   deliveries {totals['deliveries']}   value {totals['value']:.1f}   "
                             f"blocked {totals['blocked']}   invalid {totals['invalid']}   queue {state['queue_len']}")
        for i, (label, r) in enumerate(zip(self.robots, state["robots"])):
            target = tuple(r["target"]) if r["target"] else None
            doing = (f"carrying to {target}" if r["carrying"] else f"going to pick up at {target}" if target
                     else "idle")
            prio = f", priority {r['priority']}" if target else ""
            label.text = f"robot {i} ({self.colors[i][0]}, {r['kind']}): {doing}{prio}"


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Isaac's fast shutdown can end the process before an uncaught traceback is printed.
        traceback.print_exc()
        sys.stderr.flush()
    finally:
        app.close()
