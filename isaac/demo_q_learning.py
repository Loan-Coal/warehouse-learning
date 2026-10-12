import math
import os
import sys
import traceback
import numpy as np

# Correction de la configuration SimulationApp pour Isaac Sim 6.1
from isaacsim import SimulationApp

HEADLESS = os.environ.get("DEMO_HEADLESS", "").strip() == "1"

app_config = {
    "headless": HEADLESS,
    "settings": {
        "/app/exts/fold/isaacsim.sensors.rtx": False,
        "/exts/isaacsim.sensors.rtx/enabled": False,
    }
}
app = SimulationApp(app_config)

import omni.ui as ui
from isaacsim.core.utils import stage
from isaacsim.core.utils.viewports import set_camera_view

from env.layout import DEFAULT_MAP, maze_layout
from env.warehouse import Warehouse
from isaac.scene import ROBOT_COLORS, WarehouseScene
from isaac.demo import StatsPanel, draw, heading
from rl.policies.q_learning import QLearningPolicy

SEED = 0
N_ROBOTS = 2
MAX_STEPS = 200
LAYOUT = maze_layout(SEED, n_robots=N_ROBOTS)
FRAMES_PER_STEP = 20

def make_policies(env):
    try:
        q_tables = np.load("q_tables.npy", allow_pickle=True)
        return [QLearningPolicy(q_tables[i]) for i in range(env.n_robots)]
    except Exception:
        return [QLearningPolicy(np.zeros((2000000, 6))) for _ in range(env.n_robots)]

def main():
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

    while app.is_running() and not HEADLESS:
        app.update()

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
    finally:
        app.close()