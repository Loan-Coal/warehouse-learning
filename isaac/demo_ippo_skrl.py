import math
import os
import sys
import traceback
import torch

from isaacsim import SimulationApp

HEADLESS = os.environ.get("DEMO_HEADLESS", "").strip() == "1"

app_config = {
    "headless": HEADLESS,
    "settings": {
        "/app/exts/fold/isaacsim.sensors.rtx": False,
        "/exts/isaacsim.sensors.rtx/enabled": False,
        "/exts/omni.kit.window.extensions/enabled": False,
        "/exts/omni.kit.browser.core/enabled": False
    }
}

app = SimulationApp(app_config)

import numpy as np
from gymnasium import spaces

from env.layout import DEFAULT_MAP
from env.warehouse import Warehouse
from rl.policies.ippo_skrl import SKRLIPPOPolicy, OBS_DIM, N_ACTIONS

SEED = 0
N_ROBOTS = 2
MAX_STEPS = 200
LAYOUT = DEFAULT_MAP
FRAMES_PER_STEP = 20

obs_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
act_space = spaces.Discrete(N_ACTIONS)


def make_policies(env):
    policies = []
    for i in range(env.n_robots):
        weights_path = f"weights/skrl_ippo_robot_{i}.pt"
        if os.path.exists(weights_path):
            print(f"Chargement du modèle IPPO robot {i} : {weights_path}", flush=True)
            pol = SKRLIPPOPolicy(
                weights_path=weights_path,
                device="cpu",
                observation_space=obs_space,
                action_space=act_space
            )
            policies.append(pol)
        else:
            print(f"Fichier {weights_path} non trouvé. Modèle par défaut.", flush=True)
            pol = SKRLIPPOPolicy(
                device="cpu",
                observation_space=obs_space,
                action_space=act_space
            )
            policies.append(pol)
    return policies


def main():
    from isaacsim.core.utils import stage
    from isaacsim.core.utils.viewports import set_camera_view
    from isaac.scene import WarehouseScene

    env = Warehouse(LAYOUT, n_robots=N_ROBOTS, max_steps=MAX_STEPS)
    obs = env.reset(SEED)
    policies = make_policies(env)
    state = env.get_state()
    scene = WarehouseScene(state)

    app.update()
    while stage.is_stage_loading():
        app.update()

    set_camera_view(eye=scene.overview[0], target=scene.overview[1])

    yaws = [0.0] * env.n_robots
    totals = {"deliveries": 0, "blocked": 0, "invalid": 0}
    draw(scene, state, state, yaws, yaws, 1.0)

    done = False
    while not done and app.is_running():
        actions = [p.act(o) for p, o in zip(policies, obs)]
        next_obs, rewards, done, info = env.step(actions)

        for key in totals:
            totals[key] += info[key]

        new_state = env.get_state()
        new_yaws = [heading(a, b, o, yaw) for a, b, o, yaw in zip(state["robots"], new_state["robots"], obs, yaws)]

        for f in range(1, FRAMES_PER_STEP + 1):
            draw(scene, state, new_state, yaws, new_yaws, f / FRAMES_PER_STEP)
            app.update()

        state, obs, yaws = new_state, next_obs, new_yaws

    print(f"Épisode terminé: livraisons {totals['deliveries']}  bloqués {totals['blocked']}  invalides {totals['invalid']}", flush=True)

    while app.is_running() and not HEADLESS:
        app.update()


def heading(before, after, obs, yaw):
    dx, dy = after["x"] - before["x"], after["y"] - before["y"]
    if (dx, dy) == (0, 0):
        if isinstance(obs, dict):
            x, y = obs["position"]
            has_task = obs.get("carrying", False)
            tx, ty = obs.get("target", (x, y))
        else:
            x, y, _, has_task, tx, ty = obs[:6]
        if not has_task or abs(tx - x) + abs(ty - y) != 1:
            return yaw
        dx, dy = tx - x, ty - y
    return math.atan2(dx, dy)


def draw(scene, before, after, yaws_before, yaws_after, t):
    from isaac.scene import grid_to_world

    robot_xy, robot_yaw = [], []
    for a, b, ya, yb in zip(before["robots"], after["robots"], yaws_before, yaws_after):
        p = (1 - t) * grid_to_world(a["x"], a["y"]) + t * grid_to_world(b["x"], b["y"])
        turn = (yb - ya + math.pi) % (2 * math.pi) - math.pi
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
        elif cid in new:
            crates.append((cid, pose(new[cid])))
        else:
            crates.append((cid, pose(old[cid]) if t < 1.0 else None))
    scene.show(robot_xy, robot_yaw, crates)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
    finally:
        app.close()