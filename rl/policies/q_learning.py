import numpy as np

NB_STATES = 2000000

def get_target_dir(x, y, tx, ty):
    dx, dy = tx - x, ty - y
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) >= abs(dy):
        return 2 if dx > 0 else 4
    else:
        return 3 if dy > 0 else 1

def get_state(obs):
    if isinstance(obs, dict):
        x, y = obs["position"]
        orientation = obs.get("orientation", 0)
        carrying = 1 if obs.get("carrying", False) else 0
        tx, ty = obs.get("target", (x, y))
        neighbors = obs.get("neighbors", ["AISLE"] * 4)
    else:
        x, y = int(obs[0]), int(obs[1])
        carrying = 1 if obs[2] else 0
        tx, ty = int(obs[4]), int(obs[5])
        orientation = int(obs[6]) if len(obs) > 6 and isinstance(obs[6], (int, float)) else 0
        neighbors = obs[7] if len(obs) > 7 and isinstance(obs[7], (list, tuple)) else ["AISLE"] * 4

    pos_idx = (y % 20) * 20 + (x % 20)
    target_dir = get_target_dir(x, y, tx, ty)
    
    neighbor_map = {"AISLE": 0, "WALL": 1, "SHELF_EMPTY": 2, "SHELF_PACKAGE": 3, "DOCK": 4}
    n_up = neighbor_map.get(neighbors[0], 0) if len(neighbors) > 0 else 0
    n_right = neighbor_map.get(neighbors[1], 0) if len(neighbors) > 1 else 0
    n_down = neighbor_map.get(neighbors[2], 0) if len(neighbors) > 2 else 0
    n_left = neighbor_map.get(neighbors[3], 0) if len(neighbors) > 3 else 0
    neighbors_idx = n_up * 125 + n_right * 25 + n_down * 5 + n_left

    state_val = (
        (pos_idx * 5 * 4 * 2 * 625) +
        (target_dir * 4 * 2 * 625) +
        (orientation * 2 * 625) +
        (carrying * 625) +
        neighbors_idx
    )
    return state_val % NB_STATES


class QLearningPolicy:
    def __init__(self, q_table):
        self.q_table = q_table

    def act(self, obs) -> int:
        state = get_state(obs)
        if state >= len(self.q_table):
            return 0
        q_vals = self.q_table[state]
        # Si l'état n'a jamais été entraîné (toutes les Q-valeurs sont à 0)
        if np.all(q_vals == 0):
            return 0  # Avancer par défaut
        return int(np.argmax(q_vals))

    def reset(self):
        pass