import numpy as np

NB_STATES = 2000000

def get_state(obs):
    """Même encodage d'état universel que pendant l'entraînement."""
    if isinstance(obs, dict):
        x, y = obs["position"]
        orientation = obs.get("orientation", 0)
        carrying = 1 if obs["carrying"] else 0
        neighbors = obs.get("neighbors", ["AISLE"] * 4)
    else:
        x, y = int(obs[0]), int(obs[1])
        carrying = 1 if obs[2] else 0
        orientation = int(obs[6]) if len(obs) > 6 and isinstance(obs[6], (int, float)) else 0
        
        if len(obs) > 7 and isinstance(obs[7], (list, tuple)):
            neighbors = obs[7]
        else:
            neighbors = ["AISLE"] * 4

    pos_idx = (y % 20) * 20 + (x % 20)

    neighbor_map = {"AISLE": 0, "WALL": 1, "SHELF_EMPTY": 2, "SHELF_PACKAGE": 3, "DOCK": 4}
    n_up = neighbor_map.get(neighbors[0], 0) if len(neighbors) > 0 else 0
    n_right = neighbor_map.get(neighbors[1], 0) if len(neighbors) > 1 else 0
    n_down = neighbor_map.get(neighbors[2], 0) if len(neighbors) > 2 else 0
    n_left = neighbor_map.get(neighbors[3], 0) if len(neighbors) > 3 else 0

    neighbors_idx = n_up * 125 + n_right * 25 + n_down * 5 + n_left

    state_val = (
        (pos_idx * 4 * 2 * 625) +
        (orientation * 2 * 625) +
        (carrying * 625) +
        neighbors_idx
    )
    return state_val % NB_STATES


class QLearningPolicy:
    def __init__(self, q_table):
        self.q_table = q_table

    def get_state_index(self, obs):
        return get_state(obs)

    def act(self, obs) -> int:
        """Exploitation pure (100% déterministe)."""
        state = self.get_state_index(obs)
        if state >= len(self.q_table):
            return 0
        return int(np.argmax(self.q_table[state]))

    def reset(self):
        pass