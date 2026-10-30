import numpy as np
import math
from env.warehouse import Warehouse
from env.layout import maze_layout

N_ACTIONS = 6
N_ROBOTS = 2

def get_target_dir(x, y, tx, ty):
    """Calcule la direction relative vers la cible (0: sur place, 1: Haut, 2: Droite, 3: Bas, 4: Gauche)."""
    dx, dy = tx - x, ty - y
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) >= abs(dy):
        return 2 if dx > 0 else 4
    else:
        return 3 if dy > 0 else 1

def get_state(obs):
    """Encode l'état en incluant la direction de la cible."""
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

    pos_idx = (y % 20) * 20 + (x % 20)  # 400
    target_dir = get_target_dir(x, y, tx, ty)  # 5
    
    neighbor_map = {"AISLE": 0, "WALL": 1, "SHELF_EMPTY": 2, "SHELF_PACKAGE": 3, "DOCK": 4}
    n_up = neighbor_map.get(neighbors[0], 0) if len(neighbors) > 0 else 0
    n_right = neighbor_map.get(neighbors[1], 0) if len(neighbors) > 1 else 0
    n_down = neighbor_map.get(neighbors[2], 0) if len(neighbors) > 2 else 0
    n_left = neighbor_map.get(neighbors[3], 0) if len(neighbors) > 3 else 0
    neighbors_idx = n_up * 125 + n_right * 25 + n_down * 5 + n_left  # 625

    # Indexation unique
    state_val = (
        (pos_idx * 5 * 4 * 2 * 625) +
        (target_dir * 4 * 2 * 625) +
        (orientation * 2 * 625) +
        (carrying * 625) +
        neighbors_idx
    )
    return state_val % 2000000

# Initialisation des tables Q
NB_STATES = 2000000
Q = [np.zeros((NB_STATES, N_ACTIONS)) for _ in range(N_ROBOTS)]

env = Warehouse(layout=maze_layout(42, n_robots=N_ROBOTS), n_robots=N_ROBOTS, max_steps=500)
alpha, gamma, epsilon = 0.2, 0.95, 1.0

print("Début de l'entraînement Q-Learning / IQL...")
for episode in range(1500):
    obs = env.reset(seed=42 + episode)
    done = False
    step = 0
    
    while not done and step < 500:
        actions = []
        states = []
        
        for i in range(env.n_robots):
            state = get_state(obs[i])
            states.append(state)
            
            if np.random.random() < epsilon:
                act = np.random.choice(N_ACTIONS)
            else:
                act = int(np.argmax(Q[i][state]))
            actions.append(act)
            
        next_obs, raw_rewards, done, info = env.step(actions)
        
        for i in range(env.n_robots):
            st = states[i]
            st2 = get_state(next_obs[i])
            act = actions[i]
            r = raw_rewards[i]
            
            # --- Reward Shaping basé sur la distance à la cible ---
            if isinstance(obs[i], dict):
                x1, y1 = obs[i]["position"]
                tx, ty = obs[i].get("target", (x1, y1))
                x2, y2 = next_obs[i]["position"]
            else:
                x1, y1, _, _, tx, ty = obs[i][:6]
                x2, y2 = next_obs[i][0], next_obs[i][1]
                
            dist_before = abs(tx - x1) + abs(ty - y1)
            dist_after = abs(tx - x2) + abs(ty - y2)
            
            if r == -1.0:
                shaped_reward = -1.0
            elif r > 0:
                shaped_reward = r  # Prise/Livraison récompensée (+10)
            else:
                # Bonus si on s'approche de la cible, malus si on s'en éloigne
                progress = dist_before - dist_after
                shaped_reward = 0.2 * progress - 0.05

            Q[i][st, act] += alpha * (shaped_reward + gamma * np.max(Q[i][st2]) - Q[i][st, act])
            
        obs = next_obs
        step += 1
        
    epsilon = max(0.05, epsilon - 0.001)
    
    if (episode + 1) % 100 == 0:
        print(f"Épisode {episode + 1}/1500 terminé - Epsilon: {epsilon:.2f}")

np.save("q_tables.npy", np.array(Q, dtype=object))
print("Entraînement terminé ! Fichier 'q_tables.npy' sauvegardé.")