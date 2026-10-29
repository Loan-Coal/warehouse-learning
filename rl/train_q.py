import numpy as np
from env.warehouse import Warehouse
from env.layout import maze_layout

N_ACTIONS = 6
NB_STATES = 2000000
N_ROBOTS = 2

def get_state(obs):
    """Calcule l'index d'état sur 2 000 000 de combinaisons."""
    if isinstance(obs, dict):
        x, y = obs["position"]
        orientation = obs.get("orientation", 0)
        carrying = 1 if obs["carrying"] else 0
        neighbors = obs.get("neighbors", ["AISLE"] * 4)
    else:
        # Tuple d'observation local : (x, y, carrying, has_task, tx, ty, orientation, neighbors)
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


# Initialisation des tables Q
Q = [np.zeros((NB_STATES, N_ACTIONS)) for _ in range(N_ROBOTS)]

env = Warehouse(layout=maze_layout(42, n_robots=N_ROBOTS), n_robots=N_ROBOTS, max_steps=500)
alpha, gamma, epsilon = 0.1, 0.9, 1.0

print("Début de l'entraînement...")
for episode in range(1000):
    obs = env.reset(seed=42 + episode)
    done = False
    step = 0
    
    while not done and step < 500:
        actions = []
        states = []
        
        for i in range(env.n_robots):
            state = get_state(obs[i])
            states.append(state)
            
            # --- Exploration epsilon-greedy ---
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
            
            # --- Reward Shaping (Coût d'action pour décourager les boucles) ---
            if r == -1.0:
                shaped_reward = -1.0        # Action invalide (collision/mur)
            elif r > 0:
                shaped_reward = r           # Récompense de prise/livraison (+10)
            else:
                if act in [2, 3]:           # TURN_LEFT, TURN_RIGHT
                    shaped_reward = -0.15   # Pénalité plus forte pour les virages
                else:
                    shaped_reward = -0.05   # Coût de déplacement standard

            # Mise à jour de la Q-table
            Q[i][st, act] += alpha * (shaped_reward + gamma * np.max(Q[i][st2]) - Q[i][st, act])
            
        obs = next_obs
        step += 1
        
    # Décroissance progressive d'epsilon
    epsilon = max(0.01, epsilon - 0.002)
    
    if (episode + 1) % 50 == 0:
        print(f"Épisode {episode + 1}/1000 terminé - Epsilon: {epsilon:.2f}")

# Sauvegarde des tables Q
np.save("q_tables.npy", np.array(Q, dtype=object))
print("Entraînement terminé ! Fichier 'q_tables.npy' sauvegardé.")