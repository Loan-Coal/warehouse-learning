import numpy as np
from env.warehouse import Warehouse
from env.layout import DEFAULT_MAP

N_ACTIONS = 6
NB_STATES = 2000000
N_ROBOTS = 2

# Initialisation des tables Q pour chaque robot
Q = [np.zeros((NB_STATES, N_ACTIONS)) for _ in range(N_ROBOTS)]

env = Warehouse(layout=DEFAULT_MAP, n_robots=N_ROBOTS, max_steps=500)
alpha, gamma, epsilon = 0.1, 0.9, 1.0

print("Début de l'entraînement...")
for episode in range(100):
    obs = env.reset(seed=42)
    done = False
    step = 0
    
    while not done and step < 500:
        actions = []
        for i in range(env.n_robots):
            x, y = obs[i][0], obs[i][1]
            state = (((y % 20) * 20 + (x % 20)) * 8 + int(obs[i][2]) * 4) % NB_STATES
            
            if np.random.random() < epsilon:
                act = np.random.choice(N_ACTIONS)
            else:
                act = int(np.argmax(Q[i][state]))
            actions.append(act)
            
        next_obs, rewards, done, info = env.step(actions)
        
        for i in range(env.n_robots):
            x, y = obs[i][0], obs[i][1]
            st = (((y % 20) * 20 + (x % 20)) * 8 + int(obs[i][2]) * 4) % NB_STATES
            
            x2, y2 = next_obs[i][0], next_obs[i][1]
            st2 = (((y2 % 20) * 20 + (x2 % 20)) * 8 + int(next_obs[i][2]) * 4) % NB_STATES
            
            Q[i][st, actions[i]] += alpha * (rewards[i] + gamma * np.max(Q[i][st2]) - Q[i][st, actions[i]])
            
        obs = next_obs
        step += 1
        
    epsilon = max(0.01, epsilon - 0.01)

# Sauvegarde des tables Q
np.save("q_tables.npy", np.array(Q, dtype=object))
print("Entraînement terminé ! Fichier 'q_tables.npy' sauvegardé.")