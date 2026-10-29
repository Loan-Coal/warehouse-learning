import os
from isaacsim import SimulationApp

# Instanciation de SimulationApp avant tout autre import Omniverse
simulation_app = SimulationApp({"headless": False})

import torch
import numpy as np
from env.warehouse import Warehouse
from env.layout import maze_layout
from rl.policies.mappo_skrl import SKRLMAPPOPolicy

N_ROBOTS = 2

# Initialisation de l'environnement de simulation 3D
env = Warehouse(layout=maze_layout(42, n_robots=N_ROBOTS), n_robots=N_ROBOTS, max_steps=500)

# Chargement des politiques MAPPO entraînées
policies = []
for i in range(N_ROBOTS):
    weights_path = f"weights/skrl_mappo_robot_{i}.pt"
    if os.path.exists(weights_path):
        print(f"Chargement du modèle MAPPO robot {i} : {weights_path}")
        pol = SKRLMAPPOPolicy(weights_path=weights_path, device="cpu")
    else:
        print(f"Poids non trouvés pour robot {i}, politique par défaut.")
        pol = SKRLMAPPOPolicy(device="cpu")
    policies.append(pol)

# Réinitialisation de la scène
obs = env.reset(seed=100)
done = False
step_count = 0
total_deliveries = 0
total_invalid = 0

print("\n--- Début de la démonstration MAPPO ---")

while simulation_app.is_running() and not done:
    actions = []
    for i in range(N_ROBOTS):
        # Sélection de l'action optimale déterministe via la politique
        act = policies[i].act(obs[i])
        actions.append(act)

    # Exécution du pas de temps
    next_obs, rewards, done, info = env.step(actions)
    env.render()

    for i in range(N_ROBOTS):
        if rewards[i] >= 5.0:
            total_deliveries += 1
        elif rewards[i] == -1.0:
            total_invalid += 1

    obs = next_obs
    step_count += 1

print(f"\nÉpisode terminé : Pas = {step_count} | Livraisons = {total_deliveries} | Invalides = {total_invalid}")

simulation_app.close()