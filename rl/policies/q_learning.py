import numpy as np

class QLearningPolicy:
    def __init__(self, q_table):
        self.q_table = q_table

    def get_state_index(self, obs):
        """Convertit l'observation du dépôt (tuple) en index d'état pour la table Q."""
        x, y = obs[0], obs[1]
        carrying = int(obs[2])
        
        # Mappage sur la grille 20x20
        pos_idx = (y % 20) * 20 + (x % 20)
        state_idx = (pos_idx * 2 * 4) + (carrying * 4)
        return state_idx % self.q_table.shape[0]

    def act(self, obs) -> int:
        """Méthode requise par le dépôt : renvoie l'action choisie (0 à 5)."""
        state = self.get_state_index(obs)
        return int(np.argmax(self.q_table[state]))

    def reset(self):
        pass