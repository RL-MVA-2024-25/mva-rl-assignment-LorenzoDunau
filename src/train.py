from gymnasium.wrappers import TimeLimit
import gymnasium as gym
from env_hiv import HIVPatient
from stable_baselines3 import DQN
import torch
import os
import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim
print(os.path.abspath("./dqn_hiv_logs/"))

print(torch.cuda.is_available()) 

# Limitation du nombre de pas dans un épisode
env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)

class DiscretizeObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, log_bins):
        """
        Wrapper pour discrétiser les observations d'un environnement Gym.

        Args:
            env (gym.Env): L'environnement à wrapper.
            bins (list of array-like): Une liste où chaque élément est un array
                                       définissant les bornes pour chaque dimension.
        """
        super().__init__(env)
        self.log_bins = log_bins
        self.observation_space = gym.spaces.MultiDiscrete([len(b) + 1 for b in log_bins])

    def observation(self, observation):
        """
        Transforme une observation continue en une observation discrète.

        Args:
            observation (np.ndarray): Observation continue.

        Returns:
            np.ndarray: Observation discrète.
        """
        discretized = [
            np.digitize(observation[i], self.log_bins[i]) for i in range(len(self.log_bins))
        ]
        return np.array(discretized, dtype=np.int32)

    def step(self, action):
        """
        Surcharge de la méthode step pour discrétiser l'observation retournée.

        Args:
            action: L'action à exécuter.

        Returns:
            tuple: Observation discrète, récompense, indicateurs de fin, et infos.
        """
        obs, reward, done, truncated, info = self.env.step(action)
        obs = self.observation(obs)  # Discrétiser l'observation
        return obs, reward, done, truncated, info


log_bins = [
    np.logspace(np.log10(1e5), np.log10(3e6), 12),  # T1 (10 bins entre 0.1 et 1e6) 12
    np.logspace(np.log10(1), np.log10(5e4), 10),      # T1* (8 bins entre 1 et 5e4) 10
    np.logspace(np.log10(1), np.log10(3200), 8),     # T2 (6 bins entre 1 et 3200) 8
    np.logspace(np.log10(1), np.log10(80), 7),       # T2* (5 bins entre 1 et 80) 7
    np.logspace(np.log10(1), np.log10(2.5e3), 20),   # V (10 bins entre 1 et 2.5e5) 12
    np.logspace(np.log10(1), np.log10(353200), 200),   # E (7 bins entre 1 et 353200) 9
]

env = DiscretizeObservationWrapper(env=env, log_bins=log_bins)

class ProjectAgent:
    def __init__(self):
        # Configurer l'architecture du réseau
        policy_kwargs = dict(
            net_arch=[128, 128, 128],  # [128, 128, 128]
        )

        # Créer et entraîner le modèle DQN
        self.model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=5e-4,         # Taux d'apprentissage 1e-3
            buffer_size=100000,         # Taille du buffer 100000
            learning_starts=1000,       # Pas avant le début de l'apprentissage
            batch_size=64,              # Taille des batchs pour l'entraînement
            gamma=0.99,                 # Facteur de discount
            target_update_interval=500, # Fréquence de mise à jour du réseau cible
            train_freq=4,               # Fréquence d'entraînement
            gradient_steps=1,           # Nombre de mises à jour par pas d'entraînement
            policy_kwargs=policy_kwargs, # Architecture du réseau
            verbose=1,                  # Niveau de verbosité
            tensorboard_log="./dqn_hiv_logs",  # Log pour TensorBoard
            exploration_final_eps=0.05 #0.05
        )

    def act(self, observation, use_random=False):
        # Prédire une action avec le modèle
        discretized = [
            np.digitize(observation[i], log_bins[i]) for i in range(len(log_bins))
        ]
        obs = np.array(discretized)
        action, _ = self.model.predict(obs, deterministic=not use_random)
        return action

    def save(self, path):
        # Sauvegarder le modèle
        self.model.save(path)

    def load(self):
        # Charger le modèle
        self.model = DQN.load("dqn_hiv_patient.zip", env=env)

    def train(self):
        # Entraîner le modèle
        self.model.learn(total_timesteps=3200000)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    


class CustomEnv(HIVPatient):
    def __init__(self, observation):
        super().__init__()
        self.T1, self.T1star, self.T2, self.T2star, self.V, self.E = observation  


def pref(x):
    if x > 82:
        if x > 284:
            return (x - 284) / 933
        else:
            return (284 - x) /202
    else:
        return 2

class ProjectAgent2:

    def act(self, observation, use_random=False):
        future_T1_T2 = []
        d = []
        for action in range(4):
            env = CustomEnv(observation=observation)
            obs, reward, done, truncated, info = env.step(action)
            future_T1_T2.append(obs[1]+ obs[3])
            d.append(pref(obs[1]+ obs[3]))
        print(future_T1_T2)
        if future_T1_T2[-1] > 1400:
            return 0
        for i in range(4):
            if future_T1_T2[3 - i] > 82:
                return 3 - i
        return 0
        


    def save(self, path):
        # Sauvegarder le modèle
        pass

    def load(self):
        # Charger le modèle
        pass



if __name__ == "__main__":
    p = ProjectAgent()
    try:
        p.load()  # Charger le modèle existant
        print("Model loaded successfully.")
        p.train()
        p.save("dqn_hiv_patient.zip")
    except FileNotFoundError:
        print("Model file not found. Training a new model.")
        p.train()
        p.save("dqn_hiv_patient.zip")

