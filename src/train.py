from gymnasium.wrappers import TimeLimit
import gymnasium as gym
from env_hiv import HIVPatient
from stable_baselines3 import DQN
import torch
import os
import numpy as np
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
    np.logspace(np.log10(1e-1), np.log10(1e6), 10),  # T1 (10 bins entre 0.1 et 1e6)
    np.logspace(np.log10(1), np.log10(5e4), 8),      # T1* (8 bins entre 1 et 5e4)
    np.logspace(np.log10(1), np.log10(3200), 6),     # T2 (6 bins entre 1 et 3200)
    np.logspace(np.log10(1), np.log10(80), 5),       # T2* (5 bins entre 1 et 80)
    np.logspace(np.log10(1), np.log10(2.5e5), 10),   # V (10 bins entre 1 et 2.5e5)
    np.logspace(np.log10(1), np.log10(353200), 7),   # E (7 bins entre 1 et 353200)
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
            learning_rate=1e-3,         # Taux d'apprentissage
            buffer_size=100000,         # Taille du buffer
            learning_starts=1000,       # Pas avant le début de l'apprentissage
            batch_size=64,              # Taille des batchs pour l'entraînement
            gamma=0.99,                 # Facteur de discount
            target_update_interval=500, # Fréquence de mise à jour du réseau cible
            train_freq=4,               # Fréquence d'entraînement
            gradient_steps=1,           # Nombre de mises à jour par pas d'entraînement
            policy_kwargs=policy_kwargs, # Architecture du réseau
            verbose=1,                  # Niveau de verbosité
            tensorboard_log="./dqn_hiv_logs",  # Log pour TensorBoard
            exploration_final_eps=0.05
        )

    def act(self, observation, use_random=False):
        # Prédire une action avec le modèle
        action, _ = self.model.predict(observation, deterministic=not use_random)
        return action

    def save(self, path):
        # Sauvegarder le modèle
        self.model.save(path)

    def load(self):
        # Charger le modèle
        self.model = DQN.load("dqn_hiv_patient.zip", env=env)

    def train(self):
        # Entraîner le modèle
        self.model.learn(total_timesteps=100000)


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

