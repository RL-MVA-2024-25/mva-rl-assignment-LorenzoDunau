from env_hiv import HIVPatient
from gymnasium.wrappers import TimeLimit
import copy
import getch
env = TimeLimit(
    env=HIVPatient(),max_episode_steps=200
)

obs, info = env.reset()
reward=0
R =0
for N in range(200):

    # Choisir une action basée sur la position du chariot (première valeur de l'observation)
    cart_position = obs[0]
    
    print("N:          ", N)
    print("T1* =       ", obs[1])
    print("T2* =       ", obs[3])
    print("V =         ", obs[4])

    print("T1* + T2* = ", obs[1]+obs[3])
    print("E =         ", obs[5], "\n\n")

    best_next = []
    for i in range(4):
        env_clone = copy.deepcopy(env)
        obs, reward, done, truncated, info = env_clone.step(i)
        best_next.append(obs[1]+obs[3])
    print(best_next)

    action = int(getch.getch())
    print(action)
    obs, reward, done, truncated, info = env.step(action)

    R += reward

    if done or truncated:
        obs, info = env.reset()

print("R = ", R)
env.close()