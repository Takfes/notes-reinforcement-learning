import numpy as np
import matplotlib.pyplot as plt
import gym

env = gym.make('FrozenLake-v1') # is_slippery=False

n_games = 1_000
win_pct = []
scores = []

for i in range(n_games):
    done = False
    obs = env.reset()
    score = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info, prob = env.step(action)
        score += reward
    scores.append(score)
    if i % 10 == 0 :
        average = np.mean(scores[-10:])
        win_pct.append(average)
plt.plot(win_pct)