#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

# Load data
data = pd.read_csv('Milano_timeseries.csv', header=None).values
num_servers = data.shape[1]
print(num_servers)

# Define environment class for expert case
class ExpertEnvironment:
    def __init__(self, data):
        self.data = data
    
    def get_loads(self):
        return self.data
    
    def get_least_loaded_server(self):
        return np.argmin(np.mean(self.data, axis=0))

# Define environment class for bandit case
class BanditEnvironment:
    def __init__(self, data):
        self.data = data
    
    def get_load(self, arm):
        return self.data[arm][-1]
    
    def get_least_loaded_server(self):
        return np.argmin(self.data[:, -1])

def MW_expert(T):
    env = ExpertEnvironment(data)
    num_servers = env.get_loads().shape[1]
    experts = np.arange(num_servers)
    weights = np.ones(num_servers)
    weights /= weights.sum()
    regrets = np.zeros(T)
    for t in range(T):
        if t == 0:
            eta = math.sqrt(math.log(num_servers))
        else:
            eta = math.sqrt(math.log(num_servers) / t)
        least_loaded_server = env.get_least_loaded_server()
        reward = 1 - np.mean(env.get_loads()[:, least_loaded_server])
        losses = np.zeros(num_servers)
        losses[experts != least_loaded_server] = 1
        weights *= np.exp(-eta * losses)
        weights /= weights.sum()
        regrets[t] = reward
    return np.cumsum(regrets)

def MW_bandit(T):
    env = BanditEnvironment(data)
    num_servers = env.data.shape[0]
    weights = np.ones(num_servers)
    weights /= weights.sum()
    regrets = np.zeros(T)
    for t in range(T):
        if t == 0:
            eta = math.sqrt(math.log(num_servers))
        else:
            eta = math.sqrt(math.log(num_servers) / t)
        p = weights.copy()
        p /= p.sum()
        arm = np.random.choice(num_servers, p=p)
        least_loaded_server = env.get_least_loaded_server()
        reward = 1 - env.get_load(arm)
        losses = np.zeros(num_servers)
        losses[arm] = 1
        weights *= np.exp(-eta * losses)
        weights /= weights.sum()
        regrets[t] = reward
    return np.cumsum(regrets)

def UCB(T):
    env = BanditEnvironment(data)
    num_servers = env.data.shape[0]
    Q = np.zeros(num_servers)
    N = np.zeros(num_servers)
    regrets = np.zeros(T)
    for t in range(T):
        if t < num_servers:
            arm = t
        else:
            ucb_values = Q + np.sqrt(2*np.log(t)/N)
            arm = np.argmax(ucb_values)
        least_loaded_server = env.get_least_loaded_server()
        loss = 1 - env.get_load(arm)
        reward = -loss
        Q[arm] = (Q[arm]*N[arm] + reward) / (N[arm] + 1)
        N[arm] += 1
        regrets[t] = loss
    return np.cumsum(regrets)





# Compare performance of MW algorithm in expert and bandit cases
T = [1000, 7000]
for t in T:
    regrets_expert = MW_expert(t)
    regrets_bandit = MW_bandit(t)
    regrets_ucb = UCB(t)

    plt.figure(figsize=(8, 6))
    plt.plot(regrets_expert, label='MW Expert')
    plt.plot(regrets_bandit, label='MW Bandit')
    plt.plot(regrets_ucb, label='UCB-Bandit')
    plt.legend()
    plt.title(f"T={t}")
    plt.xlabel('Time')
    plt.ylabel('Cumulative regret')
    plt.yscale('log')  # set y-axis to logarithmic scale
    plt.show()


