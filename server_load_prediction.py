#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


def mw_expert(dataset_path, T):
    data = pd.read_csv(dataset_path,header=None)
    data = data.T
    n_servers = data.shape[1]
    weights = np.ones(n_servers)
    regrets = []
    for t in range(T):
        if t == 0:
            eta= np.sqrt(np.log(n_servers))
        else:
            eta=np.sqrt(np.log(n_servers)/t)
        # pick a server with probability proportional to the weights
        choice = np.random.choice(n_servers, p=weights/np.sum(weights))
        # observe the loss of all servers
        losses = data.iloc[t,:].values
        # calculate the regret
        regret = np.max(losses) - losses[choice]
        regrets.append(regret)
        # update the weights of all servers
        weights = weights * np.exp(-eta* losses)
    return np.cumsum(regrets)



def mw_adversarial(dataset_path, T):
    data = pd.read_csv(dataset_path,header=None)
    data = data.T
    n_servers = data.shape[1]
    weights = np.ones(n_servers)
    regrets = []
    for t in range(T):
        if t == 0:
            eta= np.sqrt(np.log(n_servers))
            epsilon = 1
        else:
            eta= np.sqrt(np.log(n_servers)/t* n_servers)
            epsilon = np.sqrt(np.log(n_servers) / (t * n_servers))
        # pick a server with probability epsilon, and otherwise with probability proportional to the weights
        if np.random.random() < epsilon:
            choice = np.random.choice(n_servers)
        else:
            choice = np.random.choice(n_servers, p=weights/np.sum(weights))
        # observe the loss of the chosen server
        loss = data.iloc[t,choice]
        # calculate the regret
        regret = np.max(data.iloc[t,:].values) - loss
        regrets.append(regret)
        # update the weights of the chosen server using the MW update rule
        weights *= np.exp(-eta * loss)
        # renormalize the weights to ensure they sum to 1
        weights /= np.sum(weights)
    return np.cumsum(regrets)






# Compare performance of MW algorithm in expert and bandit cases
T = [1000, 7000]
for t in T:
    regrets_expert = mw_expert("Milano_timeseries.csv",t)
    regrets_bandit = mw_adversarial("Milano_timeseries.csv",t)
    #regrets_ucb = UCB(t)

    plt.figure(figsize=(8, 6))
    # plt.plot(regrets_expert, label='MW Expert')
    # plt.plot(regrets_bandit, label='MW Bandit')
    #plt.plot(regrets_ucb, label='UCB-Bandit')
    plt.semilogy(regrets_expert, label='MW Expert')
    plt.semilogy(regrets_bandit, label='MW Bandit')
    plt.legend()
    plt.title(f"T={t}")
    plt.xlabel('Time')
    plt.ylabel('Cumulative regret')
    #plt.yscale('log')  # set y-axis to logarithmic scale
    plt.show()


