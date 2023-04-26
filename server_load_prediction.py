#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


def mw_expert(dataset_path, T):
    data = pd.read_csv(dataset_path,header=None)
    n_servers = data.shape[0] # Number of servers
    weights = np.ones(n_servers)
    regrets = []
    eta=np.sqrt(np.log(n_servers)/T)
    for t in range(T):
        # pick a server with probability proportional to the weights
        choice = np.random.choice(n_servers, p=weights/np.sum(weights))
        # observe the loss of all servers
        losses = data.iloc[:,t].values
        # calculate the regret
        regret = losses[choice] -np.min(losses)
        regrets.append(regret)
        # update the weights of all servers 
        weights = weights * np.exp(-eta* losses)
    return np.cumsum(regrets)



def mw_adversarial(dataset_path, T):
    data = pd.read_csv(dataset_path, header=None)
    n_servers = data.shape[0] # Number of servers
    weights = np.ones(n_servers)
    regrets = []
    eta = np.sqrt(np.log(n_servers) / T)
    for t in range(T):
        # calculate the probability of selecting each action i
        pi = weights / np.sum(weights)
        # select an action i with probability proportional to pi
        choice = np.random.choice(n_servers, p=pi)
        # observe the loss of the chosen action
        loss = data.iloc[choice,t]
        # calculate the regret
        regret = np.min(data.iloc[:,t].values) - loss
        regrets.append(np.abs(regret))
        # update the weights of all actions using the MW update rule
        for i in range(n_servers):
            if i == choice:
                weights[i] *= np.exp(-eta * loss / pi[i])
            else:
                weights[i] *= np.exp(eta * loss / ((1 - pi[choice]) * (n_servers - 1)))
                
        # renormalize the weights to ensure they sum to 1
        weights /= np.sum(weights)
    return np.cumsum(regrets)


def ucb_losses(dataset_path, T):
    data = pd.read_csv(dataset_path, header=None)
    n_servers = data.shape[0] # Number of servers
    n = np.zeros(n_servers) # Number of times each server has been selected
    losses = np.zeros(n_servers) # Cumulative losses for each server
    est_means = np.zeros(n_servers) # Estimated mean loss for each server
    regrets = np.zeros(T) # Initialize regrets array

    # Initialize the estimates and UCB values randomly
    for i in range(n_servers):
        loss = data.iloc[:,i].sum()
        n[i] += 1
        losses[i] += loss
        est_means[i] = losses[i] / n[i]

    # Run the UCB algorithm for T time steps
    for t in range(n_servers, T):
        # Choose the server with the highest UCB value
        ucb_values = est_means + np.sqrt(2 * np.log(t) / (n + 1e-6))
        server = np.argmax(ucb_values)

        # Update the estimates and UCB values
        loss = data.iloc[server,t]
        n[server] += 1
        losses[server] -= loss
        est_means[server] = losses[server] / n[server]

        # Calculate and record the regret
        optimal_loss = data.iloc[:,t].min()
        regret = optimal_loss - loss
        regrets[t] = np.abs(regret)

    # Calculate and return the cumulative regrets
    return np.cumsum(regrets)






# Compare performance of MW algorithm in expert and bandit cases
T = [1000, 7000]
for t in T:
    regrets_expert = mw_expert("Milano_timeseries.csv",t)
    regrets_bandit = mw_adversarial("Milano_timeseries.csv",t)
    regrets_ucb = ucb_losses("Milano_timeseries.csv",t)

    plt.figure(figsize=(8, 6))
    plt.plot(regrets_expert, label='MW Expert')
    plt.plot(regrets_bandit, label='MW Bandit')
    plt.plot(regrets_ucb, label='UCB')
    # plt.semilogy(regrets_expert, label='MW Expert')
    # plt.semilogy(regrets_bandit, label='MW Bandit')
    # plt.semilogy(regrets_ucb, label='UCB')
    plt.legend()
    plt.title(f"T={t}")
    plt.xlabel('Time')
    plt.ylabel('Cumulative regret')
    plt.show()


