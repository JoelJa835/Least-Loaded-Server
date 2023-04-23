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
    eta=np.sqrt(np.log(n_servers)/T)
    for t in range(T):
        # pick a server with probability proportional to the weights
        choice = np.random.choice(n_servers, p=weights/np.sum(weights))
        # observe the loss of all servers
        losses = data.iloc[t,:].values
        # calculate the regret
        regret = losses[choice] -np.min(losses)
        regrets.append(regret)
        # update the weights of all servers
        weights = weights * np.exp(-eta* losses)
    return np.cumsum(regrets)



def mw_adversarial(dataset_path, T):
    data = pd.read_csv(dataset_path, header=None)
    data = data.T
    n_servers = data.shape[1]
    weights = np.ones(n_servers)
    regrets = []
    eta = np.sqrt(np.log(n_servers) / T)
    for t in range(T):
        # calculate the probability of selecting each action i
        pi = weights / np.sum(weights)
        # select an action i with probability proportional to pi
        choice = np.random.choice(n_servers, p=pi)
        # observe the loss of the chosen action
        loss = data.iloc[t, choice]
        # calculate the regret
        regret = np.min(data.iloc[t, :].values) - loss
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
    data = data.T
    k = data.shape[1] # Number of servers
    n = [0] * k # Number of times each server has been selected
    losses = [0] * k # Cumulative losses for each server
    est_means = [0] * k # Estimated mean loss for each server
    regrets = []
    for t in range(T):
        if t < k:
            # Select each server k times to initialize the estimates and UCB values
            loss = data.iloc[t, :].sum() # Sum of the loads of the servers for the selected row
            n[t] += 1
            losses[t] += loss
            est_means[t] = losses[t] / n[t]
            regrets.append(0)
        else:
            # Choose the server with the highest UCB value
            ucb_values = [est_means[i] + np.sqrt(2*np.log(t) / n[i]) for i in range(k)] # Calculate UCB values for each server
            server = np.argmax(ucb_values) # Select the server with the highest UCB value
            loss = data.iloc[t, server] # Select the load of the chosen server for the current time step
            n[server] += 1 # Increment the count of times the chosen server has been selected
            losses[server] += loss # Add the observed loss to the cumulative losses for the chosen server
            est_means[server] = losses[server] / n[server] # Update the estimated mean loss for the chosen server
            optimal_loss = data.iloc[t, :].min() # Find the minimum load among all servers for the current time step
            regret = optimal_loss - loss # Calculate regret for the chosen server
            regrets.append(np.abs(regret)) # Add the regret to the list of regrets
    return np.cumsum(regrets)






# Compare performance of MW algorithm in expert and bandit cases
T = [1000, 7000]
for t in T:
    regrets_expert = mw_expert("Milano_timeseries.csv",t)
    regrets_bandit = mw_adversarial("Milano_timeseries.csv",t)
    regrets_ucb = ucb_losses("Milano_timeseries.csv",t)

    plt.figure(figsize=(8, 6))
    # plt.plot(regrets_expert, label='MW Expert')
    # plt.plot(regrets_bandit, label='MW Bandit')
    # plt.plot(regrets_ucb, label='UCB')
    plt.semilogy(regrets_expert, label='MW Expert')
    plt.semilogy(regrets_bandit, label='MW Bandit')
    plt.semilogy(regrets_ucb, label='UCB')
    plt.legend()
    plt.title(f"T={t}")
    plt.xlabel('Time')
    plt.ylabel('Cumulative regret')
    #plt.yscale('log')  # set y-axis to logarithmic scale
    plt.show()


