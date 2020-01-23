import argparse
import bandit_gen
import numpy as np
import random
import sys

from averagebandit import *

def simulate_bandit(bandit, data):
    num_epochs, cols = data.shape
    estimates = [ColumnEstimate() for col in range(cols)]
    rewards = []

    for epoch in range(num_epochs):
        ax = bandit.get_action()
        rx = data[epoch, ax]
        bandit.observe(rx)
        rewards.append(rx)

    return rewards


def simulate_averages(bandit, data=None, reps=1, epochs=1000):
    simulations = []
    generate = data is None
    for rep in range(reps):
        bandit.reset()
        if generate:
            data = bandit_gen.generate_data(10, epochs, 1)
        rewards = simulate_bandit(bandit, data)
        best_possible = np.max(data, axis=1)
        picked_best = np.equal(best_possible, rewards)
        simulations.append((rewards, picked_best))

    ave_rx = np.average(simulations, axis=0)
    return compute_cumulative_average(ave_rx)


def compute_cumulative_average(a):
    return np.divide(np.cumsum(a, axis=1), np.arange(1, a.shape[1] + 1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulates an n-armed bandit')
    parser.add_argument('-e', nargs='+', type=float, default=[0],
                        help='epsilon for random non-greediness (default=0)')
    parser.add_argument('-r', nargs='?', type=int, default=1,
                        help='how many repeated trials to simulate (default=0)')
    parser.add_argument('infile', nargs='?', type=argparse.FileType('r'),
                        default=sys.stdin, help='File containing N numerical '
                        'values per step. The number of values must be constant '
                        'across all rows.')
    argv = parser.parse_args()

    data = np.loadtxt(argv.infile, ndmin=2)
    if data.shape[1] == 0:
        exit('Each row must contain at least 1 value')

    sim_data = np.array([simulate_averages(AverageBandit(10, e=e), data, argv.r) for e in argv.e])[:,0,:]
    epoch_major = np.transpose(sim_data)
    print(epoch_major)

    np.savetxt(sys.stdout, epoch_major, fmt='%.6f')
    print(argv.e)

