import argparse
import numpy as np
import random
import sys

class ColumnEstimate:
    def __init__(self):
        self.values = []
        self.value_sum = 0

    def add_value(self, value):
        self.values.append(value)
        self.value_sum += value

    def mean(self):
        return self.value_sum / len(self.values) if self.values else 999


def best_lever_index(estimates):
    best_levers = [0]
    best_mean = estimates[0].mean()

    for lever_idx, lever in enumerate(estimates):
        cur_mean = lever.mean()
        if best_mean < cur_mean:
            best_levers = [lever_idx]
            best_mean = cur_mean
        elif best_mean == cur_mean:
            best_levers.append(lever_idx)

    return random.choice(best_levers)


def simulate_bandit(data, e=0):
    num_epochs, cols = data.shape
    estimates = [ColumnEstimate() for col in range(cols)]
    ax_rx = []

    for epoch in range(num_epochs):
        if random.random() < e:
            ax = random.randrange(cols)
        else:
            ax = best_lever_index(estimates)
        
        rx = data[epoch, ax]
        estimates[ax].add_value(rx)
        ax_rx.append((ax, rx))

    return sum(rx for ax, rx in ax_rx), np.array(ax_rx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulates an n-armed bandit')
    parser.add_argument('-e', nargs='?', type=int, default=0,
                        help='epsilon for random non-greediness (default=0)')
    parser.add_argument('infile', nargs='?', type=argparse.FileType('r'),
                        default=sys.stdin, help='File containing N numerical '
                        'values per step. The number of values must be constant '
                        'across all rows.')
    argv = parser.parse_args()

    data = np.loadtxt(argv.infile, ndmin=2)
    if data.shape[1] == 0:
        exit('Each row must contain at least 1 value')

    score, ax_rx = simulate_bandit(data, argv.e)
    cumave = np.divide(np.cumsum(ax_rx[:,1]), np.arange(1, len(ax_rx) + 1))
    print(ax_rx)
    print(cumave)
