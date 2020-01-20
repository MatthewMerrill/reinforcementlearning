import argparse
import bandit_gen
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
        return (self.value_sum / len(self.values)) if self.values else 0


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
    ax_rx = [(-1, 0)]

    for epoch in range(num_epochs):
        if random.random() < e:
            ax = random.randrange(cols)
        else:
            ax = best_lever_index(estimates)
        
        rx = data[epoch, ax]
        estimates[ax].add_value(rx)
        ax_rx.append((ax, rx))

    return ax_rx


def simulate_for_average_rx(data=None, e=0, reps=1):
    simulations = []
    generate = data is None
    for rep in range(reps):
        if generate:
            data = bandit_gen.generate_data(10, 1000, 1)
        simulations.append(np.array(simulate_bandit(data, e))[:,1])

    ave_rx = np.average(simulations, axis=0)
    return np.divide(np.cumsum(ave_rx), np.arange(1, len(ave_rx) + 1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulates an n-armed bandit')
    parser.add_argument('-e', nargs='+', type=float, default=0,
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

    np.savetxt(sys.stdout, np.transpose([simulate_for_average_rx(data, e, argv.r) for e in argv.e]), fmt='%.6f')
    print(argv.e)

